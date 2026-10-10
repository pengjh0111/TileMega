// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Backend/DmActivationScale.h>
#include <tilemega/Target/ArchDispatch.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <type_traits>

namespace tilemega::backend {

// DM's additional tiles keep four computing warps. The 16x16 case assigns
// two warps to N and two to K; its disjoint K partials meet only in scratch.
template<class Arch,int TM,int TN,int TK,int Stages>
struct ServingDmGemm {
  static_assert(arch::Caps<Arch>::kCpAsync && arch::Caps<Arch>::kBf16TensorCore);
  static_assert(solver::DmServingBF16ShapeLegal(TM,TN,TK,Stages));
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<Arch>;
  static constexpr int kThreads=128,kKSplits=TM==16 && TN==16?2:1;
  static constexpr int kSlots=Stages<8?Stages:8;
  static constexpr int kAtomK=TK<64?TK:64;
  static constexpr int kSwizzle=TK==16?1:TK==32?2:3;
  using SmemLayoutAtom=decltype(cute::composition(cute::Swizzle<kSwizzle,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::Int<kAtomK>>,
                   cute::Stride<cute::Int<kAtomK>,cute::_1>>{}));
  using LayoutA=decltype(cute::tile_to_shape(SmemLayoutAtom{},
      cute::Shape<cute::Int<TM>,cute::Int<TK>>{}));
  using LayoutB=decltype(cute::tile_to_shape(SmemLayoutAtom{},
      cute::Shape<cute::Int<TN>,cute::Int<TK>>{}));
  static_assert(cute::cosize_v<LayoutA> == TM*TK && cute::cosize_v<LayoutB> == TN*TK);
  using WarpLayout=std::conditional_t<TN==16,
      std::conditional_t<TM>=64,cute::Layout<cute::Shape<cute::_4,cute::_1,cute::_1>>,
          std::conditional_t<TM==32,cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
              cute::Layout<cute::Shape<cute::_1,cute::_2,cute::_2>>>>,
      std::conditional_t<TM==16,cute::Layout<cute::Shape<cute::_1,cute::_4,cute::_1>>,
          cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>>>;
  using ComputeWarpLayout=std::conditional_t<kKSplits==2,
      cute::Layout<cute::Shape<cute::_1,cute::_2,cute::_1>>,WarpLayout>;
  static constexpr int kMmaM=TN==16?(TM>=64?64:TM):(TM==16?16:32);
  static constexpr int kMmaN=TN==16?16:(TM==16?TN:32);
  using TiledMma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      WarpLayout{},cute::Tile<cute::Int<kMmaM>,cute::Int<kMmaN>,cute::Int<16*kKSplits>>{}));
  // With TK=16 each K warp consumes alternating 16-wide iterations. Partition
  // operands with the 64-thread N layout, avoiding a fictitious 32-wide tile.
  using ComputeMma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      ComputeWarpLayout{},cute::Tile<cute::Int<kMmaM>,cute::Int<kMmaN>,cute::_16>{}));
  using SmemCopyAtom=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  using SmemCopyAtomB=std::conditional_t<(TM==16 && TN<=32) || (TM==32 && TN==16),
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N,Element>,SmemCopyAtom>;
  static constexpr int kMainloopBytes=2*Stages*TK*(TM+TN);
  static constexpr int kReduceBytes=4*kKSplits*TM*TN;
  static constexpr int kEpilogueBytes=4*TM*TN+8*TM;
  static constexpr int kSharedBytes=kMainloopBytes>kReduceBytes?
      (kMainloopBytes>kEpilogueBytes?kMainloopBytes:kEpilogueBytes):
      (kReduceBytes>kEpilogueBytes?kReduceBytes:kEpilogueBytes);
  static_assert(kSharedBytes==solver::DmServingBF16SmemBytes(TM,TN,TK,Stages));

  template<class Operands>
  __device__ static float* Dense(Operands const& p,int tm,int tn,char* shared,
                                bool tiled_b=false) {
    using namespace cute;
    using codegen::executor::ComputeThread;
    using codegen::executor::ComputeSync;
    auto* a=reinterpret_cast<Element*>(shared);
    auto* b=a+Stages*TM*TK;
    int const a_pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    int const b_pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    if(!p.a || !p.b || p.k_begin<0 || p.k_count<=0 || p.k_begin%8 ||
       p.k_begin+p.k_count>p.k_total || a_pitch%8 || b_pitch%8 ||
       a_pitch<p.k_total || b_pitch<p.k_total ||
       (tiled_b && (!p.weight_base || p.k_total_full<=0 || p.k_begin%TK))) {
      asm volatile("trap;");return nullptr;
    }
    int const iterations=(p.k_count+TK-1)/TK;
    auto issue=[&](int it) {
      for(int v=ComputeThread()*8;v<TM*TK;v+=128*8) {
        int row=tm*TM+v/TK,col=it*TK+v%TK;
        bool valid=row<p.m && col<p.k_count;
        auto source_row=valid?DmSourceRow(p,row):0;
        auto* source=valid?p.a+std::size_t(source_row)*a_pitch+p.k_begin+col:p.a;
        Async::Copy16Bytes(a+(it%kSlots)*TM*TK+LayoutA{}(v/TK,v%TK),source,
            valid?min(8,p.k_count-col)*int(sizeof(Element)):0);
      }
      if(tiled_b) {
        int kt=(p.k_total_full+TK-1)/TK;
        auto* source=p.weight_base+(std::size_t(tn)*kt+p.k_begin/TK+it)*TN*TK;
        for(int v=ComputeThread()*8;v<TN*TK;v+=128*8)
          Async::Copy16Bytes(b+(it%kSlots)*TN*TK+v,source+v,16);
      }else for(int v=ComputeThread()*8;v<TN*TK;v+=128*8) {
        int row=tn*TN+v/TK,col=it*TK+v%TK;
        bool valid=row<p.n && col<p.k_count;
        auto* source=valid?p.b+std::size_t(row)*b_pitch+p.k_begin+col:p.b;
        Async::Copy16Bytes(b+(it%kSlots)*TN*TK+LayoutB{}(v/TK,v%TK),source,
            valid?min(8,p.k_count-col)*int(sizeof(Element)):0);
      }
    };
    if constexpr(HasDmActivationScale<Operands>::value) {
      if(p.a_scale)return Pipeline(issue,iterations,shared,DmScaledActivation<Operands,TM,TK>{p,tm});
    }
    return Pipeline(issue,iterations,shared);
  }
  // The callback issues copies; this loop commits exactly one group per stage.
  // Both dense and gathered operands share the register-pipelined MMA loop.
  template<class Issue,class Transform=DmUnscaledActivation>
  __device__ static float* Pipeline(Issue&& issue,int iterations,char* shared,Transform transform={}) {
    using namespace cute;
    using codegen::executor::ComputeThread;
    using codegen::executor::ComputeSync;
    auto* a=reinterpret_cast<Element*>(shared);
    auto* b=a+Stages*TM*TK;
    for(int it=0;it<kSlots-1;++it) {
      if(it<iterations)issue(it);
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    ComputeMma mma;
    int lane=ComputeThread()%(128/kKSplits),split=ComputeThread()/(128/kKSplits);
    auto thread=mma.get_slice(lane);
    auto coordinates_a=thread.partition_A(make_identity_tensor(Shape<Int<TM>,Int<TK>>{}));
    auto accum=partition_fragment_C(mma,Shape<Int<TM>,Int<TN>>{});clear(accum);
    auto ca=make_tiled_copy_A(SmemCopyAtom{},mma);
    auto cb=make_tiled_copy_B(SmemCopyAtomB{},mma);
    for(int it=0;it<iterations;++it) {
      // Once no future copy will be issued, empty groups must not stand in
      // for pending data. Drain the tail while keeping steady-state lookahead.
      if(it+kSlots-1>=iterations)asm volatile("cp.async.wait_all;" ::: "memory");
      else asm volatile("cp.async.wait_group %0;" :: "n"(kSlots-2):"memory");
      ComputeSync();
      int ahead=it+kSlots-1;
      if(ahead<iterations)issue(ahead);
      asm volatile("cp.async.commit_group;" ::: "memory");
      auto sa=make_tensor(make_smem_ptr(a+(it%kSlots)*TM*TK),LayoutA{});
      auto sb=make_tensor(make_smem_ptr(b+(it%kSlots)*TN*TK),LayoutB{});
      auto ra=thread.partition_fragment_A(sa);
      auto rb=thread.partition_fragment_B(sb);
      auto as=ca.get_slice(lane).partition_S(sa);
      auto bs=cb.get_slice(lane).partition_S(sb);
      auto ad=ca.get_slice(lane).retile_D(ra);
      auto bd=cb.get_slice(lane).retile_D(rb);
      copy(SmemCopyAtom{},as(_,_,0),ad(_,_,0));
      copy(SmemCopyAtomB{},bs(_,_,0),bd(_,_,0));
      for_each(make_int_sequence<size<2>(ra)>{},[&](auto k) {
        constexpr int next=decltype(k)::value+1;
        if constexpr(next<size<2>(ra)) {
          copy(SmemCopyAtom{},as(_,_,Int<next>{}),ad(_,_,Int<next>{}));
          copy(SmemCopyAtomB{},bs(_,_,Int<next>{}),bd(_,_,Int<next>{}));
        }
        transform(ra(_,_,k),coordinates_a(_,_,k),it);
        if constexpr(kKSplits==1)gemm(mma,ra(_,_,k),rb(_,_,k),accum);
        else if((it*(TK/16)+int(k))%kKSplits==split)
          gemm(mma,ra(_,_,k),rb(_,_,k),accum);
      });
    }
    asm volatile("cp.async.wait_all;" ::: "memory");ComputeSync();
    auto coordinates=make_identity_tensor(Shape<Int<TM>,Int<TN>>{});
    auto owned=thread.partition_C(coordinates);
    float* tile=reinterpret_cast<float*>(shared);
    for(int i=0;i<size(accum);++i)
      tile[split*TM*TN+get<0>(owned(i))*TN+get<1>(owned(i))]=accum(i);
    ComputeSync();
    if constexpr(kKSplits==2) {
      for(int i=ComputeThread();i<TM*TN;i+=128)tile[i]+=tile[TM*TN+i];
      ComputeSync();
    }
    return tile;
  }
};

} // namespace tilemega::backend
