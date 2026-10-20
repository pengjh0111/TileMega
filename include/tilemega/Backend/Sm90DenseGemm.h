// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <cute/tensor.hpp>

namespace tilemega::backend {

struct Sm90DenseOperands {
  cutlass::bfloat16_t const* activation;
  cutlass::bfloat16_t const* weight;
  cutlass::bfloat16_t const* residual;
  cutlass::bfloat16_t* output;
  int* argmax_index;
  float const* norm_sums;
  float* output_sums;
};


template<int Batch,int N,int K,int TN,int TK,int Slots,int WarpsN,int WarpsK,
         ServingEpilogueOp Op,int U>
struct Sm90DenseGemm {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Batch>0 && Batch<=16 && N%TN==0 && TN%8==0 && TK%64==0 && K%TK==0);
  static_assert(WarpsN*WarpsK==4 && TN%(8*WarpsN)==0 && Slots>=2 && Slots<=8);
  static_assert(Op!=ServingEpilogueOp::kPartial,"dense decode never splits K across tasks");
  static constexpr int kThreads=160,kIterations=K/TK;
  static constexpr int kOperandBytes=2*Slots*TK*(16+TN);
  static constexpr int kSharedBytes=1024+kOperandBytes+4*WarpsK*16*TN+128;
  using Atom=decltype(cute::composition(cute::Swizzle<3,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::_64>,cute::Stride<cute::_64,cute::_1>>{}));
  using LayoutA=decltype(cute::tile_to_shape(Atom{},cute::Shape<cute::_16,cute::Int<TK>>{}));
  using LayoutB=decltype(cute::tile_to_shape(Atom{},cute::Shape<cute::Int<TN>,cute::Int<TK>>{}));
  using Mma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1,cute::Int<WarpsN>,cute::_1>>{},
      cute::Tile<cute::_16,cute::Int<TN>,cute::_16>{}));
  using CopyA=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  using CopyB=std::conditional_t<(TN/WarpsN<=16 || (TN/WarpsN)%16!=0),
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N,Element>,CopyA>;
  static_assert(cute::cosize_v<LayoutA> ==16*TK && cute::cosize_v<LayoutB> ==TN*TK);

  __device__ static __noinline__ void Produce(Sm90DenseOperands p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase) {
    int lane=threadIdx.x-128;
    for(int sequence=0;sequence<tasks*kIterations;++sequence) {
      int slot=sequence%Slots,task=int(blockIdx.x)+(sequence/kIterations)*int(gridDim.x);
      int tile=sequence%kIterations;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      if(lane==0) {
        Async::ExpectTx(state.full+slot,2*TN*TK);
        Async::Bulk(b+slot*TN*TK,p.weight+(std::size_t(task)*kIterations+tile)*TN*TK,
            2*TN*TK,state.full+slot);
      }
      for(int v=lane*8;v<16*TK;v+=32*8) {
        int row=v/TK,col=v%TK;
        auto* source=row<Batch?p.activation+std::size_t(row)*K+tile*TK+col:p.activation;
        Async::Copy16Bytes(a+slot*16*TK+LayoutA{}(row,col),source,row<Batch?16:0);
      }
      // One bulk arrival plus 32 cp.async arrivals make the full barrier cover
      // both operands, including predicated zero-fill copies for inactive rows.
      Async::CompleteCopies(state.full+slot);
      phase^=1u<<slot;
    }
    __syncwarp();
    if(lane==0 && tasks)for(int slot=0;slot<Slots;++slot)
      state.producer_generations[slot]=(phase>>slot)&1;
  }

  __device__ static __noinline__ void Consume(Sm90DenseOperands p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,int tasks,
      unsigned phase,float epsilon) {
    using namespace cute;
    int tid=threadIdx.x,lane=tid%(32*WarpsN),split=tid/(32*WarpsN);
    Mma mma;auto thread=mma.get_slice(lane);
    auto copy_a=make_tiled_copy_A(CopyA{},mma);
    auto copy_b=make_tiled_copy_B(CopyB{},mma);
    auto owned=thread.partition_C(make_identity_tensor(Shape<_16,Int<TN>>{}));
    for(int local=0;local<tasks;++local) {
      auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
      for(int iteration=0;iteration<kIterations;++iteration) {
        int sequence=local*kIterations+iteration,slot=sequence%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        auto sa=make_tensor(make_smem_ptr(a+slot*16*TK),LayoutA{});
        auto sb=make_tensor(make_smem_ptr(b+slot*TN*TK),LayoutB{});
        auto ra=thread.partition_fragment_A(sa);auto rb=thread.partition_fragment_B(sb);
        auto as=copy_a.get_slice(lane).partition_S(sa);auto bs=copy_b.get_slice(lane).partition_S(sb);
        auto ad=copy_a.get_slice(lane).retile_D(ra);auto bd=copy_b.get_slice(lane).retile_D(rb);
        for_each(make_int_sequence<size<2>(ra)>{},[&](auto k) {
          if(int(k)%WarpsK==split) {
            copy(CopyA{},as(_,_,k),ad(_,_,k));copy(CopyB{},bs(_,_,k),bd(_,_,k));
            gemm(mma,ra(_,_,k),rb(_,_,k),accum);
          }
        });
        // The empty barrier counts every reader; a separate CTA rendezvous
        // would serialize otherwise independent consumer warps on every slot.
        Async::Arrive(state.empty+slot);
      }
      for(int i=0;i<size(accum);++i)
        scratch[split*16*TN+get<0>(owned(i))*TN+get<1>(owned(i))]=accum(i);
      codegen::executor::ComputeSync();
      if constexpr(WarpsK>1) {
        for(int i=tid;i<16*TN;i+=128) {
          float sum=scratch[i];
          #pragma unroll
          for(int rank=1;rank<WarpsK;++rank)sum+=scratch[rank*16*TN+i];
          scratch[i]=sum;
        }
        codegen::executor::ComputeSync();
      }
      int task=int(blockIdx.x)+local*int(gridDim.x);
      constexpr int stride=Op==ServingEpilogueOp::kSwiGLU?N/2:
          Op==ServingEpilogueOp::kArgmaxPartial?N/TN:N;
      ServingEpilogue<Op,16,TN,U>::template RunFromTile<false>(scratch,0,task,Batch,N,
          stride,N,p.output,p.residual,nullptr,reinterpret_cast<float*>(p.output),
          p.argmax_index,p.norm_sums,p.output_sums,K,epsilon);
      codegen::executor::ComputeSync();
    }
    if(tid==0 && tasks)for(int slot=0;slot<Slots;++slot)
      state.consumer_generations[slot]=(phase>>slot)&1;
  }

  __device__ static void Run(Sm90DenseOperands p,char* shared,float epsilon) {
#if defined(__CUDA_ARCH__)
    static_assert(__CUDA_ARCH__==900,"this body requires Hopper sm_90a");
#endif
    auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*16*TK;
    auto* scratch=reinterpret_cast<float*>(shared+1024+kOperandBytes);
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    // Each role publishes only its own phases, after its readers have joined.
    // The previous stage's final CTA join makes the two snapshots consistent.
    int tasks=(N/TN)/int(gridDim.x)+(int(blockIdx.x)<(N/TN)%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,tasks,phase,epsilon);
    else Produce(p,state,a,b,tasks,phase);
    __syncthreads();
  }
};

}  // namespace tilemega::backend
