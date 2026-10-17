// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ServingTaskProfile.h>

#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/Prefetch.cuh>

#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingGemm.h>
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Backend/Sm90DecodePipeline.h>
#ifndef TILEMEGA_ACTIVATION_TMA
#define TILEMEGA_ACTIVATION_TMA 0
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Backend/Sm90BulkGather.h>
#endif
#ifndef TILEMEGA_MOE_WEIGHT_BULK
#define TILEMEGA_MOE_WEIGHT_BULK 0
#endif
#ifndef TILEMEGA_MOE_ACTIVATION_BULK
#define TILEMEGA_MOE_ACTIVATION_BULK 0
#endif
#include <tilemega/Codegen/executor/Async.cuh>
#if TILEMEGA_TRACE_TASK
#include <tilemega/Backend/ServingProfiledMainloop.h>
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/tasks/DmEpilogueDispatch.cuh>
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <tilemega/Backend/ServingConv.h>
#include <tilemega/Backend/Sm90Conv.h>
#include <tilemega/Backend/ServingGemv.h>
#ifndef TILEMEGA_MOE_GEMV
#define TILEMEGA_MOE_GEMV 0
#endif
#endif

#ifndef TILEMEGA_ARCH_PATH_SM80
#define TILEMEGA_ARCH_PATH_SM80 0
#endif

#ifndef TILEMEGA_NONPAGED_TILED
#define TILEMEGA_NONPAGED_TILED 0
#endif
#if TILEMEGA_NONPAGED_TILED
#include <tilemega/Backend/ServingTiledMainloop.h>
#endif

#include <cute/tensor.hpp>
#include <type_traits>

namespace tilemega::codegen {

using codegen::executor::ComputeThread;
using codegen::executor::ComputeSync;
using codegen::executor::kComputeThreads;

struct ServingGemmOperands {
  cutlass::bfloat16_t const* a = nullptr;  // logical [M,K_total]
  cutlass::bfloat16_t const* b = nullptr;  // logical [N,K_total]
  cutlass::bfloat16_t const* residual = nullptr;
  cutlass::bfloat16_t* output = nullptr;
  float* partial = nullptr;
  float* argmax_value = nullptr;
  int* argmax_index = nullptr;
  int m = 0, n = 0, k_total = 0;
  int k_begin = 0, k_count = 0;
  int output_stride = 0;
  int partial_stride = 0;
  // Physical row pitches include zero-filled vector-alignment padding when
  // logical K has a residue.  Serving weights are packed once with this pitch.
  int a_row_stride = 0, b_row_stride = 0;
  cutlass::bfloat16_t const* weight_base = nullptr;
  int k_total_full = 0;
  float const* norm_ss = nullptr;
  float* ss_out = nullptr;
  int norm_k = 0;
  float norm_eps = 0.0f;
  backend::ServingEpilogueOp epilogue = backend::ServingEpilogueOp::kStore;
  void const* tensor_map = nullptr;
  int tensor_k_begin = 0;
#if TILEMEGA_TRACE_TASK
  ServingTaskProfile* profile=nullptr;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  DmGemmAccess access{};
  DmEpilogueChain chain{};
  ConvDesc const* convolutions = nullptr;
  backend::ConvIterationGeometry conv_iteration{};
  void const* binding = nullptr;
  void const* rows = nullptr;
  float const* a_scale = nullptr;
  DmBufferView dm_buffers{};
  backend::DmMoeRows moe{};
  bool dm_partial_rows = false;
#endif
  void const* activation_tensor_map = nullptr;
};

#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
template<class Spec,class=void> struct HasTypedDmAccess:std::false_type {};
template<class Spec> struct HasTypedDmAccess<Spec,std::void_t<decltype(Spec::kAAccess)>>:std::true_type {};
template<class Spec> struct TypedDmGemmAccess : DmGemmAccess {
  static constexpr auto a=Spec::kAAccess;
  static constexpr auto b=Spec::kBAccess;
  __device__ explicit TypedDmGemmAccess(DmGemmAccess const& original):DmGemmAccess(original) {}
};
template<class Spec> struct TypedDmGemmOperands : ServingGemmOperands {
  TypedDmGemmAccess<Spec> access;
  __device__ explicit TypedDmGemmOperands(ServingGemmOperands const& original):ServingGemmOperands(original),access(original.access) {}
};
__device__ inline backend::DmEpilogueArguments DmEpilogueOperands(
    ServingGemmOperands const& p) {
  backend::DmEpilogueArguments result;
  result.buffers = p.dm_buffers;
  result.chain = p.chain;
  result.write = p.access.write;
  result.output = p.output;
  result.m = p.m;
  result.n = p.n;
  result.output_stride = p.output_stride;
  result.norm_width = p.norm_k;
  result.norm_eps = p.norm_eps;
  result.image_rows = p.access.rows_per_batch;
  result.moe=p.moe;
  result.partial_rows=p.dm_partial_rows;
  result.routing_topk=p.access.routing_topk;
  return result;
}
#endif

template <class Arch, int TileM, int TileN, int TileK, int Stages,bool UseWgmma=false>
struct ServingGemmTaskBody {
  static_assert(!UseWgmma || !TILEMEGA_ARCH_PATH_SM80,
                "WGMMA is incompatible with --arch-paths sm80");
  static_assert(!(UseWgmma && TileM==8 && TILEMEGA_MOE_ACTIVATION_BULK),
                "M8 WGMMA does not support expert bulk gather");
  using Config = backend::ServingGemmConfig<Arch, (UseWgmma && TileM==8 ? 16 : TileM), TileN, TileK, Stages>;
  using Mainloop = typename Config::Mainloop;
  static constexpr int kThreads = Config::kThreads;
  static constexpr bool kBulkGather=UseWgmma && TileM==16 && TILEMEGA_MOE_ACTIVATION_BULK;
  static constexpr bool kDecodeTma=UseWgmma && (TileM==8 || TileM==16) && TILEMEGA_ACTIVATION_TMA;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  static constexpr int kSharedBytes=UseWgmma && (TileM==8 || TileM>=64 || kBulkGather || kDecodeTma || Stages>2) ?
      solver::ServingWgmmaSmemBytes(TileM,TileN,TileK,Stages,kBulkGather,kDecodeTma):
      solver::DmServingBF16SmemBytes(TileM,TileN,TileK,Stages);
#else
  static constexpr int kSharedBytes = UseWgmma && (TileM==8 || TileM>=64 || kBulkGather || kDecodeTma || Stages>2) ?
      solver::ServingWgmmaSmemBytes(TileM,TileN,TileK,Stages,kBulkGather,kDecodeTma):Config::kSharedBytes;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  static constexpr int kTileColumns = TileN;
#endif

  template<class Emit>
  __device__ static void PrefetchRanges(ServingGemmOperands const& p,int tile_n,Emit emit) {
#if TILEMEGA_NONPAGED_TILED
    int const kt=(p.k_total_full+TileK-1)/TileK;
    emit(executor::PrefetchRange{p.weight_base+
        (std::size_t(tile_n)*kt+p.k_begin/TileK)*TileN*TileK,
        static_cast<unsigned>(((p.k_count+TileK-1)/TileK)*TileN*TileK*2)});
#else
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    for(int n=tile_n*TileN;n<min(p.n,(tile_n+1)*TileN);++n)
      emit(executor::PrefetchRange{p.b+static_cast<long long>(n)*pitch+p.k_begin,
                                  static_cast<unsigned>(2*p.k_count)});
#endif
  }
  __device__ static __forceinline__ float* ComputeForwardWgmma(ServingGemmOperands const& p,int tile_m,
      int tile_n,char* shared) {
    using namespace cute;
    using Async=executor::Async<Arch>;
    using Body=backend::Sm90ForwardGmma<Arch,TileM,TileN,TileK>;
    using Gmma=typename Body::Gmma;
    using Element=cutlass::bfloat16_t;
    auto layout_a=as_position_independent_swizzle_layout(backend::Sm90KLayout<TileM,TileK>{});
    auto layout_b=as_position_independent_swizzle_layout(backend::Sm90KLayout<TileN,TileK>{});
    // Scratch aliases the preceding task's FP32 epilogue; join its readers
    // before either producer starts writing the new operand slots.
    ComputeSync();
    auto* barriers=reinterpret_cast<std::uint64_t*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+2*TileM*TileK;
    int a_pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    int b_pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    int iterations=(p.k_count+TileK-1)/TileK;
    if(!p.a || !p.b || p.k_begin<0 || p.k_begin%8 || p.k_count<=0 ||
       p.k_begin+p.k_count>p.k_total || a_pitch<p.k_total || b_pitch<p.k_total ||
       a_pitch%8 || b_pitch%8 || (Async::Shared(shared)&1023)) {
      asm volatile("trap;");return nullptr;
    }
    bool tma=p.activation_tensor_map &&
#if TILEMEGA_NONPAGED_TILED
        p.weight_base && p.k_begin%TileK==0 &&
#else
        p.tensor_map &&
#endif
        (p.k_count%TileK==0 || p.tensor_k_begin+p.k_count==p.k_total_full);
    if(tma) {
      if(ComputeThread()==0) {Async::Init(barriers,1);Async::Init(barriers+1,1);}
      Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
    }
    auto issue=[&](int it) {
      auto* as=a+(it%2)*TileM*TileK;
      auto* bs=b+(it%2)*TileN*TileK;
      if(tma) {
        if(ComputeThread()==0) {
          auto* barrier=barriers+it%2;
          Async::ExpectTx(barrier,2*TileK*(TileM+TileN));
          Async::Tensor2D(as,p.activation_tensor_map,p.tensor_k_begin+it*TileK,tile_m*TileM,barrier);
#if TILEMEGA_NONPAGED_TILED
          int kt=(p.k_total_full+TileK-1)/TileK;
          auto* source=p.weight_base+(std::size_t(tile_n)*kt+p.k_begin/TileK+it)*TileN*TileK;
          Async::Bulk(bs,source,2*TileN*TileK,barrier);
#else
          Async::Tensor2D(bs,p.tensor_map,p.tensor_k_begin+it*TileK,tile_n*TileN,barrier);
#endif
        }
      }else {
        for(int v=ComputeThread()*8;v<TileM*TileK;v+=kComputeThreads*8) {
          int row=tile_m*TileM+v/TileK,k=it*TileK+v%TileK;
          bool valid=row<p.m && k<p.k_count;
          auto* source=valid?p.a+std::int64_t(row)*a_pitch+p.k_begin+k:p.a;
          Async::Copy16Bytes(as+layout_a(v/TileK,v%TileK),source,valid?min(8,p.k_count-k)*2:0);
        }
#if TILEMEGA_NONPAGED_TILED
        if(!p.weight_base || p.k_begin%TileK) {asm volatile("trap;");return;}
        int kt=(p.k_total_full+TileK-1)/TileK;
        auto* source=p.weight_base+(std::size_t(tile_n)*kt+p.k_begin/TileK+it)*TileN*TileK;
        for(int v=ComputeThread()*8;v<TileN*TileK;v+=kComputeThreads*8)Async::Copy16Bytes(bs+v,source+v,16);
#else
        for(int v=ComputeThread()*8;v<TileN*TileK;v+=kComputeThreads*8) {
          int row=tile_n*TileN+v/TileK,k=it*TileK+v%TileK;
          bool valid=row<p.n && k<p.k_count;
          auto* source=valid?p.b+std::int64_t(row)*b_pitch+p.k_begin+k:p.b;
          Async::Copy16Bytes(bs+layout_b(v/TileK,v%TileK),source,valid?min(8,p.k_count-k)*2:0);
        }
#endif
        cute::cp_async_fence();
      }
    };
    issue(0);if(iterations>1)issue(1);
    typename Body::Mma mma;
    auto accum=partition_fragment_C(mma,Shape<Int<TileM>,Int<TileN>>{});
    clear(accum);Gmma::Fence(accum);
    for(int it=0;it<iterations;++it) {
      if(tma)Async::Wait(barriers+it%2,(it/2)&1);
      else {cute::cp_async_wait<0>();ComputeSync();Async::FenceProxyAsyncShared();}
      ComputeSync();Body::Accumulate(b+(it%2)*TileN*TileK,a+(it%2)*TileM*TileK,accum);
      // The producer may overwrite a slot only after every WGMMA read ends.
      Gmma::template Wait<0>();ComputeSync();
      if(it+2<iterations)issue(it+2);
    }
    return Body::Materialize(accum,shared);
  }
  // Stages>2 keeps Stages-1 weight tiles in flight across WGMMA. The two-slot
  // loops below issue one tile per wgmma wait: verified on H800 expert tasks,
  // a 256 KiB TN64 weight slice streamed at ~5 GB/s per SM (52 us).
  template<class Operands>
  __device__ static __forceinline__ float* ComputePipelinedWgmma(Operands const& p,int tile_m,
      int tile_n,char* shared) {
    using namespace cute;
    using Async=executor::Async<Arch>;
    using Body=backend::Sm90DecodeGmma<Arch,TileM,TileN,TileK>;
    using Gmma=typename Body::Gmma;
    using Element=cutlass::bfloat16_t;
    constexpr int S=Stages;
    static_assert(S>2 && S*8<=1024,"pipelined decode WGMMA barriers exceed the header");
    auto layout_a=as_position_independent_swizzle_layout(backend::Sm90KLayout<TileM,TileK>{});
    int a_pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    int iterations=(p.k_count+TileK-1)/TileK;
    if(!p.a || !p.b || p.k_begin<0 || p.k_begin%8 || p.k_count<=0 ||
       p.k_begin+p.k_count>p.k_total || a_pitch<p.k_total || a_pitch%8 ||
       p.k_begin+((p.k_count+7)&~7)>a_pitch || (Async::Shared(shared)&1023)) {
      asm volatile("trap;");return nullptr;
    }
#if TILEMEGA_NONPAGED_TILED
    if(!p.weight_base || p.k_total_full<=0 || p.k_begin%TileK) {
      asm volatile("trap;");return nullptr;
    }
    int kt=(p.k_total_full+TileK-1)/TileK;
#else
    auto layout_b=as_position_independent_swizzle_layout(backend::Sm90KLayout<TileN,TileK>{});
    int b_pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    if(b_pitch<p.k_total || b_pitch%8 || p.k_begin+((p.k_count+7)&~7)>b_pitch) {
      asm volatile("trap;");return nullptr;
    }
#endif
    // Scratch aliases the preceding task's FP32 epilogue.
    ComputeSync();
    auto* barriers=reinterpret_cast<std::uint64_t*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+S*TileM*TileK;
#if TILEMEGA_NONPAGED_TILED
    if(ComputeThread()==0)for(int slot=0;slot<S;++slot)Async::Init(barriers+slot,1);
    Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
#endif
    // Every call commits one cp.async group and group g carries tile g, so
    // wait_group<S-2> at iteration it covers tile it (S+it-1 groups committed).
    auto issue=[&](int it) {
      if(it<iterations) {
        auto* as=a+(it%S)*TileM*TileK;
        auto* bs=b+(it%S)*TileN*TileK;
        for(int v=ComputeThread()*8;v<TileM*TileK;v+=kComputeThreads*8) {
          int row=tile_m*TileM+v/TileK,k=it*TileK+v%TileK;
          bool valid=row<p.m && k<p.k_count;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
          int source_row=valid?backend::DmSourceRow(p,row):0;
#else
          int source_row=row;
#endif
          auto* source=valid?p.a+std::int64_t(source_row)*a_pitch+p.k_begin+k:p.a;
          Async::Copy16Bytes(as+layout_a(v/TileK,v%TileK),source,valid?min(8,p.k_count-k)*2:0);
        }
#if TILEMEGA_NONPAGED_TILED
        if(ComputeThread()==0) {
          auto* barrier=barriers+it%S;
          Async::ExpectTx(barrier,2*TileN*TileK);
          Async::Bulk(bs,p.weight_base+(std::size_t(tile_n)*kt+p.k_begin/TileK+it)*TileN*TileK,
                      2*TileN*TileK,barrier);
        }
#else
        for(int v=ComputeThread()*8;v<TileN*TileK;v+=kComputeThreads*8) {
          int row=tile_n*TileN+v/TileK,k=it*TileK+v%TileK;
          bool valid=row<p.n && k<p.k_count;
          auto* source=valid?p.b+std::int64_t(row)*b_pitch+p.k_begin+k:p.b;
          Async::Copy16Bytes(bs+layout_b(v/TileK,v%TileK),source,valid?min(8,p.k_count-k)*2:0);
        }
#endif
      }
      cute::cp_async_fence();
    };
    for(int slot=0;slot<S;++slot)issue(slot);
    typename Body::Mma mma;
    auto accum=partition_fragment_C(mma,Shape<Int<TileN>,Int<TileM>>{});
    clear(accum);Gmma::Fence(accum);
    for(int it=0;it<iterations;++it) {
      cute::cp_async_wait<S-2>();ComputeSync();
      Async::FenceProxyAsyncShared();
#if TILEMEGA_NONPAGED_TILED
      Async::Wait(barriers+it%S,(it/S)&1);
#endif
      ComputeSync();
      Body::Accumulate(b+(it%S)*TileN*TileK,a+(it%S)*TileM*TileK,accum);
      // wait_group<1> retires iteration it-1, whose slot is refilled next.
      Gmma::template Wait<1>();ComputeSync();
      if(it>0)issue(it-1+S);
    }
    return Body::Materialize(accum,shared);
  }
  template<class Operands>
  __device__ static __forceinline__ float* ComputeWgmma(Operands const& p,int tile_m,
      int tile_n,char* shared) {
    if constexpr(TileM>=64)return ComputeForwardWgmma(p,tile_m,tile_n,shared);
    else if constexpr(Stages>2)return ComputePipelinedWgmma(p,tile_m,tile_n,shared);
    else {
    if constexpr(kDecodeTma) {
      bool dense=true;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
      dense=p.access.a==DmAAccess::kDense && p.access.b==DmBAccess::kDense && !p.a_scale;
#endif
      if(dense)return backend::Sm90DecodePipeline<Arch,TileM,TileN,TileK>::Run(p,tile_m,tile_n,shared);
    }
    using namespace cute;
    using Async=executor::Async<Arch>;
    using Body=backend::Sm90DecodeGmma<Arch,TileM,TileN,TileK>;
    using Gmma=typename Body::Gmma;
    using Element=cutlass::bfloat16_t;
    using LayoutA=backend::Sm90KLayout<TileM,TileK>;
    using LayoutB=backend::Sm90KLayout<TileN,TileK>;
    auto layout_a=as_position_independent_swizzle_layout(LayoutA{});
    auto layout_b=as_position_independent_swizzle_layout(LayoutB{});
    if constexpr(kBulkGather)ComputeSync();
    auto* a=reinterpret_cast<Element*>(shared+(kBulkGather?1024:0));
    auto* b=a+2*TileM*TileK;
    typename Body::Mma mma;
    auto accum=partition_fragment_C(mma,Shape<Int<TileN>,Int<TileM>>{});
    clear(accum);Gmma::Fence(accum);
    int a_pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    int b_pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    int iterations=(p.k_count+TileK-1)/TileK;
    if(!p.a || !p.b || p.k_begin<0 || p.k_begin%8 || p.k_count<=0 ||
       p.k_begin+p.k_count>p.k_total || a_pitch<p.k_total || b_pitch<p.k_total ||
       a_pitch%8 || b_pitch%8 || p.k_begin+((p.k_count+7)&~7)>a_pitch ||
       p.k_begin+((p.k_count+7)&~7)>b_pitch) {
      asm volatile("trap;");return nullptr;
    }
    for(int it=0;it<iterations;++it) {
      auto* as=a+(it%2)*TileM*TileK;
      auto* bs=b+(it%2)*TileN*TileK;
      bool gathered=false;
#if TILEMEGA_MOE_WEIGHT_BULK
      bool weight_loaded=false,weight_wait=false;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
      if constexpr(kBulkGather) {
#if TILEMEGA_MOE_WEIGHT_BULK
        static_assert(TILEMEGA_NONPAGED_TILED, "expert weight bulk requires packed weights");
        if(p.access.b==DmBAccess::kExpertIndirect) {
          if(!p.weight_base || p.k_total_full<=0 || p.k_begin%TileK ||
             p.k_begin+it*TileK>=p.k_total_full) {
            asm volatile("trap;");return nullptr;
          }
          int kt=(p.k_total_full+TileK-1)/TileK;
          auto* source=p.weight_base+(std::size_t(tile_n)*kt+p.k_begin/TileK+it)*TileN*TileK;
          if(p.access.a==DmAAccess::kRowGather && (it+1)*TileK<=p.k_count)
            gathered=backend::Sm90BulkGather<Arch,TileM,TileK>::template Load<true>(
                p,tile_m,it,as,b+2*TileN*TileK,reinterpret_cast<std::uint64_t*>(shared),
                bs,source,TileN*TileK*sizeof(Element));
          // Dense expert intermediates and masked A tails still own a packed
          // B tile. Its DMA overlaps the original cp.async A path.
          if(!gathered) {
            backend::Sm90BulkGather<Arch,TileM,TileK>::BeginWeight(
                bs,source,TileN*TileK*sizeof(Element),reinterpret_cast<std::uint64_t*>(shared));
            weight_wait=true;
          }
          weight_loaded=true;
        }else
#endif
          gathered=backend::Sm90BulkGather<Arch,TileM,TileK>::Load(
              p,tile_m,it,as,b+2*TileN*TileK,reinterpret_cast<std::uint64_t*>(shared));
      }
#endif
      if(!gathered)for(int v=ComputeThread()*8;v<TileM*TileK;v+=kComputeThreads*8) {
        int row=tile_m*TileM+v/TileK,k=it*TileK+v%TileK;
        bool valid=row<p.m && k<p.k_count;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
        int source_row=valid?backend::DmSourceRow(p,row):0;
#else
        int source_row=row;
#endif
        auto* source=valid?p.a+std::int64_t(source_row)*a_pitch+p.k_begin+k:p.a;
        Async::Copy16Bytes(as+layout_a(v/TileK,v%TileK),source,
            valid?min(8,p.k_count-k)*2:0);
      }
#if TILEMEGA_NONPAGED_TILED
      if(!p.weight_base || p.k_total_full<=0 || p.k_begin%TileK) {
        asm volatile("trap;");return nullptr;
      }
      int kt=(p.k_total_full+TileK-1)/TileK;
      auto* source=p.weight_base+(std::size_t(tile_n)*kt+p.k_begin/TileK+it)*TileN*TileK;
#if TILEMEGA_MOE_WEIGHT_BULK && defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
      if(!weight_loaded)
#endif
      for(int v=ComputeThread()*8;v<TileN*TileK;v+=kComputeThreads*8)
        Async::Copy16Bytes(bs+v,source+v,16);
#else
      for(int v=ComputeThread()*8;v<TileN*TileK;v+=kComputeThreads*8) {
        int row=tile_n*TileN+v/TileK,k=it*TileK+v%TileK;
        bool valid=row<p.n && k<p.k_count;
        auto* source=valid?p.b+std::int64_t(row)*b_pitch+p.k_begin+k:p.b;
        Async::Copy16Bytes(bs+layout_b(v/TileK,v%TileK),source,
            valid?min(8,p.k_count-k)*2:0);
      }
#endif
      cute::cp_async_fence();cute::cp_async_wait<0>();ComputeSync();
#if TILEMEGA_MOE_WEIGHT_BULK && defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
      if(weight_wait)backend::Sm90BulkGather<Arch,TileM,TileK>::WaitWeight(
          reinterpret_cast<std::uint64_t*>(shared));
#endif
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      ComputeSync();Body::Accumulate(bs,as,accum);Gmma::template Wait<1>();
    }
    return Body::Materialize(accum,shared);
    }
  }
#if TILEMEGA_DM_WGMMA_INLINE
  __device__ static __forceinline__
#else
  __device__ static TILEMEGA_HOPPER_TASK_INLINE
#endif
  void Run(ServingGemmOperands const& p, int tile_m,
                             int tile_n, char* shared) {
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    if(p.access.a==DmAAccess::kIm2Col || p.access.a==DmAAccess::kRowGather ||
       p.access.b==DmBAccess::kExpertIndirect || p.a_scale) {
      if(p.epilogue!=backend::ServingEpilogueOp::kPartial || !p.partial) {
        asm volatile("trap;");return;
      }
      auto partial=p;
      partial.output=reinterpret_cast<cutlass::bfloat16_t*>(p.partial);
      partial.output_stride=p.partial_stride;
      partial.chain={};partial.chain.store_rounding=DmRounding::kFP32;
      partial.access.write={};
      partial.dm_partial_rows=true;
      using Spec=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kDense,1,DmRounding::kFP32>;
      RunDm<Spec>(partial,tile_m,tile_n,shared);
      return;
    }
    if constexpr(TileN==16 || TileK<64) {
      auto* tile=Config::Dense(p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
      auto finish=[&](auto op) {
        backend::ServingEpilogue<decltype(op)::value,TileM,TileN>::template RunFromTile<false>(
            tile,tile_m,tile_n,p.m,p.n,p.output_stride,p.partial_stride,
            p.output,p.residual,p.partial,p.argmax_value,p.argmax_index,
            p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
      };
      switch(p.epilogue) {
        case backend::ServingEpilogueOp::kStore:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kStore>{});break;
        case backend::ServingEpilogueOp::kResidual:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kResidual>{});break;
        case backend::ServingEpilogueOp::kSwiGLU:
          if constexpr(TileN%32==0)finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kSwiGLU>{});
          else asm volatile("trap;");break;
        case backend::ServingEpilogueOp::kArgmaxPartial:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kArgmaxPartial>{});break;
        case backend::ServingEpilogueOp::kPartial:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kPartial>{});break;
      }
    }else {
#endif
    if constexpr(UseWgmma) {
      auto* tile=ComputeWgmma(p,tile_m,tile_n,shared);
      auto finish=[&](auto op) {
        backend::ServingEpilogue<decltype(op)::value,TileM,TileN>::template RunFromTile<false>(
            tile,tile_m,tile_n,p.m,p.n,p.output_stride,p.partial_stride,p.output,
            p.residual,p.partial,p.argmax_value,p.argmax_index,p.norm_ss,p.ss_out,
            p.norm_k,p.norm_eps);
      };
      switch(p.epilogue) {
        case backend::ServingEpilogueOp::kStore:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kStore>{});break;
        case backend::ServingEpilogueOp::kResidual:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kResidual>{});break;
        case backend::ServingEpilogueOp::kSwiGLU:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kSwiGLU>{});break;
        case backend::ServingEpilogueOp::kArgmaxPartial:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kArgmaxPartial>{});break;
        case backend::ServingEpilogueOp::kPartial:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kPartial>{});break;
      }
      ComputeSync();
      return;
    }
    using namespace cute;
    if (!p.a || !p.b || p.k_begin < 0 || p.k_count <= 0 ||
        p.k_begin + p.k_count > p.k_total) {
      asm volatile("trap;");
      return;
    }
    int a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    int copy_k_count = (p.k_count + 7) & ~7;
    if (a_pitch < p.k_total || b_pitch < p.k_total ||
        a_pitch % 8 != 0 || b_pitch % 8 != 0 || p.k_begin % 8 != 0 ||
        p.k_begin + copy_k_count > a_pitch ||
        p.k_begin + copy_k_count > b_pitch) {
      asm volatile("trap;");
      return;
    }
    constexpr auto tile_shape = typename Mainloop::TileShape{};
    typename Mainloop::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape));
    clear(accum);
#if TILEMEGA_NONPAGED_TILED
    backend::ServingTiledMainloop<Arch,Config,TileM,TileN,TileK,Stages>::Run(
        p,tile_m,tile_n,shared,accum);
#else
    auto dA = make_stride(int64_t(a_pitch), _1{},
                          int64_t(p.m) * a_pitch);
    auto dB = make_stride(int64_t(b_pitch), _1{},
                          int64_t(p.n) * b_pitch);
    auto a = make_tensor(make_gmem_ptr(p.a + p.k_begin),
                         make_shape(p.m, copy_k_count, 1), dA);
    auto b = make_tensor(make_gmem_ptr(p.b + p.k_begin),
                         make_shape(p.n, copy_k_count, 1), dB);
    auto coordinate = make_coord(tile_m, tile_n, _, 0);
    auto gA = local_tile(a(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<_1, X, _1>{});
    auto gB = local_tile(b(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<X, _1, _1>{});
    auto residue = make_tuple(p.m - size<0>(gA) * tile_m,
                              p.n - size<0>(gB) * tile_n,
                              copy_k_count - size<1>(gA) * size<2>(gA));
    auto k_iter = make_coord_iterator(shape<2>(gA));
#if TILEMEGA_TRACE_TASK
    cutlass::gemm::collective::ProfiledServingMainloop<Mainloop>{}(accum, gA,
#else
    Mainloop{}(accum, gA,
#endif
               gB, accum, k_iter, size<2>(gA), residue,
               ComputeThread(), shared
#if TILEMEGA_TRACE_TASK
               , FirstTileReadyProfile{p.profile}
#endif
               );
#endif
#if TILEMEGA_TRACE_TASK
    auto epilogue_begin=TaskProfileNow(p.profile);
#endif
    auto finish = [&](auto op) {
      backend::ServingEpilogue<decltype(op)::value, TileM, TileN>::Run(
          accum, mma, shared, tile_m, tile_n, p.m, p.n,
          p.output_stride, p.partial_stride, p.output, p.residual, p.partial,
          p.argmax_value, p.argmax_index,p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
    };
    switch (p.epilogue) {
      case backend::ServingEpilogueOp::kStore:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kStore>{}); break;
      case backend::ServingEpilogueOp::kResidual:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kResidual>{}); break;
      case backend::ServingEpilogueOp::kSwiGLU:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kSwiGLU>{}); break;
      case backend::ServingEpilogueOp::kArgmaxPartial:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kArgmaxPartial>{}); break;
      case backend::ServingEpilogueOp::kPartial:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kPartial>{}); break;
    }
#if TILEMEGA_TRACE_TASK
    if(p.profile)p.profile->epilogue_ns=TaskProfileNow(p.profile)-epilogue_begin;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    }
#endif
  }
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  template <class Spec>
#if TILEMEGA_DM_WGMMA_INLINE
  __device__ static __forceinline__
#else
  __device__ static
#endif
  void RunDm(ServingGemmOperands const& p, int tile_m,
                             int tile_n, char* shared) {
    auto resolved=p;
    if(!backend::ResolveDmMoeTile(resolved,tile_m,TileM))return;
    RunDmResolved<Spec>(resolved,tile_m,tile_n,shared);
  }
  template <class Spec>
#if TILEMEGA_DM_WGMMA_INLINE
  __device__ static __forceinline__
#else
  __device__ static
#endif
  void RunDmResolved(ServingGemmOperands const& original, int tile_m,
                                     int tile_n, char* shared) {
    if constexpr(HasTypedDmAccess<Spec>::value) if(original.access.a!=Spec::kAAccess || original.access.b!=Spec::kBAccess) {
      asm volatile("trap;");return;
    }
    using Typed=std::conditional_t<HasTypedDmAccess<Spec>::value,TypedDmGemmOperands<Spec>,ServingGemmOperands>;
    using View=std::conditional_t<HasTypedDmAccess<Spec>::value,Typed,ServingGemmOperands const&>;
    View typed(original);
    auto const& p=typed;
    if constexpr(UseWgmma) {
      if(p.a_scale) {asm volatile("trap;");return;}
      float* tile=nullptr;
      if(p.access.a==DmAAccess::kIm2Col) {
#if TILEMEGA_CONV_TMA_IM2COL
        if constexpr(TileM==64 && TileK==64) {
          if(!p.convolutions || p.access.conv==kDmNoIndex || !p.dm_buffers.layouts) {
            asm volatile("trap;");return;
          }
          auto const& conv=p.convolutions[p.access.conv];
          if(conv.input_layout>=p.dm_buffers.count){asm volatile("trap;");return;}
          tile=backend::Sm90Conv<Arch,TileM,TileN,TileK>::template Run<TILEMEGA_NONPAGED_TILED!=0>(
              p,conv,p.dm_buffers.layouts[conv.input_layout],tile_m,tile_n,shared);
        }else {asm volatile("trap;");return;}
#else
        asm volatile("trap;");return;
#endif
      }else tile=ComputeWgmma(p,tile_m,tile_n,shared);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
      ComputeSync();
      return;
    }
#if TILEMEGA_MOE_GEMV
    if constexpr(TileN<=32 && TileK>=32) {
      using Gemv=backend::ServingGemv<Arch,TileM,TileN,TileK>;
      if(Gemv::Supported(p,tile_m)) {
        auto* tile=Gemv::Dense(p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
        backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
            tile,DmEpilogueOperands(p),tile_m,tile_n);
        return;
      }
    }
#endif
    if((p.a_scale && p.access.a==DmAAccess::kDense) ||
       p.access.a==DmAAccess::kRowGather || p.access.b==DmBAccess::kExpertIndirect) {
      auto* tile=backend::ServingDmGemm<Arch,TileM,TileN,TileK,Stages>::Dense(
          p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
      return;
    }
    if(p.access.a==DmAAccess::kIm2Col) {
      if(!p.convolutions || p.access.conv==kDmNoIndex ||
         p.k_begin%TileK || p.k_count%TileK || !p.conv_iteration.iterations) {
        asm volatile("trap;");return;
      }
      auto const& conv=p.convolutions[p.access.conv];
      if(!p.dm_buffers.layouts || conv.input_layout>=p.dm_buffers.count) {
        asm volatile("trap;");return;
      }
      auto* tile=backend::ServingConv<Arch,TileM,TileN,TileK,Stages>::template
          Run<TILEMEGA_NONPAGED_TILED!=0>(p,conv,p.dm_buffers.layouts[conv.input_layout],
              p.conv_iteration,tile_m,tile_n,p.k_begin/TileK,p.k_count/TileK,shared);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
      return;
    }
    if constexpr(TileN==16 || TileK<64) {
      auto* tile=Config::Dense(p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
    }else {
    using namespace cute;
    if (!p.a || !p.b || p.k_begin < 0 || p.k_count <= 0 ||
        p.k_begin + p.k_count > p.k_total) {
      asm volatile("trap;");
      return;
    }
    int a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    int copy_k_count = (p.k_count + 7) & ~7;
    if (a_pitch < p.k_total || b_pitch < p.k_total ||
        a_pitch % 8 != 0 || b_pitch % 8 != 0 || p.k_begin % 8 != 0 ||
        p.k_begin + copy_k_count > a_pitch ||
        p.k_begin + copy_k_count > b_pitch) {
      asm volatile("trap;");
      return;
    }
    constexpr auto tile_shape = typename Mainloop::TileShape{};
    typename Mainloop::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape));
    clear(accum);
#if TILEMEGA_NONPAGED_TILED
    backend::ServingTiledMainloop<Arch,Config,TileM,TileN,TileK,Stages>::Run(
        p,tile_m,tile_n,shared,accum);
#else
    auto dA = make_stride(int64_t(a_pitch), _1{},
                          int64_t(p.m) * a_pitch);
    auto dB = make_stride(int64_t(b_pitch), _1{},
                          int64_t(p.n) * b_pitch);
    auto a = make_tensor(make_gmem_ptr(p.a + p.k_begin),
                         make_shape(p.m, copy_k_count, 1), dA);
    auto b = make_tensor(make_gmem_ptr(p.b + p.k_begin),
                         make_shape(p.n, copy_k_count, 1), dB);
    auto coordinate = make_coord(tile_m, tile_n, _, 0);
    auto gA = local_tile(a(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<_1, X, _1>{});
    auto gB = local_tile(b(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<X, _1, _1>{});
    auto residue = make_tuple(p.m - size<0>(gA) * tile_m,
                              p.n - size<0>(gB) * tile_n,
                              copy_k_count - size<1>(gA) * size<2>(gA));
    auto k_iter = make_coord_iterator(shape<2>(gA));
    Mainloop{}(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
               ComputeThread(), shared);
#endif
    backend::ServingDmEpilogue<Arch, Spec, TileM, TileN>::Run(
        accum, mma, shared, DmEpilogueOperands(p), tile_m, tile_n);
    }
  }
#endif

};

}  // namespace tilemega::codegen
