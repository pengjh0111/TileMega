// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Backend/DmMoeOperands.h>
#include <tilemega/Codegen/executor/Async.cuh>

#ifndef TILEMEGA_MOE_ACTIVATION_BULK
#define TILEMEGA_MOE_ACTIVATION_BULK 0
#endif
namespace tilemega::backend {
template<class Arch,int M,int K>
struct Sm90BulkGather {
  static_assert(M==16 && (K==64 || K==128));
  using E=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<Arch>;
  __device__ static __forceinline__ void BeginWeight(E* destination,E const* source,
      unsigned bytes,std::uint64_t* barrier) {
    static_assert(arch::Caps<Arch>::kBulkCopy);
    using codegen::executor::ComputeThread;using codegen::executor::ComputeSync;
    ComputeSync();
    if(ComputeThread()==0)Async::Init(barrier,1);
    Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
    if(ComputeThread()==0) {
      Async::ExpectTx(barrier,bytes);
      Async::Bulk(destination,source,bytes,barrier);
    }
  }
  __device__ static __forceinline__ void WaitWeight(std::uint64_t* barrier) {
    Async::Wait(barrier,0);codegen::executor::ComputeSync();
    if(codegen::executor::ComputeThread()==0)Async::Invalidate(barrier);
    codegen::executor::ComputeSync();
  }
  template<bool WeightBulk=false,class Operands>
  __device__ static __forceinline__ bool Load(Operands const& p,int tile_m,int iteration,
      E* destination,E* linear,std::uint64_t* barrier,
      E* weight_destination=nullptr,E const* weight_source=nullptr,unsigned weight_bytes=0) {
    static_assert(arch::Caps<Arch>::kBulkCopy);
    using codegen::executor::ComputeThread;using codegen::executor::ComputeSync;
    if(p.access.a!=codegen::DmAAccess::kRowGather || (iteration+1)*K>p.k_count)return false;
    int rows=min(M,p.m-tile_m*M),pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    if(rows<=0 || (!p.rows && !p.moe.rows)) {asm volatile("trap;");return false;}
    ComputeSync();
    if(ComputeThread()==0)Async::Init(barrier,1);
    Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
    if(ComputeThread()==0) {
      // Both operands share one completion phase; the slot remains owned until WGMMA wait.
      Async::ExpectTx(barrier,rows*K*sizeof(E)+(WeightBulk?weight_bytes:0));
      if constexpr(WeightBulk)Async::Bulk(weight_destination,weight_source,weight_bytes,barrier);
      for(int row=0;row<rows;++row) {
        auto source_row=DmSourceRow(p,tile_m*M+row);
        Async::Bulk(linear+row*K,p.a+std::size_t(source_row)*pitch+p.k_begin+iteration*K,
                    K*sizeof(E),barrier);
      }
    }
    Async::Wait(barrier,0);ComputeSync();
    if(ComputeThread()==0)Async::Invalidate(barrier);
    // Linear bulk has no SW128 transform. Repack its 16-byte vectors before
    // publishing generic shared writes to the WGMMA async proxy.
    auto layout=cute::as_position_independent_swizzle_layout(Sm90KLayout<M,K>{});
    for(int v=ComputeThread()*8;v<M*K;v+=128*8) {
      uint4 value=v/K<rows?*reinterpret_cast<uint4 const*>(linear+v):make_uint4(0,0,0,0);
      *reinterpret_cast<uint4*>(destination+layout(v/K,v%K))=value;
    }
    ComputeSync();Async::FenceProxyAsyncShared();ComputeSync();
    return true;
  }
};
} // namespace tilemega::backend
