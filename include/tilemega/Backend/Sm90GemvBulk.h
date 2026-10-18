// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingGemv.h>

namespace tilemega::backend {
// Two 32-KiB weight slots overlap the next bulk transfer with unchanged SIMT
// arithmetic. Ordinary shared loads finish before reusing a slot.
template<class Arch,int TileM,int TileN,int TileK>
struct Sm90GemvBulk {
  using Base=ServingGemv<Arch,TileM,TileN,TileK>;
  using Element=typename Base::Element;
  using Copy=codegen::executor::Async<Arch>;
  static_assert(arch::Caps<Arch>::kBulkCopy && TileN<=32,
                "bulk GEMV requires bulk-copy capability and TN8/16/32");
  static constexpr int kWeightBytes=2*TileN*Base::kBatchK;
  static_assert(kWeightBytes==32768);
  static constexpr int kWeightOffset=(Base::kSharedBytes+15)/16*16;
  static constexpr int kBarrierOffset=kWeightOffset+2*kWeightBytes;
  static constexpr int kSharedBytes=kBarrierOffset+2*sizeof(std::uint64_t);
  static_assert(kSharedBytes==solver::ServingGemvSmemBytes(TileM,TileN,true));

  template<class Operands>
  __device__ static void IssueB(Operands const& p,int tile_n,int first,int slot,
      char* shared,bool tiled,std::uint64_t policy) {
    using namespace codegen::executor;
    if(ComputeThread()!=0)return;
    auto* destination=reinterpret_cast<Element*>(shared+kWeightOffset+slot*kWeightBytes);
    auto* barrier=reinterpret_cast<std::uint64_t*>(shared+kBarrierOffset)+slot;
    int count=min(Base::kBatchK,p.k_count-first);
    if(tiled) {
      int kt=(p.k_total_full+TileK-1)/TileK;
      auto* source=p.weight_base+(std::uint64_t(tile_n)*kt+(p.k_begin+first)/TileK)*TileN*TileK;
      unsigned bytes=2*TileN*count;
      Copy::ExpectTx(barrier,bytes);
      Copy::BulkHint(destination,source,bytes,barrier,policy);
    }else {
      int columns=min(TileN,p.n-tile_n*TileN);
      int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
      unsigned bytes=2*count;
      Copy::ExpectTx(barrier,columns*bytes);
      for(int col=0;col<columns;++col)
        Copy::BulkHint(destination+col*Base::kBatchK,
            p.b+std::uint64_t(tile_n*TileN+col)*pitch+p.k_begin+first,
            bytes,barrier,policy);
    }
  }

  template<bool Swizzled=false,class Operands>
  __device__ static float* Dense(Operands const& p,int tile_m,int tile_n,
      char* shared,bool tiled) {
    using namespace codegen::executor;
    Base::Validate(p,tile_m);
    // Bulk byte counts and packed tile boundaries must be exact. These are
    // plan legality conditions, not a silent fallback to the direct loader.
    if(p.k_begin%TileK || p.k_count%TileK || tile_n*TileN>=p.n ||
        (tiled && (!p.weight_base || p.k_total_full<=0)))asm volatile("trap;");
    auto* barriers=reinterpret_cast<std::uint64_t*>(shared+kBarrierOffset);
    if(ComputeThread()==0){Copy::Init(barriers,1);Copy::Init(barriers+1,1);}
    Copy::InitFence();ComputeSync();
    Copy::ProxyAsyncGlobalFence();
    std::uint64_t policy=0;
    if(ComputeThread()==0)policy=Copy::EvictFirst();
    IssueB(p,tile_n,0,0,shared,tiled,policy);
    if(Base::kBatchK<p.k_count)IssueB(p,tile_n,Base::kBatchK,1,shared,tiled,policy);
    float acc[Base::kColumns][Base::kRows]{};
    for(int first=0,iteration=0;first<p.k_count;first+=Base::kBatchK,++iteration) {
      int slot=iteration%2;
      Copy::Wait(barriers+slot,unsigned(iteration/2)%2);
      auto const* page=reinterpret_cast<Element const*>(shared+kWeightOffset+slot*kWeightBytes);
      uint4 weights[Base::kUnroll][Base::kColumns]{};
      #pragma unroll
      for(int u=0;u<Base::kUnroll;++u) {
        #pragma unroll
        for(int c=0;c<Base::kColumns;++c) {
          int col=(ComputeThread()/32)*Base::kColumns+c;
          int k=u*Base::kAChunk+(ComputeThread()%32)*8;
          if(tile_n*TileN+col<p.n && first+k<p.k_count) {
            int offset=tiled?(k/TileK)*TileN*TileK+Base::PackedIndex(col,k%TileK):
                col*Base::kBatchK+k;
            weights[u][c]=Base::LoadShared16(page+offset);
          }
        }
      }
      ComputeSync();
      if(first+2*Base::kBatchK<p.k_count)
        IssueB(p,tile_n,first+2*Base::kBatchK,slot,shared,tiled,policy);
      Base::Fold(p,tile_m,first,weights,acc,shared);
    }
    ComputeSync();
    if(ComputeThread()==0){Copy::Invalidate(barriers);Copy::Invalidate(barriers+1);}
    ComputeSync();
    return Base::template Materialize<Swizzled>(acc,shared);
  }
};
} // namespace tilemega::backend
