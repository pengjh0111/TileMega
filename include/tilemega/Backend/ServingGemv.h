#pragma once
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Backend/DmServingGemv.h>
#else
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Codegen/tasks/ServingTaskProfile.h>
namespace tilemega::backend {
// Direct register streaming avoids a shared-memory copy for M<=4. Each warp
// reduces one column at a time; no MMA accumulator rows are allocated.
template<class Arch,int TileN,int TileK,bool Tiled>
struct ServingGemv {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(TileN==8 || TileN==16 || TileN==32);
  static_assert(TileK==64 || TileK==128);
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=128;
  static constexpr int kSharedBytes=ServingEpilogueScratchBytes(16,TileN,TILEMEGA_EP_PARALLEL_ARGMAX);
  __host__ __device__ static constexpr int PackedIndex(int n,int k) {
    return 512*(n/8+(TileN/8)*(k/64))+64*(n%8)+8*((k%64/8)^(n%8))+k%8;
  }
  template<class Operands>
  __device__ static void Accumulate(Operands const& p,int tile_m,int tile_n,float* tile) {
    if(p.m>4 || p.m<1 || tile_m!=0 || p.k_count<1 || p.k_begin<0 ||
       p.k_begin+p.k_count>p.k_total){asm volatile("trap;");return;}
    using Epi=ServingEpilogue<ServingEpilogueOp::kStore,16,TileN>;
    int lane=codegen::executor::ComputeThread()%32,warp=codegen::executor::ComputeThread()/32;
    int ap=p.a_row_stride?p.a_row_stride:p.k_total;
    int bp=p.b_row_stride?p.b_row_stride:p.k_total;
    int full=p.k_total_full?p.k_total_full:p.k_total;
    int kt=(full+TileK-1)/TileK;
    for(int i=codegen::executor::ComputeThread();i<16*TileN;i+=128)tile[Epi::SharedIndex(i/TileN,i%TileN)]=0;
    codegen::executor::ComputeSync();
    for(int local=warp;local<TileN;local+=4) {
      int col=tile_n*TileN+local;
      float acc[4]={};
      if(col<p.n)for(int k=lane*8;k<p.k_count;k+=256) {
        int kg=p.k_begin+k;
        alignas(16) Element b[8];
        auto const* src=p.b+std::int64_t(col)*bp+kg;
        if constexpr(Tiled)src=p.weight_base+(std::int64_t(tile_n)*kt+kg/TileK)*TileN*TileK+PackedIndex(local,kg%TileK);
        int valid=min(8,p.k_count-k);
        if(valid==8 && kg%8==0 && (Tiled || bp%8==0))*reinterpret_cast<uint4*>(b)=LoadGlobal16(src);
        else for(int e=0;e<valid;++e) {
          if constexpr(Tiled)b[e]=p.weight_base[(std::int64_t(tile_n)*kt+(kg+e)/TileK)*TileN*TileK+PackedIndex(local,(kg+e)%TileK)];
          else b[e]=src[e];
        }
        #pragma unroll
        for(int row=0;row<4;++row)if(row<p.m) {
          alignas(16) Element a[8];
          auto const* ar=p.a+std::int64_t(row)*ap+kg;
          if(valid==8 && kg%8==0 && ap%8==0)*reinterpret_cast<uint4*>(a)=LoadGlobal16(ar);
          else for(int e=0;e<valid;++e)a[e]=ar[e];
          #pragma unroll
          for(int e=0;e<8;++e)if(e<valid)acc[row]=fmaf(float(a[e]),float(b[e]),acc[row]);
        }
#if TILEMEGA_TRACE_TASK
        // A streaming implementation has no page; observe its first A/B vector.
        if(local==0 && k==0)codegen::FirstTileReadyProfile{p.profile}();
#endif
      }
      #pragma unroll
      for(int row=0;row<4;++row) {
        for(int offset=16;offset;offset>>=1)acc[row]+=__shfl_down_sync(0xffffffff,acc[row],offset);
        if(lane==0)tile[Epi::SharedIndex(row,local)]=acc[row];
      }
    }
    codegen::executor::ComputeSync();
  }
};
}

#endif
