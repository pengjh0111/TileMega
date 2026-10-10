// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/DmMoeOperands.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace tilemega::backend {
// Four warps reduce independent columns. Physical M padding can retain the
// MMA family's task ownership, but a live binding contains at most four rows.
// BF16 products accumulate in FP32; only the shared epilogue rounds outputs.
template<class Arch,int TileM,int TileN,int TileK>
struct ServingGemv {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(TileM>=1 && (TileN==8 || TileN==16 || TileN==32));
  static_assert(TileK==32 || TileK==64 || TileK==128);
  using Element=cutlass::bfloat16_t;
  static constexpr int kRows=TileM<4?TileM:4;
  static constexpr int kWarps=4,kThreads=128;
  static constexpr int kValues=(kRows*TileN+kWarps-1)/kWarps;
  static constexpr int kSharedBytes=(TileM*TileN+2*TileM)*sizeof(float);
  static constexpr int kAtomK=TileK<64?TileK:64;
  static constexpr int kSwizzle=TileK==32?2:3;
  using Atom=decltype(cute::composition(cute::Swizzle<kSwizzle,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::Int<kAtomK>>,
                   cute::Stride<cute::Int<kAtomK>,cute::_1>>{}));
  using LayoutB=decltype(cute::tile_to_shape(Atom{},
      cute::Shape<cute::Int<TileN>,cute::Int<TileK>>{}));

  template<class Operands>
  __device__ static bool Supported(Operands const& p,int tile_m) {
    return p.access.a!=codegen::DmAAccess::kIm2Col && !p.a_scale &&
        p.m-tile_m*TileM>0 && p.m-tile_m*TileM<=kRows;
  }
  template<class Operands>
  __device__ static float A(Operands const& p,unsigned row,unsigned k) {
    auto source=DmSourceRow(p,row);
    return float(p.a[std::uint64_t(source)*(p.a_row_stride?p.a_row_stride:p.k_total)+k]);
  }
  template<class Operands>
  __device__ static void Validate(Operands const& p,int tile_m) {
    if(!Supported(p,tile_m) || !p.a || !p.b || p.k_begin<0 || p.k_count<=0 ||
        p.k_begin+p.k_count>p.k_total)asm volatile("trap;");
  }
  template<class Operands>
  __device__ static float* Materialize(Operands const& p,int tile_m,
                                      float (&values)[kValues],char* shared) {
    using namespace codegen::executor;
    auto* tile=reinterpret_cast<float*>(shared);
    for(int i=ComputeThread();i<TileM*TileN;i+=kThreads)tile[i]=0;
    ComputeSync();
    int lane=ComputeThread()%32,warp=ComputeThread()/32;
    for(int i=0;i<kValues;++i) {
      auto sum=values[i];
      for(int shift=16;shift;shift/=2)sum+=__shfl_down_sync(0xffffffffu,sum,shift);
      int output=warp+i*kWarps;
      if(!lane && output<kRows*TileN && tile_m*TileM+output/TileN<p.m)
        tile[output]=sum;
    }
    ComputeSync();return tile;
  }
  template<class Operands>
  __device__ static float* Dense(Operands const& p,int tile_m,int tile_n,
                                char* shared,bool tiled) {
    using namespace codegen::executor;
    Validate(p,tile_m);
    if(tiled && (!p.weight_base || p.k_total_full<=0 || p.k_begin%TileK))asm volatile("trap;");
    float values[kValues]{};
    int lane=ComputeThread()%32,warp=ComputeThread()/32;
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    int kt=(p.k_total_full+TileK-1)/TileK;
    for(int i=0;i<kValues;++i) {
      int output=warp+i*kWarps,row=tile_m*TileM+output/TileN,col=output%TileN;
      if(output>=kRows*TileN || row>=p.m || tile_n*TileN+col>=p.n)continue;
      for(int k=lane;k<p.k_count;k+=32) {
        int global=p.k_begin+k;
        auto b=tiled?p.weight_base[(std::uint64_t(tile_n)*kt+global/TileK)*TileN*TileK+
            LayoutB{}(col,global%TileK)]:p.b[std::uint64_t(tile_n*TileN+col)*pitch+global];
        values[i]=fmaf(A(p,row,global),float(b),values[i]);
      }
    }
    return Materialize(p,tile_m,values,shared);
  }
  template<class Operands,class Ring,class Gate>
  __device__ static float* Paged(Operands const& p,int tile_m,int tile_n,
      Ring const& ring,std::uint64_t& sequence,char* shared,Gate gate) {
    using namespace codegen::executor;
    Validate(p,tile_m);
    constexpr int bytes=TileN*TileK*sizeof(Element);
    constexpr int stages=Ring::kPageBytes/bytes;
    static_assert(stages>=1 && Ring::kPageBytes%bytes==0);
    float values[kValues]{};
    int lane=ComputeThread()%32,warp=ComputeThread()/32;
    int iterations=(p.k_count+TileK-1)/TileK;
    for(int first=0;first<iterations;first+=stages) {
      ring.AwaitFull(sequence);
      for(int stage=0;stage<stages && first+stage<iterations;++stage) {
        int iteration=first+stage;
        if(gate.enabled)gate.Wait(p.k_begin/TileK+iteration);
        auto* b=reinterpret_cast<Element const*>(ring.Page(sequence))+stage*TileN*TileK;
        for(int i=0;i<kValues;++i) {
          int output=warp+i*kWarps,row=tile_m*TileM+output/TileN,col=output%TileN;
          if(output>=kRows*TileN || row>=p.m || tile_n*TileN+col>=p.n)continue;
          for(int k=lane;k<TileK && iteration*TileK+k<p.k_count;k+=32)
            values[i]=fmaf(A(p,row,p.k_begin+iteration*TileK+k),float(b[LayoutB{}(col,k)]),values[i]);
        }
      }
      // Every warp finishes its B reads before any lane releases the page.
      ComputeSync();ring.Release(sequence++);
    }
    return Materialize(p,tile_m,values,shared);
  }
};
} // namespace tilemega::backend
