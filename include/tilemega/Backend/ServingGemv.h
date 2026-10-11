// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/DmMoeOperands.h>
#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/tasks/ServingTaskProfile.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <cute/tensor.hpp>

namespace tilemega::backend {
// One arithmetic path for row-major, packed, paged and resolved expert B.
// Each warp owns TN/4 columns. A is copied once per 256-element K block,
// shared by every output column, and double buffered. B loads for a whole
// unrolled batch precede its FMAs: 16 vectors/lane (32 KiB/CTA).
template<class Arch,int TileM,int TileN,int TileK>
struct ServingGemv {
  static_assert(arch::Caps<Arch>::kCpAsync);
  static_assert(TileM>=1 && (TileN==8 || TileN==16 || TileN==32 || TileN==64));
  static_assert(TileK==32 || TileK==64 || TileK==128);
  using Element=cutlass::bfloat16_t;
  using Copy=codegen::executor::Async<Arch>;
  static constexpr int kRows=TileM<4?TileM:4,kThreads=128,kWarps=4;
  static constexpr int kColumns=TileN/kWarps,kUnroll=16/kColumns;
  static constexpr int kAChunk=256,kBatchK=kAChunk*kUnroll;
  static constexpr int kSharedBytes=solver::ServingGemvSmemBytes(TileM,TileN);
  static constexpr int kAtomK=TileK<64?TileK:64,kSwizzle=TileK==32?2:3;
  using Atom=decltype(cute::composition(cute::Swizzle<kSwizzle,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::Int<kAtomK>>,
                   cute::Stride<cute::Int<kAtomK>,cute::_1>>{}));
  using LayoutB=decltype(cute::tile_to_shape(Atom{},
      cute::Shape<cute::Int<TileN>,cute::Int<TileK>>{}));
  using Epi=ServingEpilogue<ServingEpilogueOp::kStore,TileM,TileN>;
  __host__ __device__ static constexpr int PackedIndex(int n,int k) {
    return LayoutB{}(n,k);
  }
  template<class Operands>
  __device__ static bool Supported(Operands const& p,int tile_m) {
    bool supported=p.m-tile_m*TileM>0 && p.m-tile_m*TileM<=kRows &&
        p.k_begin>=0 && p.k_count>0 && p.k_begin+p.k_count<=p.k_total &&
        p.k_begin%8==0 && (p.a_row_stride?p.a_row_stride:p.k_total)%8==0 &&
        (p.b_row_stride?p.b_row_stride:p.k_total)%8==0;
    if constexpr(HasDmOperandRows<Operands>::value)
      supported=supported && p.access.a!=codegen::DmAAccess::kIm2Col && !p.a_scale;
    return supported;
  }
  template<class Operands>
  __device__ static void Validate(Operands const& p,int tile_m) {
    if(!Supported(p,tile_m) || !p.a || !p.b)asm volatile("trap;");
  }
  __device__ static uint4 LoadShared16(void const* pointer) {
    uint4 value;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
        :"=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w)
        :"r"(Copy::Shared(pointer)):"memory");
    return value;
  }
  template<class Operands>
  __device__ static void IssueA(Operands const& p,int tile_m,int offset,
                                Element* shared,int slot) {
    using namespace codegen::executor;
    for(int vector=ComputeThread();vector<kRows*kAChunk/8;vector+=kThreads) {
      int row=vector/(kAChunk/8),k=(vector%(kAChunk/8))*8;
      unsigned bytes=0;
      Element const* source=p.a;
      if(tile_m*TileM+row<p.m && offset+k<p.k_count) {
        unsigned source_row=DmSourceRow(p,tile_m*TileM+row);
        source=p.a+std::uint64_t(source_row)*(p.a_row_stride?p.a_row_stride:p.k_total)+
            p.k_begin+offset+k;
        bytes=min(8,p.k_count-offset-k)*sizeof(Element);
      }
      Copy::Copy16Bytes(shared+(slot*kRows+row)*kAChunk+k,source,bytes);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  template<class Operands>
  __device__ static void Fold(Operands const& p,int tile_m,int first,
      uint4 const (&weights)[kUnroll][kColumns],float (&acc)[kColumns][kRows],
      char* shared) {
    using namespace codegen::executor;
    auto* a=reinterpret_cast<Element*>(shared);
    IssueA(p,tile_m,first,a,0);
    if constexpr(kUnroll>1)IssueA(p,tile_m,first+kAChunk,a,1);
    #pragma unroll
    for(int u=0;u<kUnroll;++u) {
      if(u+1<kUnroll)asm volatile("cp.async.wait_group 1;" ::: "memory");
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
      ComputeSync();
#if TILEMEGA_TRACE_TASK
      if(first==0 && u==0)codegen::FirstTileReadyProfile{p.profile}();
#endif
      alignas(16) Element activation[kRows][8];
      #pragma unroll
      for(int row=0;row<kRows;++row)
        *reinterpret_cast<uint4*>(activation[row])=LoadShared16(
            a+((u%2)*kRows+row)*kAChunk+(ComputeThread()%32)*8);
      #pragma unroll
      for(int col=0;col<kColumns;++col) {
        alignas(16) Element b[8];
        *reinterpret_cast<uint4*>(b)=weights[u][col];
        #pragma unroll
        for(int row=0;row<kRows;++row) {
          #pragma unroll
          for(int e=0;e<8;++e)
            if(first+u*kAChunk+(ComputeThread()%32)*8+e<p.k_count)
              acc[col][row]=fmaf(float(activation[row][e]),float(b[e]),acc[col][row]);
        }
      }
      // All warps finish with this A slot before its next asynchronous copy.
      ComputeSync();
      if(u+2<kUnroll)IssueA(p,tile_m,first+(u+2)*kAChunk,a,u%2);
    }
  }
  template<bool Swizzled=false>
  __device__ static float* Materialize(float (&acc)[kColumns][kRows],char* shared) {
    using namespace codegen::executor;
    auto* tile=reinterpret_cast<float*>(shared);
    for(int i=ComputeThread();i<TileM*TileN;i+=kThreads)tile[i]=0;
    ComputeSync();
    #pragma unroll
    for(int c=0;c<kColumns;++c) {
      #pragma unroll
      for(int row=0;row<kRows;++row) {
        float sum=acc[c][row];
        #pragma unroll
        for(int shift=16;shift;shift/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,shift);
        if(ComputeThread()%32==0) {
          int col=(ComputeThread()/32)*kColumns+c;
          tile[Swizzled?Epi::SharedIndex(row,col):row*TileN+col]=sum;
        }
      }
    }
    ComputeSync();return tile;
  }
  template<bool Swizzled=false,class Operands>
  __device__ static float* Dense(Operands const& p,int tile_m,int tile_n,
                                char* shared,bool tiled) {
    using namespace codegen::executor;
    Validate(p,tile_m);
    if(tiled && (!p.weight_base || p.k_total_full<=0 || p.k_begin%TileK))asm volatile("trap;");
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    int kt=(p.k_total_full+TileK-1)/TileK;
    float acc[kColumns][kRows]{};
    for(int first=0;first<p.k_count;first+=kBatchK) {
      uint4 weights[kUnroll][kColumns]{};
      #pragma unroll
      for(int u=0;u<kUnroll;++u) {
        #pragma unroll
        for(int c=0;c<kColumns;++c) {
          int col=(ComputeThread()/32)*kColumns+c;
          int k=first+u*kAChunk+(ComputeThread()%32)*8,global=p.k_begin+k;
          if(tile_n*TileN+col<p.n && k<p.k_count) {
            auto const* source=tiled?p.weight_base+
                (std::uint64_t(tile_n)*kt+global/TileK)*TileN*TileK+PackedIndex(col,global%TileK):
                p.b+std::uint64_t(tile_n*TileN+col)*pitch+global;
            weights[u][c]=LoadGlobal16(source);
          }
        }
      }
      Fold(p,tile_m,first,weights,acc,shared);
    }
    return Materialize<Swizzled>(acc,shared);
  }
  template<class Ring>
  static constexpr bool kPagedSupported=Ring::kPages*Ring::kPageBytes>=32768 &&
      Ring::kPageBytes%(TileN*TileK*sizeof(Element))==0;
  template<bool Swizzled=false,class Operands,class Ring,class Gate>
  __device__ static float* Paged(Operands const& p,int tile_m,int tile_n,
      Ring const& ring,std::uint64_t& sequence,char* shared,Gate gate) {
    using namespace codegen::executor;
    static_assert(kPagedSupported<Ring>,"GEMV needs 32 KiB of B pages in flight");
    Validate(p,tile_m);
    if(p.k_begin%TileK)asm volatile("trap;");
    constexpr int page_k=Ring::kPageBytes/(TileN*sizeof(Element));
    static_assert(kBatchK%page_k==0);
    float acc[kColumns][kRows]{};
    for(int first=0;first<p.k_count;first+=kBatchK) {
      int pages=(min(kBatchK,p.k_count-first)+page_k-1)/page_k;
      for(int page=0;page<pages;++page)ring.AwaitFull(sequence+page);
      for(int k=first;k<min(first+kBatchK,p.k_count);k+=TileK)
        if(gate.enabled)gate.Wait((p.k_begin+k)/TileK);
      uint4 weights[kUnroll][kColumns]{};
      #pragma unroll
      for(int u=0;u<kUnroll;++u) {
        #pragma unroll
        for(int c=0;c<kColumns;++c) {
          int col=(ComputeThread()/32)*kColumns+c,k=u*kAChunk+(ComputeThread()%32)*8;
          if(tile_n*TileN+col<p.n && first+k<p.k_count) {
            auto const* page=reinterpret_cast<Element const*>(ring.Page(sequence+k/page_k));
            weights[u][c]=LoadShared16(page+(k%page_k/TileK)*TileN*TileK+PackedIndex(col,k%TileK));
          }
        }
      }
      // B is now in registers; the loader may reuse these slots while A and
      // the arithmetic progress. Every compute lane contributes one release.
      ComputeSync();
      for(int page=0;page<pages;++page)ring.Release(sequence++);
      Fold(p,tile_m,first,weights,acc,shared);
    }
    return Materialize<Swizzled>(acc,shared);
  }
};
} // namespace tilemega::backend
