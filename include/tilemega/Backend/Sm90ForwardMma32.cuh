#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Backend/Sm90Gmma.h>
#include <cutlass/bfloat16.h>

namespace tilemega::backend {
template<int TN>
struct Sm90ForwardMma32 {
  static_assert(TN==32 || TN==64 || TN==128 || TN==256);
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  struct Accumulator {float value[TN/32][8]{};};

  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return row*128+((((col/8)^(row%8))*16))+(col%8)*2;
  }
  __device__ static __forceinline__ void LoadX4(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void Mma(float* c,unsigned const (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
  __device__ static __forceinline__ void Accumulate(
      Element const* weights,Element const* activation,Accumulator& accum) {
    int warp=int(threadIdx.x)/32,lane=int(threadIdx.x)%32;
    // Four warps cover disjoint m16 x n16 output tiles. Both input panels
    // keep the existing K-major SW128 storage used by the forward ring.
    #pragma unroll
    for(int owned=0;owned<TN/32;++owned) {
      int tile=warp+4*owned,m16=(tile/(TN/16))*16,n16=(tile%(TN/16))*16;
      #pragma unroll
      for(int k16=0;k16<64;k16+=16) {
        unsigned a[4],b[4];
        LoadX4(a,Async::Shared(activation)+Swizzled(m16+lane%16,k16+(lane/16)*8));
        int j=lane/8,row=n16+(j/2)*8+lane%8,col=k16+(j%2)*8;
        LoadX4(b,Async::Shared(weights)+Swizzled(row,col));
        Mma(accum.value[owned],a,b[0],b[1]);
        Mma(accum.value[owned]+4,a,b[2],b[3]);
      }
    }
  }
  __device__ static __forceinline__ float* Materialize(Accumulator const& accum,char* workspace) {
    auto* tile=reinterpret_cast<float*>(workspace);
    int warp=int(threadIdx.x)/32,lane=int(threadIdx.x)%32;
    #pragma unroll
    for(int owned=0;owned<TN/32;++owned) {
      int index=warp+4*owned,m16=(index/(TN/16))*16,n16=(index%(TN/16))*16;
      #pragma unroll
      for(int e=0;e<8;++e) {
        int row=m16+(lane/4)+8*((e%4)/2);
        int col=n16+(e/4)*8+2*(lane%4)+(e&1);
        tile[row*TN+col]=accum.value[owned][e];
      }
    }
    codegen::executor::ComputeSync();
    return tile;
  }
};

template<int TM,int TN>
struct Sm90ForwardMath:Sm90ForwardGmma<arch::Sm90,TM,TN,64> {
  using Base=Sm90ForwardGmma<arch::Sm90,TM,TN,64>;
  __device__ static __forceinline__ auto Make() {
    auto accum=cute::partition_fragment_C(typename Base::Mma{},cute::Shape<cute::Int<TM>,cute::Int<TN>>{});
    cute::clear(accum);return accum;
  }
  __device__ static __forceinline__ void Wait() {Sm90Gmma<arch::Sm90>::template Wait<0>();}
};
template<int TN>
struct Sm90ForwardMath<32,TN>:Sm90ForwardMma32<TN> {
  using Accumulator=typename Sm90ForwardMma32<TN>::Accumulator;
  __device__ static __forceinline__ Accumulator Make() {return {};}
  __device__ static __forceinline__ void Wait() {}
};
} // namespace tilemega::backend
