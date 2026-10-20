// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cutlass/bfloat16.h>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>

namespace tilemega::backend {

struct Sm90DenseAttentionMath {
  using Element=cutlass::bfloat16_t;
  __device__ static __forceinline__ float WarpSum(float x) {
    for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);
    return __shfl_sync(0xffffffff,x,0);
  }
  __device__ static __forceinline__ float WarpMax(float x) {
    for(int d=16;d;d>>=1)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));
    return __shfl_sync(0xffffffff,x,0);
  }
  template<int I>
  __device__ static __forceinline__ float Get(uint4 x) {
    unsigned word;
    if constexpr(I<2)word=x.x;
    else if constexpr(I<4)word=x.y;
    else if constexpr(I<6)word=x.z;
    else word=x.w;
    return __uint_as_float(I%2?word&0xffff0000u:word<<16);
  }
  template<int I>
  __device__ static __forceinline__ void Put(uint4& x,Element value) {
    unsigned bits=unsigned(value.raw())<<(16*(I%2));
    if constexpr(I<2)x.x|=bits;
    else if constexpr(I<4)x.y|=bits;
    else if constexpr(I<6)x.z|=bits;
    else x.w|=bits;
  }
  __device__ static __forceinline__ float2 Pair(Element const* x) {
    unsigned bits=*reinterpret_cast<unsigned const*>(x);
    return make_float2(__uint_as_float(bits<<16),__uint_as_float(bits&0xffff0000u));
  }
  template<int D,bool Norm>
  __device__ static __forceinline__ void Rotate(Element const* x,Element const* norm,
      Element const* cosine,Element const* sine,float epsilon,Element* shared,
      Element* cache=nullptr) {
    int d=(threadIdx.x&31)*8;
    uint4 a{},b{},w{},partner_w{},cos{},sin{};
    float square=0;
    if(d<D) {
      a=*reinterpret_cast<uint4 const*>(x+d);
      b=*reinterpret_cast<uint4 const*>(x+(d+D/2)%D);
      cos=*reinterpret_cast<uint4 const*>(cosine+d);sin=*reinterpret_cast<uint4 const*>(sine+d);
      if constexpr(Norm) {
        w=*reinterpret_cast<uint4 const*>(norm+d);
        partner_w=*reinterpret_cast<uint4 const*>(norm+(d+D/2)%D);
        cute::for_each(cute::make_int_sequence<8>{},[&](auto i) {
          float value=Get<int(i)>(a);square+=value*value;
        });
      }
    }
    float inverse=1;
    if constexpr(Norm)inverse=rsqrtf(WarpSum(square)/D+epsilon);
    if(d<D) {
      uint4 output{};
      cute::for_each(cute::make_int_sequence<8>{},[&](auto i) {
        float first=Get<int(i)>(a),second=Get<int(i)>(b);
        if constexpr(Norm) {
          first=float(Element(float(Element(first*inverse))*Get<int(i)>(w)));
          second=float(Element(float(Element(second*inverse))*Get<int(i)>(partner_w)));
        }
        Element left(first*Get<int(i)>(cos));
        Element right((d<D/2?-second:second)*Get<int(i)>(sin));
        Put<int(i)>(output,Element(float(left)+float(right)));
      });
      *reinterpret_cast<uint4*>(shared+d)=output;
      if(cache)*reinterpret_cast<uint4*>(cache+d)=output;
    }
  }
};

}  // namespace tilemega::backend
