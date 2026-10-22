// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseAttention.h>

namespace tilemega::codegen::native_dense {
struct ReadSources {
  cutlass::bfloat16_t const *first=nullptr,*second=nullptr;
};

template<class Body,int Tasks>
struct GemmRead {
  template<int Bytes>
  __device__ static __forceinline__ void Prepare(ReadSources source,int,char* shared,
      backend::Sm90DenseRingState& ring) {
    if constexpr(Bytes>0)if(blockIdx.x<Tasks)
      Body::template Preissue<Bytes>(source.first,shared,ring);
  }
  template<int Bytes,int Prefix>
  __device__ static __forceinline__ void Prefetch(ReadSources source,int) {
    if constexpr(Bytes>0)if(blockIdx.x<Tasks)Body::template Prefetch<Bytes,Prefix>(source.first);
  }
};

template<class Loader,int Tasks>
struct KvRead {
  template<int Bytes>
  __device__ static __forceinline__ void Prepare(ReadSources source,int past,char* shared,
      backend::Sm90DenseRingState& ring) {
    if constexpr(Bytes>0)if(blockIdx.x<Tasks)
      Loader::template Prepare<Bytes>(source.first,source.second,past,shared,ring);
  }
  template<int Bytes,int Prefix>
  __device__ static __forceinline__ void Prefetch(ReadSources source,int past) {
    if constexpr(Bytes>0)if(blockIdx.x<Tasks)
      Loader::template Prefetch<Bytes,Prefix>(source.first,source.second,past);
  }
};

// Buffer IDs stay in the caller. This type is shared by equal stage geometry
// across layers, so the outlined entry does not duplicate the ring machinery.
template<class Loader,int Offset,int Prefix,int L2Bytes,bool Overlap,int RingSlots,bool Late=false>
struct RawProducerTail {
  static constexpr bool kBefore=!Overlap && !Late;
  ReadSources source;
  int past;
  char* shared;
  backend::Sm90DenseRingState *current,*next;
  __device__ __forceinline__ void Issue() const {
    if constexpr(Overlap)current->template RetireOperands<RingSlots>();
    Loader::template Prepare<Prefix>(source,past,shared+Offset,*next);
    Loader::template Prefetch<L2Bytes,Prefix>(source,past);
  }
  __device__ __forceinline__ void Before() const {if constexpr(kBefore)Issue();}
  __device__ __forceinline__ void operator()() const {if constexpr(!kBefore)Issue();}
};

template<class Body>
struct StageEntry {
  template<int Prefix,bool Initialize=true,class Tail>
  __device__ static __noinline__ void Gemm(backend::Sm90DenseOperands p,char* shared,
      float epsilon,backend::Sm90DenseRingState* ring,Tail tail) {
    Body::template Run<Prefix,Tail,Initialize>(p,shared,epsilon,ring,tail);
  }
  template<class Tail>
  __device__ static __noinline__ void Attention(backend::Sm90DenseAttentionOperands p,
      char* shared,backend::Sm90DenseRingState* ring,Tail tail) {
    Body::Run(p,shared,ring,tail);
  }
};
}  // namespace tilemega::codegen::native_dense
