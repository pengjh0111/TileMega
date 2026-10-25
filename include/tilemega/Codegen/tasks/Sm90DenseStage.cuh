// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseOperands.cuh>

namespace tilemega::codegen::native_dense {
struct ReadSources {
  cutlass::bfloat16_t const* weight=nullptr;
  CUtensorMap const *key=nullptr,*value=nullptr;
};

template<class Body,int Tasks>
struct GemmRead {
  template<int Bytes>
  __device__ static __forceinline__ void Prepare(ReadSources source,int,char* shared,
      backend::Sm90DenseRingState& ring) {
    if constexpr(Bytes>0)if(blockIdx.x<Tasks)Body::template Preissue<Bytes>(source.weight,shared,ring);
  }
};
template<class Body,int Tasks>
struct KvRead {
  template<int Bytes>
  __device__ static __forceinline__ void Prepare(ReadSources source,int past,char* shared,
      backend::Sm90DenseRingState& ring) {
    if constexpr(Bytes>0)if(blockIdx.x<Tasks)
      Body::template Preissue<Bytes>(source.key,source.value,past,shared,ring);
  }
};

// D6 (prompt §6): after its last tile the producer of stage i issues stage
// i+1's first bytes into the other arena and ring. Buffer IDs stay in the
// caller; equal next-stage geometry shares one tail type across layers.
template<class Loader,int Offset,int Prefix>
struct RawProducerTail {
  ReadSources source;
  int past;
  char* shared;
  backend::Sm90DenseRingState* next;
  __device__ __forceinline__ void operator()() const {
    Loader::template Prepare<Prefix>(source,past,shared+Offset,*next);
  }
};

template<class Body>
struct StageEntry {
  template<int Prefix,class Tail>
  __device__ static __forceinline__ void Gemm(backend::Sm90DenseOperands const& p,char* shared,
      backend::Sm90DenseRingState* ring,Tail tail) {
    Body::template Run<Prefix,Tail>(p,shared,ring,tail);
  }
  template<int Prefix,class Tail>
  __device__ static __forceinline__ void Attention(backend::Sm90DenseAttentionOperands const& p,
      char* shared,backend::Sm90DenseRingState* ring,Tail tail) {
    Body::template Run<Prefix>(p,shared,ring,tail);
  }
};
}  // namespace tilemega::codegen::native_dense
