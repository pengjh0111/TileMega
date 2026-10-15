// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ClusterMailbox.cuh>
#include <tilemega/Codegen/tasks/IndependentAttentionTaskBody.h>

namespace tilemega::codegen {
template<class Arch,int D,int Q,int Ranks,bool Norm>
struct ClusterAttentionMergeTaskBody {
  static_assert((D==64 || D==128) && Q>=1 && Q<=8);
  using Attention=IndependentAttentionTaskBody<Arch,D,Q,Norm>;
  static constexpr unsigned kElements=Attention::kClusterElements;
  using Mailbox=executor::ClusterMailbox<Arch,Ranks,kElements>;
  using Element=cutlass::bfloat16_t;
  __device__ static __forceinline__ void Merge(ServingAttentionOperands const& p,
      int batch,int group,float const* local,float const (*peers)[kElements]) {
    for(int vector=threadIdx.x;vector<Q*D/8;vector+=blockDim.x) {
      int q=vector/(D/8),d=(vector%(D/8))*8;
      float maximum=local[Q*D+q];
      #pragma unroll
      for(int rank=1;rank<Ranks;++rank)maximum=fmaxf(maximum,peers[rank-1][Q*D+q]);
      float normalizer=0,numerator[8]={};
      // Partial outputs are normalized. Their log2 normalizers determine weights.
      #pragma unroll
      for(int rank=0;rank<Ranks;++rank) {
        auto* tile=rank==0?local:peers[rank-1];
        float lse=tile[Q*D+q];
        float weight=isfinite(lse)?exp2f(lse-maximum):0;
        normalizer+=weight;
        #pragma unroll
        for(int lane=0;lane<8;++lane)numerator[lane]+=weight*tile[q*D+d+lane];
      }
      alignas(16) Element result[8];
      #pragma unroll
      for(int lane=0;lane<8;++lane)result[lane]=Element(normalizer>0?numerator[lane]/normalizer:0);
      auto offset=((std::size_t(batch)*p.heads_kv+group)*Q+q)*D+d;
      backend::StoreGlobal16(p.context+offset,*reinterpret_cast<uint4 const*>(result));
    }
  }
  __device__ static __forceinline__ bool Run(ServingAttentionOperands const& p,
      typename Attention::SharedStorage& shared,typename Mailbox::Storage& mailbox,
      int batch,int group,int split) {
    return Mailbox::template Exchange<kElements>(mailbox,[&] {
      return Attention::RunLocalPartial(p,shared,batch,group,split);
    },[&](float* local,float const (*peers)[kElements]) {
      Merge(p,batch,group,local,peers);
    });
  }
};
} // namespace tilemega::codegen
