// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/RoPETaskBody.h>

#ifndef TILEMEGA_FUSED_ROPE_KV
#define TILEMEGA_FUSED_ROPE_KV 1
#endif

namespace tilemega::codegen {
/// operand = {rotation input, frequency, retained prefix, full KV output}.
template <class Arch,class SmemUnion,int Threads>
struct FusedRoPEKVTaskBody {
  static constexpr bool kLegal = Threads > 0;
  __device__ static TaskOwnership Ownership(Params const& p,StageDesc const& stage) {
    if (p.ownership_flags & kKVTileOwnership)
      return {TaskOwnershipKind::kTilePerBlock,p.dims.seq*int(stage.extent)};
    int tokens=p.dims.seq>p.dims.past ? p.dims.seq : p.dims.past;
    return {TaskOwnershipKind::kElementChunk,(tokens*int(stage.extent)*int(stage.width)+Threads-1)/Threads};
  }
  __device__ static void RunTask(Params const& p,StageDesc const& stage,SmemUnion& smem,int task) {
    static_assert(TILEMEGA_FUSED_ROPE_KV || Threads<0,"fused RoPE/KV body disabled");
    int dim=stage.width,half_dim=dim/2,heads=stage.extent;
    bool consumer_tile=p.ownership_flags & kKVTileOwnership;
    bool producer_tile=p.ownership_flags & kRoPETileOwnership;
    int consumer_width=consumer_tile ? dim : Threads;
    int first=task*consumer_width;
    int appended=p.dims.seq*heads*dim;
    int pairs_per_task=producer_tile ? half_dim : Threads;
    int first_pair=(first/dim)*half_dim+first%half_dim;
    int producer_task=first_pair/pairs_per_task;
    if (first<appended) {
      for (int local=threadIdx.x;local<pairs_per_task;local+=Threads) {
        int pair=producer_task*pairs_per_task+local;
        if (pair>=appended/2) continue;
        int head_token=pair/half_dim,half=pair%half_dim;
        auto rotated=RotateRoPEPair(p.buffers[stage.operand[0]],p.buffers[stage.operand[1]],
            p.dims.past,head_token/heads,head_token*dim,half,half_dim);
        smem.fused_rope[local]=rotated.first;
        smem.fused_rope[pairs_per_task+local]=rotated.second;
      }
      __syncthreads();
    }
    auto* output=p.buffers[stage.operand[3]];
    for (int lane=threadIdx.x;lane<consumer_width;lane+=Threads) {
      int index=first+lane;
      if (index<appended) {
        int d=index%dim,head_token=index/dim;
        int pair=head_token*half_dim+d%half_dim;
        int local=pair-producer_task*pairs_per_task;
        if (local<0 || local>=pairs_per_task) { asm volatile("trap;"); return; }
        int token=head_token/heads,head=head_token%heads;
        output[(head*p.dims.total+p.dims.past+token)*dim+d]=
            smem.fused_rope[local+(d>=half_dim ? pairs_per_task : 0)];
      }
      // Tile-owned KV prefixes are preloaded by the host, as in KVAppend.
      int retained=heads*p.dims.past*dim;
      if (!consumer_tile && index<retained) {
        int d=index%dim,temp=index/dim,pos=temp%p.dims.past,head=temp/p.dims.past;
        output[(head*p.dims.total+pos)*dim+d]=p.buffers[stage.operand[2]][index];
      }
    }
  }
  __device__ void operator()(Params const& p,StageDesc const& stage,SmemUnion& smem) const {
    for (int task=PlacedBlock();task<Ownership(p,stage).count;task+=gridDim.x) {
      RunTask(p,stage,smem,task);
      __syncthreads();
    }
  }
};
}  // namespace tilemega::codegen
