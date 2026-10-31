// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseOperands.cuh>
#include <tilemega/Backend/Sm90DenseMark.cuh>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <climits>
#include <cmath>

namespace tilemega::backend {

// GEMM consumers call a finish hook once after their last task; only the
// lm_head stage has one.
struct Sm90DenseNoFinish {
  __device__ void operator()(Sm90DenseOperands const&,Sm90DenseRingState&,float const*,int const*,
                             char*,int) const {}
};

// D9 (prompt §7): lm_head with the argmax folded in. Each CTA has reduced its
// tasks to one (value, index) per row; it publishes that partial into the
// GEMM's [B][G] value/index outputs and counts on lm_ctr. The last arriver
// (old % G == G - 1; the counter only grows) reduces the B x G partials into
// the next tokens. With Embed it also writes the next step's embedding rows
// into x: bulk table rows into the consumed operand slots, then st.global
// (generic proxy; the next qkv reads x by TMA after acquire + proxy fence).
template<int Batch,int Width,int Capacity,bool Embed>
struct Sm90DenseHeadFinish {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Batch>0 && Batch<=16 && Width%8==0 && Capacity>=1088);
  static constexpr int kRowBytes=2*Width;
  Sm90DenseHeadOperands const* head;

  // free: operand storage no task reads any more; bytes: its extent.
  __device__ __forceinline__ void operator()(Sm90DenseOperands const& p,Sm90DenseRingState& state,
      float const* best,int const* arg,char* free,int bytes) const {
    int tid=threadIdx.x,warp=tid>>5,lane=tid&31,grid=int(gridDim.x);
    auto* values=reinterpret_cast<float*>(p.output);
    if(tid<Batch) {
      values[tid*grid+int(blockIdx.x)]=best[tid];p.argmax_index[tid*grid+int(blockIdx.x)]=arg[tid];
    }
    auto* flag=reinterpret_cast<int*>(free);auto* tokens=flag+1;
    codegen::executor::ComputeSync();
    if(tid==0) {
      unsigned long long old;
      asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;":"=l"(old):"l"(head->counter):"memory");
      *flag=old%unsigned(grid)==unsigned(grid-1);
    }
    codegen::executor::ComputeSync();
    if(tid==0)Sm90DenseMark(2);
    if(!*flag)return;
    // Ties keep the smaller vocabulary index, as within a task.
    for(int row=warp;row<Batch;row+=4) {
      float value=-INFINITY;int index=INT_MAX;
      for(int part=lane;part<grid;part+=32) {
        float v=values[row*grid+part];int id=p.argmax_index[row*grid+part];
        if(v>value || (v==value && id<index)) {value=v;index=id;}
      }
      #pragma unroll
      for(int d=16;d;d>>=1) {
        float v=__shfl_xor_sync(0xffffffff,value,d);int id=__shfl_xor_sync(0xffffffff,index,d);
        if(v>value || (v==value && id<index)) {value=v;index=id;}
      }
      // A prompt position keeps its frozen token: write the argmax only once
      // the step has left the prompt, then read back whichever token now
      // stands at past + 1, so the embedding below teacher-forces the prefill
      // (§8 A3.5). prompt_len = 0 makes the condition always hold.
      if(lane==0) {
        int next=row*Capacity+head->past+1;
        if(head->past+1>=head->prompt_len)head->tokens[next]=index;
        tokens[row]=head->tokens[next];
      }
    }
    codegen::executor::ComputeSync();
    if(tid==0)Sm90DenseMark(3);
    if constexpr(Embed) {
      constexpr int kStage=128;
      int rows=(bytes-kStage)/kRowBytes;rows=rows<Batch?rows:Batch;
      auto* staged=reinterpret_cast<Element*>(free+kStage);
      unsigned phase=state.side_generations[0];
      for(int first=0;first<Batch;first+=rows) {
        int count=Batch-first<rows?Batch-first:rows;
        if(tid==0) {
          Async::ExpectTx(state.side,count*kRowBytes);
          for(int r=0;r<count;++r)
            Async::Bulk(staged+r*Width,head->table+std::size_t(tokens[first+r])*Width,kRowBytes,state.side);
        }
        Async::Wait(state.side,phase);phase^=1;
        for(int i=tid*8;i<count*Width;i+=128*8)
          *reinterpret_cast<uint4*>(head->x+std::size_t(first)*Width+i)=
              *reinterpret_cast<uint4 const*>(staged+i);
        // Generic reads of the staged rows precede the next async refill.
        Async::FenceProxyAsyncShared();
        codegen::executor::ComputeSync();
      }
      if(tid==0)state.side_generations[0]=phase;
    }
    if(tid==0)Sm90DenseMark(4);
  }
};

}  // namespace tilemega::backend
