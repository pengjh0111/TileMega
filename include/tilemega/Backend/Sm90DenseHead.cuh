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
//
// Tp > 1 (prompt appendix C.3): each rank holds VocabLocal rows of the
// vocabulary, so its last arriver only has a per-row local winner. It sends
// (value bits << 32 | global index) to every peer, releases a per-parity flag
// at system scope and acquires every peer's, then reduces the Tp candidates
// with the same rule (larger value, then smaller index). Every rank reaches
// the same token. The exchange runs every step, prompt positions included, so
// the epoch chain that also orders the next step's all-reduce writes after
// this step's reads (plan §3.5 C2) never breaks.
template<int Batch,int Width,int Capacity,bool Embed,int Tp=1,int VocabLocal=0>
struct Sm90DenseHeadFinish {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Batch>0 && Batch<=16 && Width%8==0 && Capacity>=1088);
  static_assert(Tp>=1 && Tp<=8 && (Tp==1)==(VocabLocal==0));
  static constexpr int kRowBytes=2*Width;
  // free: [0, 4) last-arriver flag, [4, 68) tokens, [128, 256) the Tp > 1
  // local candidates; the embedding rows are staged after them.
  static constexpr int kStage=Tp>1?256:128;
  Sm90DenseHeadOperands const* head;
  Sm90DenseTpHeadOperands const* tp=nullptr;

  // Larger value wins; equal values keep the smaller vocabulary index.
  __device__ static __forceinline__ bool Better(float v,int id,float value,int index) {
    return v>value || (v==value && id<index);
  }
  __device__ static __forceinline__ std::uint64_t Candidate(float value,int index) {
    return (std::uint64_t(__float_as_uint(value))<<32)|std::uint32_t(index);
  }

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
    if constexpr(Tp>1) {
      Exchange(p,values,tokens,reinterpret_cast<std::uint64_t*>(free+128),warp,lane,grid);
    }else
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

  // The Tp > 1 replacement of the row loop above; returns after the tokens of
  // every row are written (when past + 1 leaves the prompt) and read back.
  __device__ __forceinline__ void Exchange(Sm90DenseOperands const& p,float const* values,
      int* tokens,std::uint64_t* local,int warp,int lane,int grid) const {
    int tid=threadIdx.x;
    // 1. Local winner of each row, with its index made global. The shuffle
    // reduction stays inside the warp loop; nothing below it may, because the
    // exchange needs CTA-wide joins (at B1 only warp 0 has a row).
    for(int row=warp;row<Batch;row+=4) {
      float value=-INFINITY;int index=INT_MAX;
      for(int part=lane;part<grid;part+=32) {
        float v=values[row*grid+part];int id=p.argmax_index[row*grid+part];
        if(Better(v,id,value,index)) {value=v;index=id;}
      }
      #pragma unroll
      for(int d=16;d;d>>=1) {
        float v=__shfl_xor_sync(0xffffffff,value,d);int id=__shfl_xor_sync(0xffffffff,index,d);
        if(Better(v,id,value,index)) {value=v;index=id;}
      }
      if(lane==0)local[row]=Candidate(value,index+tp->rank*VocabLocal);
    }
    codegen::executor::ComputeSync();
    unsigned parity=unsigned((tp->epoch-1)&1);
    // 2. Data into every peer's slot for this rank.
    for(int i=tid;i<Batch*Tp;i+=128) {
      int q=i/Batch,row=i%Batch;
      if(q==tp->rank)continue;
#if TILEMEGA_TP_LITMUS
      Sm90DenseTpLitmusSpin(tp->rank,tp->epoch,0xffffffffu,i);
#endif
      asm volatile("st.relaxed.sys.global.u64 [%0], %1;"::
          "l"(&tp->x[q]->data[parity][tp->rank][row]),"l"(local[row]):"memory");
    }
    codegen::executor::ComputeSync();
    // 3. One release per peer after the CTA join (the fence is cumulative over
    // the CTA's data writes), then acquire every peer's flag.
    if(tid==0) {
      asm volatile("fence.acq_rel.sys;":::"memory");
      for(int q=0;q<Tp;++q)if(q!=tp->rank)
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;"::
            "l"(&tp->x[q]->flag[parity][tp->rank]),"l"(std::uint64_t(tp->epoch)):"memory");
      for(int q=0;q<Tp;++q)if(q!=tp->rank) {
        std::uint64_t seen;
        do {asm volatile("ld.acquire.sys.global.u64 %0, [%1];":"=l"(seen)
                :"l"(&tp->x[tp->rank]->flag[parity][q]):"memory");}
        while(seen!=tp->epoch);
      }
    }
    codegen::executor::ComputeSync();
    // 4. The global winner over ranks in ascending order, then the token.
    for(int row=warp;row<Batch;row+=4)if(lane==0) {
      float value=-INFINITY;int index=INT_MAX;
      for(int q=0;q<Tp;++q) {
        std::uint64_t c=local[row];
        if(q!=tp->rank)
          asm volatile("ld.relaxed.sys.global.u64 %0, [%1];":"=l"(c)
              :"l"(&tp->x[tp->rank]->data[parity][q][row]):"memory");
        float v=__uint_as_float(std::uint32_t(c>>32));int id=int(std::uint32_t(c));
        if(Better(v,id,value,index)) {value=v;index=id;}
      }
      int next=row*Capacity+head->past+1;
      if(head->past+1>=head->prompt_len)head->tokens[next]=index;
      tokens[row]=head->tokens[next];
    }
  }
};

}  // namespace tilemega::backend
