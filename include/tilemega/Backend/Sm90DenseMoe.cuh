// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseMark.cuh>
#include <tilemega/Backend/Sm90DenseStream.cuh>
#include <climits>
#include <cmath>

namespace tilemega::backend {

// Native MoE decode (prompt §8). M4 router + top-k: a GEMM over the
// RMS-normalized x' whose last arriver selects each token's top-k, writes the
// routes and the schedule record. M5 grouped gate_up and M6 down + combine
// loop over the record's n_active x N/TN tasks. The MMA core, layouts and
// rings are Sm90DenseGemm's.
enum class Sm90DenseMoeMode {kRouter,kGateUp,kDown};

// Schedule record (i32): n_active, expert[E] (ascending), count[E],
// rows[E][16]; a row is b * TopK + k for token b routed at rank k.
template<int Experts>
struct Sm90DenseMoeSchedule {
  static constexpr int kMaxRows=16,kInts=1+2*Experts+Experts*kMaxRows;
  __host__ __device__ static constexpr int Active() {return 0;}
  __host__ __device__ static constexpr int Expert(int a) {return 1+a;}
  __host__ __device__ static constexpr int Count(int a) {return 1+Experts+a;}
  __host__ __device__ static constexpr int Row(int a,int r) {return 1+2*Experts+a*kMaxRows+r;}
};

// M4's last arriver (count modulo the router's tasks; the counter only
// grows). MR3: each lane holds E/32 keys (ord16(bf16 logit) << 16) |
// (0xFFFF - e), so one __reduce_max_sync per round picks the largest logit
// and, among equal logits, the smaller expert; a warp runs its tokens
// (b = w mod 4) interleaved. SoftmaxAll: weights are the full-E softmax at
// the selected experts (OLMoE, norm_topk_prob = false); otherwise the
// softmax of the selected logits (Qwen3, norm_topk_prob = true).
template<int Batch,int Experts,int TopK,int Tasks,bool FloatWeights,bool SoftmaxAll=false>
struct Sm90DenseRouterFinish {
  using Schedule=Sm90DenseMoeSchedule<Experts>;
  static_assert((Experts==64 || Experts==128) && TopK<=8 && Batch<=Schedule::kMaxRows);
  static constexpr int kPer=Experts/32,kTokens=(Batch+3)/4;
  // Logits are FP32 holding BF16 values, so the BF16 bits are the high half.
  // -0 is folded onto +0: the R3 float comparison treats them as equal (U5).
  __device__ static __forceinline__ unsigned Key(float logit,int e) {
    unsigned h=__float_as_uint(logit)>>16;
    if(h==0x8000u)h=0;
    unsigned order=h&0x8000u?~h&0xffffu:h|0x8000u;
    return order<<16|(0xffffu-unsigned(e));
  }
  __device__ static __forceinline__ float Value(unsigned key) {
    unsigned order=key>>16,h=order&0x8000u?order&0x7fffu:~order&0xffffu;
    return __uint_as_float(h<<16);
  }
  __device__ __forceinline__ void operator()(Sm90DenseMoeOperands const& p,int tasks,char* free) const {
    if(!tasks)return;
    int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
    auto* flag=reinterpret_cast<int*>(free);
    auto* selected=flag+4;  // [Batch][TopK]
    codegen::executor::ComputeSync();
    if(tid==0) {
      unsigned long long old;
      asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;":"=l"(old)
          :"l"(p.counter),"l"(static_cast<unsigned long long>(tasks)):"memory");
      *flag=(old+tasks)%Tasks==0;
    }
    codegen::executor::ComputeSync();
    if(tid==0)Sm90DenseMark(3);
    if(!*flag)return;
    auto const* logits=static_cast<float const*>(p.output);
    float value[kTokens][kPer];unsigned key[kTokens][kPer],chosen[kTokens][TopK];
    #pragma unroll
    for(int j=0;j<kTokens;++j)
      #pragma unroll
      for(int i=0;i<kPer;++i) {
        int b=warp+4*j;
        value[j][i]=b<Batch?logits[b*Experts+lane+32*i]:0.0f;
        key[j][i]=b<Batch?Key(value[j][i],lane+32*i):0u;
      }
    #pragma unroll
    for(int k=0;k<TopK;++k)
      #pragma unroll
      for(int j=0;j<kTokens;++j) {
        unsigned best=0;
        #pragma unroll
        for(int i=0;i<kPer;++i)best=max(best,key[j][i]);
        best=__reduce_max_sync(0xffffffffu,best);
        chosen[j][k]=best;
        // Keys are unique (they carry the expert), so one lane clears it.
        #pragma unroll
        for(int i=0;i<kPer;++i)if(key[j][i]==best)key[j][i]=0;
      }
    #pragma unroll
    for(int j=0;j<kTokens;++j) {
      int b=warp+4*j;
      if(b>=Batch)break;
      float top=Value(chosen[j][0]),sum=0;
      if constexpr(SoftmaxAll) {
        #pragma unroll
        for(int i=0;i<kPer;++i)sum+=expf(value[j][i]-top);
        #pragma unroll
        for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
      }else {
        #pragma unroll
        for(int k=0;k<TopK;++k)sum+=expf(Value(chosen[j][k])-top);
      }
      #pragma unroll
      for(int k=0;k<TopK;++k)
        if(lane==k) {
          int index=int(0xffffu-(chosen[j][k]&0xffffu));
          float weight=expf(Value(chosen[j][k])-top)/sum;
          p.route_index[b*TopK+k]=index;selected[b*TopK+k]=index;
          if constexpr(FloatWeights)static_cast<float*>(p.route_weights)[b*TopK+k]=weight;
          else static_cast<cutlass::bfloat16_t*>(p.route_weights)[b*TopK+k]=cutlass::bfloat16_t(weight);
        }
    }
    codegen::executor::ComputeSync();
    if(tid==0)Sm90DenseMark(4);
    // Thread e collects its rows; active experts take ascending slots.
    int rows[Batch],count=0;
    for(int b=0;b<Batch;++b)
      #pragma unroll
      for(int k=0;k<TopK;++k)if(selected[b*TopK+k]==tid)rows[count++]=b*TopK+k;
    unsigned ballot=__ballot_sync(0xffffffff,count>0);
    auto* offsets=flag+4+Batch*TopK;
    if(lane==0)offsets[warp]=__popc(ballot);
    codegen::executor::ComputeSync();
    int slot=__popc(ballot&((1u<<lane)-1));
    for(int w=0;w<warp;++w)slot+=offsets[w];
    if(count) {
      p.schedule[Schedule::Expert(slot)]=tid;p.schedule[Schedule::Count(slot)]=count;
      for(int r=0;r<count;++r)p.schedule[Schedule::Row(slot,r)]=rows[r];
    }
    if(tid==0)p.schedule[Schedule::Active()]=offsets[0]+offsets[1]+offsets[2]+offsets[3];
    // MR7 item 2: clear the fused stage's per-slot counters here, beside the
    // schedule record. M4's closing grid barrier orders these stores before
    // any M56 task of this layer, and this layer's M56 has not started; the
    // previous layer's finished before that barrier. Only active slots are
    // ever waited on, so a stale inactive counter is never read. A counter
    // per slot (not an epoch per expert) is required: n_active differs by
    // layer, so an inactive slot's count would not advance and an epoch
    // comparison would deadlock.
    if(p.gate_up_done && tid<Experts)p.gate_up_done[tid]=0;
    if(tid==0)Sm90DenseMark(5);
  }
};

// E2 (RC3, RC4, RC8): the routing a stage needs, computed by every CTA in its
// own shared memory. R4 had one CTA (the router's last arriver) decide the
// routes behind a grid barrier and publish them globally, so every later record
// read was a dependent global load. D0 measured that tail at 5.3 us/layer at B1
// and far more at B16, against a 3.4/7.3 us probe.
//
// Rows are ordered by s = b * TopK + k and slots by ascending expert number,
// which is exactly RouterFinish's order, so h's layout, P's writes and the
// combine order are unchanged and the tokens stay bitwise identical.
template<int Batch,int Experts,int TopK>
struct alignas(16) Sm90DenseLocalSchedule {
  static constexpr int kRows=16,kSelections=kRows*8;
  float weight[kSelections];            // s -> routing weight (bf16-rounded when !FloatWeights)
  unsigned char rows[Experts][kRows];   // active slot a, row r -> s, ascending in s
  short expert[Experts];                // slot -> expert, ascending
  unsigned char count[Experts];         // slot -> rows
  unsigned char sel[kSelections];       // s -> expert
  unsigned char cnt[4][Experts];        // warp w -> count of e among its selections
  unsigned char slot_of[Experts];       // expert -> slot
  int offsets[4];                       // warp w -> active experts it owns
  int active;
};

// Every CTA runs this on the same logits with the same instructions, so all
// CTAs agree bit for bit without any synchronization beyond the router stage's
// closing grid barrier, which already made the logits visible.
template<int Batch,int Experts,int TopK,bool FloatWeights,bool SoftmaxAll=false>
struct Sm90DenseLocalRoute {
  using Schedule=Sm90DenseMoeSchedule<Experts>;
  using LocalSchedule=Sm90DenseLocalSchedule<Batch,Experts,TopK>;
  using Finish=Sm90DenseRouterFinish<Batch,Experts,TopK,1,FloatWeights,SoftmaxAll>;
  static_assert((Experts==64 || Experts==128) && TopK<=8 && Batch<=Schedule::kMaxRows);
  static constexpr int kPer=Experts/32,kTokens=(Batch+3)/4;

  // 128 consumer threads; tid, warp and lane as in RouterFinish.
  __device__ static __forceinline__ void Build(float const* logits,LocalSchedule& s) {
    int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
    for(int i=tid;i<4*Experts;i+=128)reinterpret_cast<unsigned char*>(s.cnt)[i]=0;
    codegen::executor::ComputeSync();
    // Steps 2-3 are RouterFinish's own top-k and weights, unchanged.
    float value[kTokens][kPer];unsigned key[kTokens][kPer],chosen[kTokens][TopK];
    #pragma unroll
    for(int j=0;j<kTokens;++j)
      #pragma unroll
      for(int i=0;i<kPer;++i) {
        int b=warp+4*j;
        value[j][i]=b<Batch?logits[b*Experts+lane+32*i]:0.0f;
        key[j][i]=b<Batch?Finish::Key(value[j][i],lane+32*i):0u;
      }
    #pragma unroll
    for(int k=0;k<TopK;++k)
      #pragma unroll
      for(int j=0;j<kTokens;++j) {
        unsigned best=0;
        #pragma unroll
        for(int i=0;i<kPer;++i)best=max(best,key[j][i]);
        best=__reduce_max_sync(0xffffffffu,best);
        chosen[j][k]=best;
        #pragma unroll
        for(int i=0;i<kPer;++i)if(key[j][i]==best)key[j][i]=0;
      }
    #pragma unroll
    for(int j=0;j<kTokens;++j) {
      int b=warp+4*j;
      if(b>=Batch)break;
      float top=Finish::Value(chosen[j][0]),sum=0;
      if constexpr(SoftmaxAll) {
        #pragma unroll
        for(int i=0;i<kPer;++i)sum+=expf(value[j][i]-top);
        #pragma unroll
        for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
      }else {
        #pragma unroll
        for(int k=0;k<TopK;++k)sum+=expf(Finish::Value(chosen[j][k])-top);
      }
      #pragma unroll
      for(int k=0;k<TopK;++k)
        if(lane==k) {
          int index=int(0xffffu-(chosen[j][k]&0xffffu));
          float w=expf(Finish::Value(chosen[j][k])-top)/sum;
          s.sel[b*TopK+k]=static_cast<unsigned char>(index);
          // The down epilogue reads this value, so it is rounded here exactly
          // as R4 rounded it when storing route_weights.
          s.weight[b*TopK+k]=FloatWeights?w:float(cutlass::bfloat16_t(w));
        }
    }
    codegen::executor::ComputeSync();
    // Thread t holds selection t; __match_any_sync groups the lanes that chose
    // the same expert, giving each its rank within this warp.
    int e=tid<Batch*TopK?int(s.sel[tid]):255;
    unsigned peers=__match_any_sync(0xffffffffu,unsigned(e));
    int rank_w=__popc(peers&((1u<<lane)-1)),count_w=__popc(peers);
    if(tid<Batch*TopK && rank_w==0)s.cnt[warp][e]=static_cast<unsigned char>(count_w);
    codegen::executor::ComputeSync();
    // Thread e owns expert e: its slot is the number of active experts below it.
    // Experts is 64 or 128, so a warp is wholly inside or outside the range and
    // __ballot_sync stays convergent.
    unsigned ballot=0;
    if(tid<Experts) {
      int c=s.cnt[0][tid]+s.cnt[1][tid]+s.cnt[2][tid]+s.cnt[3][tid];
      ballot=__ballot_sync(0xffffffffu,c>0);
      if(lane==0)s.offsets[warp]=__popc(ballot);
    }else if(lane==0) {
      // E = 64 leaves warps 2 and 3 without experts; they publish zero so the
      // prefix below covers all four entries.
      s.offsets[warp]=0;
    }
    codegen::executor::ComputeSync();
    if(tid<Experts) {
      int slot=__popc(ballot&((1u<<lane)-1));
      #pragma unroll
      for(int w=0;w<4;++w)if(w<warp)slot+=s.offsets[w];
      int c=s.cnt[0][tid]+s.cnt[1][tid]+s.cnt[2][tid]+s.cnt[3][tid];
      s.slot_of[tid]=static_cast<unsigned char>(slot);
      if(c>0) {
        s.expert[slot]=static_cast<short>(tid);
        s.count[slot]=static_cast<unsigned char>(c);
      }
      if(tid==0)s.active=s.offsets[0]+s.offsets[1]+s.offsets[2]+s.offsets[3];
    }
    codegen::executor::ComputeSync();
    if(tid<Batch*TopK) {
      int r=rank_w;
      #pragma unroll
      for(int w=0;w<4;++w)if(w<warp)r+=s.cnt[w][e];
      s.rows[s.slot_of[e]][r]=static_cast<unsigned char>(tid);
    }
    codegen::executor::ComputeSync();
  }

  // CTA 0 alone republishes the schedule globally, for routing_check.py.
  __device__ static __forceinline__ void Publish(LocalSchedule const& s,
      int* schedule,void* route_weights,int* route_index) {
    int tid=threadIdx.x;
    if(tid==0)schedule[Schedule::Active()]=s.active;
    for(int a=tid;a<s.active;a+=128) {
      schedule[Schedule::Expert(a)]=s.expert[a];
      schedule[Schedule::Count(a)]=s.count[a];
      for(int r=0;r<s.count[a];++r)schedule[Schedule::Row(a,r)]=s.rows[a][r];
    }
    for(int i=tid;i<Batch*TopK;i+=128) {
      route_index[i]=s.sel[i];
      if constexpr(FloatWeights)static_cast<float*>(route_weights)[i]=s.weight[i];
      else static_cast<cutlass::bfloat16_t*>(route_weights)[i]=cutlass::bfloat16_t(s.weight[i]);
    }
  }
};

// Page: 0 when the weights are TN x TK tile pages; 64 when the experts keep
// the 64 x 64 pages the legacy MoE prefill shares (one copy of the expert
// weights fits the card; recorded deviation from §8's per-TN packing). A
// slot is then assembled from runs of whole or half pages per 64-column panel.
template<int Batch,int N,int K,int TN,int TK,int Slots,int WN,int WK,Sm90DenseMoeMode Mode,int U,
         int Experts,int TopK,bool FloatWeights,int Page=0,int RegisterStages=1,bool SoftmaxAll=false>
struct Sm90DenseMoeGemm {
  static constexpr bool kInputNorm=Mode!=Sm90DenseMoeMode::kDown;
  using Core=Sm90DenseGemm<Batch,N,K,TN,TK,Slots,WN,WK,ServingEpilogueOp::kStore,U,kInputNorm,
      RegisterStages>;
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Schedule=Sm90DenseMoeSchedule<Experts>;
  static constexpr int kTiles=N/TN,kIterations=K/TK,kRows=Core::kRows,kThreads=Core::kThreads;
  static constexpr int kSharedBytes=Core::kSharedBytes,kSlotBytes=Core::kSlotBytes;
  static constexpr auto kLayout=Core::kLayout;
  static constexpr bool kDynamic=Mode!=Sm90DenseMoeMode::kRouter;
  // M5/M6 run n_active x kTiles tasks; this bounds them.
  static constexpr int kMaxTasks=kDynamic?Experts*kTiles:kTiles;
  static_assert(Batch<=Schedule::kMaxRows && Experts<=128 && TopK<=8);
  static_assert(Mode!=Sm90DenseMoeMode::kGateUp || TN%(2*U)==0);
  static_assert(Page==0 || (Page==64 && kDynamic && TN%32==0 && N%64==0 && K%64==0));
  // Kept as the reference the E2 equivalence test compares against; the
  // router stage no longer runs it.
  using RouterFinish=Sm90DenseRouterFinish<Batch,Experts,TopK,kTiles,FloatWeights,SoftmaxAll>;

  __device__ static __forceinline__ int TotalTasks(Sm90DenseMoeOperands const& p) {
    if constexpr(kDynamic)return p.schedule[Schedule::Active()]*kTiles;
    else return kTiles;
  }
  // Weight tiles of task t: router tile t, else tile t % kTiles of the
  // expert in slot t / kTiles (one expert's pages are N x K).
  __device__ static __forceinline__ Element const* Tiles(Element const* weight,int const* schedule,int t) {
    if constexpr(!kDynamic)return weight+std::size_t(t)*kIterations*TN*TK;
    else if constexpr(Page)return weight+std::size_t(schedule[Schedule::Expert(t/kTiles)])*N*K;
    else return weight+std::size_t(schedule[Schedule::Expert(t/kTiles)])*N*K+
        std::size_t(t%kTiles)*kIterations*TN*TK;
  }
  // One slot's weights: task t's tile, k-tile `tile`; Tiles(...) is the base.
  __device__ static __forceinline__ void CopyWeights(Element* destination,Element const* tiles,int t,
      int tile,std::uint64_t* barrier) {
    // E1: the expert stack is the layer's largest read-once stream.
    std::uint64_t const policy=Sm90DenseStream::WeightPolicy();
    if constexpr(!Page)
      Sm90DenseStream::WeightBulk(destination,tiles+std::size_t(tile)*TN*TK,2*TN*TK,barrier,policy);
    else {
      // Rows of a 64-row page are contiguous in 8-row groups, and the slot's
      // [TK/64][TN] panel layout uses the same atom (TN is a multiple of 32).
      int first=(t%kTiles)*TN;
      #pragma unroll
      for(int panel=0;panel<TK/64;++panel) {
        int column=tile*TK/64+panel;
        for(int row=first;row<first+TN;) {
          int end=min(first+TN,(row/64+1)*64);
          Sm90DenseStream::WeightBulk(destination+512*((row-first)/8+(TN/8)*panel),
              tiles+(std::size_t(row/64)*(K/64)+column)*4096+512*((row%64)/8),2*(end-row)*64,
              barrier,policy);
          row=end;
        }
      }
    }
  }

  // D6 stream for this CTA's first task: into M4 from o, into M6 from M5's
  // producer (the routes are known once M4 has run).
  template<int Bytes>
  __device__ static __forceinline__ void Preissue(Element const* weight,int const* schedule,
      char* shared,Sm90DenseRingState& state) {
    static_assert(Bytes>0 && Bytes%16==0 && Bytes<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    static_assert(Mode!=Sm90DenseMoeMode::kGateUp,"M4 does not stream into M5");
    static_assert(Page==0 || Bytes%(2*TN*TK)==0,"paged experts stream whole slots");
    int t=int(blockIdx.x);
    if(t>=(kDynamic?schedule[Schedule::Active()]*kTiles:kTiles))return;
    auto* b=reinterpret_cast<Element*>(shared+1024)+Slots*kRows*TK;
    auto const* tiles=Tiles(weight,schedule,t);
    int remaining=Bytes;
    #pragma unroll
    for(int slot=0;slot<Slots && slot<kIterations;++slot) {
      int part=min(remaining,2*TN*TK);remaining-=part;
      if(part) {
        Async::ExpectTx(state.full+slot,kSlotBytes);
        if constexpr(Page)CopyWeights(b+slot*TN*TK,tiles,t,slot,state.full+slot);
        else Async::Bulk(b+slot*TN*TK,tiles+std::size_t(slot)*TN*TK,part,state.full+slot);
      }
    }
  }

  // Byte offset of (row, col) in a [TK/64][kRows][64] SWIZZLE_128B slot.
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return (col/64)*kRows*128+row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }

  // MR1: a task's schedule record in registers. Every lane holds the expert
  // and count; lane r < 16 holds row r's token (rows past count are unused).
  // Rows are loaded without waiting for count, so no load depends on another.
  struct Record {int expert,count,route;};
  __device__ static __forceinline__ Record Load(int const* schedule,int t) {
    if constexpr(!kDynamic)return {};
    else {
      // A speculative first record may lie past n_active; keep it in the table.
      int a=min(t/kTiles,Experts-1),lane=threadIdx.x&31;
      return {schedule[Schedule::Expert(a)],Mode==Sm90DenseMoeMode::kGateUp?schedule[Schedule::Count(a)]:0,
          Mode==Sm90DenseMoeMode::kGateUp && lane<Schedule::kMaxRows?schedule[Schedule::Row(a,lane)]:0};
    }
  }
  __device__ static __forceinline__ Element const* TaskTiles(Element const* weight,int expert,int t) {
    if constexpr(!kDynamic)return weight+std::size_t(t)*kIterations*TN*TK;
    else if constexpr(Page)return weight+std::size_t(expert)*N*K;
    else return weight+std::size_t(expert)*N*K+std::size_t(t%kTiles)*kIterations*TN*TK;
  }

  // first: this CTA's first record, loaded beside n_active (MR1). The next
  // task's record is loaded when a task starts and is consumed one task
  // later, so no global load sits between an empty wait and its copies.
  template<int Prefix>
  __device__ static __noinline__ void Produce(Sm90DenseMoeOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase,bool streamed,Record first) {
    Sm90DenseRoles::Producer();
    int lane=threadIdx.x&31;
    // x', h and the schedule were written with generic stores before the
    // grid barrier; the async proxy must observe them.
    if(lane==0)Async::ProxyAsyncGlobalFence();
    Record next=first;
    for(int local=0;local<tasks;++local) {
      int t=int(blockIdx.x)+local*int(gridDim.x);
      Record current=next;
      if(local+1<tasks)next=Load(p.schedule,t+int(gridDim.x));
      auto const* tiles=TaskTiles(p.weight,current.expert,t);
      int token=current.route/TopK;
      for(int tile=0;tile<kIterations;++tile) {
        int sequence=local*kIterations+tile,slot=sequence%Slots;
        Async::Wait(state.empty+slot,(phase>>slot)&1);
        if constexpr(Mode==Sm90DenseMoeMode::kGateUp) {
          // The producer lanes gather the slot's rows of x' (one TK segment
          // each) into the swizzled slot. A non-.noinc cp.async arrive adds a
          // pending arrival that its copies retire, so lane 0's single
          // expect_tx arrival must follow the warp's arrives.
          auto base=Async::Shared(a+slot*kRows*TK);
          for(int first_chunk=0;first_chunk<current.count*(TK/8);first_chunk+=32) {
            int chunk=first_chunk+lane,r=chunk/(TK/8),col=(chunk%(TK/8))*8;
            int row_token=__shfl_sync(0xffffffff,token,r&(Schedule::kMaxRows-1));
            if(chunk<current.count*(TK/8))
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(base+Swizzled(r,col)),
                  "l"(p.input+std::size_t(row_token)*K+tile*TK+col):"memory");
          }
          asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];"::
              "r"(Async::Shared(state.full+slot)):"memory");
          __syncwarp();
          if(lane==0) {
            Async::ExpectTx(state.full+slot,2*TN*TK);
            CopyWeights(b+slot*TN*TK,tiles,t,tile,state.full+slot);
          }
        }else if(lane==0) {
          // Only M6's first task can have been streamed by M5 (D6).
          unsigned prefix=0;
          if constexpr(Prefix>0)
            if(streamed && sequence<(Prefix+2*TN*TK-1)/(2*TN*TK))
              prefix=min(Prefix-sequence*2*TN*TK,2*TN*TK);
          if(!prefix)Async::ExpectTx(state.full+slot,kSlotBytes);
          if(!prefix)CopyWeights(b+slot*TN*TK,tiles,t,tile,state.full+slot);
          else if constexpr(!Page)
            if(prefix<2*TN*TK)Async::Bulk(b+slot*TN*TK+prefix/2,tiles+std::size_t(tile)*TN*TK+prefix/2,
                2*TN*TK-prefix,state.full+slot);
          // Rows past the batch (router) or the slot's count (down) are never
          // written out.
          if constexpr(Mode==Sm90DenseMoeMode::kRouter)
            Async::Tensor3D(a+slot*kRows*TK,p.activation,0,0,tile*(TK/64),state.full+slot);
          else Async::Tensor4D(a+slot*kRows*TK,p.activation,0,0,tile*(TK/64),t/kTiles,state.full+slot);
        }
        phase^=1u<<slot;
      }
    }
    if(lane==0 && tasks)for(int slot=0;slot<Slots;++slot)
      state.producer_generations[slot]=(phase>>slot)&1;
    __syncwarp();
  }

  // value(row, col): the WK partial tiles summed; inverse(row): RMS of x'.
  __device__ static __forceinline__ float Value(float const* tile,int row,int col) {
    float sum=tile[row*TN+col];
    #pragma unroll
    for(int s=1;s<WK;++s)sum+=tile[s*16*TN+row*TN+col];
    return sum;
  }
  __device__ static __forceinline__ float Inverse(float const* norms,int row,float epsilon) {
    if constexpr(!kInputNorm)return 1.0f;
    else {
      float squares=0;
      #pragma unroll
      for(int s=0;s<WK;++s)squares+=norms[s*16+row];
      return rsqrtf(squares/float(K)+epsilon);
    }
  }
  // E2: `routing` replaces every global record read in the expert epilogues.
  // The router mode never uses it and passes null.
  using LocalSchedule=Sm90DenseLocalSchedule<Batch,Experts,TopK>;
  __device__ static __forceinline__ void Epilogue(Sm90DenseMoeOperands const& p,float const* tile,
      float const* norms,int t,int* flag,LocalSchedule const* routing=nullptr) {
    int tid=threadIdx.x,group=t/kTiles,j=t%kTiles;
    if constexpr(Mode==Sm90DenseMoeMode::kRouter) {
      // fp32 logits that hold the BF16 rounding HF's router produces.
      auto* logits=static_cast<float*>(p.output);
      for(int i=tid;i<Batch*TN;i+=128) {
        int row=i/TN,col=i%TN;
        logits[row*N+t*TN+col]=float(Element(Value(tile,row,col)*Inverse(norms,row,p.epsilon)));
      }
    }else if constexpr(Mode==Sm90DenseMoeMode::kGateUp) {
      // SwiGLU (interleave U) into the slot's compact rows h[a][r][N/2].
      int count=routing->count[group];
      auto* h=static_cast<Element*>(p.output)+std::size_t(group)*Schedule::kMaxRows*(N/2)+j*(TN/2);
      for(int i=tid;i<count*(TN/2);i+=128) {
        int row=i/(TN/2),col=i%(TN/2),gate=2*U*(col/U)+col%U;
        float scale=Inverse(norms,row,p.epsilon);
        h[row*(N/2)+col]=ServingEpilogueValue<ServingEpilogueOp::kSwiGLU>::Apply(
            Value(tile,row,gate)*scale,Value(tile,row,gate+U)*scale);
      }
    }else {
      // P[b][k][j tile] = w_{b,k} y, then the n_active-th arriver of tile j
      // adds x' and the k partials in fixed k order (ctr[j] returns to 0).
      int count=routing->count[group];
      auto* partial=static_cast<float*>(p.output);
      for(int i=tid;i<count*TN;i+=128) {
        // E2: count, the row and its weight all come from shared memory, so
        // the count -> row -> weight dependent global chain (RC4) is gone.
        int row=i/TN,col=i%TN,route=routing->rows[group][row];
        float weight=routing->weight[route];
        partial[std::size_t(route)*N+j*TN+col]=weight*Value(tile,row,col);
      }
      codegen::executor::ComputeSync();
      if(tid==0) {
        unsigned long long old;
        asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;":"=l"(old):"l"(p.counter+j):"memory");
        *flag=int(old+1)==routing->active;
        if(*flag)p.counter[j]=0;
      }
      codegen::executor::ComputeSync();
      if(tid==0 && *flag)Sm90DenseMark(4);
      if(*flag) {
        // E3 (RC2): one scalar element per iteration made every round wait on
        // the previous one, because the compiler cannot prove the store to x
        // does not alias the next round's partial reads. D0 measured the
        // segment at 16.6 us/layer at B16. Each thread now takes four adjacent
        // columns as a float4 and issues a whole batch of reads before any
        // arithmetic, so the misses overlap. The accumulation order is still
        // k ascending from zero and the write is still bf16(float(x) + s), so
        // x'' is bitwise what R4 produced.
        static_assert(TN%4==0,"E3 reads the combine in float4 columns");
        constexpr int kGroups=Batch*TN/4,kPer=(kGroups+127)/128,kBatch=2;
        #pragma unroll 1
        for(int first=0;first<kPer;first+=kBatch) {
          float4 part[kBatch][TopK];uint2 old_x[kBatch];
          int row[kBatch],col[kBatch];bool ok[kBatch];
          #pragma unroll
          for(int m=0;m<kBatch;++m) {
            int g=tid+128*(first+m);
            ok[m]=first+m<kPer && g<kGroups;
            row[m]=g/(TN/4);col[m]=j*TN+4*(g%(TN/4));
            if(ok[m]) {
              #pragma unroll
              for(int k=0;k<TopK;++k)
                part[m][k]=*reinterpret_cast<float4 const*>(
                    partial+std::size_t(row[m]*TopK+k)*N+col[m]);
              old_x[m]=*reinterpret_cast<uint2 const*>(p.x+std::size_t(row[m])*N+col[m]);
            }
          }
          #pragma unroll
          for(int m=0;m<kBatch;++m)if(ok[m]) {
            float sum[4]={0,0,0,0};
            #pragma unroll
            for(int k=0;k<TopK;++k) {
              sum[0]+=part[m][k].x;sum[1]+=part[m][k].y;
              sum[2]+=part[m][k].z;sum[3]+=part[m][k].w;
            }
            // bitcast/raw keep this to the documented bf16 bit API; the four
            // results are packed into one 8-byte store, which the column
            // alignment (TN and N multiples of 4) makes well aligned.
            std::uint16_t out[4];
            #pragma unroll
            for(int c=0;c<4;++c) {
              auto bits=std::uint16_t(c<2?(old_x[m].x>>(16*c)):(old_x[m].y>>(16*(c-2))));
              out[c]=Element(float(Element::bitcast(bits))+sum[c]).raw();
            }
            uint2 stored;
            stored.x=unsigned(out[0])|(unsigned(out[1])<<16);
            stored.y=unsigned(out[2])|(unsigned(out[3])<<16);
            *reinterpret_cast<uint2*>(p.x+std::size_t(row[m])*N+col[m])=stored;
          }
        }
      }
      if(tid==0 && *flag)Sm90DenseMark(5);
    }
  }

  __device__ static __noinline__ void Consume(Sm90DenseMoeOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,int tasks,unsigned phase) {
    Sm90DenseRoles::Consumer();
    using namespace cute;
    int tid=threadIdx.x,lane=tid%(32*WN),split=tid/(32*WN);
    float* norms=scratch+WK*16*TN;
    auto* flag=reinterpret_cast<int*>(norms+64);
    typename Core::Mma mma;auto thread=mma.get_slice(lane);
    auto owned=thread.partition_C(make_identity_tensor(Shape<_16,Int<TN>>{}));
    for(int local=0;local<tasks;++local) {
      int t=int(blockIdx.x)+local*int(gridDim.x);
      auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
      float squares[2]={0,0};
      for(int iteration=0;iteration<kIterations;++iteration) {
        int sequence=local*kIterations+iteration,slot=sequence%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        if(tid==0)Sm90DenseMarkFirst(0);
        auto* sa=a+slot*kRows*TK;auto* sb=b+slot*TN*TK;
        if constexpr(WK==1)Core::template MmaSlice<0>(sa,sb,accum,squares);
        else if constexpr(WK==2) {
          if(split==0)Core::template MmaSlice<0>(sa,sb,accum,squares);
          else Core::template MmaSlice<1>(sa,sb,accum,squares);
        }else {
          if(split==0)Core::template MmaSlice<0>(sa,sb,accum,squares);
          else if(split==1)Core::template MmaSlice<1>(sa,sb,accum,squares);
          else if(split==2)Core::template MmaSlice<2>(sa,sb,accum,squares);
          else Core::template MmaSlice<3>(sa,sb,accum,squares);
        }
        // Readers fence before the async refill (see Sm90DenseGemm).
        Async::FenceProxyAsyncShared();
        Async::Arrive(state.empty+slot);
      }
      if constexpr(kInputNorm)
        #pragma unroll
        for(int r=0;r<2;++r) {
          squares[r]+=__shfl_xor_sync(0xffffffff,squares[r],1);
          squares[r]+=__shfl_xor_sync(0xffffffff,squares[r],2);
        }
      for(int i=0;i<size(accum);++i)
        scratch[split*16*TN+get<0>(owned(i))*TN+get<1>(owned(i))]=accum(i);
      if constexpr(kInputNorm)
        if((tid/32)%WN==0 && (tid&3)==0) {
          int row=(tid&31)/4;
          norms[split*16+row]=squares[0];norms[split*16+row+8]=squares[1];
        }
      if(tid==0)Sm90DenseMark(1);
      codegen::executor::ComputeSync();
      Epilogue(p,scratch,norms,t,flag);
      codegen::executor::ComputeSync();
      if(tid==0)Sm90DenseMark(2);
    }
    if(tid==0 && tasks)for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
    // Every task has been consumed: the operand slots are free.
    // E2: the router stage ends at its logits. The routing itself moved into
    // the fused expert stage, where every CTA derives it from those logits, so
    // there is no last arriver, no global schedule write and no grid-barrier
    // hop between deciding the routes and using them (RC3, RC4, RC8).
    // The per-slot counters the fused stage waits on must still be cleared
    // before it starts; the router's closing grid barrier orders this write
    // ahead of every M56 task, and the previous layer's M56 finished before
    // that barrier. Only the last CTA does it, and it owns no router task.
    if constexpr(Mode==Sm90DenseMoeMode::kRouter)
      if(p.gate_up_done && int(blockIdx.x)==int(gridDim.x)-1 && tid<Experts)
        p.gate_up_done[tid]=0;
  }

  template<int Prefix=0,class Tail=Sm90DenseNoTail>
  __device__ static __forceinline__ void Run(Sm90DenseMoeOperands const& p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={},bool streamed=true) {
    static_assert(Prefix>=0 && Prefix%16==0 && Prefix<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    static_assert(Mode!=Sm90DenseMoeMode::kGateUp || Prefix==0,"M4 does not stream into M5");
    static_assert(Page==0 || Prefix%(2*TN*TK)==0,"paged experts stream whole slots");
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*kRows*TK;
    auto* scratch=reinterpret_cast<float*>(shared+kLayout.scratch);
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    // The producer's first record is read beside n_active (MR1); a CTA past
    // the last task discards it.
    Record first{};
    if(threadIdx.x>=128)first=Load(p.schedule,int(blockIdx.x));
    int total=TotalTasks(p);
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,tasks,phase);
    else {
      Produce<Prefix>(p,state,a,b,tasks,phase,streamed,first);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

// MR7 (prompt §4 A2): M5 and M6 fused into one stage M56. Tasks [0, n T_gu)
// are gate_up and [n T_gu, n (T_gu + T_dn)) are down, so with CTA b taking
// t = b + k G every CTA's gate_up tasks precede its down tasks. gu_done[a]
// counts the gate_up tasks of active slot a that have published h; a down
// task's producer issues the weights it can (they do not depend on h), then
// waits for gu_done[a] >= T_gu before reading h by TMA.
//
// Why it cannot deadlock: a gate_up task never waits, and in every CTA the
// gate_up tasks come before the down tasks, so the CTAs holding the gate_up
// tasks of slot a always reach their releases; the producer's spin is in the
// producer warp and does not block its own CTA's consumer.
// Why x'' may be written in place: tile j's combine waits for all n_active
// down tasks (a, j), and every down task waited for all T_gu gate_up tasks of
// its slot, so every gate_up read of x' has retired before any combine writes
// x''.
template<int Batch,int NGateUp,int NDown,int Hidden,int TNGu,int TKGu,int TNDn,int TKDn,int Slots,
         int WNGu,int WKGu,int WNDn,int WKDn,int U,int Experts,int TopK,bool FloatWeights,
         int Page=0,int RegisterStages=1,bool SoftmaxAll=false>
struct Sm90DenseMoeFusedGemm {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Schedule=Sm90DenseMoeSchedule<Experts>;
  // The experts' intermediate width: gate_up emits the two interleaved halves.
  static constexpr int kIntermediate=NGateUp/2;
  using Gu=Sm90DenseMoeGemm<Batch,NGateUp,Hidden,TNGu,TKGu,Slots,WNGu,WKGu,
      Sm90DenseMoeMode::kGateUp,U,Experts,TopK,FloatWeights,Page,RegisterStages>;
  using Dn=Sm90DenseMoeGemm<Batch,NDown,kIntermediate,TNDn,TKDn,Slots,WNDn,WKDn,
      Sm90DenseMoeMode::kDown,U,Experts,TopK,FloatWeights,Page,RegisterStages>;
  static constexpr int kTilesGu=NGateUp/TNGu,kTilesDn=NDown/TNDn;
  static constexpr int kIterGu=Hidden/TKGu,kIterDn=kIntermediate/TKDn;
  static constexpr int kRows=Sm90DenseLayout::ActivationRows(Batch);
  static constexpr int kThreads=160;
  // One ring for both task types: a slot takes the larger of the two.
  static constexpr int kMaxTK=TKGu>TKDn?TKGu:TKDn;
  static constexpr int kWeightSlot=TNGu*TKGu>TNDn*TKDn?TNGu*TKGu:TNDn*TKDn;
  static constexpr int kSlotBytesGu=2*TKGu*(kRows+TNGu),kSlotBytesDn=2*TKDn*(kRows+TNDn);
  using LocalSchedule=Sm90DenseLocalSchedule<Batch,Experts,TopK>;
  using LocalRoute=Sm90DenseLocalRoute<Batch,Experts,TopK,FloatWeights,SoftmaxAll>;
  // The solver sizes this arena from Sm90DenseLayout::MoeRouteBytes, so the
  // two must agree or a geometry could pass its budget check and overflow.
  static_assert(int(sizeof(LocalSchedule))<=Sm90DenseLayout::MoeRouteBytes(Experts),
      "MoeRouteBytes understates the local schedule");
  static constexpr int kRouteBytes=(Sm90DenseLayout::MoeRouteBytes(Experts)+127)/128*128;
  static constexpr auto kLayout=Sm90DenseLayout::MoeFused(TNGu,TKGu,TNDn,TKDn,Slots,WKGu,WKDn,
      Batch,Sm90DenseLayout::MoeRouteBytes(Experts));
  static constexpr int kSharedBytes=kLayout.end;
  // E2: the route region is the tail of the arena (Sm90DenseLayout::MoeFused).
  static constexpr int kRouteOffset=kLayout.end-kRouteBytes;
  static constexpr int kMaxTasks=Experts*(kTilesGu+kTilesDn);
  // M4's routes are unknown to its predecessor, so M56 receives no D6 stream.
  static constexpr int kMaxPreissue=0;
  static_assert(Batch<=Schedule::kMaxRows && Experts<=128 && TopK<=8);
  static_assert(TNGu%(2*U)==0,"gate_up interleaves the two halves by U");
  static_assert(Page==0 || (Page==64 && TNGu%32==0 && TNDn%32==0));

  __device__ static __forceinline__ int Active(Sm90DenseMoeFusedOperands const& p) {
    return p.schedule[Schedule::Active()];
  }
  __device__ static __forceinline__ int TotalTasks(Sm90DenseMoeFusedOperands const& p) {
    return Active(p)*(kTilesGu+kTilesDn);
  }
  // Task t: gate_up while t < n T_gu, else down. `slot` is the active expert
  // slot a and `tile` the column tile j of that task's own family.
  struct Task {int slot,tile;bool down;};
  // MR1's record, read once per task. The component bodies each nest their
  // own, so the fused stage carries one for both task types.
  struct Record {int expert,count,route;};
  __device__ static __forceinline__ Task Of(int t,int active) {
    int split=active*kTilesGu;
    if(t<split)return {t/kTilesGu,t%kTilesGu,false};
    int u=t-split;return {u/kTilesDn,u%kTilesDn,true};
  }
  // The component bodies' epilogues read only these fields.
  __device__ static __forceinline__ Sm90DenseMoeOperands GateUpOperands(
      Sm90DenseMoeFusedOperands const& p) {
    return {nullptr,p.weight_gate_up,p.input,p.hidden,p.schedule,p.route_weights,p.route_index,p.x,
        p.counter,p.epsilon,nullptr};
  }
  __device__ static __forceinline__ Sm90DenseMoeOperands DownOperands(
      Sm90DenseMoeFusedOperands const& p) {
    return {p.activation,p.weight_down,p.input,p.output,p.schedule,p.route_weights,p.route_index,p.x,
        p.counter,p.epsilon,nullptr};
  }

  // MR7 item 4: acquire gu_done[a] >= T_gu, then order the async proxy after
  // it so the TMA of h observes the gate_up stores.
  __device__ static __forceinline__ void WaitGateUp(unsigned const* gate_up_done,int slot) {
    unsigned seen;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(seen)
          :"l"(gate_up_done+slot):"memory");
    } while(seen<unsigned(kTilesGu));
    Async::ProxyAsyncGlobalFence();
  }

  __device__ static __noinline__ void ProduceGateUp(Sm90DenseMoeFusedOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,Task task,int lane,unsigned& phase,
      int& sequence,Record record) {
    auto const* tiles=Gu::TaskTiles(p.weight_gate_up,record.expert,task.slot*kTilesGu+task.tile);
    int token=record.route/TopK;
    for(int tile=0;tile<kIterGu;++tile) {
      int slot=(sequence++)%Slots;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      // The producer lanes gather the slot's rows of x' (one TK segment each)
      // into the swizzled slot; a non-.noinc cp.async arrive adds a pending
      // arrival its copies retire, so lane 0's expect_tx must follow them.
      auto base=Async::Shared(a+slot*kRows*kMaxTK);
      for(int first=0;first<record.count*(TKGu/8);first+=32) {
        int chunk=first+lane,r=chunk/(TKGu/8),col=(chunk%(TKGu/8))*8;
        int row_token=__shfl_sync(0xffffffff,token,r&(Schedule::kMaxRows-1));
        if(chunk<record.count*(TKGu/8))
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::
              "r"(base+Gu::Swizzled(r,col)),
              "l"(p.input+std::size_t(row_token)*Hidden+tile*TKGu+col):"memory");
      }
      asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];"::
          "r"(Async::Shared(state.full+slot)):"memory");
      __syncwarp();
      if(lane==0) {
        Async::ExpectTx(state.full+slot,2*TNGu*TKGu);
        Gu::CopyWeights(b+slot*kWeightSlot,tiles,task.slot*kTilesGu+task.tile,tile,state.full+slot);
      }
      phase^=1u<<slot;
    }
  }

  // The weights do not depend on h, so they are issued for the slots this
  // task can claim before the producer waits on gu_done (MR7 item 4).
  __device__ static __noinline__ void ProduceDown(Sm90DenseMoeFusedOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,Task task,int lane,unsigned& phase,
      int& sequence,Record record) {
    auto const* tiles=Dn::TaskTiles(p.weight_down,record.expert,task.slot*kTilesDn+task.tile);
    constexpr int kAhead=Slots<kIterDn?Slots:kIterDn;
    int first=sequence;
    for(int tile=0;tile<kAhead;++tile) {
      int slot=(sequence++)%Slots;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      if(lane==0) {
        Async::ExpectTx(state.full+slot,kSlotBytesDn);
        Dn::CopyWeights(b+slot*kWeightSlot,tiles,task.slot*kTilesDn+task.tile,tile,state.full+slot);
      }
      phase^=1u<<slot;
    }
    // The whole warp waits. A lane-0-only spin let lanes 1-31 run ahead, and the
    // warp could reach the stage's final __syncthreads (bar.sync is .aligned,
    // counted per warp execution) split into groups that completed the CTA
    // barrier without the consumers: the MR7 hang, about one stall in 25k
    // fused stages, verified by per-thread barrier markers.
    WaitGateUp(p.gate_up_done,task.slot);
    // Producer side: thread 128 owns the mark (appendix D).
    if(threadIdx.x==128)Sm90DenseMarkFirst(3);
    if(lane==0) {
      // h of the slots whose weights are already in flight, then the rest.
      for(int tile=0;tile<kAhead;++tile)
        Async::Tensor4D(a+((first+tile)%Slots)*kRows*kMaxTK,p.activation,0,0,tile*(TKDn/64),
            task.slot,state.full+(first+tile)%Slots);
    }
    __syncwarp();
    for(int tile=kAhead;tile<kIterDn;++tile) {
      int slot=(sequence++)%Slots;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      if(lane==0) {
        Async::ExpectTx(state.full+slot,kSlotBytesDn);
        Dn::CopyWeights(b+slot*kWeightSlot,tiles,task.slot*kTilesDn+task.tile,tile,state.full+slot);
        Async::Tensor4D(a+slot*kRows*kMaxTK,p.activation,0,0,tile*(TKDn/64),task.slot,
            state.full+slot);
      }
      phase^=1u<<slot;
    }
  }

  __device__ static __forceinline__ void Produce(Sm90DenseMoeFusedOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase,int active,
      LocalSchedule const& routing) {
    Sm90DenseRoles::Producer();
    int lane=threadIdx.x&31;
    // x', h and the schedule were written with generic stores before the grid
    // barrier; the async proxy must observe them.
    if(lane==0)Async::ProxyAsyncGlobalFence();
    int sequence=0;
    for(int local=0;local<tasks;++local) {
      int t=int(blockIdx.x)+local*int(gridDim.x);
      Task task=Of(t,active);
      // E2: the record is read from this CTA's own shared schedule, so the
      // per-task global loads of RC8 are gone along with their latency.
      int a_index=min(task.slot,Experts-1);
      Record record{routing.expert[a_index],routing.count[a_index],
          lane<LocalSchedule::kRows?routing.rows[a_index][lane]:0};
      if(task.down)ProduceDown(p,state,a,b,task,lane,phase,sequence,record);
      else ProduceGateUp(p,state,a,b,task,lane,phase,sequence,record);
    }
    if(lane==0 && tasks)for(int slot=0;slot<Slots;++slot)
      state.producer_generations[slot]=(phase>>slot)&1;
    __syncwarp();
  }

  // U7's preset: the two MMA configurations are never live together, so each
  // task type runs in its own __noinline__ consumer.
  template<class Body,int Iterations,int TN,int TK,int WN,int WK>
  __device__ static __forceinline__ void ConsumeTask(Sm90DenseMoeOperands const& q,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,int t,unsigned& phase,
      int& sequence,int* flag,LocalSchedule const& routing) {
    using namespace cute;
    int tid=threadIdx.x,lane=tid%(32*WN),split=tid/(32*WN);
    float* norms=scratch+WK*16*TN;
    typename Body::Core::Mma mma;auto thread=mma.get_slice(lane);
    auto owned=thread.partition_C(make_identity_tensor(Shape<_16,Int<TN>>{}));
    auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
    float squares[2]={0,0};
    for(int iteration=0;iteration<Iterations;++iteration) {
      int slot=(sequence++)%Slots;
      Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
      if(tid==0)Sm90DenseMarkFirst(1);
      auto* sa=a+slot*kRows*kMaxTK;auto* sb=b+slot*kWeightSlot;
      if constexpr(WK==1)Body::Core::template MmaSlice<0>(sa,sb,accum,squares);
      else if constexpr(WK==2) {
        if(split==0)Body::Core::template MmaSlice<0>(sa,sb,accum,squares);
        else Body::Core::template MmaSlice<1>(sa,sb,accum,squares);
      }else {
        if(split==0)Body::Core::template MmaSlice<0>(sa,sb,accum,squares);
        else if(split==1)Body::Core::template MmaSlice<1>(sa,sb,accum,squares);
        else if(split==2)Body::Core::template MmaSlice<2>(sa,sb,accum,squares);
        else Body::Core::template MmaSlice<3>(sa,sb,accum,squares);
      }
      Async::FenceProxyAsyncShared();
      Async::Arrive(state.empty+slot);
    }
    if constexpr(Body::kInputNorm)
      #pragma unroll
      for(int r=0;r<2;++r) {
        squares[r]+=__shfl_xor_sync(0xffffffff,squares[r],1);
        squares[r]+=__shfl_xor_sync(0xffffffff,squares[r],2);
      }
    for(int i=0;i<size(accum);++i)
      scratch[split*16*TN+get<0>(owned(i))*TN+get<1>(owned(i))]=accum(i);
    if constexpr(Body::kInputNorm)
      if((tid/32)%WN==0 && (tid&3)==0) {
        int row=(tid&31)/4;
        norms[split*16+row]=squares[0];norms[split*16+row+8]=squares[1];
      }
    codegen::executor::ComputeSync();
    Body::Epilogue(q,scratch,norms,t,flag,&routing);
    codegen::executor::ComputeSync();
  }
  __device__ static __noinline__ void ConsumeGateUp(Sm90DenseMoeFusedOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,Task task,unsigned& phase,
      int& sequence,int* flag,LocalSchedule const& routing) {
    auto q=GateUpOperands(p);
    ConsumeTask<Gu,kIterGu,TNGu,TKGu,WNGu,WKGu>(q,state,a,b,scratch,
        task.slot*kTilesGu+task.tile,phase,sequence,flag,routing);
    // MR7 item 3: h's rows are published; release them to the down tasks.
    if(threadIdx.x==0) {
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;"::
          "l"(p.gate_up_done+task.slot):"memory");
      Sm90DenseMark(2);
    }
  }
  __device__ static __noinline__ void ConsumeDown(Sm90DenseMoeFusedOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,Task task,unsigned& phase,
      int& sequence,int* flag,LocalSchedule const& routing) {
    auto q=DownOperands(p);
    ConsumeTask<Dn,kIterDn,TNDn,TKDn,WNDn,WKDn>(q,state,a,b,scratch,
        task.slot*kTilesDn+task.tile,phase,sequence,flag,routing);
  }

  __device__ static __forceinline__ void Consume(Sm90DenseMoeFusedOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,int tasks,unsigned phase,
      int active,LocalSchedule const& routing) {
    Sm90DenseRoles::Consumer();
    constexpr int kScratchTN=TNGu>TNDn?TNGu:TNDn,kScratchWK=WKGu>WKDn?WKGu:WKDn;
    auto* flag=reinterpret_cast<int*>(scratch+kScratchWK*16*kScratchTN+64);
    int sequence=0;
    for(int local=0;local<tasks;++local) {
      int t=int(blockIdx.x)+local*int(gridDim.x);
      Task task=Of(t,active);
      if(task.down)ConsumeDown(p,state,a,b,scratch,task,phase,sequence,flag,routing);
      else ConsumeGateUp(p,state,a,b,scratch,task,phase,sequence,flag,routing);
    }
    if(threadIdx.x==0 && tasks)
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
  }

  template<int Prefix=0,class Tail=Sm90DenseNoTail>
  __device__ static __forceinline__ void Run(Sm90DenseMoeFusedOperands const& p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={},bool streamed=true) {
    static_assert(Prefix==0,"M56 receives no stream: M4 does not know the routes");
    (void)streamed;
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*kRows*kMaxTK;
    auto* scratch=reinterpret_cast<float*>(shared+kLayout.scratch);
    auto& routing=*reinterpret_cast<LocalSchedule*>(shared+kRouteOffset);
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    // E2: the consumers derive this CTA's copy of the routing from the logits
    // the router stage left in L2; the closing grid barrier of that stage has
    // already made them visible. The producer waits for it because every task
    // index depends on n_active.
    if(threadIdx.x<128)LocalRoute::Build(p.logits,routing);
    __syncthreads();
    int active=routing.active;
    int total=active*(kTilesGu+kTilesDn);
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    if(threadIdx.x<128) {
      // One CTA republishes the global schedule for the routing check; it is
      // off the task path because no stage reads it any more.
      if(blockIdx.x==0)LocalRoute::Publish(routing,p.schedule,p.route_weights,p.route_index);
      Consume(p,state,a,b,scratch,tasks,phase,active,routing);
    }else {
      Produce(p,state,a,b,tasks,phase,active,routing);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
