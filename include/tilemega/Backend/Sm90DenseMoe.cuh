// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseGemm.h>
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
// grows). Ties keep the smaller expert and the probabilities are the
// softmax of the selected BF16 logits, as MoETopKTaskBody computes them.
template<int Batch,int Experts,int TopK,int Tasks,bool FloatWeights>
struct Sm90DenseRouterFinish {
  using Schedule=Sm90DenseMoeSchedule<Experts>;
  static_assert(Experts==128 && TopK<=8 && Batch<=Schedule::kMaxRows);
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
    if(!*flag)return;
    auto const* logits=static_cast<float const*>(p.output);
    for(int b=warp;b<Batch;b+=4) {
      float value[Experts/32];
      #pragma unroll
      for(int i=0;i<Experts/32;++i)value[i]=logits[b*Experts+lane+32*i];
      float chosen[TopK];int index[TopK];
      #pragma unroll
      for(int k=0;k<TopK;++k) {
        float best=-INFINITY;int id=INT_MAX;
        #pragma unroll
        for(int i=0;i<Experts/32;++i)
          if(value[i]>best || (value[i]==best && lane+32*i<id)) {best=value[i];id=lane+32*i;}
        #pragma unroll
        for(int d=16;d;d>>=1) {
          float v=__shfl_xor_sync(0xffffffff,best,d);int e=__shfl_xor_sync(0xffffffff,id,d);
          if(v>best || (v==best && e<id)) {best=v;id=e;}
        }
        chosen[k]=best;index[k]=id;
        #pragma unroll
        for(int i=0;i<Experts/32;++i)if(id==lane+32*i)value[i]=-INFINITY;
      }
      if(lane==0) {
        float probability[TopK],sum=0;
        #pragma unroll
        for(int k=0;k<TopK;++k) {probability[k]=expf(chosen[k]-chosen[0]);sum+=probability[k];}
        #pragma unroll
        for(int k=0;k<TopK;++k) {
          p.route_index[b*TopK+k]=index[k];selected[b*TopK+k]=index[k];
          float weight=probability[k]/sum;
          if constexpr(FloatWeights)static_cast<float*>(p.route_weights)[b*TopK+k]=weight;
          else static_cast<cutlass::bfloat16_t*>(p.route_weights)[b*TopK+k]=cutlass::bfloat16_t(weight);
        }
      }
    }
    codegen::executor::ComputeSync();
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
  }
};

template<int Batch,int N,int K,int TN,int TK,int Slots,int WN,int WK,Sm90DenseMoeMode Mode,int U,
         int Experts,int TopK,bool FloatWeights,int RegisterStages=1>
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
  using RouterFinish=Sm90DenseRouterFinish<Batch,Experts,TopK,kTiles,FloatWeights>;

  __device__ static __forceinline__ int TotalTasks(Sm90DenseMoeOperands const& p) {
    if constexpr(kDynamic)return p.schedule[Schedule::Active()]*kTiles;
    else return kTiles;
  }
  // Weight tiles of task t: router tile t, else tile t % kTiles of the
  // expert in slot t / kTiles (one expert's pages are N x K).
  __device__ static __forceinline__ Element const* Tiles(Element const* weight,int const* schedule,int t) {
    if constexpr(!kDynamic)return weight+std::size_t(t)*kIterations*TN*TK;
    else return weight+std::size_t(schedule[Schedule::Expert(t/kTiles)])*N*K+
        std::size_t(t%kTiles)*kIterations*TN*TK;
  }

  // D6 stream into M6 from M5's producer, for this CTA's first task.
  template<int Bytes>
  __device__ static __forceinline__ void Preissue(Element const* weight,int const* schedule,
      char* shared,Sm90DenseRingState& state) {
    static_assert(Bytes>0 && Bytes%16==0 && Bytes<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    int t=int(blockIdx.x);
    if(t>=schedule[Schedule::Active()]*kTiles)return;
    auto* b=reinterpret_cast<Element*>(shared+1024)+Slots*kRows*TK;
    auto const* tiles=Tiles(weight,schedule,t);
    int remaining=Bytes;
    #pragma unroll
    for(int slot=0;slot<Slots && slot<kIterations;++slot) {
      int part=min(remaining,2*TN*TK);remaining-=part;
      if(part) {
        Async::ExpectTx(state.full+slot,kSlotBytes);
        Async::Bulk(b+slot*TN*TK,tiles+std::size_t(slot)*TN*TK,part,state.full+slot);
      }
    }
  }

  // Byte offset of (row, col) in a [TK/64][kRows][64] SWIZZLE_128B slot.
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return (col/64)*kRows*128+row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }

  template<int Prefix>
  __device__ static __noinline__ void Produce(Sm90DenseMoeOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase,bool streamed) {
    Sm90DenseRoles::Producer();
    int lane=threadIdx.x&31;
    // x', h and the schedule were written with generic stores before the
    // grid barrier; the async proxy must observe them.
    if(lane==0)Async::ProxyAsyncGlobalFence();
    for(int sequence=0;sequence<tasks*kIterations;++sequence) {
      int slot=sequence%Slots,tile=sequence%kIterations;
      int t=int(blockIdx.x)+(sequence/kIterations)*int(gridDim.x);
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      auto const* weights=Tiles(p.weight,p.schedule,t)+std::size_t(tile)*TN*TK;
      if constexpr(Mode==Sm90DenseMoeMode::kGateUp) {
        // The producer lanes gather the slot's rows of x' (one TK segment
        // each) into the swizzled slot. A non-.noinc cp.async arrive adds a
        // pending arrival that its copies retire, so lane 0's single
        // expect_tx arrival must follow the warp's arrives.
        int group=t/kTiles,count=p.schedule[Schedule::Count(group)];
        auto base=Async::Shared(a+slot*kRows*TK);
        for(int chunk=lane;chunk<count*(TK/8);chunk+=32) {
          int r=chunk/(TK/8),col=(chunk%(TK/8))*8;
          int token=p.schedule[Schedule::Row(group,r)]/TopK;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(base+Swizzled(r,col)),
              "l"(p.input+std::size_t(token)*K+tile*TK+col):"memory");
        }
        asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];"::
            "r"(Async::Shared(state.full+slot)):"memory");
        __syncwarp();
        if(lane==0) {
          Async::ExpectTx(state.full+slot,2*TN*TK);
          Async::Bulk(b+slot*TN*TK,weights,2*TN*TK,state.full+slot);
        }
      }else if(lane==0) {
        // Only M6's first task can have been streamed by M5 (D6).
        unsigned prefix=0;
        if constexpr(Prefix>0)
          if(streamed && sequence<(Prefix+2*TN*TK-1)/(2*TN*TK))
            prefix=min(Prefix-sequence*2*TN*TK,2*TN*TK);
        if(!prefix)Async::ExpectTx(state.full+slot,kSlotBytes);
        if(prefix<2*TN*TK)Async::Bulk(b+slot*TN*TK+prefix/2,weights+prefix/2,2*TN*TK-prefix,
            state.full+slot);
        // Rows past the batch (router) or the slot's count (down) are never
        // written out.
        if constexpr(Mode==Sm90DenseMoeMode::kRouter)
          Async::Tensor3D(a+slot*kRows*TK,p.activation,0,0,tile*(TK/64),state.full+slot);
        else Async::Tensor4D(a+slot*kRows*TK,p.activation,0,0,tile*(TK/64),t/kTiles,state.full+slot);
      }
      phase^=1u<<slot;
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
  __device__ static __forceinline__ void Epilogue(Sm90DenseMoeOperands const& p,float const* tile,
      float const* norms,int t,int* flag) {
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
      int count=p.schedule[Schedule::Count(group)];
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
      int count=p.schedule[Schedule::Count(group)];
      auto* partial=static_cast<float*>(p.output);
      for(int i=tid;i<count*TN;i+=128) {
        int row=i/TN,col=i%TN,route=p.schedule[Schedule::Row(group,row)];
        float weight=FloatWeights?static_cast<float const*>(p.route_weights)[route]:
            float(static_cast<Element const*>(p.route_weights)[route]);
        partial[std::size_t(route)*N+j*TN+col]=weight*Value(tile,row,col);
      }
      codegen::executor::ComputeSync();
      if(tid==0) {
        unsigned long long old;
        asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;":"=l"(old):"l"(p.counter+j):"memory");
        *flag=int(old+1)==p.schedule[Schedule::Active()];
        if(*flag)p.counter[j]=0;
      }
      codegen::executor::ComputeSync();
      if(*flag)
        for(int i=tid;i<Batch*TN;i+=128) {
          int row=i/TN,col=j*TN+i%TN;
          float sum=0;
          #pragma unroll
          for(int k=0;k<TopK;++k)sum+=partial[std::size_t(row*TopK+k)*N+col];
          p.x[std::size_t(row)*N+col]=Element(float(p.x[std::size_t(row)*N+col])+sum);
        }
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
      codegen::executor::ComputeSync();
      Epilogue(p,scratch,norms,t,flag);
      codegen::executor::ComputeSync();
    }
    if(tid==0 && tasks)for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
    // Every task has been consumed: the operand slots are free.
    if constexpr(Mode==Sm90DenseMoeMode::kRouter)RouterFinish{}(p,tasks,reinterpret_cast<char*>(a));
  }

  template<int Prefix=0,class Tail=Sm90DenseNoTail>
  __device__ static __forceinline__ void Run(Sm90DenseMoeOperands const& p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={},bool streamed=true) {
    static_assert(Prefix>=0 && Prefix%16==0 && Prefix<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    static_assert(Mode==Sm90DenseMoeMode::kDown || Prefix==0,
        "M4's routes are unknown to its predecessor and M4 does not stream into M5");
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*kRows*TK;
    auto* scratch=reinterpret_cast<float*>(shared+kLayout.scratch);
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    int total=TotalTasks(p);
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,tasks,phase);
    else {
      Produce<Prefix>(p,state,a,b,tasks,phase,streamed);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
