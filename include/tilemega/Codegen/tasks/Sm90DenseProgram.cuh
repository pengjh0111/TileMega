// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseAttentionStream.cuh>
#include <tilemega/Backend/Sm90DenseToken.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <array>
#include <tuple>
#include <utility>

#ifndef TILEMEGA_TRACE_STAGE
#define TILEMEGA_TRACE_STAGE 0
#endif
#ifndef TILEMEGA_TRACE_STEP
#define TILEMEGA_TRACE_STEP 0
#endif

namespace tilemega::codegen::native_dense {

using Element=cutlass::bfloat16_t;
inline constexpr unsigned kAbsent=~0u;

template<unsigned Count>
struct Bindings {
  Element* buffers[Count];
  template<unsigned Id>
  __device__ __forceinline__ Element* Get() const {
    if constexpr(Id==kAbsent)return nullptr;
    else {static_assert(Id<Count);return buffers[Id];}
  }
};

struct alignas(64) GridCounter {unsigned long long arrivals;};
struct StageInfo {
  char const *kind,*name;
  unsigned long long weight_bytes;
  unsigned tasks,heads,group,width,chunks;
};
struct StageTrace {
  unsigned long long iteration,past,begin,arrive,release,tasks,smid,stage;
};
struct StepTrace {
  unsigned long long iteration,past,begin,end,first,last,token_lag,kv_lag,barrier_wait,has_tasks;
};
struct TraceState {
  StageTrace* stages=nullptr;
  StepTrace* steps=nullptr;
  unsigned slots=0;
};
__device__ inline unsigned long long Now() {
  unsigned long long time;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(time));return time;
}

template<int Batch,int N,int K,int TN,int TK,int Slots,int WN,int WK,
         backend::ServingEpilogueOp Op,int U,unsigned A,unsigned W,unsigned C,
         unsigned D,unsigned Index,unsigned Norm,unsigned Sums,int RegisterStages=1>
struct Gemm {
  using Body=backend::Sm90DenseGemm<Batch,N,K,TN,TK,Slots,WN,WK,Op,U,RegisterStages>;
  static constexpr int kTasks=N/TN,kSharedBytes=Body::kSharedBytes;
  static constexpr int kRingSlots=Slots;
  static constexpr int kMaxPreissue=(Slots<Body::kIterations?Slots:Body::kIterations)*2*TN*TK;
  static constexpr auto kLayout=backend::Sm90DenseLayout::Gemm(TN,TK,Slots,WK);
  static_assert(kLayout.end==kSharedBytes);
  template<int Bytes>
  static constexpr auto Payload() {
    return backend::Sm90DenseLayout::GemmPayload(TN,TK,Slots,K,Bytes);
  }
  template<int Bytes,class B>
  __device__ static __forceinline__ void Prepare(B const& b,int,char* shared,
      backend::Sm90DenseRingState& ring) {
    if constexpr(Bytes>0)if(blockIdx.x<kTasks) {
      using Async=typename Body::Async;
      int remaining=Bytes;
      #pragma unroll
      for(int slot=0;slot<Slots && slot<Body::kIterations;++slot) {
        int part=min(remaining,2*TN*TK);remaining-=part;
        if(part) {
          Async::Wait(ring.empty+slot,ring.producer_generations[slot]);
          Async::ExpectTx(ring.full+slot,2*TN*TK);
          Async::Bulk(shared+kLayout.key+slot*2*TN*TK,
              b.template Get<W>()+(std::size_t(blockIdx.x)*Body::kIterations+slot)*TN*TK,
              part,ring.full+slot);
          ring.prefix_bytes[slot]=part;
        }
      }
    }
  }
  template<int Bytes,int Prefix,class B>
  __device__ static __forceinline__ void Prefetch(B const& b,int) {
    if constexpr(Bytes>0)if(blockIdx.x<kTasks) {
      constexpr int bytes=Bytes<2*TN*K-Prefix?Bytes:2*TN*K-Prefix;
      if constexpr(bytes>0)Body::Async::Prefetch(
          b.template Get<W>()+std::size_t(blockIdx.x)*TN*K+Prefix/2,bytes);
    }
  }
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail) {
    backend::Sm90DenseOperands p{b.template Get<A>(),b.template Get<W>(),
        b.template Get<C>(),b.template Get<D>(),reinterpret_cast<int*>(b.template Get<Index>()),
        reinterpret_cast<float const*>(b.template Get<Norm>()),
        reinterpret_cast<float*>(b.template Get<Sums>())};
    Body::Run(p,shared,epsilon,&ring,tail);
  }
};

template<int Batch,int D,int Q,bool Norm,int Chunks,int Keys,int Slots,int RowWork,
         int Capacity,unsigned Qkv,unsigned Key,unsigned Value,unsigned Output,
         unsigned Cosine,unsigned Sine,unsigned QNorm,unsigned KNorm,
         unsigned Partial,unsigned Lse,unsigned Arrivals>
struct Attention {
  using Body=std::conditional_t<(Batch<=4),
      backend::Sm90DenseAttention<Batch,D,Q,Norm,Chunks,RowWork,Capacity>,
      backend::Sm90DenseAttentionStream<Batch,D,Q,Norm,Chunks,Keys,Slots,RowWork,Capacity>>;
  static constexpr int kTasks=Batch*8*Chunks,kSharedBytes=Body::kSharedBytes;
  static constexpr int kRingSlots=Batch<=4?1:Slots;
  static constexpr int kMaxPreissue=4*D*(Batch<=4?
      Body::Partition::MaxKeys(Capacity,RowWork):Slots*Keys);
  static constexpr auto kLayout=backend::Sm90DenseLayout::Attention(
      Batch,D,Q,Chunks,Keys,Slots,RowWork,Capacity);
  static_assert(kLayout.end==kSharedBytes);
  using Storage=typename Body::Storage;
  static_assert(kLayout.key==1024+offsetof(Storage,key) &&
      kLayout.value==1024+offsetof(Storage,value) && kLayout.scratch==1024+offsetof(Storage,output));
  template<int Bytes>
  static constexpr auto Payload() {
    return backend::Sm90DenseLayout::AttentionPayload(
        Batch,D,Q,Chunks,Keys,Slots,RowWork,Capacity,Bytes);
  }
  template<int Bytes,class B>
  __device__ static __forceinline__ void Prepare(B const& b,int past,char* shared,
      backend::Sm90DenseRingState& ring) {
    if constexpr(Bytes>0)if(blockIdx.x<kTasks) {
      using Async=typename Body::Async;
      auto range=Body::Partition::Range(past,int(blockIdx.x)%Chunks,RowWork);
      int bg=int(blockIdx.x)/Chunks,remaining=Bytes;
      constexpr int rows_per_slot=Batch<=4?Body::Partition::MaxKeys(Capacity,RowWork):Keys;
      constexpr int count=Batch<=4?1:Slots;
      Async::ProxyAsyncGlobalFence();
      #pragma unroll
      for(int slot=0;slot<count;++slot) {
        int begin=range.begin+slot*rows_per_slot;
        int history=max(0,min(begin+rows_per_slot,min(range.end,past))-begin);
        int rows=min(history,remaining/(4*D));remaining-=rows*4*D;
        if(rows) {
          Async::Wait(ring.empty+slot,ring.producer_generations[slot]);
          Async::ExpectTx(ring.full+slot,4*history*D);
          std::size_t offset=(std::size_t(bg)*Capacity+begin)*D;
          Async::Bulk(shared+kLayout.key+slot*2*rows_per_slot*D,
              b.template Get<Key>()+offset,2*rows*D,ring.full+slot);
          Async::Bulk(shared+kLayout.value+slot*2*rows_per_slot*D,
              b.template Get<Value>()+offset,2*rows*D,ring.full+slot);
          ring.prefix_bytes[slot]=4*rows*D;
        }
      }
    }
  }
  template<int Bytes,int Prefix,class B>
  __device__ static __forceinline__ void Prefetch(B const& b,int past) {
    if constexpr(Bytes>0)if(blockIdx.x<kTasks) {
      auto range=Body::Partition::Range(past,int(blockIdx.x)%Chunks,RowWork);
      int history=min(range.end,past)-range.begin,done=min(history,Prefix/(4*D));
      int bytes=min(Bytes/2,2*D*(history-done))&~15;
      if(bytes) {
        Body::Async::ProxyAsyncGlobalFence();
        std::size_t offset=(std::size_t(int(blockIdx.x)/Chunks)*Capacity+range.begin+done)*D;
        Body::Async::Prefetch(b.template Get<Key>()+offset,bytes);
        Body::Async::Prefetch(b.template Get<Value>()+offset,bytes);
      }
    }
  }
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail) {
    backend::Sm90DenseAttentionOperands p{b.template Get<Qkv>(),b.template Get<Key>(),
        b.template Get<Value>(),b.template Get<Output>(),b.template Get<Cosine>(),
        b.template Get<Sine>(),b.template Get<QNorm>(),b.template Get<KNorm>(),
        reinterpret_cast<float*>(b.template Get<Partial>()),
        reinterpret_cast<float*>(b.template Get<Lse>()),
        reinterpret_cast<std::uint64_t*>(b.template Get<Arrivals>()),past,epsilon};
    Body::Run(p,shared,&ring,tail);
  }
};

template<int Batch,int Width,int Vocab,int Capacity,unsigned Tokens,unsigned Table,
         unsigned Output,unsigned Sums>
struct Embedding {
  static constexpr int kTasks=Batch,kSharedBytes=1024;
  static constexpr int kRingSlots=0;
  static constexpr int kMaxPreissue=0;
  static constexpr backend::Sm90DenseLayout kLayout{};
  template<int Bytes> static constexpr backend::Sm90DensePayload Payload() {return {};}
  template<int Bytes,class B>
  __device__ static void Prepare(B const&,int,char*,backend::Sm90DenseRingState&) {}
  template<int Bytes,int Prefix,class B>
  __device__ static void Prefetch(B const&,int) {}
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char*,float,
      backend::Sm90DenseRingState&,Tail tail) {
    if(threadIdx.x<128)for(int row=blockIdx.x;row<Batch;row+=gridDim.x) {
      auto* tokens=reinterpret_cast<int const*>(b.template Get<Tokens>());
      backend::Sm90DenseEmbedding<Width,Vocab,Capacity>::Run(tokens,b.template Get<Table>(),
          b.template Get<Output>(),reinterpret_cast<float*>(b.template Get<Sums>()),row,past);
    }
    if(threadIdx.x==128)tail();
    __syncthreads();
  }
};

template<int Batch,int Parts,int Capacity,unsigned Values,unsigned Indices,unsigned Tokens>
struct Argmax {
  using Body=backend::Sm90DenseArgmax<Parts,Capacity>;
  static constexpr int kTasks=Batch,kSharedBytes=1024+sizeof(typename Body::Storage);
  static constexpr int kRingSlots=0;
  static constexpr int kMaxPreissue=0;
  static constexpr backend::Sm90DenseLayout kLayout{0,0,1024,kSharedBytes};
  template<int Bytes> static constexpr backend::Sm90DensePayload Payload() {return {};}
  template<int Bytes,class B>
  __device__ static void Prepare(B const&,int,char*,backend::Sm90DenseRingState&) {}
  template<int Bytes,int Prefix,class B>
  __device__ static void Prefetch(B const&,int) {}
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float,
      backend::Sm90DenseRingState&,Tail tail) {
    if(threadIdx.x<128)for(int row=blockIdx.x;row<Batch;row+=gridDim.x)
      Body::Run(reinterpret_cast<float const*>(b.template Get<Values>()),
          reinterpret_cast<int const*>(b.template Get<Indices>()),
          reinterpret_cast<int*>(b.template Get<Tokens>()),
          *reinterpret_cast<typename Body::Storage*>(shared+1024),row,past);
    if(threadIdx.x==128)tail();
    __syncthreads();
  }
};

template<class Stage,int Offset=0,int Prefix=0,int L2Bytes=0>
struct Placement:Stage {
  static_assert(Offset>=0 && Offset%1024==0 && Prefix>=0 && Prefix%16==0 &&
      Prefix<=Stage::kMaxPreissue && L2Bytes>=0 && L2Bytes%16==0);
  static constexpr int kOffset=Offset,kPrefix=Prefix,kL2Bytes=L2Bytes;
  static constexpr int kSharedBytes=Offset+Stage::kSharedBytes;
  static constexpr auto kPayload=Stage::template Payload<Prefix>().Shift(Offset);
  static constexpr auto kLive=Stage::kLayout.Live(Offset);
  static constexpr auto kScratch=Stage::kLayout.Scratch(Offset);
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail) {
    Stage::Run(b,past,shared+Offset,epsilon,ring,tail);
  }
  template<class B>
  __device__ static __forceinline__ void Issue(B const& b,int past,char* shared,
      backend::Sm90DenseRingState& ring) {
    Stage::template Prepare<Prefix>(b,past,shared+Offset,ring);
    Stage::template Prefetch<L2Bytes,Prefix>(b,past);
  }
};

template<class Current,class Next,class Binding>
struct ProducerTail {
  static_assert(!Next::kPayload.Overlaps(Current::kScratch),
      "preissue aliases live current-stage epilogue scratch");
  Binding const& binding;
  int past;
  char* shared;
  backend::Sm90DenseRingState &current,&next;
  __device__ __forceinline__ void operator()() const {
    if constexpr(Next::kPrefix || Next::kL2Bytes) {
      // Empty covers operand readers, but not epilogue scratch. Its liveness
      // is checked independently at plan creation and again by this type.
      if constexpr(Next::kPayload.Overlaps(Current::kLive))
        current.template RetireOperands<Current::kRingSlots>();
      Next::Issue(binding,past,shared,next);
    }
  }
};

template<int Batch,int Capacity,int Residency,unsigned Buffers,class... Stage>
struct Program {
  using Binding=Bindings<Buffers>;
  using Stages=std::tuple<Stage...>;
  static constexpr int kBatch=Batch,kCapacity=Capacity,kResidency=Residency,kThreads=160;
  static constexpr unsigned kBuffers=Buffers,kStages=sizeof...(Stage);
  static constexpr int kSharedBytes=[] {
    int value=1024;((value=value<Stage::kSharedBytes?Stage::kSharedBytes:value),...);return value;
  }();
  static_assert(sizeof(Binding)+256<32764 && kStages>0 && (Residency==1 || Residency==2));

  template<unsigned Index>
  __device__ static __forceinline__ void Boundary(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace) {
    using Body=std::tuple_element_t<Index,Stages>;
    using Next=std::tuple_element_t<(Index+1)%kStages,Stages>;
#if TILEMEGA_TRACE_STAGE
    StageTrace* record=trace.stages && trace.slots?
        trace.stages+(iteration%trace.slots)*kStages*gridDim.x+Index*gridDim.x+blockIdx.x:nullptr;
    if(threadIdx.x==0 && record) {
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      *record={iteration,static_cast<unsigned long long>(past),Now(),0,0,
          Body::kTasks/gridDim.x+(blockIdx.x<Body::kTasks%gridDim.x),sm,Index};
    }
#endif
    // Every stage returns only after all 160 threads have retired its storage.
    auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
    if constexpr(Index+1<kStages) {
      ProducerTail<Body,Next,Binding> tail{b,past,shared,rings[Index%2],rings[(Index+1)%2]};
      Body::Run(b,past,shared,epsilon,rings[Index%2],tail);
    }else Body::Run(b,past,shared,epsilon,rings[Index%2],backend::Sm90DenseNoTail{});
    if(threadIdx.x==0) {
#if TILEMEGA_TRACE_STAGE
      if(record)record->arrive=Now();
#endif
      auto* counter=&events[Index].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
      unsigned long long seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
      while(seen<(iteration+1)*gridDim.x);
    }
    __syncthreads();
#if TILEMEGA_TRACE_STAGE
    if(threadIdx.x==0 && record)record->release=Now();
#endif
  }
  template<std::size_t... I>
  __device__ static __forceinline__ void Sequence(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,
      std::index_sequence<I...>) {
    (Boundary<I>(b,past,shared,epsilon,events,iteration,trace),...);
  }
  __device__ static __forceinline__ void Step(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace) {
#if TILEMEGA_TRACE_STEP
    StepTrace* record=trace.steps && trace.slots?
        trace.steps+(iteration%trace.slots)*gridDim.x+blockIdx.x:nullptr;
    if(threadIdx.x==0 && record)*record={iteration,static_cast<unsigned long long>(past),Now(),0,0,0,0,0,0,0};
#endif
    Sequence(b,past,shared,epsilon,events,iteration,trace,std::make_index_sequence<kStages>{});
#if TILEMEGA_TRACE_STEP
    if(threadIdx.x==0 && record)record->end=Now();
#endif
  }
};

template<class P>
__global__ __launch_bounds__(P::kThreads,P::kResidency)
void sm90_dense_native_kernel(const __grid_constant__ typename P::Binding b,
    int const* pasts,unsigned first,unsigned steps,float epsilon,GridCounter* events,
    unsigned long long iteration,unsigned long long* step_ns,TraceState trace) {
  static_assert(!arch::kDevicePass || arch::kSm90AllAvailable,
                "native dense programs require sm_90a");
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  rings[0].Initialize();rings[1].Initialize();
  if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[first]=Now();
  for(unsigned step=0;step<steps;++step) {
    int past=pasts[first+step];
    P::Step(b,past,shared,epsilon,events,iteration+step,trace);
    if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[first+step+1]=Now();
  }
  rings[0].Invalidate();rings[1].Invalidate();
}

}  // namespace tilemega::codegen::native_dense
