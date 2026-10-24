// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Codegen/Sm90DenseEpilogueContract.h>
#include <tilemega/Backend/Sm90DenseAttentionStream.cuh>
#include <tilemega/Backend/Sm90DenseToken.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90DensePreissue.cuh>
#include <tilemega/Codegen/tasks/Sm90DenseStage.cuh>
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
#ifndef TILEMEGA_TRACE_NATIVE_BOUNDARY
#define TILEMEGA_TRACE_NATIVE_BOUNDARY 0
#endif
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES
#error "native boundary diagnostic must be isolated from other trace modes"
#endif
#include <tilemega/Codegen/tasks/Sm90DenseBoundaryDiagnostic.h>
#endif
#if defined(TILEMEGA_TRACE_NATIVE_RELEASE) && TILEMEGA_TRACE_NATIVE_RELEASE
#if TILEMEGA_TRACE_NATIVE_RELEASE != 1 || TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY
#error "native release trace must be isolated from other trace modes"
#endif
#ifndef TILEMEGA_NATIVE_RELEASE_SHARED_BUDGET
#error "native release trace requires its target/residency shared-memory budget"
#endif
#define TILEMEGA_NATIVE_RELEASE_ENABLED 1
#include <tilemega/Codegen/tasks/Sm90DenseReleaseTrace.cuh>
#endif

namespace tilemega::codegen::native_dense {

using Element=cutlass::bfloat16_t;
inline constexpr unsigned kAbsent=~0u;

// Passed as one __grid_constant__ kernel parameter: the tensor maps keep
// their 64-byte alignment and TMA reads them from parameter space.
template<unsigned Count,unsigned Maps>
struct alignas(64) Bindings {
  CUtensorMap maps[Maps];
  Element* buffers[Count];
  template<unsigned Id>
  __device__ __forceinline__ Element* Get() const {
    if constexpr(Id==kAbsent)return nullptr;
    else {static_assert(Id<Count);return buffers[Id];}
  }
  template<unsigned Id>
  __device__ __forceinline__ CUtensorMap const* Map() const {
    static_assert(Id<Maps);return &maps[Id];
  }
};

struct alignas(64) GridCounter {unsigned long long arrivals;};
struct NativeTensorMap {unsigned buffer,k,tile_k,rows;};
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
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
  unsigned long long* boundary_releases=nullptr;
  unsigned boundary_observe=0,boundary_probe_mask=0;
#endif
#if TILEMEGA_TRACE_NATIVE_PHASES
  backend::Sm90DenseDiagnosticRecord* diagnostics=nullptr;
#endif
};
__device__ inline unsigned long long Now() {
  unsigned long long time;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(time));return time;
}

template<int Batch,int N,int K,int TN,int TK,int Slots,int WN,int WK,
         backend::ServingEpilogueOp Op,int U,unsigned A,unsigned W,unsigned C,
         unsigned D,unsigned Index,unsigned Norm,unsigned Sums,int RegisterStages,unsigned Map>
struct Gemm:epilogue_contract::Presence<Norm,Sums> {
  using Features=epilogue_contract::Presence<Norm,Sums>;
  static constexpr bool kInputNorm=Features::kInputNorm,kOutputSums=Features::kOutputSums;
  static constexpr epilogue_contract::Gemm kEpilogueBinding{Norm,Sums,Batch,K,N,TN,
      bool(TILEMEGA_DN_VECTOR_SUMS),Op==backend::ServingEpilogueOp::kResidual,true};
  using Body=backend::Sm90DenseGemm<Batch,N,K,TN,TK,Slots,WN,WK,Op,U,RegisterStages>;
  static constexpr int kTasks=N/TN,kSharedBytes=Body::kSharedBytes;
  static constexpr bool kArgmax=false;
  static constexpr bool kUsesOperandRecord=true;
  static constexpr int kRingSlots=Slots;
  static constexpr int kMaxPreissue=(Slots<Body::kIterations?Slots:Body::kIterations)*2*TN*TK;
  static constexpr auto kLayout=backend::Sm90DenseLayout::Gemm(TN,TK,Slots,WK,Batch);
  static_assert(kLayout.end==kSharedBytes);
  using Preissue=GemmRead<Body,kTasks>;
  template<int Bytes>
  static constexpr auto Payload() {
    return backend::Sm90DenseLayout::GemmPayload(TN,TK,Slots,K,Batch,Bytes);
  }
  template<class B>
  __device__ static __forceinline__ ReadSources Sources(B const& b) {
    return {b.template Get<W>(),nullptr};
  }
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,int,float epsilon,
      backend::Sm90DenseOperandBank& record) {
    record.gemm.activation=b.template Map<Map>();record.gemm.weight=b.template Get<W>();
    record.gemm.residual=b.template Get<C>();record.gemm.output=b.template Get<D>();
    record.gemm.argmax_index=reinterpret_cast<int*>(b.template Get<Index>());
    record.gemm.norm_sums=reinterpret_cast<float const*>(b.template Get<Norm>());
    record.gemm.output_sums=reinterpret_cast<float*>(b.template Get<Sums>());
    record.gemm.epsilon=epsilon;
  }
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int,char* shared,float,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record) {
    auto const& p=record.gemm;
    StageEntry<Body>::template Gemm<Prefix,kInputNorm,kOutputSums>(p,shared,&ring,tail);
  }
};

template<int Batch,int D,int Q,bool Norm,int Chunks,int Keys,int Slots,int RowWork,
         int Capacity,unsigned Qkv,unsigned Key,unsigned Value,unsigned Output,
         unsigned Cosine,unsigned Sine,unsigned QNorm,unsigned KNorm,
         unsigned Partial,unsigned Lse,unsigned Arrivals>
struct Attention {
  static constexpr bool kArgmax=false;
  static constexpr bool kUsesOperandRecord=true;
  using Body=std::conditional_t<(Batch<=4),
      backend::Sm90DenseAttention<Batch,D,Q,Norm,Chunks,RowWork,Capacity>,
      backend::Sm90DenseAttentionStream<Batch,D,Q,Norm,Chunks,Keys,Slots,RowWork,Capacity>>;
  using Loader=backend::Sm90DenseKvPreissue<(Batch>=8),D,Q,Chunks,Keys,Slots,RowWork,Capacity>;
  static constexpr int kTasks=Batch*8*Chunks,kSharedBytes=Body::kSharedBytes;
  using Preissue=KvRead<Loader,kTasks>;
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
  template<class B>
  __device__ static __forceinline__ ReadSources Sources(B const& b) {
    return {b.template Get<Key>(),b.template Get<Value>()};
  }
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,int past,float epsilon,
      backend::Sm90DenseOperandBank& record) {
    record.attention.qkv=b.template Get<Qkv>();record.attention.key=b.template Get<Key>();
    record.attention.value=b.template Get<Value>();record.attention.output=b.template Get<Output>();
    record.attention.cosine=b.template Get<Cosine>();record.attention.sine=b.template Get<Sine>();
    record.attention.q_norm=b.template Get<QNorm>();record.attention.k_norm=b.template Get<KNorm>();
    record.attention.partial=reinterpret_cast<float*>(b.template Get<Partial>());
    record.attention.lse=reinterpret_cast<float*>(b.template Get<Lse>());
    record.attention.arrivals=reinterpret_cast<std::uint64_t*>(b.template Get<Arrivals>());
    record.attention.past=past;record.attention.epsilon=epsilon;
  }
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record) {
    auto const& p=record.attention;
    StageEntry<Body>::Attention(p,shared,&ring,tail);
  }
};

template<int Batch,int Width,int Vocab,int Capacity,unsigned Tokens,unsigned Table,
         unsigned Output,unsigned Sums>
struct Embedding {
  static constexpr bool kArgmax=false;
  static constexpr bool kUsesOperandRecord=false;
  static constexpr int kTasks=Batch,kSharedBytes=1024;
  static constexpr int kRingSlots=0;
  static constexpr int kMaxPreissue=0;
  static constexpr backend::Sm90DenseLayout kLayout{};
  template<int Bytes> static constexpr backend::Sm90DensePayload Payload() {return {};}
  template<int Bytes,class B>
  __device__ static void Prepare(B const&,int,char*,backend::Sm90DenseRingState&) {}
  template<int Bytes,int Prefix,class B>
  __device__ static void Prefetch(B const&,int) {}
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char*,float,
      backend::Sm90DenseRingState&,Tail tail) {
    if constexpr(Tail::kBefore)if(threadIdx.x==128)tail.Before();
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
  static constexpr bool kArgmax=true;
  static constexpr bool kUsesOperandRecord=false;
  using Body=backend::Sm90DenseArgmax<Parts,Capacity>;
  static constexpr int kTasks=Batch,kSharedBytes=1024+sizeof(typename Body::Storage);
  static constexpr int kRingSlots=0;
  static constexpr int kMaxPreissue=0;
  static constexpr auto kLayout=backend::Sm90DenseLayout::Argmax();
  static_assert(kLayout.end==kSharedBytes);
  template<int Bytes> static constexpr backend::Sm90DensePayload Payload() {return {};}
  template<int Bytes,class B>
  __device__ static void Prepare(B const&,int,char*,backend::Sm90DenseRingState&) {}
  template<int Bytes,int Prefix,class B>
  __device__ static void Prefetch(B const&,int) {}
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float,
      backend::Sm90DenseRingState&,Tail tail) {
    if constexpr(Tail::kBefore)if(threadIdx.x==128)tail.Before();
    if(threadIdx.x<128)for(int row=blockIdx.x;row<Batch;row+=gridDim.x)
      Body::Run(reinterpret_cast<float const*>(b.template Get<Values>()),
          reinterpret_cast<int const*>(b.template Get<Indices>()),
          reinterpret_cast<int*>(b.template Get<Tokens>()),
          *reinterpret_cast<typename Body::Storage*>(shared+1024),row,past);
    if(threadIdx.x==128)tail();
    __syncthreads();
  }
};

template<class Stage,int Offset=0,int Prefix=0,int L2Bytes=0,bool Late=false>
struct Placement:Stage {
  static_assert(Offset>=0 && Offset%1024==0 && Prefix>=0 && Prefix%16==0 &&
      Prefix<=Stage::kMaxPreissue && L2Bytes>=0 && L2Bytes%16==0);
  static_assert(Stage::kMaxPreissue>0 || L2Bytes==0,"this stage has no data to prefetch");
  static constexpr int kOffset=Offset,kPrefix=Prefix,kL2Bytes=L2Bytes;
  static constexpr bool kLate=Late;
  static constexpr int kSharedBytes=Offset+Stage::kSharedBytes;
  static constexpr auto kPayload=Stage::template Payload<Prefix>().Shift(Offset);
  static constexpr auto kLive=Stage::kLayout.Live(Offset);
  static constexpr auto kScratch=Stage::kLayout.Scratch(Offset);
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record) {
    if constexpr(Stage::kUsesOperandRecord)
      Stage::template Run<Prefix,B,Tail>(b,past,shared+Offset,epsilon,ring,tail,record);
    else Stage::template Run<Prefix,B,Tail>(b,past,shared+Offset,epsilon,ring,tail);
  }
};

template<class Current,class Next,class Binding>
struct ProducerTail {
  static_assert(!Next::kPayload.Overlaps(Current::kScratch),
      "preissue aliases live current-stage epilogue scratch");
  Binding const& binding;
  static constexpr bool kBefore=(Next::kPrefix || Next::kL2Bytes) &&
      !Next::kPayload.Overlaps(Current::kLive) && !Next::kLate;
  int past;
  char* shared;
  backend::Sm90DenseRingState &current,&next;
  __device__ __forceinline__ auto Raw() const {
    if constexpr(!(Next::kPrefix || Next::kL2Bytes))return backend::Sm90DenseNoTail{};
    else {
      using Tail=RawProducerTail<typename Next::Preissue,Next::kOffset,Next::kPrefix,
          Next::kL2Bytes,Next::kPayload.Overlaps(Current::kLive),Current::kRingSlots,Next::kLate>;
      // Uniform call arguments avoid per-layer conditional pointer lifetimes.
      auto source=Next::Sources(binding);
      return Tail{source,past,shared,&current,&next};
    }
  }
};

template<int Batch,int Capacity,int Residency,unsigned Buffers,unsigned Maps,class... Stage>
struct Program {
  using Binding=Bindings<Buffers,Maps>;
  using Stages=std::tuple<Stage...>;
  static constexpr int kBatch=Batch,kCapacity=Capacity,kResidency=Residency,kThreads=160;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps,kStages=sizeof...(Stage);
  using RoleLaunch=backend::Sm90DenseRoleLaunch<kThreads>;
  static_assert(RoleLaunch::kThreads==kThreads);
  static constexpr std::array<epilogue_contract::Gemm,kStages> kEpilogueBindings{{
      epilogue_contract::SpecFor<Stage>::value...}};
  using FirstStage=std::tuple_element_t<0,Stages>;
  static_assert(!FirstStage::kUsesOperandRecord,
      "native shared-operand ABI requires an operand-free first stage; no implicit startup publication barrier");
  static_assert(((Stage::kOffset>=0 && Stage::kLayout.Live(Stage::kOffset).begin>=1024) && ...));

#if TILEMEGA_TRACE_NATIVE_PHASES
  static constexpr unsigned long long kTotalTasks=(0ull+...+Stage::kTasks);
  static constexpr auto kTaskOffsets=[] {
    std::array<unsigned long long,kStages> offsets{};
    std::array<unsigned long long,kStages> tasks{{Stage::kTasks...}};
    for(unsigned i=1;i<kStages;++i)offsets[i]=offsets[i-1]+tasks[i-1];
    return offsets;
  }();
#endif
#if TILEMEGA_NATIVE_RELEASE_ENABLED
  static constexpr int kArenaSharedBytes=[] {
    int value=1024;((value=value<Stage::kSharedBytes?Stage::kSharedBytes:value),...);return value;
  }();
  using ReleaseRecord=release_trace::Record<kStages>;
  static constexpr int kReleaseOffset=release_trace::Align16(kArenaSharedBytes);
  static_assert(backend::Sm90DenseOperandHeader::kEnd<=kReleaseOffset);
  static constexpr int kSharedBytes=kReleaseOffset+sizeof(ReleaseRecord);
  static constexpr backend::Sm90DenseInterval kReleaseInterval{kReleaseOffset,kSharedBytes};
  static_assert(kSharedBytes<=TILEMEGA_NATIVE_RELEASE_SHARED_BUDGET,
      "native release trace exceeds the selected target/residency shared-memory budget");
  static_assert(((!Stage::kPayload.Overlaps(kReleaseInterval)) && ...));

  __device__ static __forceinline__ void BeginRelease(char* shared,int past,
                                                     unsigned long long iteration) {
    auto* header=reinterpret_cast<volatile release_trace::Header*>(shared+kReleaseOffset);
    if(threadIdx.x==0) {
      header->iteration=iteration;header->past=past;header->valid=0;
      unsigned long long begin;
      asm volatile("mov.u64 %0, %%globaltimer;":"=l"(begin)::"memory");
      header->begin=begin;
    }
  }
  __device__ static __noinline__ void FlushRelease(char* shared,
      release_trace::Binding const& binding,unsigned long long iteration) {
    auto* header=reinterpret_cast<volatile release_trace::Header*>(shared+kReleaseOffset);
    if(threadIdx.x==0) {
      unsigned long long end;
      asm volatile("mov.u64 %0, %%globaltimer;":"=l"(end)::"memory");
      header->end=end;header->valid=1;
    }
    // The last release is published by thread zero after the existing V3 join.
    __syncthreads();
    auto* source=reinterpret_cast<volatile std::uint32_t*>(shared+kReleaseOffset);
    auto* destination=binding.words+
        ((iteration%binding.slots)*gridDim.x+blockIdx.x)*(sizeof(ReleaseRecord)/4);
    for(unsigned word=threadIdx.x;word<sizeof(release_trace::Header)/4+kStages;word+=blockDim.x)
      destination[word]=source[word];
    __syncthreads();
  }
#else
  static constexpr int kSharedBytes=[] {
    int value=1024;((value=value<Stage::kSharedBytes?Stage::kSharedBytes:value),...);return value;
  }();
#endif
  static_assert(sizeof(Binding)+256<32764 && kStages>0 && (Residency==1 || Residency==2));
  template<unsigned Index>
  __device__ static __forceinline__ void Boundary(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace) {
    using Body=std::tuple_element_t<Index,Stages>;
    using Next=std::tuple_element_t<(Index+1)%kStages,Stages>;
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    static_assert(boundary_diagnostic::ValidStages(kStages));
    constexpr unsigned bit=boundary_diagnostic::Bit(boundary_diagnostic::ClassOf(Index,kStages));
    bool observe=trace.boundary_observe && trace.slots && trace.boundary_releases;
    bool probe=observe && trace.stages && (trace.boundary_probe_mask&bit);
    auto boundary_index=trace.slots?
        (iteration%trace.slots)*kStages*gridDim.x+Index*gridDim.x+blockIdx.x:0;
    if(threadIdx.x==0 && probe) {
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      trace.stages[boundary_index]={iteration,static_cast<unsigned long long>(past),Now(),0,0,
          Body::kTasks/gridDim.x+(blockIdx.x<Body::kTasks%gridDim.x),sm,Index};
    }
#endif
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
      Body::Run(b,past,shared,epsilon,rings[Index%2],tail.Raw(),
          backend::Sm90DenseOperandHeader::Bank(shared,Index));
    }else Body::Run(b,past,shared,epsilon,rings[Index%2],backend::Sm90DenseNoTail{},
        backend::Sm90DenseOperandHeader::Bank(shared,Index));
    if(threadIdx.x==0) {
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
      if(probe)trace.stages[boundary_index].arrive=Now();
#endif
#if TILEMEGA_TRACE_STAGE
      if(record)record->arrive=Now();
#endif
      // Current Run has joined all producer/consumer tasks, epilogues and tails.
      // Publish the next immutable bank before the existing V3 CTA join.
      // Next's preissue tail reads original bindings and never this header.
      if constexpr(Next::kUsesOperandRecord)
        Next::PublishOperands(b,past,epsilon,
            backend::Sm90DenseOperandHeader::Bank(shared,(Index+1)%kStages));
      auto* counter=&events[Index].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
      unsigned long long seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
      while(seen<(iteration+1)*gridDim.x);
#if TILEMEGA_TRACE_NATIVE_PHASES
      constexpr unsigned next_index=(Index+1)%kStages;
      auto next_iteration=iteration+(next_index==0);
      auto& context=backend::Sm90DenseDiagnostic::Context(shared);
      context.records=trace.diagnostics && trace.slots?trace.diagnostics+
          (next_iteration%trace.slots)*kTotalTasks+kTaskOffsets[next_index]:nullptr;
      context.iteration=next_iteration;context.past=past;context.stage=next_index;
#endif
    }
    __syncthreads();
#if TILEMEGA_NATIVE_RELEASE_ENABLED
    if(threadIdx.x==0) {
      unsigned low;
      asm volatile("mov.u32 %0, %%globaltimer_lo;":"=r"(low)::"memory");
      reinterpret_cast<volatile std::uint32_t*>(
          shared+kReleaseOffset+sizeof(release_trace::Header))[Index]=low;
    }
#endif
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0) {
      auto& context=backend::Sm90DenseDiagnostic::Context(shared);
      context.previous_release_global=Now();
    }
    if(threadIdx.x<128 && (threadIdx.x&31)==0)
      backend::Sm90DenseDiagnostic::Context(shared).previous_release_clock[threadIdx.x/32]=
          backend::Sm90DenseDiagnostic::Clock();
#endif
#if TILEMEGA_TRACE_STAGE
    if(threadIdx.x==0 && record)record->release=Now();
#endif
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    // Every mode uses the same observer; class probes add only begin/arrival records.
    if(threadIdx.x==0 && observe) {
      auto release=Now();
      trace.boundary_releases[boundary_index]=release;
      if(probe)trace.stages[boundary_index].release=release;
    }
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
    unsigned long long iteration,unsigned long long* step_ns,TraceState trace
#if TILEMEGA_NATIVE_RELEASE_ENABLED
    ,const __grid_constant__ release_trace::Binding release_binding
#endif
    ) {
  using RoleLaunch=backend::Sm90DenseRoleLaunch<P::kThreads>;
  static_assert(RoleLaunch::kThreads==P::kThreads);
  static_assert(!arch::kDevicePass || arch::kSm90AllAvailable,
                "native dense programs require sm_90a");
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  rings[0].Initialize();
#if TILEMEGA_TRACE_NATIVE_PHASES
  if(threadIdx.x==0) {
    auto& context=backend::Sm90DenseDiagnostic::Context(shared);
    context={trace.diagnostics && trace.slots?trace.diagnostics+
        (iteration%trace.slots)*P::kTotalTasks:nullptr,iteration,
        static_cast<unsigned long long>(pasts[first]),0,Now(),{}};
  }
#endif
  rings[1].Initialize();
#if TILEMEGA_TRACE_NATIVE_PHASES
  if(threadIdx.x<128 && (threadIdx.x&31)==0)
    backend::Sm90DenseDiagnostic::Context(shared).previous_release_clock[threadIdx.x/32]=
        backend::Sm90DenseDiagnostic::Clock();
#endif
  if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[first]=Now();
  for(unsigned step=0;step<steps;++step) {
    int past=pasts[first+step];
#if TILEMEGA_NATIVE_RELEASE_ENABLED
    P::BeginRelease(shared,past,iteration+step);
#endif
    P::Step(b,past,shared,epsilon,events,iteration+step,trace);
#if TILEMEGA_NATIVE_RELEASE_ENABLED
    P::FlushRelease(shared,release_binding,iteration+step);
#endif
    if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[first+step+1]=Now();
  }
  rings[0].Invalidate();rings[1].Invalidate();
}

}  // namespace tilemega::codegen::native_dense
