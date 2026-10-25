// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Codegen/Sm90DenseEpilogueContract.h>
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <tilemega/Backend/Sm90DenseToken.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
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
// A stage operand that differs between layers names a column of the
// per-layer pointer table instead of a buffer (D7).
inline constexpr unsigned kLayerSlot=1u<<30;

// Passed as one __grid_constant__ kernel parameter: the tensor maps keep
// their 64-byte alignment and TMA reads them from parameter space. Thread 0
// reads a layer's row of `layers` when it publishes that layer's operands.
template<unsigned Count,unsigned Maps,unsigned Layers,unsigned Slots>
struct alignas(64) Bindings {
  CUtensorMap maps[Maps];
  Element* buffers[Count];
  Element* layers[Layers][Slots];
  template<unsigned Id>
  __device__ __forceinline__ Element* Get(int layer=0) const {
    if constexpr(Id==kAbsent)return nullptr;
    else if constexpr((Id&kLayerSlot)!=0) {
      static_assert((Id&~kLayerSlot)<Slots);return layers[layer][Id&~kLayerSlot];
    }else {static_assert(Id<Count);return buffers[Id];}
  }
  template<unsigned Id>
  __device__ __forceinline__ CUtensorMap const* Map() const {
    static_assert(Id<Maps);return &maps[Id];
  }
};

struct alignas(64) GridCounter {unsigned long long arrivals;};
// One rank-3/4 SWIZZLE_128B descriptor over a bound buffer (codegen-emitted).
struct NativeTensorMap {
  unsigned buffer,rank;
  unsigned long long dims[4],strides[3];
  unsigned box[4];
};
// Each layer's K and V are views of one [L][2][B][Hkv][Cap][D] allocation
// (F19); the two stacked 4D maps index them by layer (D7).
template<unsigned Layers>
struct NativeStackedKv {
  std::array<unsigned,Layers> key,value;
  unsigned long long layer_bytes,value_bytes;
};
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

// InputNorm: the stage's input is RMS-normalized in the consumers; its gamma
// is folded into the packed weight (qkv, gate_up, lm_head).
template<int Batch,int N,int K,int TN,int TK,int Slots,int WN,int WK,
         backend::ServingEpilogueOp Op,int U,bool InputNorm,unsigned W,unsigned C,
         unsigned D,unsigned Index,int RegisterStages,unsigned Map>
struct Gemm {
  using Body=backend::Sm90DenseGemm<Batch,N,K,TN,TK,Slots,WN,WK,Op,U,InputNorm,RegisterStages>;
  static constexpr int kTasks=N/TN,kSharedBytes=Body::kSharedBytes;
  static constexpr bool kArgmax=false;
  static constexpr bool kUsesOperandRecord=true;
  static constexpr int kRingSlots=Slots;
  static constexpr int kMaxPreissue=(Slots<Body::kIterations?Slots:Body::kIterations)*2*TN*TK;
  static constexpr auto kLayout=Body::kLayout;
  using Preissue=GemmRead<Body,kTasks>;
  template<int Bytes>
  static constexpr auto Payload() {
    return backend::Sm90DenseLayout::GemmPayload(TN,TK,Slots,K,Batch,Bytes);
  }
  template<class B>
  __device__ static __forceinline__ ReadSources Sources(B const& b,int layer) {
    return {b.template Get<W>(layer),nullptr,nullptr,layer};
  }
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,int layer,int,float epsilon,
      backend::Sm90DenseOperandBank& record) {
    record.gemm.activation=b.template Map<Map>();record.gemm.weight=b.template Get<W>(layer);
    record.gemm.residual=b.template Get<C>(layer);record.gemm.output=b.template Get<D>(layer);
    record.gemm.argmax_index=reinterpret_cast<int*>(b.template Get<Index>(layer));
    record.gemm.epsilon=epsilon;
  }
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int,char* shared,float,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record,
      bool streamed) {
    StageEntry<Body>::template Gemm<Prefix>(record.gemm,shared,&ring,tail,streamed);
  }
};

template<int Batch,int Hkv,int D,int Q,bool Norm,int Chunks,int Keys,int Slots,int RowWork,
         int Capacity,unsigned Qkv,unsigned Key,unsigned Value,unsigned Output,
         unsigned Cosine,unsigned Sine,unsigned QNorm,unsigned KNorm,
         unsigned Partial,unsigned Lse,unsigned Arrivals,unsigned KeyMap,unsigned ValueMap>
struct Attention {
  static constexpr bool kArgmax=false;
  static constexpr bool kUsesOperandRecord=true;
  using Body=backend::Sm90DenseAttentionBody<Batch,Hkv,D,Q,Norm,Chunks,Keys,Slots,RowWork,Capacity>;
  static constexpr int kTasks=Body::kTasks,kSharedBytes=Body::kSharedBytes;
  static constexpr int kRingSlots=Slots,kMaxPreissue=Body::kMaxPreissue;
  static constexpr auto kLayout=backend::Sm90DenseLayout::Attention(D,Q,Chunks,Keys,Slots);
  static_assert(kLayout.end==kSharedBytes);
  using Preissue=KvRead<Body,kTasks>;
  template<int Bytes>
  static constexpr auto Payload() {
    return backend::Sm90DenseLayout::AttentionPayload(D,Keys,Slots,Bytes);
  }
  template<class B>
  __device__ static __forceinline__ ReadSources Sources(B const& b,int layer) {
    return {nullptr,b.template Map<KeyMap>(),b.template Map<ValueMap>(),layer};
  }
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,int layer,int past,float epsilon,
      backend::Sm90DenseOperandBank& record) {
    record.attention.key_map=b.template Map<KeyMap>();record.attention.value_map=b.template Map<ValueMap>();
    record.attention.qkv=b.template Get<Qkv>(layer);record.attention.key=b.template Get<Key>(layer);
    record.attention.value=b.template Get<Value>(layer);record.attention.output=b.template Get<Output>(layer);
    record.attention.cosine=b.template Get<Cosine>(layer);record.attention.sine=b.template Get<Sine>(layer);
    record.attention.q_norm=b.template Get<QNorm>(layer);record.attention.k_norm=b.template Get<KNorm>(layer);
    record.attention.partial=reinterpret_cast<float*>(b.template Get<Partial>(layer));
    record.attention.lse=reinterpret_cast<float*>(b.template Get<Lse>(layer));
    record.attention.arrivals=reinterpret_cast<std::uint64_t*>(b.template Get<Arrivals>(layer));
    record.attention.past=past;record.attention.epsilon=epsilon;record.attention.layer=layer;
  }
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record,
      bool streamed) {
    StageEntry<Body>::template Attention<Prefix>(record.attention,shared,&ring,tail,streamed);
  }
};

template<int Batch,int Width,int Vocab,int Capacity,unsigned Tokens,unsigned Table,
         unsigned Output>
struct Embedding {
  static constexpr bool kArgmax=false;
  static constexpr bool kUsesOperandRecord=false;
  static constexpr int kTasks=Batch,kSharedBytes=1024;
  static constexpr int kRingSlots=0;
  static constexpr int kMaxPreissue=0;
  static constexpr backend::Sm90DenseLayout kLayout{};
  template<int Bytes> static constexpr backend::Sm90DensePayload Payload() {return {};}
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char*,float,
      backend::Sm90DenseRingState&,Tail tail) {
    if(threadIdx.x<128)for(int row=blockIdx.x;row<Batch;row+=gridDim.x) {
      auto* tokens=reinterpret_cast<int const*>(b.template Get<Tokens>());
      backend::Sm90DenseEmbedding<Width,Vocab,Capacity>::Run(tokens,b.template Get<Table>(),
          b.template Get<Output>(),row,past);
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
  template<int Prefix,class B,class Tail>
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

template<class Stage,int Offset=0,int Prefix=0>
struct Placement:Stage {
  static_assert(Offset>=0 && Offset%1024==0 && Prefix>=0 && Prefix%16==0 &&
      Prefix<=Stage::kMaxPreissue);
  static constexpr int kOffset=Offset,kPrefix=Prefix;
  static constexpr int kSharedBytes=Offset+Stage::kSharedBytes;
  static constexpr auto kPayload=Stage::template Payload<Prefix>().Shift(Offset);
  static constexpr auto kLive=Stage::kLayout.Live(Offset);
  static constexpr auto kScratch=Stage::kLayout.Scratch(Offset);
  template<class B,class Tail>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record,
      bool streamed) {
    if constexpr(Stage::kUsesOperandRecord)
      Stage::template Run<Prefix,B,Tail>(b,past,shared+Offset,epsilon,ring,tail,record,streamed);
    else Stage::template Run<Prefix,B,Tail>(b,past,shared+Offset,epsilon,ring,tail);
  }
};

// D6 issues into the alternate arena only: the payload never overlaps the
// current stage's live region, so the tail needs no wait on the current ring.
// A stage without a ring (the prologue) issues nothing.
template<class Current,class Next,class Binding>
struct ProducerTail {
  static_assert(!Next::kPayload.Overlaps(Current::kLive),
      "preissue payload overlaps the current stage's arena");
  Binding const& binding;
  int layer;
  int past;
  char* shared;
  backend::Sm90DenseRingState& next;
  __device__ __forceinline__ auto Raw() const {
    if constexpr(!Next::kPrefix || !Current::kRingSlots)return backend::Sm90DenseNoTail{};
    else {
      using Tail=RawProducerTail<typename Next::Preissue,Next::kOffset,Next::kPrefix>;
      // Uniform call arguments avoid per-layer conditional pointer lifetimes.
      return Tail{Next::Sources(binding,layer),past,shared,&next};
    }
  }
};

template<class... Stage>
struct LayerPair {
  static_assert(sizeof...(Stage)==10,"a layer pair holds two layers of five typed stages");
  using Stages=std::tuple<Stage...>;
};

// D7 (prompt §6): Prologue (embedding) -> layer pairs -> Head (lm_head) ->
// Final (argmax). A pair is ten typed stages run for layers l and l+1; two
// layers per pair keep every typed stage's arena parity fixed. The last pair
// runs separately so that its down streams into the head. Executed
// boundaries keep their global index g (grid counters, banks, traces);
// Loop selects a runtime pair loop over the same pair code that the
// unrolled form instantiates once per pair (DT-3).
template<int Batch,int Capacity,int Residency,unsigned Buffers,unsigned Maps,int Arena,
         int Layers,unsigned LayerSlots,bool Loop,class Prologue,class Pair,class Head,class Final>
struct Program {
  using Binding=Bindings<Buffers,Maps,Layers,LayerSlots>;
  using PairStages=typename Pair::Stages;
  template<unsigned J> using PairStage=std::tuple_element_t<J,PairStages>;
  static constexpr int kBatch=Batch,kCapacity=Capacity,kResidency=Residency,kThreads=160;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps,kLayers=Layers;
  static constexpr unsigned kStages=3+5*Layers,kHead=1+5*Layers;
  using RoleLaunch=backend::Sm90DenseRoleLaunch<kThreads>;
  static_assert(RoleLaunch::kThreads==kThreads);
  static_assert(Layers>=2 && Layers%2==0,"layer pairs need an even layer count");
  static_assert(!Prologue::kUsesOperandRecord,
      "native shared-operand ABI requires an operand-free first stage; no implicit startup publication barrier");
  // D5: a stage of parity P sits in arena P (offset P*Arena) and fits it;
  // only a stage at offset 0 may extend over both arenas (full width).
  static_assert(Arena>0 && Arena%1024==0);
  template<class S,unsigned Parity>
  static constexpr bool Placed() {
    return S::kOffset>=0 && S::kLayout.Live(S::kOffset).begin>=1024 &&
        S::kSharedBytes<=1024+2*Arena &&
        (S::kOffset==int(Parity)*Arena || (S::kOffset==0 && S::kSharedBytes>1024+Arena));
  }
  template<unsigned... J>
  static constexpr bool PairPlaced(std::integer_sequence<unsigned,J...>) {
    return (Placed<PairStage<J>,(1+J)%2>() && ...);
  }
  static_assert(Placed<Prologue,0>() && Placed<Head,kHead%2>() && Placed<Final,(kHead+1)%2>() &&
      PairPlaced(std::make_integer_sequence<unsigned,10>{}));
  template<unsigned... J>
  static constexpr int PairBytes(std::integer_sequence<unsigned,J...>) {
    int value=1024;((value=value<PairStage<J>::kSharedBytes?PairStage<J>::kSharedBytes:value),...);
    return value;
  }
  static constexpr int kArenaSharedBytes=[] {
    int value=PairBytes(std::make_integer_sequence<unsigned,10>{});
    for(int bytes:{Prologue::kSharedBytes,Head::kSharedBytes,Final::kSharedBytes})
      value=value<bytes?bytes:value;
    return value;
  }();
  // Tasks of executed boundary g.
  template<unsigned... J>
  static constexpr unsigned long long PairTasks(unsigned j,std::integer_sequence<unsigned,J...>) {
    unsigned long long value=0;((value=j==J?PairStage<J>::kTasks:value),...);return value;
  }
  static constexpr unsigned long long Tasks(unsigned g) {
    if(g==0)return Prologue::kTasks;
    if(g==kHead)return Head::kTasks;
    if(g==kHead+1)return Final::kTasks;
    return PairTasks((g-1)%10,std::make_integer_sequence<unsigned,10>{});
  }

#if TILEMEGA_TRACE_NATIVE_PHASES
  static constexpr auto kTaskOffsets=[] {
    std::array<unsigned long long,kStages> offsets{};
    for(unsigned i=1;i<kStages;++i)offsets[i]=offsets[i-1]+Tasks(i-1);
    return offsets;
  }();
  static constexpr unsigned long long kTotalTasks=kTaskOffsets[kStages-1]+Tasks(kStages-1);
#endif
#if TILEMEGA_NATIVE_RELEASE_ENABLED
  using ReleaseRecord=release_trace::Record<kStages>;
  static constexpr int kReleaseOffset=release_trace::Align16(kArenaSharedBytes);
  static_assert(backend::Sm90DenseOperandHeader::kEnd<=kReleaseOffset);
  static constexpr int kSharedBytes=kReleaseOffset+sizeof(ReleaseRecord);
  static_assert(kSharedBytes<=TILEMEGA_NATIVE_RELEASE_SHARED_BUDGET,
      "native release trace exceeds the selected target/residency shared-memory budget");

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
  static constexpr int kSharedBytes=kArenaSharedBytes;
#endif
  static_assert(sizeof(Binding)+256<32764 && (Residency==1 || Residency==2));
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
  static_assert(boundary_diagnostic::ValidStages(kStages));
#endif

  // One executed boundary g of typed stage Body (arena and ring parity
  // Parity) for runtime layer `layer`; Next runs next for `next_layer`.
  // streamed: Body's predecessor issued Body's stream prefix.
  template<class Body,class Next,unsigned Parity,bool Last=false>
  __device__ static __forceinline__ void Boundary(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,
      unsigned g,int layer,int next_layer,bool streamed) {
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    unsigned bit=boundary_diagnostic::Bit(boundary_diagnostic::ClassOf(g,kStages));
    bool observe=trace.boundary_observe && trace.slots && trace.boundary_releases;
    bool probe=observe && trace.stages && (trace.boundary_probe_mask&bit);
    auto boundary_index=trace.slots?
        (iteration%trace.slots)*kStages*gridDim.x+g*gridDim.x+blockIdx.x:0;
    if(threadIdx.x==0 && probe) {
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      trace.stages[boundary_index]={iteration,static_cast<unsigned long long>(past),Now(),0,0,
          Body::kTasks/gridDim.x+(blockIdx.x<Body::kTasks%gridDim.x),sm,g};
    }
#endif
#if TILEMEGA_TRACE_STAGE
    StageTrace* record=trace.stages && trace.slots?
        trace.stages+(iteration%trace.slots)*kStages*gridDim.x+g*gridDim.x+blockIdx.x:nullptr;
    if(threadIdx.x==0 && record) {
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      *record={iteration,static_cast<unsigned long long>(past),Now(),0,0,
          Body::kTasks/gridDim.x+(blockIdx.x<Body::kTasks%gridDim.x),sm,g};
    }
#endif
    // Every stage returns only after all 160 threads have retired its storage.
    auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
    auto& bank=backend::Sm90DenseOperandHeader::Bank(shared,Parity);
    if constexpr(!Last) {
      ProducerTail<Body,Next,Binding> tail{b,next_layer,past,shared,rings[Parity^1]};
      Body::Run(b,past,shared,epsilon,rings[Parity],tail.Raw(),bank,streamed);
    }else Body::Run(b,past,shared,epsilon,rings[Parity],backend::Sm90DenseNoTail{},bank,streamed);
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
        Next::PublishOperands(b,next_layer,past,epsilon,
            backend::Sm90DenseOperandHeader::Bank(shared,Parity^1));
      auto* counter=&events[g].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
      unsigned long long seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
      while(seen<(iteration+1)*gridDim.x);
#if TILEMEGA_TRACE_NATIVE_PHASES
      unsigned next_index=(g+1)%kStages;
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
          shared+kReleaseOffset+sizeof(release_trace::Header))[g]=low;
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
  // Layers 2*pair and 2*pair+1. Only layer 0's qkv follows the ring-less
  // prologue and receives no stream.
  template<bool LastPair,unsigned... J>
  __device__ static __forceinline__ void RunPair(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,int pair,
      std::integer_sequence<unsigned,J...>) {
    (Boundary<PairStage<J>,std::conditional_t<J==9,
        std::conditional_t<LastPair,Head,PairStage<0>>,PairStage<(J+1)%10>>,(1+J)%2>(
        b,past,shared,epsilon,events,iteration,trace,1+10*pair+J,2*pair+int(J/5),
        J==9?(LastPair?0:2*pair+2):2*pair+int((J+1)/5),J!=0 || pair!=0),...);
  }
  template<int... P>
  __device__ static __forceinline__ void Unrolled(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,
      std::integer_sequence<int,P...>) {
    (RunPair<false>(b,past,shared,epsilon,events,iteration,trace,P,
        std::make_integer_sequence<unsigned,10>{}),...);
  }
  __device__ static __forceinline__ void Step(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace) {
#if TILEMEGA_TRACE_STEP
    StepTrace* record=trace.steps && trace.slots?
        trace.steps+(iteration%trace.slots)*gridDim.x+blockIdx.x:nullptr;
    if(threadIdx.x==0 && record)*record={iteration,static_cast<unsigned long long>(past),Now(),0,0,0,0,0,0,0};
#endif
    Boundary<Prologue,PairStage<0>,0>(b,past,shared,epsilon,events,iteration,trace,0,0,0,false);
    if constexpr(Loop) {
      #pragma unroll 1
      for(int pair=0;pair<Layers/2-1;++pair)
        RunPair<false>(b,past,shared,epsilon,events,iteration,trace,pair,
            std::make_integer_sequence<unsigned,10>{});
    }else Unrolled(b,past,shared,epsilon,events,iteration,trace,
        std::make_integer_sequence<int,Layers/2-1>{});
    RunPair<true>(b,past,shared,epsilon,events,iteration,trace,Layers/2-1,
        std::make_integer_sequence<unsigned,10>{});
    Boundary<Head,Final,kHead%2>(b,past,shared,epsilon,events,iteration,trace,kHead,0,0,true);
    Boundary<Final,Prologue,(kHead+1)%2,true>(b,past,shared,epsilon,events,iteration,trace,
        kHead+1,0,0,true);
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
