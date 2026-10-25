// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Codegen/Sm90DenseEpilogueContract.h>
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <tilemega/Backend/Sm90DenseMoe.cuh>
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
  // lm_ctr of the folded argmax (D9); runtime-owned and only ever increased.
  std::uint64_t* lm_counter;
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
  static constexpr int kBatch=Batch,kTasks=N/TN,kSharedBytes=Body::kSharedBytes;
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

// Native MoE stages (prompt §8): M4 router + top-k, M5 grouped gate_up, M6
// down + combine. W: the (per-layer) weights; X: the residual stream x' (M5
// gathers its rows, M6's last arrivers add into it); Out: logits, h or P;
// Schedule, Routes (weights) and Index (this layer's top-k, for the routing
// check); Counter: the router's or the per-tile combine counters; Map: x'
// (router) or h (down).
template<int Batch,int N,int K,int TN,int TK,int Slots,int WN,int WK,backend::Sm90DenseMoeMode Mode,
         int U,int Experts,int TopK,bool FloatWeights,int Page,unsigned W,unsigned X,unsigned Out,
         unsigned Schedule,unsigned Routes,unsigned Index,unsigned Counter,unsigned Map,
         int RegisterStages>
struct Moe {
  using Body=backend::Sm90DenseMoeGemm<Batch,N,K,TN,TK,Slots,WN,WK,Mode,U,Experts,TopK,FloatWeights,
      Page,RegisterStages>;
  static constexpr bool kUsesOperandRecord=true;
  static constexpr int kTasks=Body::kMaxTasks,kSharedBytes=Body::kSharedBytes,kRingSlots=Slots;
  // M4's routes are unknown to its predecessor's producer: M5 receives no stream.
  static constexpr int kMaxPreissue=Mode==backend::Sm90DenseMoeMode::kGateUp?0:
      (Slots<Body::kIterations?Slots:Body::kIterations)*2*TN*TK;
  static constexpr auto kLayout=Body::kLayout;
  using Preissue=MoeRead<Body>;
  template<int Bytes>
  static constexpr auto Payload() {
    return backend::Sm90DenseLayout::GemmPayload(TN,TK,Slots,K,Batch,Bytes);
  }
  template<class B>
  __device__ static __forceinline__ ReadSources Sources(B const& b,int layer) {
    return {b.template Get<W>(layer),nullptr,nullptr,layer,
        reinterpret_cast<int const*>(b.template Get<Schedule>(layer))};
  }
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,int layer,int,float epsilon,
      backend::Sm90DenseOperandBank& record) {
    CUtensorMap const* map=nullptr;
    if constexpr(Map!=kAbsent)map=b.template Map<Map>();
    record.moe={map,b.template Get<W>(layer),b.template Get<X>(layer),b.template Get<Out>(layer),
        reinterpret_cast<int*>(b.template Get<Schedule>(layer)),b.template Get<Routes>(layer),
        reinterpret_cast<int*>(b.template Get<Index>(layer)),b.template Get<X>(layer),
        reinterpret_cast<std::uint64_t*>(b.template Get<Counter>(layer)),epsilon};
  }
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const&,int,char* shared,float,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record,
      bool streamed) {
    StageEntry<Body>::template Moe<Prefix>(record.moe,shared,&ring,tail,streamed);
  }
};

// lm_head with the folded argmax and, with Embed, the next step's embedding
// (D9). Its value/index outputs hold the per-CTA [B][G] partials.
template<class G,unsigned Tokens,unsigned Table,unsigned X,int Width,int Capacity,bool Embed>
struct Head:G {
  static_assert(G::kUsesOperandRecord);
  using Finish=backend::Sm90DenseHeadFinish<G::kBatch,Width,Capacity,Embed>;
  // The [B][G] partials reuse the [B][tasks] argmax outputs.
  static constexpr int kMaxGrid=G::kTasks;
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,int layer,int past,float epsilon,
      backend::Sm90DenseOperandBank& record) {
    G::PublishOperands(b,layer,past,epsilon,record);
    record.head={reinterpret_cast<int*>(b.template Get<Tokens>()),b.lm_counter,
        b.template Get<Table>(),b.template Get<X>(),past};
  }
  template<int Prefix,class B,class Tail>
  __device__ static __forceinline__ void Run(B const&,int,char* shared,float,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& record,
      bool streamed) {
    StageEntry<typename G::Body>::template Head<Prefix>(record.gemm,shared,&ring,tail,streamed,
        Finish{&record.head});
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

// The typed stages of PerGroup consecutive layers; an even stage count
// keeps every typed stage's arena parity fixed (dense: two layers of five,
// MoE: one layer of six).
template<int PerGroup,class... Stage>
struct LayerGroup {
  static constexpr int kLayers=PerGroup,kSize=int(sizeof...(Stage));
  static_assert(PerGroup>0 && kSize%PerGroup==0 && kSize%2==0,
      "a layer group holds whole layers and an even number of typed stages");
  using Stages=std::tuple<Stage...>;
};

// D7 (prompt §6): Prologue (embedding) -> layer groups -> LmHead (lm_head
// with the folded argmax, D9). The last group runs separately so that its
// final stage streams into the head. Executed boundaries keep their global
// index g (grid counters, banks, traces); Loop selects a runtime group loop
// over the same group code that the unrolled form instantiates once per
// group (DT-3). With Embed the head's last arriver writes the next step's
// embedding and the prologue runs only before a request's step 0.
template<int Batch,int Capacity,int Residency,unsigned Buffers,unsigned Maps,int Arena,
         int Layers,unsigned LayerSlots,bool Loop,bool Embed,class Prologue,class Group,class LmHead>
struct Program {
  using Binding=Bindings<Buffers,Maps,Layers,LayerSlots>;
  using GroupStages=typename Group::Stages;
  template<unsigned J> using GroupStage=std::tuple_element_t<J,GroupStages>;
  static constexpr int kBatch=Batch,kCapacity=Capacity,kResidency=Residency,kThreads=160;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps,kLayers=Layers;
  static constexpr unsigned kGroup=Group::kSize,kPerGroup=Group::kLayers,kPerLayer=kGroup/kPerGroup;
  static constexpr unsigned kStages=2+kPerLayer*Layers,kHead=1+kPerLayer*Layers;
  static_assert(Layers%Group::kLayers==0,"layer groups cover the layers exactly");
  static constexpr bool kEmbed=Embed;
  // The head's per-CTA partials reuse its [B][tasks] outputs.
  static constexpr int kHeadMaxGrid=LmHead::kMaxGrid;
  using RoleLaunch=backend::Sm90DenseRoleLaunch<kThreads>;
  static_assert(RoleLaunch::kThreads==kThreads);
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
  static constexpr bool GroupPlaced(std::integer_sequence<unsigned,J...>) {
    return (Placed<GroupStage<J>,(1+J)%2>() && ...);
  }
  static_assert(Placed<Prologue,0>() && Placed<LmHead,kHead%2>() &&
      GroupPlaced(std::make_integer_sequence<unsigned,kGroup>{}));
  template<unsigned... J>
  static constexpr int GroupBytes(std::integer_sequence<unsigned,J...>) {
    int value=1024;((value=value<GroupStage<J>::kSharedBytes?GroupStage<J>::kSharedBytes:value),...);
    return value;
  }
  static constexpr int kArenaSharedBytes=[] {
    int value=GroupBytes(std::make_integer_sequence<unsigned,kGroup>{});
    for(int bytes:{Prologue::kSharedBytes,LmHead::kSharedBytes})
      value=value<bytes?bytes:value;
    return value;
  }();
  // Tasks of executed boundary g.
  template<unsigned... J>
  static constexpr unsigned long long GroupTasks(unsigned j,std::integer_sequence<unsigned,J...>) {
    unsigned long long value=0;((value=j==J?GroupStage<J>::kTasks:value),...);return value;
  }
  static constexpr unsigned long long Tasks(unsigned g) {
    if(g==0)return Prologue::kTasks;
    if(g==kHead)return LmHead::kTasks;
    return GroupTasks((g-1)%kGroup,std::make_integer_sequence<unsigned,kGroup>{});
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
  // Parity) for runtime layer `layer`; Next runs next for `next_layer` with
  // parity NextParity (the same parity only after the head when the
  // prologue is skipped). streamed: Body's predecessor issued its prefix.
  template<class Body,class Next,unsigned Parity,bool Last=false,unsigned NextParity=Parity^1>
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
            backend::Sm90DenseOperandHeader::Bank(shared,NextParity));
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
  // Layers kPerGroup*group onwards. Only layer 0's first stage follows the
  // ring-less prologue and receives no stream.
  template<bool LastGroup,unsigned... J>
  __device__ static __forceinline__ void RunGroup(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,int group,
      std::integer_sequence<unsigned,J...>) {
    int layer=int(kPerGroup)*group;
    (Boundary<GroupStage<J>,std::conditional_t<J+1==kGroup,
        std::conditional_t<LastGroup,LmHead,GroupStage<0>>,GroupStage<(J+1)%kGroup>>,(1+J)%2>(
        b,past,shared,epsilon,events,iteration,trace,1+kGroup*group+J,layer+int(J/kPerLayer),
        J+1==kGroup?(LastGroup?0:layer+int(kPerGroup)):layer+int((J+1)/kPerLayer),
        J!=0 || group!=0),...);
  }
  template<int... P>
  __device__ static __forceinline__ void Unrolled(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,
      std::integer_sequence<int,P...>) {
    (RunGroup<false>(b,past,shared,epsilon,events,iteration,trace,P,
        std::make_integer_sequence<unsigned,kGroup>{}),...);
  }
  // step: the request's step number (pasts index), independent of launches.
  __device__ static __forceinline__ void Step(Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,
      unsigned step) {
#if TILEMEGA_TRACE_STEP
    StepTrace* record=trace.steps && trace.slots?
        trace.steps+(iteration%trace.slots)*gridDim.x+blockIdx.x:nullptr;
    if(threadIdx.x==0 && record)*record={iteration,static_cast<unsigned long long>(past),Now(),0,0,0,0,0,0,0};
#endif
    if(!Embed || step==0)
      Boundary<Prologue,GroupStage<0>,0>(b,past,shared,epsilon,events,iteration,trace,0,0,0,false);
    else if(threadIdx.x==0) {
      // The skipped prologue still counts once per step, keeping its target.
      auto* counter=&events[0].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
    }
    if constexpr(Loop) {
      #pragma unroll 1
      for(int group=0;group<Layers/int(kPerGroup)-1;++group)
        RunGroup<false>(b,past,shared,epsilon,events,iteration,trace,group,
            std::make_integer_sequence<unsigned,kGroup>{});
    }else Unrolled(b,past,shared,epsilon,events,iteration,trace,
        std::make_integer_sequence<int,Layers/int(kPerGroup)-1>{});
    RunGroup<true>(b,past,shared,epsilon,events,iteration,trace,Layers/int(kPerGroup)-1,
        std::make_integer_sequence<unsigned,kGroup>{});
    // No stream crosses the step boundary; with Embed the head publishes the
    // next step's layer-0 qkv bank in place of the skipped prologue.
    Boundary<LmHead,std::conditional_t<Embed,GroupStage<0>,Prologue>,kHead%2,true,
        Embed?1u:(kHead%2)^1u>(b,past,shared,epsilon,events,iteration,trace,kHead,0,0,true);
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
  // A launch that starts after step 0 without the prologue needs the
  // layer-0 qkv bank the previous launch's head published (D9 ③).
  if constexpr(P::kEmbed)
    if(first!=0) {
      if(threadIdx.x==0)P::template GroupStage<0>::PublishOperands(b,0,pasts[first],epsilon,
          backend::Sm90DenseOperandHeader::Bank(shared,1));
      __syncthreads();
    }
  if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[first]=Now();
  for(unsigned step=0;step<steps;++step) {
    int past=pasts[first+step];
#if TILEMEGA_NATIVE_RELEASE_ENABLED
    P::BeginRelease(shared,past,iteration+step);
#endif
    P::Step(b,past,shared,epsilon,events,iteration+step,trace,first+step);
#if TILEMEGA_NATIVE_RELEASE_ENABLED
    P::FlushRelease(shared,release_binding,iteration+step);
#endif
    if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[first+step+1]=Now();
  }
  rings[0].Invalidate();rings[1].Invalidate();
}

}  // namespace tilemega::codegen::native_dense
