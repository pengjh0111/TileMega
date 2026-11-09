// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#ifndef TILEMEGA_F3_REGISTER_BUDGET
#define TILEMEGA_F3_REGISTER_BUDGET 0
#endif
#if TILEMEGA_F3_REGISTER_BUDGET < 0 || TILEMEGA_F3_REGISTER_BUDGET > 3
#error "F3 register budget must be0 (56/224),1 (64/216),2 (72/208) or3 (72/216)"
#endif

#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Backend/Sm90F3SwapAB.cuh>
#include <tilemega/Backend/Sm90F3Small.cuh>
#include <tilemega/Backend/Sm90F3Attention.cuh>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <type_traits>
#include <utility>
#include <array>

#ifndef TILEMEGA_F3_PREFETCH_DISTANCE
#define TILEMEGA_F3_PREFETCH_DISTANCE 2
#endif
#ifndef TILEMEGA_F3_REPEAT_ARITHMETIC
#define TILEMEGA_F3_REPEAT_ARITHMETIC 0
#endif
#if TILEMEGA_F3_REPEAT_ARITHMETIC != 0 && TILEMEGA_F3_REPEAT_ARITHMETIC != 1
#error "F3 Repeat arithmetic must be0 or1"
#endif

// The F3 forward program (DNN3 prompt §5). One pass over a compile-time stage
// list on a 384-thread CTA: two consumer warpgroups and one producer warpgroup walk
// the same list, each calling its own half of every stage, and a stage's
// boundary is the epoch-counted grid gate of §5.5.
//
// There is no operand bank. A body reads its pointers and tensor maps straight
// out of the bindings, so no thread has to publish a record before the gate.
namespace tilemega::codegen::forward_f3 {

using Element=cutlass::bfloat16_t;
inline constexpr unsigned kAbsent=~0u;

struct RepeatWeight {unsigned buffer,bytes;};

// The tensor maps travel as grid constants, as the TMA contract requires; the
// bound pointers travel as a device array, because a model with hundreds of
// stages has more buffers than the 32 KB parameter space holds.
template<unsigned Count,unsigned Maps>
struct alignas(64) Bindings {
  CUtensorMap maps[Maps];
  Element* const* buffers;
  unsigned const* repeat_indices=nullptr;
  RepeatWeight const* repeat_weights=nullptr;
  template<unsigned Id>
  __device__ __forceinline__ Element* Get() const {
    if constexpr(Id==kAbsent)return nullptr;
    else {static_assert(Id<Count);return buffers[Id];}
  }
  template<unsigned Id>
  __device__ __forceinline__ CUtensorMap const* Map() const {
    if constexpr(Id==kAbsent)return nullptr;
    else {static_assert(Id<Maps);return &maps[Id];}
  }
};

struct alignas(64) ForwardCounter {unsigned long long arrivals;};

// One descriptor over a bound buffer, encoded on the host: either a tiled
// rank-2/3 view (`kind` 0) at an element offset into the buffer, or the TMA
// im2col view of a convolution's input (`kind` 1), which the driver builds
// from the plan's own ConvDesc and buffer layout.
struct ForwardMapSpec {
  // `swizzle` is 128 for a WGMMA operand or an epilogue tile read through an
  // SW128 descriptor, and 0 for a tile a CUDA-core body reads row-major.
  unsigned buffer,kind,rank,swizzle;
  unsigned long long offset;
  unsigned long long dims[4],strides[3];
  unsigned box[4];
  codegen::ConvDesc conv;
  codegen::DmBufferLayout layout;
};

// A stage's epilogue operands, assembled from the bindings at the point of
// use. Per-channel vectors are FP32 in the plan's buffer table.
template<class B,class Bind,class Shape>
__device__ __forceinline__ backend::Sm90F3EpilogueOperands EpilogueOf(B const& b) {
  backend::Sm90F3EpilogueOperands e{};
  e.vectors[0]=reinterpret_cast<float const*>(b.template Get<Bind::kVector0>());
  e.vectors[1]=reinterpret_cast<float const*>(b.template Get<Bind::kVector1>());
  e.vectors[2]=reinterpret_cast<float const*>(b.template Get<Bind::kVector2>());
  e.vectors[3]=reinterpret_cast<float const*>(b.template Get<Bind::kVector3>());
  e.residual=b.template Get<Bind::kResidual>();
  e.stats=reinterpret_cast<float const*>(b.template Get<Bind::kStats>());
  e.row_stats=reinterpret_cast<float*>(b.template Get<Bind::kRowStats>());
  e.row_partials=reinterpret_cast<float*>(b.template Get<Bind::kRowPartials>());
  e.row_counters=reinterpret_cast<unsigned long long*>(b.template Get<Bind::kRowCounters>());
  e.channel_partial=reinterpret_cast<float*>(b.template Get<Bind::kChannelPartial>());
  e.partial=reinterpret_cast<float*>(b.template Get<Bind::kPartial>());
  e.counters=reinterpret_cast<unsigned long long*>(b.template Get<Bind::kCounters>());
  e.output=b.template Get<Bind::kOutput>();
  e.output_stride=Shape::kOutputStride;
  e.residual_stride=Shape::kResidualStride;
  e.norm_width=Shape::kNormWidth;
  e.epsilon=Shape::kEpsilon;
  return e;
}

// The three stage wrappers. Each one builds its body's operand struct and
// forwards the role call; nothing else sits between the list and the body.
template<class Body,class Bind,class Shape>
struct GemmStage {
  using Kind=Body;
  static constexpr bool kMatrix=true;
  static constexpr std::size_t kWeightBytes=Body::kWeightBytes;
  static constexpr bool kUsesRing=true;
  static constexpr bool kCanPreissue=Body::kCanPreissue;
  static constexpr int kSharedBytes=Body::kLayoutBytes;
  template<class B>
  __device__ static __forceinline__ void PrefetchWeights(B const& b,std::size_t limit) {
    constexpr std::size_t kChunk=64*1024;
    std::size_t part=((limit+gridDim.x-1)/gridDim.x+15)/16*16;
    std::size_t begin=std::size_t(blockIdx.x)*part,end=begin+part;
    if(end>limit)end=limit;
    auto* weights=reinterpret_cast<char const*>(b.template Get<Bind::kWeight>());
    for(std::size_t offset=begin;offset<end;offset+=kChunk) {
      unsigned bytes=unsigned(end-offset<kChunk?end-offset:kChunk);
      backend::Sm90F3Async::Base::Prefetch(weights+offset,bytes);
    }
  }
  template<class B>
  __device__ static __forceinline__ void Preissue(B const& b,backend::Sm90F3Ring& ring,
      char* arena,backend::Sm90F3Phase& phase) {
    Body::Preissue(Operands(b),ring,reinterpret_cast<Element*>(arena),phase);
  }
  template<class B>
  __device__ static __forceinline__ backend::Sm90F3GemmOperands Operands(B const& b) {
    backend::Sm90F3GemmOperands p{};
    p.activation=b.template Map<Bind::kMap>();
    p.residual=b.template Map<Bind::kResidualMap>();
    p.weight=b.template Get<Bind::kWeight>();
    p.input=b.template Get<Bind::kInput>();
    p.epilogue=EpilogueOf<B,Bind,Shape>(b);
    if constexpr(Body::kTmaStore)p.output_map=b.template Map<Bind::kOutputMap>();
    return p;
  }
  template<class B>
  __device__ static __forceinline__ void Produce(B const& b,backend::Sm90F3Ring& ring,
      char* arena,char* side,backend::Sm90F3Phase& phase,int prearmed) {
    Body::Produce(Operands(b),ring,reinterpret_cast<Element*>(arena),side,phase,prearmed);
  }
  template<class B>
  __device__ static __forceinline__ void Consume(B const& b,backend::Sm90F3Ring& ring,
      char* arena,char* side,backend::Sm90F3Phase& phase,backend::Sm90F3Header* header,int wg) {
    Body::Consume(Operands(b),ring,reinterpret_cast<Element*>(arena),side,phase,header,wg);
  }
};

template<class Body,class Bind,class Shape>
struct AttentionStage {
  using Kind=Body;
  static constexpr bool kMatrix=false;
  static constexpr bool kUsesRing=true,kCanPreissue=false;
  static constexpr int kSharedBytes=Body::kLayoutBytes;
  template<class B>
  __device__ static __forceinline__ backend::Sm90F3AttentionOperands Operands(B const& b) {
    return {b.template Map<Bind::kQueryMap>(),b.template Map<Bind::kMap>(),
            b.template Map<Bind::kValueMap>(),b.template Get<Bind::kOutput>()};
  }
  template<class B>
  __device__ static __forceinline__ void Produce(B const& b,backend::Sm90F3Ring& ring,
      char* arena,char*,backend::Sm90F3Phase& phase,int) {
    Body::Produce(Operands(b),ring,arena,phase);
  }
  template<class B>
  __device__ static __forceinline__ void Consume(B const& b,backend::Sm90F3Ring& ring,
      char* arena,char*,backend::Sm90F3Phase& phase,backend::Sm90F3Header*,int) {
    Body::Consume(Operands(b),ring,arena,phase);
  }
};

template<class Body,class Bind,class Shape>
struct SmallStage {
  using Kind=Body;
  static constexpr bool kMatrix=false;
  static constexpr bool kUsesRing=false;
  static constexpr bool kCanPreissue=false;
  static constexpr int kSharedBytes=Body::kSharedBytes;
  template<class B>
  __device__ static __forceinline__ backend::Sm90F3SmallOperands Operands(B const& b) {
    backend::Sm90F3SmallOperands p{};
    p.input=b.template Get<Bind::kInput>();
    p.weight=b.template Get<Bind::kWeight>();
    p.table=b.template Get<Bind::kTable>();
    p.vector=reinterpret_cast<float const*>(b.template Get<Bind::kVector0>());
    p.stats=reinterpret_cast<float*>(b.template Get<Bind::kRowStats>());
    p.accumulator=reinterpret_cast<float*>(b.template Get<Bind::kPartial>());
    p.output=b.template Get<Bind::kOutput>();
    p.index0=reinterpret_cast<long long const*>(b.template Get<Bind::kIndex0>());
    p.index1=reinterpret_cast<long long const*>(b.template Get<Bind::kIndex1>());
    p.width=Shape::kWidth;p.height=Shape::kHeight;
    p.channels=Shape::kChannels;p.out_channels=Shape::kOutChannels;
    p.parts=Shape::kPoolParts;p.rows=Shape::kRows;
    p.epsilon=Shape::kEpsilon;
    return p;
  }
  template<class B>
  __device__ static __forceinline__ void Produce(B const&,backend::Sm90F3Ring&,
      char*,char*,backend::Sm90F3Phase&,int) {}
  template<class B>
  __device__ static __forceinline__ void Consume(B const& b,backend::Sm90F3Ring& ring,
      char* arena,char*,backend::Sm90F3Phase& phase,backend::Sm90F3Header* header,int wg) {
    TM_F3_TRACE_BEGIN(kSmallCompute);
    Body::Consume(Operands(b),ring,arena,phase,header,wg);
    TM_F3_TRACE_END(kSmallCompute);
  }
};

// A stage's placement: arena (i % 2), or the whole payload when the stage
// spans both (§5.2 item 3).
template<class Stage,int Offset>
struct Placement:Stage {
  static_assert(Offset>=0 && Offset%1024==0);
  static constexpr int kOffset=Offset;
  static constexpr bool kSpanning=Offset==backend::Sm90F3Budget::kHeader &&
      Stage::kSharedBytes>backend::Sm90F3Budget::kArena;
};

// Indexing the stage list by inheritance rather than recursion: a model with
// hundreds of stages exceeds the standard library's tuple_element depth.
template<unsigned I,class T>
struct Entry {using Type=T;};
template<class Sequence,class... Stage>
struct Lookup;
template<unsigned... I,class... Stage>
struct Lookup<std::integer_sequence<unsigned,I...>,Stage...>:Entry<I,Stage>... {};
template<unsigned I,class T>
Entry<I,T> Find(Entry<I,T> const&);

template<class S,class=void>struct MulticastStage:std::false_type {};
template<class S>struct MulticastStage<S,std::void_t<decltype(S::Kind::kMulticast)>>
    :std::bool_constant<S::Kind::kMulticast> {};

template<int Batch,unsigned Buffers,unsigned Maps,class... Stage>
struct Program {
  using Binding=Bindings<Buffers,Maps>;
  using Map=Lookup<std::make_integer_sequence<unsigned,sizeof...(Stage)>,Stage...>;
  template<unsigned I>
  using At=typename decltype(Find<I>(Map{}))::Type;
  static constexpr unsigned kStages=sizeof...(Stage);
  static constexpr bool kRepeated=false;
  static constexpr int kBatch=Batch,kThreads=384,kResidency=1;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps;
  static constexpr int kCluster=(MulticastStage<Stage>::value || ...)?2:1;
  static_assert(kStages>=1);
  static_assert(TILEMEGA_F3_PREFETCH_DISTANCE==1 || TILEMEGA_F3_PREFETCH_DISTANCE==2);
  static constexpr int kSharedBytes=backend::Sm90F3Budget::kBudget;
  static_assert(sizeof(Binding)+256<32764);

  // Both roles meet here. The consumers publish this stage's writes and count
  // one arrival per CTA against the epoch; the producer, which has already
  // pre-issued the next stage, joins the same 384-thread barrier and only then
  // fences its async proxy against the activations the gate just published.
  template<unsigned I>
  __device__ static __forceinline__ void Gate(ForwardCounter* events,
      unsigned long long epoch,bool consumer) {
    using Async=backend::Sm90F3Async;
    if constexpr(I+1<kStages) {
      if(consumer) {
        Async::template NamedBarrier<Async::kBarConsumers,256>();
        if(tilemega::backend::Sm90F3Async::ThreadIndex()==0) {
          Async::GateArrive(&events[I].arrivals);
          Async::GateWait(&events[I].arrivals,epoch*gridDim.x);
        }
        Async::template NamedBarrier<Async::kBarGate,384>();
      }else {
        Async::template NamedBarrier<Async::kBarGate,384>();
        if(Async::ThreadIndex()==256 || Async::ThreadIndex()==288)
          Async::Base::ProxyAsyncGlobalFence();
        __syncwarp();
      }
    }
  }

  __device__ static __forceinline__ char* Arena(char* shared,int offset) {
    return shared+offset;
  }

  // The consumer loop: warps 0-7. Each stage's own phase bits live in the
  // registers of the role, indexed by the arena the stage sits in, so they
  // continue across the stages that share one arena (§5.3 item 2).
  template<unsigned... I>
  __device__ static __forceinline__ void ConsumerLoop(Binding const& b,char* shared,
      ForwardCounter* events,unsigned long long epoch,
      std::integer_sequence<unsigned,I...>) {
    auto* header=reinterpret_cast<backend::Sm90F3Header*>(shared);
    backend::Sm90F3Phase phase[2]{};
    int wg=int(threadIdx.x)/128;
    ((ConsumeOne<I>(b,shared,header,phase,wg),Gate<I>(events,epoch,true),TraceLeave()),...);
  }
  __device__ static __forceinline__ void TraceLeave() {TM_F3_TRACE_LEAVE();}
  template<unsigned I>
  __device__ static __forceinline__ void ConsumeOne(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],int wg) {
    using S=At<I>;
    TM_F3_TRACE_ENTER(I);
    auto* arena=Arena(shared,S::kOffset);
    // A spanning stage owns both arenas, so its slots start at the header.
    auto* side=arena+SideOffset<I>();
    S::template Consume<Binding>(b,header->rings[I%2],arena,side,phase[I%2],header,wg);
    TM_F3_TRACE_ARRIVE();
  }
  template<unsigned I>
  static constexpr int Prefix() {
    if constexpr(I==0 || !At<I>::kCanPreissue)return 0;
    else if constexpr(At<I-1>::kSpanning || At<I>::kSpanning)return 0;
    else return At<I>::Kind::kSlots<At<I>::Kind::kIterations?
        At<I>::Kind::kSlots:At<I>::Kind::kIterations;
  }
  template<unsigned I>
  __device__ static __forceinline__ void PreissueNext(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2]) {
    if constexpr(I+1<kStages) {
      if constexpr(Prefix<I+1>()>0) {
        using S=At<I+1>;
        S::template Preissue<Binding>(b,header->rings[(I+1)%2],
            Arena(shared,S::kOffset),phase[(I+1)%2]);
      }
    }
  }
  template<unsigned I>
  static constexpr unsigned NextMatrix() {
    if constexpr(I>=kStages)return kStages;
    else if constexpr(At<I>::kMatrix)return I;
    else return NextMatrix<I+1>();
  }
  template<unsigned I>
  __device__ static __forceinline__ void PrefetchNextWeights(Binding const& b) {
    constexpr unsigned next=NextMatrix<I+1>();
    constexpr unsigned first=TILEMEGA_F3_PREFETCH_DISTANCE==1?next:NextMatrix<next+1>();
    constexpr std::size_t kBudget=24*1024*1024;
    if constexpr(first<kStages) {
      constexpr auto bytes=At<first>::kWeightBytes;
      At<first>::template PrefetchWeights<Binding>(b,bytes<kBudget?bytes:kBudget);
      if constexpr(bytes<=8*1024*1024) {
        constexpr unsigned second=NextMatrix<first+1>();
        if constexpr(second<kStages) {
          if constexpr(bytes+At<second>::kWeightBytes<=kBudget)
            At<second>::template PrefetchWeights<Binding>(b,At<second>::kWeightBytes);
        }
      }
    }
  }
  template<unsigned I>
  __device__ static __forceinline__ void ProduceOne(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2]) {
    using S=At<I>;
    auto* arena=Arena(shared,S::kOffset);
    auto* side=arena+SideOffset<I>();
    if(backend::Sm90F3Async::ThreadIndex()==288)PrefetchNextWeights<I>(b);
    S::template Produce<Binding>(b,header->rings[I%2],arena,side,phase[I%2],Prefix<I>());
  }
  template<unsigned I>
  static constexpr int SideOffset() {return At<I>::Kind::kSideOffset;}

  template<unsigned... I>
  __device__ static __forceinline__ void ProducerLoop(Binding const& b,char* shared,
      ForwardCounter* events,unsigned long long epoch,
      std::integer_sequence<unsigned,I...>) {
    auto* header=reinterpret_cast<backend::Sm90F3Header*>(shared);
    backend::Sm90F3Phase phase[2]{};
    ((ProduceOne<I>(b,shared,header,phase),PreissueNext<I>(b,shared,header,phase),
        Gate<I>(events,epoch,false)),...);
  }
};

template<class Table,unsigned Stride,unsigned R,unsigned Column>
struct RepeatIndexPattern {
  static_assert(R>1 && Column<Stride);
  static constexpr unsigned kFirst=Table::indices[Column];
  static constexpr unsigned kSecond=Table::indices[Stride+Column];
  // Unsigned differences also encode descending columns; every row is checked.
  static constexpr unsigned kDelta=kSecond-kFirst;
  static constexpr unsigned Classify() {
    bool constant=true,affine=kFirst!=kAbsent,periodic=true;
    for(unsigned r=0;r<R;++r) {
      unsigned value=Table::indices[r*Stride+Column];
      constant&=value==kFirst;
      affine&=value!=kAbsent && value==kFirst+r*kDelta;
      periodic&=value==(r%2?kSecond:kFirst);
    }
    return constant?1:affine?2:periodic?3:0;
  }
  static constexpr unsigned kKind=Classify();
};

// Canonical binding IDs are translated at run time; maps stay grid constants.
template<class Base,unsigned Count,unsigned Maps,class Table,unsigned R>
struct RepeatBindings {
  Base const& base;
  unsigned repetition;
  template<unsigned Column>
  __device__ __forceinline__ unsigned Index() const {
#if TILEMEGA_F3_REPEAT_ARITHMETIC
    using P=RepeatIndexPattern<Table,Count+Maps,R,Column>;
    if constexpr(P::kKind==1)return P::kFirst;
    else if constexpr(P::kKind==2)return P::kFirst+repetition*P::kDelta;
    else if constexpr(P::kKind==3)return repetition%2?P::kSecond:P::kFirst;
    else
#endif
      return base.repeat_indices[repetition*(Count+Maps)+Column];
  }
  template<unsigned Id>
  __device__ __forceinline__ Element* Get() const {
    if constexpr(Id==kAbsent)return nullptr;
    else {
      static_assert(Id<Count);
      unsigned id=Index<Id>();
      return id==kAbsent?nullptr:base.buffers[id];
    }
  }
  template<unsigned Id>
  __device__ __forceinline__ CUtensorMap const* Map() const {
    if constexpr(Id==kAbsent)return nullptr;
    else {
      static_assert(Id<Maps);
      unsigned id=Index<Count+Id>();
      return id==kAbsent?nullptr:&base.maps[id];
    }
  }
};

// First/last variants preserve the embedding's NT=1 statistics and the final
// FFN's lack of a statistics output without duplicating the middle layers.
template<class Normal,class First=Normal,class Last=Normal>
struct RepeatStage {
#if TILEMEGA_F3_REGISTER_BUDGET >= 2
  // Explicit selection keeps the two arena histories in registers.
  __device__ static __forceinline__ backend::Sm90F3Phase LoadPhase(
      backend::Sm90F3Phase const (&phase)[2],unsigned stage) {
    return stage%2?phase[1]:phase[0];
  }
  __device__ static __forceinline__ void StorePhase(backend::Sm90F3Phase (&phase)[2],
      unsigned stage,backend::Sm90F3Phase const& current) {
    if(stage%2)phase[1]=current;else phase[0]=current;
  }
#endif
  template<class S>static constexpr bool Span() {
    return S::kSharedBytes>backend::Sm90F3Budget::kArena;
  }
  __device__ static __forceinline__ bool Spanning(unsigned rep,unsigned repeats) {
    if(rep==0)return Span<First>();
    if(rep+1==repeats)return Span<Last>();
    return Span<Normal>();
  }
  template<class S,class B>
  __device__ static __forceinline__ int PreissueAs(B const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],
      unsigned stage,bool previous_spanning) {
    if constexpr(S::kCanPreissue && !Span<S>()) {
      if(!previous_spanning) {
        unsigned arena=stage%2;
#if TILEMEGA_F3_REGISTER_BUDGET >= 2
        auto current=LoadPhase(phase,stage);
        S::Preissue(b,header->rings[arena],
            shared+backend::Sm90F3Budget::Offset(stage,S::kSharedBytes),current);
        StorePhase(phase,stage,current);
#else
        S::Preissue(b,header->rings[arena],
            shared+backend::Sm90F3Budget::Offset(stage,S::kSharedBytes),phase[arena]);
#endif
        return S::Kind::kSlots<S::Kind::kIterations?S::Kind::kSlots:S::Kind::kIterations;
      }
    }
    return 0;
  }
  template<class B>
  __device__ static __forceinline__ int Preissue(B const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],
      unsigned stage,unsigned rep,unsigned repeats,bool previous_spanning) {
    if constexpr(!std::is_same_v<First,Normal>)
      if(rep==0)return PreissueAs<First>(b,shared,header,phase,stage,previous_spanning);
    if constexpr(!std::is_same_v<Last,Normal>)
      if(rep+1==repeats)return PreissueAs<Last>(b,shared,header,phase,stage,previous_spanning);
    return PreissueAs<Normal>(b,shared,header,phase,stage,previous_spanning);
  }
  template<class S,class B>
  __device__ static __forceinline__ void ProduceAs(B const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],unsigned stage,int prefix) {
    auto* arena=shared+backend::Sm90F3Budget::Offset(stage,S::kSharedBytes);
#if TILEMEGA_F3_REGISTER_BUDGET >= 2
    auto current=LoadPhase(phase,stage);
    S::Produce(b,header->rings[stage%2],arena,arena+S::Kind::kSideOffset,current,prefix);
    StorePhase(phase,stage,current);
#else
    S::Produce(b,header->rings[stage%2],arena,arena+S::Kind::kSideOffset,phase[stage%2],prefix);
#endif
  }
  template<class B>
  __device__ static __forceinline__ void Produce(B const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],
      unsigned stage,unsigned rep,unsigned repeats,int prefix) {
    if constexpr(!std::is_same_v<First,Normal>)
      if(rep==0) {ProduceAs<First>(b,shared,header,phase,stage,prefix);return;}
    if constexpr(!std::is_same_v<Last,Normal>)
      if(rep+1==repeats) {ProduceAs<Last>(b,shared,header,phase,stage,prefix);return;}
    ProduceAs<Normal>(b,shared,header,phase,stage,prefix);
  }
  template<class S,class B>
  __device__ static __forceinline__ void ConsumeAs(B const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],unsigned stage,int wg) {
    auto* arena=shared+backend::Sm90F3Budget::Offset(stage,S::kSharedBytes);
#if TILEMEGA_F3_REGISTER_BUDGET >= 2
    auto current=LoadPhase(phase,stage);
    S::Consume(b,header->rings[stage%2],arena,arena+S::Kind::kSideOffset,current,header,wg);
    StorePhase(phase,stage,current);
#else
    S::Consume(b,header->rings[stage%2],arena,arena+S::Kind::kSideOffset,phase[stage%2],header,wg);
#endif
  }
  template<class B>
  __device__ static __forceinline__ void Consume(B const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],
      unsigned stage,unsigned rep,unsigned repeats,int wg) {
    if constexpr(!std::is_same_v<First,Normal>)
      if(rep==0) {ConsumeAs<First>(b,shared,header,phase,stage,wg);return;}
    if constexpr(!std::is_same_v<Last,Normal>)
      if(rep+1==repeats) {ConsumeAs<Last>(b,shared,header,phase,stage,wg);return;}
    ConsumeAs<Normal>(b,shared,header,phase,stage,wg);
  }
};

template<unsigned R,class... Stage>struct Repeat {};
template<unsigned R,class... Stage>struct RepeatTables;

template<int Batch,unsigned Buffers,unsigned Maps,class Before,unsigned R,
    class... Body,class After,class Final>
struct Program<Batch,Buffers,Maps,Before,Repeat<R,Body...>,After,Final> {
  using Binding=Bindings<Buffers,Maps>;
  using Tables=RepeatTables<R,Body...>;
  using View=RepeatBindings<Binding,Buffers,Maps,Tables,R>;
  using BodyMap=Lookup<std::make_integer_sequence<unsigned,sizeof...(Body)>,Body...>;
  template<unsigned I>using At=typename decltype(Find<I>(BodyMap{}))::Type;
  static constexpr bool kRepeated=true;
  static constexpr unsigned kBody=sizeof...(Body),kStages=1+R*kBody+2;
  static constexpr int kBatch=Batch,kThreads=384,kResidency=1,kCluster=1;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps;
  static constexpr int kSharedBytes=backend::Sm90F3Budget::kBudget;
  // Arena and phase selection use the logical stage, including odd bodies
  // whose first stage alternates arenas on successive repetitions.
  static_assert(R>1 && kBody>0);
  static_assert(sizeof(Binding)+256<32764);
  static auto const& RepeatIndices() {return Tables::indices;}
  static auto const& RepeatWeights() {return Tables::weights;}

  __device__ static __forceinline__ void Gate(unsigned stage,ForwardCounter* events,
      unsigned long long epoch,bool consumer) {
    using A=backend::Sm90F3Async;
    if(stage+1<kStages) {
      if(consumer) {
        A::template NamedBarrier<A::kBarConsumers,256>();
        if(A::ThreadIndex()==0) {
          A::GateArrive(&events[stage].arrivals);
          A::GateWait(&events[stage].arrivals,epoch*gridDim.x);
        }
        A::template NamedBarrier<A::kBarGate,384>();
      }else {
        A::template NamedBarrier<A::kBarGate,384>();
        if(A::ThreadIndex()==256 || A::ThreadIndex()==288)A::Base::ProxyAsyncGlobalFence();
        __syncwarp();
      }
    }
  }
  __device__ static __forceinline__ void Prefetch(Binding const& b,unsigned stage) {
    constexpr unsigned budget=24*1024*1024;
    unsigned found=0,used=0;
    for(unsigned i=stage+1;i<kStages;++i) {
      auto info=b.repeat_weights[i];
      if(!info.bytes)continue;
      if(++found<TILEMEGA_F3_PREFETCH_DISTANCE)continue;
      if(used+info.bytes>budget && used)break;
      unsigned bytes=info.bytes<budget?info.bytes:budget;
      unsigned part=((bytes+gridDim.x-1)/gridDim.x+15)/16*16;
      unsigned begin=blockIdx.x*part,end=begin+part;
      if(end>bytes)end=bytes;
      auto* weight=reinterpret_cast<char const*>(b.buffers[info.buffer]);
      for(unsigned offset=begin;offset<end;offset+=64*1024)
        backend::Sm90F3Async::Base::Prefetch(weight+offset,
            end-offset<64*1024?end-offset:64*1024);
      used+=bytes;
      if(found>TILEMEGA_F3_PREFETCH_DISTANCE || bytes>8*1024*1024)break;
    }
  }
  template<class S>
  __device__ static __forceinline__ void StaticConsume(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],
      unsigned stage,int wg,ForwardCounter* events,unsigned long long epoch) {
    TM_F3_TRACE_ENTER(stage);
    auto* arena=shared+backend::Sm90F3Budget::Offset(stage,S::kSharedBytes);
    S::Consume(b,header->rings[stage%2],arena,arena+S::Kind::kSideOffset,phase[stage%2],header,wg);
    TM_F3_TRACE_ARRIVE();Gate(stage,events,epoch,true);TM_F3_TRACE_LEAVE();
  }
  template<unsigned I>
  __device__ static __forceinline__ void BodyConsume(View const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],unsigned rep,
      int wg,ForwardCounter* events,unsigned long long epoch) {
    unsigned stage=1+rep*kBody+I;
    TM_F3_TRACE_ENTER(stage);
    At<I>::Consume(b,shared,header,phase,stage,rep,R,wg);
    TM_F3_TRACE_ARRIVE();Gate(stage,events,epoch,true);TM_F3_TRACE_LEAVE();
  }
  template<unsigned... I>
  __device__ static __forceinline__ void RepeatedConsumer(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],int wg,
      ForwardCounter* events,unsigned long long epoch,std::integer_sequence<unsigned,I...>) {
    #pragma unroll 1
    for(unsigned rep=0;rep<R;++rep) {
      View view{b,rep};
      (BodyConsume<I>(view,shared,header,phase,rep,wg,events,epoch),...);
    }
  }
  template<class Sequence>
  __device__ static __forceinline__ void ConsumerLoop(Binding const& b,char* shared,
      ForwardCounter* events,unsigned long long epoch,Sequence) {
    auto* header=reinterpret_cast<backend::Sm90F3Header*>(shared);
    backend::Sm90F3Phase phase[2]{};int wg=threadIdx.x/128;
    StaticConsume<Before>(b,shared,header,phase,0,wg,events,epoch);
    RepeatedConsumer(b,shared,header,phase,wg,events,epoch,
        std::make_integer_sequence<unsigned,kBody>{});
    StaticConsume<After>(b,shared,header,phase,kStages-2,wg,events,epoch);
    StaticConsume<Final>(b,shared,header,phase,kStages-1,wg,events,epoch);
  }
  template<class S>
  __device__ static __forceinline__ void StaticProduce(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],unsigned stage,int prefix) {
    if(backend::Sm90F3Async::ThreadIndex()==288)Prefetch(b,stage);
    auto* arena=shared+backend::Sm90F3Budget::Offset(stage,S::kSharedBytes);
    S::Produce(b,header->rings[stage%2],arena,arena+S::Kind::kSideOffset,phase[stage%2],prefix);
  }
  template<unsigned I>
  __device__ static __forceinline__ void BodyProduce(View const& view,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],unsigned rep,
      int& prefix,ForwardCounter* events,unsigned long long epoch) {
    unsigned stage=1+rep*kBody+I;
    if(backend::Sm90F3Async::ThreadIndex()==288)Prefetch(view.base,stage);
    At<I>::Produce(view,shared,header,phase,stage,rep,R,prefix);
    bool spanning=At<I>::Spanning(rep,R);
    if constexpr(I+1<kBody)
      prefix=At<I+1>::Preissue(view,shared,header,phase,stage+1,rep,R,spanning);
    else {
      if(rep+1<R) {
        View next{view.base,rep+1};
        prefix=At<0>::Preissue(next,shared,header,phase,stage+1,rep+1,R,spanning);
      }else prefix=RepeatStage<After>::Preissue(view.base,shared,header,phase,stage+1,0,1,spanning);
    }
    Gate(stage,events,epoch,false);
  }
  template<unsigned... I>
  __device__ static __forceinline__ void RepeatedProducer(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2],int& prefix,
      ForwardCounter* events,unsigned long long epoch,std::integer_sequence<unsigned,I...>) {
    #pragma unroll 1
    for(unsigned rep=0;rep<R;++rep) {
      View view{b,rep};
      (BodyProduce<I>(view,shared,header,phase,rep,prefix,events,epoch),...);
    }
  }
  template<class Sequence>
  __device__ static __forceinline__ void ProducerLoop(Binding const& b,char* shared,
      ForwardCounter* events,unsigned long long epoch,Sequence) {
    auto* header=reinterpret_cast<backend::Sm90F3Header*>(shared);
    backend::Sm90F3Phase phase[2]{};
    StaticProduce<Before>(b,shared,header,phase,0,0);
    View first{b,0};
    int prefix=At<0>::Preissue(first,shared,header,phase,1,0,R,RepeatStage<Before>::template Span<Before>());
    Gate(0,events,epoch,false);
    RepeatedProducer(b,shared,header,phase,prefix,events,epoch,
        std::make_integer_sequence<unsigned,kBody>{});
    StaticProduce<After>(b,shared,header,phase,kStages-2,prefix);
    prefix=RepeatStage<Final>::Preissue(b,shared,header,phase,kStages-1,0,1,
        RepeatStage<After>::template Span<After>());
    Gate(kStages-2,events,epoch,false);
    StaticProduce<Final>(b,shared,header,phase,kStages-1,prefix);
  }
};

template<class P>
__global__ __launch_bounds__(384,1)
void sm90_forward_f3_kernel(const __grid_constant__ typename P::Binding b,
    ForwardCounter* events,unsigned long long epoch) {
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<backend::Sm90F3Header*>(shared);
  // Thread zero initializes both arenas' barriers, one fence publishes them,
  // then every consumer warp primes the empty barriers so the producer's first
  // wait on phase 0 passes (§5.3 item 1).
  if(tilemega::backend::Sm90F3Async::ThreadIndex()==0) {
    header->rings[0].Initialize();
    header->rings[1].Initialize();
    header->last_arriver=0;
    backend::Sm90F3Async::Base::InitFence();
  }
  __syncthreads();
  if constexpr(P::kCluster==2)backend::Sm90F3Async::Base::ClusterSync();
  if(threadIdx.x<256 && (tilemega::backend::Sm90F3Async::ThreadIndex()&31)==0) {
    header->rings[0].Prime();
    header->rings[1].Prime();
    if constexpr(P::kCluster==2) {
      header->rings[0].PrimeCluster();header->rings[1].PrimeCluster();
    }
  }
  __syncthreads();
  if constexpr(P::kCluster==2)backend::Sm90F3Async::Base::ClusterSync();
  auto sequence=std::make_integer_sequence<unsigned,P::kStages>{};
  if(threadIdx.x>=256) {
    // Profile3 permits R211 in tuned B16 consumers:72+216+216=504 per SMSP.
    backend::Sm90F3Async::template RegDealloc<(TILEMEGA_F3_REGISTER_BUDGET>=2?72:
        (TILEMEGA_F3_REGISTER_BUDGET?64:56))>();
    P::ProducerLoop(b,shared,events,epoch,sequence);
  }else {
    backend::Sm90F3Async::template RegAlloc<(TILEMEGA_F3_REGISTER_BUDGET==2?208:
        (TILEMEGA_F3_REGISTER_BUDGET?216:224))>();
    P::ConsumerLoop(b,shared,events,epoch,sequence);
  }
  __syncthreads();
  if constexpr(P::kCluster==2)backend::Sm90F3Async::Base::ClusterSync();
  if(tilemega::backend::Sm90F3Async::ThreadIndex()==0) {header->rings[0].Invalidate();header->rings[1].Invalidate();}
}

}  // namespace tilemega::codegen::forward_f3
