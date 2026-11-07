// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Backend/Sm90F3SwapAB.cuh>
#include <tilemega/Backend/Sm90F3Small.cuh>
#include <tilemega/Backend/Sm90F3Attention.cuh>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <type_traits>
#include <utility>

// The F3 forward program (DNN3 prompt §5). One pass over a compile-time stage
// list on a 288-thread CTA: two consumer warpgroups and one producer warp walk
// the same list, each calling its own half of every stage, and a stage's
// boundary is the epoch-counted grid gate of §5.5.
//
// There is no operand bank. A body reads its pointers and tensor maps straight
// out of the bindings, so no thread has to publish a record before the gate.
namespace tilemega::codegen::forward_f3 {

using Element=cutlass::bfloat16_t;
inline constexpr unsigned kAbsent=~0u;

// The tensor maps travel as grid constants, as the TMA contract requires; the
// bound pointers travel as a device array, because a model with hundreds of
// stages has more buffers than the 32 KB parameter space holds.
template<unsigned Count,unsigned Maps>
struct alignas(64) Bindings {
  CUtensorMap maps[Maps];
  Element* const* buffers;
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
  static constexpr bool kUsesRing=true;
  static constexpr bool kCanPreissue=Body::kCanPreissue;
  static constexpr int kSharedBytes=Body::kLayoutBytes;
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

template<int Batch,unsigned Buffers,unsigned Maps,class... Stage>
struct Program {
  using Binding=Bindings<Buffers,Maps>;
  using Map=Lookup<std::make_integer_sequence<unsigned,sizeof...(Stage)>,Stage...>;
  template<unsigned I>
  using At=typename decltype(Find<I>(Map{}))::Type;
  static constexpr unsigned kStages=sizeof...(Stage);
  static constexpr int kBatch=Batch,kThreads=288,kResidency=1;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps;
  static_assert(kStages>=1);
  static constexpr int kSharedBytes=backend::Sm90F3Budget::kBudget;
  static_assert(sizeof(Binding)+256<32764);

  // Both roles meet here. The consumers publish this stage's writes and count
  // one arrival per CTA against the epoch; the producer, which has already
  // pre-issued the next stage, joins the same 288-thread barrier and only then
  // fences its async proxy against the activations the gate just published.
  template<unsigned I>
  __device__ static __forceinline__ void Gate(ForwardCounter* events,
      unsigned long long epoch,bool consumer) {
    using Async=backend::Sm90F3Async;
    if constexpr(I+1<kStages) {
      if(consumer) {
        Async::template NamedBarrier<Async::kBarConsumers,256>();
        if(threadIdx.x==0) {
          Async::GateArrive(&events[I].arrivals);
          Async::GateWait(&events[I].arrivals,epoch*gridDim.x);
        }
        Async::template NamedBarrier<Async::kBarGate,288>();
      }else {
        Async::template NamedBarrier<Async::kBarGate,288>();
        if(threadIdx.x==256)Async::Base::ProxyAsyncGlobalFence();
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
  __device__ static __forceinline__ void ProduceOne(Binding const& b,char* shared,
      backend::Sm90F3Header* header,backend::Sm90F3Phase (&phase)[2]) {
    using S=At<I>;
    auto* arena=Arena(shared,S::kOffset);
    auto* side=arena+SideOffset<I>();
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

template<class P>
__global__ __launch_bounds__(288,1)
void sm90_forward_f3_kernel(const __grid_constant__ typename P::Binding b,
    ForwardCounter* events,unsigned long long epoch) {
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<backend::Sm90F3Header*>(shared);
  // Thread zero initializes both arenas' barriers, one fence publishes them,
  // then every consumer warp primes the empty barriers so the producer's first
  // wait on phase 0 passes (§5.3 item 1).
  if(threadIdx.x==0) {
    header->rings[0].Initialize();
    header->rings[1].Initialize();
    header->last_arriver=0;
    backend::Sm90F3Async::Base::InitFence();
  }
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0) {
    header->rings[0].Prime();
    header->rings[1].Prime();
  }
  __syncthreads();
  auto sequence=std::make_integer_sequence<unsigned,P::kStages>{};
  if(threadIdx.x>=256)P::ProducerLoop(b,shared,events,epoch,sequence);
  else P::ConsumerLoop(b,shared,events,epoch,sequence);
  __syncthreads();
  if(threadIdx.x==0) {header->rings[0].Invalidate();header->rings[1].Invalidate();}
}

}  // namespace tilemega::codegen::forward_f3
