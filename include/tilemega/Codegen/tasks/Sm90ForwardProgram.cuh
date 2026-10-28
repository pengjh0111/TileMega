// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90ForwardGemm.cuh>
#include <tilemega/Backend/Sm90ForwardDepthwise.cuh>
#include <tilemega/Backend/Sm90ForwardFused.cuh>
#include <tilemega/Backend/Sm90ForwardEncoderAttention.cuh>
#include <tilemega/Backend/Sm90ForwardSmall.cuh>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tuple>
#include <type_traits>
#include <utility>

// Native DNN forward (prompt §10 N-1). One pass over a compile-time stage
// list: no step loop and no KV state, so a stage's boundary is the V3 grid
// barrier alone. The ring state, the two arenas, the operand bank and the
// barrier protocol are the native dense program's (Sm90DenseProgram.cuh);
// only the per-stage bodies and the absence of a request loop differ.
namespace tilemega::codegen::native_forward {

using Element=cutlass::bfloat16_t;
inline constexpr unsigned kAbsent=~0u;

// Forward bindings carry no per-layer table: every stage names its own
// buffers, and a conv stage names its im2col or tiled tensor map.
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
    static_assert(Id<Maps);return &maps[Id];
  }
};

struct alignas(64) ForwardCounter {unsigned long long arrivals;};
// A side output the epilogue accumulates into, cleared before every launch.
struct ForwardZeroed {unsigned buffer;unsigned long long elements;};
// Thread zero of the publishing CTA fills the next stage's bank before the V3
// barrier; from the join to that stage's return the record is immutable. A
// stage reads exactly one member, so the five operand kinds share storage.
union alignas(16) ForwardOperandBank {
  backend::Sm90ForwardGemmOperands gemm;
  backend::Sm90ForwardDepthwiseOperands depthwise;
  backend::Sm90ForwardFusedOperands fused;
  backend::Sm90ForwardEncoderAttentionOperands attention;
  backend::Sm90ForwardSmallOperands small;
};
struct ForwardOperandHeader {
  static constexpr int kOffset=2*sizeof(backend::Sm90DenseRingState);
  static constexpr int kEnd=kOffset+2*int(sizeof(ForwardOperandBank));
  static_assert(kOffset==512 && kEnd<=1024,"the forward header shares the dense 1 KB prologue");
  static_assert(std::is_trivially_copyable_v<ForwardOperandBank>);
  __device__ static __forceinline__ ForwardOperandBank& Bank(char* arena,unsigned stage) {
    return reinterpret_cast<ForwardOperandBank*>(arena+kOffset)[stage%2];
  }
};

// A stage's buffer and map identifiers come from the plan; its scalar
// geometry and epsilon come from the same plan as a generated trait class,
// because C++17 has no floating-point template parameter.
template<class B,class Bind>
__device__ __forceinline__ backend::Sm90ForwardEpilogueOperands EpilogueOf(B const& b,
    float epsilon,int residual_stride,int norm_width=0) {
  backend::Sm90ForwardEpilogueOperands e{};
  // Per-channel vectors are FP32 in the plan's buffer table.
  e.vectors[0]=reinterpret_cast<float const*>(b.template Get<Bind::kVector0>());
  e.vectors[1]=reinterpret_cast<float const*>(b.template Get<Bind::kVector1>());
  e.vectors[2]=reinterpret_cast<float const*>(b.template Get<Bind::kVector2>());
  e.vectors[3]=reinterpret_cast<float const*>(b.template Get<Bind::kVector3>());
  e.residual=b.template Get<Bind::kResidual>();
  e.stats=reinterpret_cast<float const*>(b.template Get<Bind::kStats>());
  e.row_stats=reinterpret_cast<float*>(b.template Get<Bind::kRowStats>());
  e.row_partials=reinterpret_cast<float*>(b.template Get<Bind::kRowPartials>());
  e.row_counters=reinterpret_cast<unsigned long long*>(b.template Get<Bind::kRowCounters>());
  e.row_tiles=1;e.row_tile=0;
  e.channel_partial=reinterpret_cast<float*>(b.template Get<Bind::kChannelPartial>());
  e.epsilon=epsilon;e.residual_stride=residual_stride;e.norm_width=norm_width;
  e.row_base=0;
  return e;
}

template<class Body,class Bind,class Shape>
struct ForwardGemmStage {
  static constexpr int kSharedBytes=Body::kSharedBytes;
  static constexpr auto kLayout=Body::kLayout;
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,ForwardOperandBank& bank) {
    auto& p=bank.gemm;
    p.activation=b.template Map<Bind::kMap>();
    p.weight=b.template Get<Bind::kWeight>();
    p.output=b.template Get<Bind::kOutput>();
    p.partial=reinterpret_cast<float*>(b.template Get<Bind::kPartial>());
    p.counters=reinterpret_cast<std::uint64_t*>(b.template Get<Bind::kCounters>());
    p.epilogue=EpilogueOf<B,Bind>(b,Shape::kEpsilon,Shape::kResidualStride,Shape::kNormWidth);
    // The convolution geometry is a compile-time trait of the body itself.
    p.output_stride=Shape::kOutputStride;
  }
  template<class B>
  __device__ static __forceinline__ void Run(B const&,char* shared,int offset,
      backend::Sm90DenseRingState& ring,ForwardOperandBank const& bank) {
    Body::Run(bank.gemm,shared+offset,ring);
  }
};

template<class Body,class Bind,class Shape>
struct ForwardDepthwiseStage {
  static constexpr int kSharedBytes=Body::kSharedBytes;
  static constexpr auto kLayout=Body::kLayout;
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,ForwardOperandBank& bank) {
    auto& p=bank.depthwise;
    p.activation=b.template Map<Bind::kMap>();
    p.weight=b.template Get<Bind::kWeight>();
    p.output=b.template Get<Bind::kOutput>();
    p.epilogue=EpilogueOf<B,Bind>(b,Shape::kEpsilon,Shape::kResidualStride,Shape::kNormWidth);
    p.width=Shape::kWidth;p.height=Shape::kHeight;p.channels=Shape::kChannels;
    p.stride=Shape::kStrideH;
    p.output_width=Shape::kOutputWidth;p.output_height=Shape::kOutputHeight;
    p.padded_channels=Shape::kPaddedChannels;p.output_stride=Shape::kOutputStride;
  }
  template<class B>
  __device__ static __forceinline__ void Run(B const&,char* shared,int offset,
      backend::Sm90DenseRingState& ring,ForwardOperandBank const& bank) {
    Body::Run(bank.depthwise,shared+offset,ring);
  }
};

template<class Body,class Bind,class Shape>
struct ForwardFusedStage {
  static constexpr int kSharedBytes=Body::kSharedBytes;
  static constexpr auto kLayout=Body::kLayout;
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,ForwardOperandBank& bank) {
    auto& p=bank.fused;
    p.activation=b.template Map<Bind::kMap>();
    p.depthwise=b.template Get<Bind::kDepthwise>();
    p.depthwise_pack=Shape::kDepthwisePack;
    p.dw_vectors[0]=reinterpret_cast<float const*>(b.template Get<Bind::kDwVector0>());
    p.dw_vectors[1]=reinterpret_cast<float const*>(b.template Get<Bind::kDwVector1>());
    p.dw_vectors[2]=reinterpret_cast<float const*>(b.template Get<Bind::kDwVector2>());
    p.dw_vectors[3]=reinterpret_cast<float const*>(b.template Get<Bind::kDwVector3>());
    p.pointwise=b.template Get<Bind::kWeight>();
    p.expand=b.template Get<Bind::kExpand>();
    p.output=b.template Get<Bind::kOutput>();
    p.partial=reinterpret_cast<float*>(b.template Get<Bind::kPartial>());
    p.counters=reinterpret_cast<std::uint64_t*>(b.template Get<Bind::kCounters>());
    p.epilogue=EpilogueOf<B,Bind>(b,Shape::kEpsilon,Shape::kResidualStride,Shape::kNormWidth);
    p.width=Shape::kWidth;p.height=Shape::kHeight;p.channels=Shape::kChannels;
    p.out_channels=Shape::kOutChannels;
    p.output_width=Shape::kOutputWidth;p.output_height=Shape::kOutputHeight;
    p.stride=Shape::kStrideH;p.padded_channels=Shape::kPaddedChannels;
    p.output_stride=Shape::kOutputStride;
  }
  template<class B>
  __device__ static __forceinline__ void Run(B const&,char* shared,int offset,
      backend::Sm90DenseRingState& ring,ForwardOperandBank const& bank) {
    Body::Run(bank.fused,shared+offset,ring);
  }
};

template<class Body,class Bind,class Shape>
struct ForwardAttentionStage {
  static constexpr int kSharedBytes=Body::kSharedBytes;
  static constexpr auto kLayout=Body::kLayout;
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,ForwardOperandBank& bank) {
    auto& p=bank.attention;
    p.key=b.template Map<Bind::kMap>();
    p.value=b.template Map<Bind::kValueMap>();
    p.query=b.template Map<Bind::kQueryMap>();
    p.output=b.template Get<Bind::kOutput>();
    p.scale=Shape::kScale;
    p.heads=Shape::kHeads;p.sequence=Shape::kSequence;
  }
  template<class B>
  __device__ static __forceinline__ void Run(B const&,char* shared,int offset,
      backend::Sm90DenseRingState& ring,ForwardOperandBank const& bank) {
    Body::Run(bank.attention,shared+offset,ring);
  }
};

template<class Body,class Bind,class Shape>
struct ForwardSmallStage {
  static constexpr int kSharedBytes=Body::kSharedBytes;
  static constexpr auto kLayout=Body::kLayout;
  template<class B>
  __device__ static __forceinline__ void PublishOperands(B const& b,ForwardOperandBank& bank) {
    auto& p=bank.small;
    p.input=b.template Get<Bind::kInput>();
    p.weight=b.template Get<Bind::kWeight>();
    p.gamma=reinterpret_cast<float const*>(b.template Get<Bind::kVector1>());
    p.bias=reinterpret_cast<float const*>(b.template Get<Bind::kVector0>());
    p.table=b.template Get<Bind::kTable>();
    p.stats=reinterpret_cast<float*>(b.template Get<Bind::kRowStats>());
    p.output=b.template Get<Bind::kOutput>();
    p.accumulator=reinterpret_cast<float*>(b.template Get<Bind::kPartial>());
    p.counter=reinterpret_cast<std::uint64_t*>(b.template Get<Bind::kCounters>());
    p.index0=reinterpret_cast<long long const*>(b.template Get<Bind::kIndex0>());
    p.index1=reinterpret_cast<long long const*>(b.template Get<Bind::kIndex1>());
    p.width=Shape::kWidth;p.height=Shape::kHeight;
    p.channels=Shape::kChannels;p.out_channels=Shape::kOutChannels;
    p.epsilon=Shape::kEpsilon;
  }
  template<class B>
  __device__ static __forceinline__ void Run(B const&,char* shared,int offset,
      backend::Sm90DenseRingState& ring,ForwardOperandBank const& bank) {
    Body::Run(bank.small,shared+offset,ring);
  }
};

// One descriptor over a bound buffer, encoded on the host: either a tiled
// rank-2/3/4 view (`kind` 0) at an element offset into the buffer, or the TMA
// im2col view of a convolution's input (`kind` 1), which the driver builds
// from the plan's own ConvDesc and buffer layout.
struct ForwardMapSpec {
  // `swizzle` is 128 for a WGMMA operand and 0 for a tile a CUDA-core body
  // reads row-major.
  unsigned buffer,kind,rank,swizzle;
  unsigned long long offset;
  unsigned long long dims[4],strides[3];
  unsigned box[4];
  codegen::ConvDesc conv;
  codegen::DmBufferLayout layout;
};

// A stage's shared-memory placement: arena (i % 2) unless the stage's
// storage exceeds one arena, in which case it takes the whole budget
// (prompt §6 D5 item 3, reused unchanged).
template<class Stage,int Offset=0>
struct Placement:Stage {
  static_assert(Offset>=0 && Offset%1024==0);
  static constexpr int kOffset=Offset;
  static constexpr int kSharedBytes=Offset+Stage::kSharedBytes;
  static constexpr auto kLive=Stage::kLayout.Live(Offset);
  // The body addresses its arena from its own base; the 1 KB prologue of a
  // placed stage overlaps the program header, which the body never reads.
  template<class B>
  __device__ static __forceinline__ void Run(B const& b,char* shared,
      backend::Sm90DenseRingState& ring,ForwardOperandBank const& bank) {
    Stage::template Run<B>(b,shared,Offset,ring,bank);
  }
};

// Indexing the stage list by inheritance rather than by recursion: a model
// with hundreds of stages exceeds the standard library's tuple_element depth.
template<unsigned I,class T>
struct ForwardStageEntry {using Type=T;};
template<class Sequence,class... Stage>
struct ForwardStageMap;
template<unsigned... I,class... Stage>
struct ForwardStageMap<std::integer_sequence<unsigned,I...>,Stage...>
    :ForwardStageEntry<I,Stage>... {};
template<unsigned I,class T>
ForwardStageEntry<I,T> ForwardStageLookup(ForwardStageEntry<I,T> const&);

template<int Batch,int Residency,unsigned Buffers,unsigned Maps,int Arena,class... Stage>
struct Program {
  using Binding=Bindings<Buffers,Maps>;
  using Map=ForwardStageMap<std::make_integer_sequence<unsigned,sizeof...(Stage)>,Stage...>;
  template<unsigned I>
  using At=typename decltype(ForwardStageLookup<I>(Map{}))::Type;
  static constexpr unsigned kStages=sizeof...(Stage);
  static constexpr int kBatch=Batch,kResidency=Residency,kThreads=160;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps;
  static_assert(kStages>=1 && (Residency==1 || Residency==2));
  static_assert(Arena>0 && Arena%1024==0);

  template<unsigned... I>
  static constexpr bool Placed(std::integer_sequence<unsigned,I...>) {
    return ((At<I>::kOffset==int(I%2)*Arena ||
             (At<I>::kOffset==0 && At<I>::kSharedBytes>1024+Arena)) && ...);
  }
  static_assert(Placed(std::make_integer_sequence<unsigned,kStages>{}),
      "every forward stage sits in its parity's arena or spans the budget");
  template<unsigned... I>
  static constexpr int Bytes(std::integer_sequence<unsigned,I...>) {
    int value=1024;((value=value<At<I>::kSharedBytes?At<I>::kSharedBytes:value),...);return value;
  }
  static constexpr int kSharedBytes=Bytes(std::make_integer_sequence<unsigned,kStages>{});
  static_assert(sizeof(Binding)+256<32764);

  // One stage and its boundary: run, publish the next stage's bank, then the
  // V3 grid barrier on this stage's counter.
  template<unsigned I>
  __device__ static __forceinline__ void Boundary(Binding const& b,char* shared,
      ForwardCounter* events) {
    using Body=At<I>;
    auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
    auto const& bank=ForwardOperandHeader::Bank(shared,I%2);
    Body::template Run<Binding>(b,shared,rings[I%2],bank);
    if(threadIdx.x==0) {
      if constexpr(I+1<kStages)At<I+1>::template PublishOperands<Binding>(b,
          ForwardOperandHeader::Bank(shared,(I+1)%2));
      auto* counter=&events[I].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
      unsigned long long seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
      while(seen<gridDim.x);
    }
    __syncthreads();
  }
  template<unsigned... I>
  __device__ static __forceinline__ void Run(Binding const& b,char* shared,ForwardCounter* events,
      std::integer_sequence<unsigned,I...>) {
    (Boundary<I>(b,shared,events),...);
  }
};

template<class P>
__global__ __launch_bounds__(P::kThreads,P::kResidency)
void sm90_forward_native_kernel(const __grid_constant__ typename P::Binding b,
    ForwardCounter* events) {
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  rings[0].Initialize();rings[1].Initialize();
  // The first stage's operands are published before any barrier: its bank is
  // immutable for the whole pass and no predecessor can publish it.
  if(threadIdx.x==0)P::template At<0>::template PublishOperands<typename P::Binding>(b,
      ForwardOperandHeader::Bank(shared,0));
  __syncthreads();
  P::Run(b,shared,events,std::make_integer_sequence<unsigned,P::kStages>{});
  rings[0].Invalidate();rings[1].Invalidate();
}

}  // namespace tilemega::codegen::native_forward
