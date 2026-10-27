// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tuple>
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

struct alignas(64) ForwardCounter {unsigned long long arrivals;};

// A stage's shared-memory placement: arena (i % 2) unless the stage's
// storage exceeds one arena, in which case it takes the whole budget
// (prompt §6 D5 item 3, reused unchanged).
template<class Stage,int Offset=0>
struct Placement:Stage {
  static_assert(Offset>=0 && Offset%1024==0);
  static constexpr int kOffset=Offset;
  static constexpr int kSharedBytes=Offset+Stage::kSharedBytes;
  static constexpr auto kLive=Stage::kLayout.Live(Offset);
};

template<int Residency,unsigned Buffers,unsigned Maps,int Arena,class... Stage>
struct Program {
  using Binding=Bindings<Buffers,Maps>;
  using Stages=std::tuple<Stage...>;
  template<unsigned I> using At=std::tuple_element_t<I,Stages>;
  static constexpr unsigned kStages=sizeof...(Stage);
  static constexpr int kResidency=Residency,kThreads=160;
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
    auto& bank=backend::Sm90DenseOperandHeader::Bank(shared,I%2);
    Body::Run(b,shared,rings[I%2],bank);
    if(threadIdx.x==0) {
      if constexpr(I+1<kStages)At<I+1>::PublishOperands(b,
          backend::Sm90DenseOperandHeader::Bank(shared,(I+1)%2));
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
  if(threadIdx.x==0)P::template At<0>::PublishOperands(b,
      backend::Sm90DenseOperandHeader::Bank(shared,0));
  __syncthreads();
  P::Run(b,shared,events,std::make_integer_sequence<unsigned,P::kStages>{});
  rings[0].Invalidate();rings[1].Invalidate();
}

}  // namespace tilemega::codegen::native_forward
