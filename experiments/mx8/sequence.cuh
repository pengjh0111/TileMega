// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <unordered_map>

namespace mx8_sequence {
using P = tm_mx_plan::Plan;
#if MX8_MOE
namespace Base = tm_mx_moe_runtime_v3;
using Ops = tilemega::codegen::mx::MoeOperandsV3<P>;
using Control = tilemega::codegen::mx::MoeControlV3<P>;
#else
namespace Base = tm_mx_runtime_v2;
using Ops = tilemega::codegen::mx::DenseOperandsV2<P>;
using Control = tilemega::codegen::mx::DecodeControlV2<P>;
#endif
using Sync = typename Control::Sync;
using R = Base::R;
struct Cursor { unsigned index; unsigned long long iteration; };
struct EmptyPrefetch { static constexpr int kLoads = 0; };
struct State {
  Cursor* cursor = nullptr;
  unsigned* routes = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
};
inline std::unordered_map<void*, State> states;

__global__ void Begin(Cursor* cursor, unsigned first, unsigned long long iteration) {
  if (threadIdx.x == 0) *cursor = {first, iteration};
}
__global__ void Advance(Cursor* cursor) {
  if (threadIdx.x == 0) { ++cursor->index; ++cursor->iteration; }
}

template <class Body, class Shape, class Operand>
__device__ __forceinline__ void Run(Operand const& operand, char* shared) {
  auto prefetched = Body::Prefetch(operand, Shape{});
  Body::Run(prefetched, operand, Shape{}, shared, [] { return EmptyPrefetch{}; });
}

// A stage boundary discards LDS and the incoming prefetch ring. MoE routing
// is materialized per CTA so both execution variants consume identical bits.
template <int Stage, bool Last = false>
__global__ void __launch_bounds__(P::kThreads) StageKernel(
    P::Binding binding, int layer, int const* pasts, Cursor const* cursor, unsigned* routes) {
  extern __shared__ __align__(16) char shared[];
  using GlobalInt = __attribute__((address_space(1))) int const;
  int const past = __builtin_mxc_readfirstlane(*((GlobalInt*)pasts + cursor->index));
  auto* tokens = reinterpret_cast<int*>(shared + Control::kTokenOffset);
  auto* shared_tokens = (tilemega::backend::mx::MxSharedTokensV2*)tokens;
#if MX8_MOE
  if (Sync::IsRole()) Control::BroadcastTokens(binding, past, tokens);
#else
  if (Sync::IsRole()) Control::Broadcast(binding, past, tokens);
#endif
#if MX8_MOE
  using Table = tilemega::backend::mx::MxRouteTableV3;
  auto* table = (tilemega::codegen::mx::MxMoeSharedRouteTableV3*)(shared + Control::kRouteOffset);
  auto* combine = (tilemega::backend::mx::MxSharedCombineStateV3*)(shared + Control::kCombineOffset);
  constexpr unsigned words = sizeof(Table) / 4;
  using GlobalWord = __attribute__((address_space(1))) unsigned;
  using SharedWord = __attribute__((address_space(3))) unsigned;
  if constexpr (Stage == 5) {
    for (unsigned i = threadIdx.x; i < words; i += P::kThreads)
      ((SharedWord*)table)[i] = ((GlobalWord*)routes)[blockIdx.x * words + i];
    tilemega::backend::mx::MxMoeCombineTileV3<P::Down::kTN, false, P::Down::kCombineMode>::Initialize(combine, P::kHidden);
  }
#endif
  Sync::CtaSync();
  if constexpr (Stage == 0) Run<typename P::Qkv, typename P::QkvShape>(Ops::Qkv(binding, layer, past, shared_tokens), shared);
  if constexpr (Stage == 1) Run<typename P::Attn, typename P::AttnShape>(Ops::Attn(binding, layer, past), shared);
  if constexpr (Stage == 2) Run<typename P::O, typename P::OShape>(Ops::O(binding, layer, shared_tokens), shared);
#if MX8_MOE
  if constexpr (Stage == 3) Run<typename P::Router, typename P::RouterShape>(Ops::Router(binding, layer, past), shared);
  if constexpr (Stage == 4) {
    Run<typename P::GateUp, typename P::GateUpShape>(Ops::template Grouped<true>(binding, layer, past, table), shared);
    for (unsigned i = threadIdx.x; i < words; i += P::kThreads)
      ((GlobalWord*)routes)[blockIdx.x * words + i] = ((SharedWord*)table)[i];
  }
  if constexpr (Stage == 5) {
    if constexpr (Last) Run<typename P::DownLast, typename P::DownShape>(Ops::template Grouped<false>(binding, layer, past, table), shared);
    else Run<typename P::Down, typename P::DownShape>(Ops::template Grouped<false>(binding, layer, past, table), shared);
  }
  if constexpr (Stage == 6) Run<typename P::Head, typename P::HeadShape>(Ops::Head(binding, past), shared);
#else
  if constexpr (Stage == 3) Run<typename P::GateUp, typename P::GateUpShape>(Ops::GateUp(binding, layer), shared);
  if constexpr (Stage == 4) {
    if constexpr (Last) Run<typename P::DownLast, typename P::DownShape>(Ops::Down(binding, layer), shared);
    else Run<typename P::Down, typename P::DownShape>(Ops::Down(binding, layer), shared);
  }
  if constexpr (Stage == 5) Run<typename P::Head, typename P::HeadShape>(Ops::Head(binding, past), shared);
#endif
}

template <int Stage, bool Last = false> int Configure() {
  auto kernel = StageKernel<Stage, Last>;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, R::kSmem) != cudaSuccess) return -10;
  int resident = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, P::kThreads, R::kSmem) != cudaSuccess || resident < 1) return -11;
  return 0;
}
template <class S, int Stage, bool Last = false>
void Enqueue(S* s, State const& state, int layer, cudaStream_t stream) {
  StageKernel<Stage, Last><<<P::kGrid, P::kThreads, R::kSmem, stream>>>(
      *static_cast<P::Binding const*>(s->binding), layer, s->pasts, state.cursor, state.routes);
}
template <class S> void Step(S* s, State const& state, cudaStream_t stream) {
  for (int layer = 0; layer < P::kLayers; ++layer) {
    Enqueue<S, 0>(s, state, layer, stream);
    Enqueue<S, 1>(s, state, layer, stream);
    Enqueue<S, 2>(s, state, layer, stream);
    Enqueue<S, 3>(s, state, layer, stream);
#if MX8_MOE
    Enqueue<S, 4>(s, state, layer, stream);
    if (layer + 1 == P::kLayers) Enqueue<S, 5, true>(s, state, layer, stream);
    else Enqueue<S, 5>(s, state, layer, stream);
#else
    if (layer + 1 == P::kLayers) Enqueue<S, 4, true>(s, state, layer, stream);
    else Enqueue<S, 4>(s, state, layer, stream);
#endif
  }
#if MX8_MOE
  Enqueue<S, 6>(s, state, P::kLayers - 1, stream);
#else
  Enqueue<S, 5>(s, state, P::kLayers - 1, stream);
#endif
  Advance<<<1, 1, 0, stream>>>(state.cursor);
}
}  // namespace mx8_sequence

template <class S> void Mx8Release(S* s) {
  using namespace mx8_sequence;
  auto i = states.find(s);
  if (i == states.end()) return;
  auto const& state = i->second;
  if (state.executable) cudaGraphExecDestroy(state.executable);
  if (state.graph) cudaGraphDestroy(state.graph);
  if (state.routes) cudaFree(state.routes);
  if (state.cursor) cudaFree(state.cursor);
  states.erase(i);
}

template <class S> int Mx8Prepare(S* s) {
  using namespace mx8_sequence;
  if (states.count(s)) return -12;
  auto& state = states[s];
  if (cudaMalloc(&state.cursor, sizeof(Cursor)) != cudaSuccess) return -13;
#if MX8_MOE
  static_assert(!P::kPrompt);
  if (cudaMalloc(&state.routes, P::kGrid * sizeof(tilemega::backend::mx::MxRouteTableV3)) != cudaSuccess) return -14;
#endif
  int status = Configure<0>() | Configure<1>() | Configure<2>() | Configure<3>() | Configure<4>() | Configure<5>();
#if MX8_MOE
  status |= Configure<5, true>() | Configure<6>();
#else
  status |= Configure<4, true>();
#endif
  if (status) return status;
#if !MX8_HOST_LAUNCH
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return -15;
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { cudaStreamDestroy(stream); return -16; }
  Step(s, state, stream);
  auto end = cudaStreamEndCapture(stream, &state.graph);
  cudaStreamDestroy(stream);
  if (end != cudaSuccess) return -17;
  if (cudaGraphInstantiate(&state.executable, state.graph, nullptr, nullptr, 0) != cudaSuccess) return -18;
#endif
  return 0;
}

template <class S> int Mx8Launch(S* s, unsigned first, unsigned count, unsigned long long iteration, void* raw_stream) {
  using namespace mx8_sequence;
  if (iteration != s->next_iteration || !states.count(s)) return -20;
  if (iteration > UINT64_MAX - count) return -21;
  auto stream = static_cast<cudaStream_t>(raw_stream);
  auto const& state = states.at(s);
  Begin<<<1, 1, 0, stream>>>(state.cursor, first, iteration);
  for (unsigned step = 0; step < count; ++step) {
#if MX8_HOST_LAUNCH
    Step(s, state, stream);
#else
    if (cudaGraphLaunch(state.executable, stream) != cudaSuccess) return 1;
#endif
  }
  if (cudaGetLastError() != cudaSuccess) return 1;
  s->next_iteration += count;
  return 0;
}
