// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/DecodeRuntimeV2.cuh
//
// The serving C ABI (ServingAbi.h) for an MX-5 static decode plan (MX-5 §4.13). The generated source defines, before including this file,
//   namespace tm_mx_plan {
//     struct Plan { ... };                 // DecodeProgramV2.cuh's plan type
//     inline constexpr MxBufferRowV2 kBuffers[] = {...};   // external buffers
//   }
// Every buffer of the table is external (weights, KV, tokens, RoPE): Python
// binds them by name; the program's scratch is internal to the plan, made
// and cleared here. Only L1 exists: one launch per step or a device loop of
// steps (launch_steps); there is no L2 and no paging.
#pragma once

#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh>

#include <cuda_runtime.h>
#if TILEMEGA_MX_TP > 1
#include <mcr/mc_runtime_api.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#ifndef TILEMEGA_WATCHDOG
#define TILEMEGA_WATCHDOG 1
#endif

namespace tilemega::codegen::mx {

template <class P, class = void> struct MxTpOf : std::integral_constant<int, 1> {};
template <class P> struct MxTpOf<P, std::void_t<decltype(P::kTp)>> : std::integral_constant<int, P::kTp> {};

template <int TN, class Shape>
constexpr std::size_t MxGemmPartialFloatsV2(int grid, bool stream_k) {
  if (!stream_k) return 0;
  int const tiles = Shape::n / TN;
  // This is the slot stride used by the production GEMM body.
  return std::size_t(tiles) * backend::mx::MxMaxContributors(tiles, grid) * Shape::b * (TN + 1);
}
constexpr std::size_t MxMaxV2(std::size_t a, std::size_t b) { return a > b ? a : b; }

template <class P>
struct MxRuntimeV2 {
  static constexpr int kTp = MxTpOf<P>::value;
  static constexpr int kGrid = P::kGrid, kStages = 5 * P::kLayers + 1;
  static constexpr int kGenerationOffset = (P::kBodySmemBytes + P::kBatch * 4 + 15) / 16 * 16;
  static constexpr int kSmem = kGenerationOffset + 16;
  static_assert(P::kThreads == 512 || P::kThreads == 576);
  static_assert(P::kGrid == 104 && kSmem <= 65536);
  static_assert(P::kBodySmemBytes >= 32784 && P::kBodySmemBytes % 16 == 0);
  static_assert(kSmem == DecodeControlV2<P>::kSharedBytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::Qkv::kTN, P::Qkv::kWN).bytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::O::kTN, P::O::kWN).bytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::GateUp::kTN, P::GateUp::kWN).bytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::Down::kTN, P::Down::kWN).bytes);
  static_assert(P::kBodySmemBytes >= P::Head::SmemBytes(P::kBatch));
  static_assert(P::kBodySmemBytes >= P::Attn::kSharedBytes);
  static constexpr std::size_t kPartial = MxMaxV2(
      MxMaxV2(MxGemmPartialFloatsV2<P::Qkv::kTN, typename P::QkvShape>(kGrid, P::kQkvStreamK),
              MxGemmPartialFloatsV2<P::O::kTN, typename P::OShape>(kGrid, P::kOStreamK)),
      MxMaxV2(MxGemmPartialFloatsV2<P::GateUp::kTN, typename P::GateUpShape>(kGrid, P::kGateUpStreamK),
              MxGemmPartialFloatsV2<P::Down::kTN, typename P::DownShape>(kGrid, P::kDownStreamK)));
};

struct MxPlanStateV2 {
  void* binding = nullptr;  // P::Binding, host copy passed by value
  std::vector<void*> owned;
  int* pasts = nullptr;
  std::uint32_t steps = 0;
  unsigned long long* step_ns = nullptr;
  std::uint64_t next_iteration = 0;
  WatchdogRecord* watchdog = nullptr;
  MxTraceV2 trace{nullptr, 0};
  int device = 0;
  bool allocation_failed = false;
#if TILEMEGA_MX_TP > 1
  void* tp_ar = nullptr;
  void* tp_x = nullptr;
  bool tp_bound = false, poisoned = false;
#endif
};

}  // namespace tilemega::codegen::mx

namespace tm_mx_runtime_v2 {
using namespace tilemega::codegen::mx;
using P = tm_mx_plan::Plan;
using R = MxRuntimeV2<P>;
inline constexpr std::uint32_t kBufferCount = sizeof(tm_mx_plan::kBuffers) / sizeof(tm_mx_plan::kBuffers[0]);

#if TILEMEGA_MX_TP > 1
inline void* MxTpAlloc(std::size_t bytes) {
  void* pointer = nullptr;
  if (mcExtMallocWithFlags(&pointer, bytes, mcDeviceMallocFinegrained) != mcSuccess) return nullptr;
  if (cudaMemset(pointer, 0, bytes) != cudaSuccess) {
    cudaFree(pointer);
    return nullptr;
  }
  return pointer;
}
inline void MxTpFree(void* pointer) { cudaFree(pointer); }
static_assert(R::kTp == TILEMEGA_MX_TP);
#endif

inline int LaunchError(MxPlanStateV2* s, int code) {
#if TILEMEGA_MX_TP > 1
  if (s && code) s->poisoned = true;
#endif
  return code;
}

inline void* Alloc(MxPlanStateV2* s, std::size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes ? bytes : 16) != cudaSuccess) {
    s->allocation_failed = true;
    return nullptr;
  }
  if (cudaMemset(p, 0, bytes ? bytes : 16) != cudaSuccess) {
    cudaFree(p);
    s->allocation_failed = true;
    return nullptr;
  }
  s->owned.push_back(p);
  return p;
}
inline void Free(MxPlanStateV2* s) {
  if (!s) return;
#if TILEMEGA_MX_TP > 1
  cudaSetDevice(s->device);
  if (s->tp_ar) MxTpFree(s->tp_ar);
  if (s->tp_x) MxTpFree(s->tp_x);
#endif
  for (void* p : s->owned) cudaFree(p);
  if (s->pasts) cudaFree(s->pasts);
  if (s->step_ns) cudaFree(s->step_ns);
  if (s->watchdog) cudaFreeHost(s->watchdog);
  delete static_cast<P::Binding*>(s->binding);
  delete s;
}
}  // namespace tm_mx_runtime_v2

extern "C" int tm_plan_query(tm_plan_info* out) {
  if (!out) return -1;
  *out = {};
  out->abi_version = TM_SERVING_ABI_VERSION;
  out->phase = TM_SERVING_DECODE;
  out->batch_lo = out->batch_hi = tm_mx_runtime_v2::P::kBatch;
  out->seq = 1;
  out->past_lo = tm_mx_runtime_v2::P::kPastLo;
  out->past_hi = tm_mx_runtime_v2::P::kPastHi;
  out->capacity = tm_mx_runtime_v2::P::kCapacity;
  out->grid = tm_mx_runtime_v2::P::kGrid;
  out->residency = tm_mx_runtime_v2::P::kGrid / 104;
  out->modes = TM_SERVING_L1;
  out->buffer_count = tm_mx_runtime_v2::kBufferCount;
  return 0;
}

extern "C" int tm_plan_buffer(std::uint32_t index, tm_buffer_info* out) {
  if (!out || index >= tm_mx_runtime_v2::kBufferCount) return -1;
  auto const& row = tm_mx_plan::kBuffers[index];
  *out = {};
  out->name = row.name;
  out->role = TM_BUFFER_EXTERNAL;
  out->dtype = row.dtype;
  out->elements_constant = row.constant;
  out->elements_per_batch = row.per_batch;
  out->pack_json = row.pack_json;
  return 0;
}

extern "C" std::uint64_t tm_plan_shared_bytes() { return tm_mx_runtime_v2::R::kSmem; }

extern "C" void* tm_plan_create(int batch, void* const* external, int device) {
  using namespace tm_mx_runtime_v2;
  if (batch != P::kBatch || !external) return nullptr;
  if (cudaSetDevice(device) != cudaSuccess) return nullptr;
  auto* s = new MxPlanStateV2;
  s->device = device;
  auto* b = new P::Binding{};
  s->binding = b;
  for (std::uint32_t i = 0; i < kBufferCount; ++i) {
    auto const& row = tm_mx_plan::kBuffers[i];
    void* p = external[i];
    if (!p || (reinterpret_cast<std::uintptr_t>(p) & 15)) {
      std::fprintf(stderr, "tm_plan_create: buffer %s is null or not 16 B aligned\n", row.name);
      Free(s);
      return nullptr;
    }
    auto* e = static_cast<tilemega::Bf16*>(p);
    auto& layer = b->layer[row.layer];
    switch (row.slot) {
      case kSlotTokensV2: b->tokens = static_cast<int*>(p); break;
      case kSlotRopeCosV2: b->rope_cos = e; break;
      case kSlotRopeSinV2: b->rope_sin = e; break;
      case kSlotEmbedV2: b->embed = e; break;
      case kSlotHeadV2: b->w_head = e; break;
      case kSlotQkvV2: layer.w_qkv = e; break;
      case kSlotOV2: layer.w_o = e; break;
      case kSlotGateUpV2: layer.w_gu = e; break;
      case kSlotDownV2: layer.w_down = e; break;
      case kSlotQNormV2: layer.q_norm = e; break;
      case kSlotKNormV2: layer.k_norm = e; break;
      case kSlotKCacheV2: layer.k_cache = e; break;
      case kSlotVtCacheV2: layer.vt_cache = e; break;
      default: Free(s); return nullptr;
    }
  }
  auto& sc = b->s;
  std::size_t const B = P::kBatch;
  sc.x = static_cast<tilemega::Bf16*>(Alloc(s, B * P::kHidden * 2));
  sc.qkv = static_cast<tilemega::Bf16*>(Alloc(s, B * P::QkvShape::n * 2));
  sc.ctx = static_cast<tilemega::Bf16*>(Alloc(s, B * P::OShape::k * 2));
  sc.h = static_cast<tilemega::Bf16*>(Alloc(s, B * P::DownShape::k * 2));
  sc.gemm_partial = static_cast<float*>(Alloc(s, R::kPartial * 4));
  sc.attn_partial = static_cast<float*>(
      Alloc(s, B * P::AttnShape::hkv * P::Attn::kSlots * P::kQ * (P::kHeadDim + 2) * 4));
  sc.head_partial = static_cast<tilemega::backend::mx::MxHeadPartialV2*>(
      Alloc(s, B * P::kGrid * sizeof(tilemega::backend::mx::MxHeadPartialV2)));
  sc.t_qkv = static_cast<unsigned*>(Alloc(s, (P::QkvShape::n / P::Qkv::kTN) * 4));
  sc.t_o = static_cast<unsigned*>(Alloc(s, (P::OShape::n / P::O::kTN) * 4));
  sc.t_gu = static_cast<unsigned*>(Alloc(s, (P::GateUpShape::n / P::GateUp::kTN) * 4));
  sc.t_down = static_cast<unsigned*>(Alloc(s, (P::DownShape::n / P::Down::kTN) * 4));
  sc.t_attn = static_cast<unsigned*>(Alloc(s, B * P::AttnShape::hkv * 4));
  sc.t_head = static_cast<unsigned*>(Alloc(s, 64));
  sc.barrier = static_cast<unsigned long long*>(Alloc(s, std::size_t(R::kStages) * 64));
#if TILEMEGA_MX_TP > 1
  s->tp_ar = MxTpAlloc(2 * std::size_t(P::kLayers) * R::kTp * B * 4 * P::kHidden);
  s->tp_x = MxTpAlloc(sizeof(tilemega::backend::mx::MxTpExchangeV2));
  if (!s->tp_ar || !s->tp_x) s->allocation_failed = true;
#endif
  if (s->allocation_failed) {
    Free(s);
    return nullptr;
  }
#if TILEMEGA_WATCHDOG
  {
    unsigned long long ms = 10000;
    if (auto const env = std::getenv("TILEMEGA_WATCHDOG_MS")) ms = std::strtoull(env, nullptr, 10);
    if (ms) {
      if (ms > UINT64_MAX / 1000000ull ||
          cudaHostAlloc(&s->watchdog, sizeof(tilemega::codegen::WatchdogRecord), cudaHostAllocMapped) != cudaSuccess) {
        Free(s);
        return nullptr;
      }
      *s->watchdog = {};
      if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&sc.watchdog), s->watchdog, 0) != cudaSuccess) {
        Free(s);
        return nullptr;
      }
      sc.watchdog_ns = ms * 1000000ull;
    }
  }
#endif
#if TILEMEGA_MX_DIAG
  if (auto const env = std::getenv("TILEMEGA_MX_DIAG_SKIP")) sc.diag_skip = unsigned(std::strtoul(env, nullptr, 0));
#endif
#if TILEMEGA_TRACE_STAGE
  {
    unsigned slots = 16;
    if (auto const env = std::getenv("TILEMEGA_TRACE_LAUNCHES")) slots = unsigned(std::strtoul(env, nullptr, 10));
    s->trace.slots = slots;
    s->trace.data = static_cast<unsigned long long*>(
        Alloc(s, std::size_t(slots) * R::kStages * P::kGrid * 3 * 8));
  }
#endif
  if (s->allocation_failed) {
    Free(s);
    return nullptr;
  }
  auto kernel = mx_decode_kernel_v2<P>;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, R::kSmem) != cudaSuccess) {
    Free(s);
    return nullptr;
  }
  int blocks = 0;
  auto const occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, P::kThreads, R::kSmem);
  if (occupancy != cudaSuccess || blocks != 1) {
    std::fprintf(stderr, "tm_plan_create: occupancy %d, the program needs %d CTAs per AP\n", blocks, P::kGrid / 104);
    Free(s);
    return nullptr;
  }
  std::printf("E2E_LOOP_MODES 1\nE2E_PDL 0\n");
  return s;
}

extern "C" int tm_plan_set_steps(void* opaque, std::int32_t const* past, std::uint32_t count) {
  using namespace tm_mx_runtime_v2;
  auto* s = static_cast<MxPlanStateV2*>(opaque);
  if (!s || !past || !count || s->pasts) return -1;
  for (std::uint32_t i = 0; i < count; ++i)
    if (past[i] < P::kPastLo || past[i] > P::kPastHi) return -1;
  if (cudaMalloc(&s->pasts, std::size_t(count) * 4) != cudaSuccess) return -1;
  if (cudaMemcpy(s->pasts, past, std::size_t(count) * 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
#if TILEMEGA_MX_DIAG
  if (cudaMalloc(&s->step_ns, (std::size_t(count) + 1) * 8) != cudaSuccess) return -1;
  if (cudaMemset(s->step_ns, 0, (std::size_t(count) + 1) * 8) != cudaSuccess) return -1;
#endif
  s->steps = count;
  return 0;
}

namespace tm_mx_runtime_v2 {
inline int Launch(MxPlanStateV2* s, std::uint32_t first, std::uint32_t steps, std::uint64_t iteration,
                  unsigned long long* step_ns, void* stream) {
#if TILEMEGA_MX_TP > 1
  if (cudaSetDevice(s->device) != cudaSuccess) return LaunchError(s, 1);
  if (s->poisoned) return -5;
  if (!s->tp_bound) return LaunchError(s, -4);
#endif
  if (steps > UINT32_MAX / R::kStages || iteration != s->next_iteration || iteration > UINT64_MAX / P::kGrid ||
      steps > UINT64_MAX / P::kGrid - iteration) return LaunchError(s, -2);
  mx_decode_kernel_v2<P><<<P::kGrid, P::kThreads, R::kSmem, static_cast<cudaStream_t>(stream)>>>(
      *static_cast<P::Binding const*>(s->binding), StageLaunchV2{s->pasts, first, steps, iteration, step_ns, s->trace});
  if (cudaGetLastError() != cudaSuccess) return LaunchError(s, 1);
  s->next_iteration += steps;
  return 0;
}
}  // namespace tm_mx_runtime_v2

extern "C" int tm_plan_launch(void* opaque, std::uint32_t step, std::uint32_t mode, std::uint64_t iteration,
                              void* stream) {
  using namespace tm_mx_runtime_v2;
  auto* s = static_cast<MxPlanStateV2*>(opaque);
  if (!s || mode != TM_SERVING_L1 || step >= s->steps) return LaunchError(s, -1);
  return Launch(s, step, 1, iteration, nullptr, stream);
}

extern "C" int tm_plan_launch_steps(void* opaque, std::uint32_t first_step, std::uint32_t steps, std::uint32_t mode,
                                    std::uint64_t base_iteration, void* stream) {
  using namespace tm_mx_runtime_v2;
  auto* s = static_cast<MxPlanStateV2*>(opaque);
  if (!s) return -1;
  if (mode != TM_SERVING_L1) return LaunchError(s, -3);
  if (!steps || first_step >= s->steps || steps > s->steps - first_step) return LaunchError(s, -1);
  return Launch(s, first_step, steps, base_iteration, s->step_ns ? s->step_ns + first_step : nullptr, stream);
}

#if TILEMEGA_MX_TP > 1
extern "C" unsigned tm_plan_tp_world() { return tm_mx_runtime_v2::R::kTp; }

extern "C" int tm_plan_tp_buffers(void* opaque, void** ar, void** x) {
  auto* s = static_cast<tilemega::codegen::mx::MxPlanStateV2*>(opaque);
  if (!s || !ar || !x) return -1;
  *ar = s->tp_ar;
  *x = s->tp_x;
  return 0;
}

extern "C" int tm_plan_bind_tp(void* opaque, int rank, void* const* ar, void* const* x) {
  using namespace tm_mx_runtime_v2;
  auto* s = static_cast<MxPlanStateV2*>(opaque);
  if (!s || !ar || !x || rank < 0 || rank >= R::kTp) return -1;
  if (ar[rank] != s->tp_ar || x[rank] != s->tp_x || cudaSetDevice(s->device) != cudaSuccess) return -1;
  for (int q = 0; q < R::kTp; ++q) {
    if (!ar[q] || !x[q] || (reinterpret_cast<std::uintptr_t>(ar[q]) & 63) ||
        (reinterpret_cast<std::uintptr_t>(x[q]) & 63)) return -1;
    if (q == rank) continue;
    cudaPointerAttributes attributes{}, exchange{};
    if (cudaPointerGetAttributes(&attributes, ar[q]) != cudaSuccess ||
        cudaPointerGetAttributes(&exchange, x[q]) != cudaSuccess || attributes.device != exchange.device) return -1;
    int access = 0;
    if (cudaDeviceCanAccessPeer(&access, s->device, attributes.device) != cudaSuccess || !access) return -1;
    auto const status = cudaDeviceEnablePeerAccess(attributes.device, 0);
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) return -1;
    cudaGetLastError();
  }
  auto* b = static_cast<P::Binding*>(s->binding);
  for (int q = 0; q < R::kTp; ++q) {
    b->ar[q] = static_cast<std::uint64_t*>(ar[q]);
    b->x[q] = static_cast<tilemega::backend::mx::MxTpExchangeV2*>(x[q]);
  }
  b->rank = rank;
  s->tp_bound = true;
  return 0;
}

extern "C" int tm_plan_set_prompt_len(void* opaque, int prompt_len) {
  using namespace tm_mx_runtime_v2;
  auto* s = static_cast<MxPlanStateV2*>(opaque);
  if (!s || prompt_len < 0 || prompt_len > P::kCapacity) return -1;
  static_cast<P::Binding*>(s->binding)->prompt_len = prompt_len;
  return 0;
}
#endif

extern "C" unsigned tm_plan_loop_modes(void* opaque) { return opaque ? TM_SERVING_L1 : 0; }
extern "C" unsigned tm_plan_pdl_enabled(void*) { return 0; }
extern "C" std::uint64_t tm_plan_execution_smem(void* opaque, unsigned mode) {
  return opaque && mode == TM_SERVING_L1 ? tm_mx_runtime_v2::R::kSmem : 0;
}

extern "C" int tm_plan_read_step_ns(void* opaque, std::uint32_t first, std::uint32_t count, std::uint64_t* host) {
  auto* s = static_cast<tilemega::codegen::mx::MxPlanStateV2*>(opaque);
  if (!s || !s->step_ns || !host || first > s->steps || count > s->steps - first + 1) return -1;
  return int(cudaMemcpy(host, s->step_ns + first, std::size_t(count) * 8, cudaMemcpyDeviceToHost));
}

extern "C" int tm_plan_read_buffer(void* opaque, std::uint32_t index, void* host, std::uint64_t bytes) {
  // Every buffer is external and owned by the caller.
  (void)opaque, (void)index, (void)host, (void)bytes;
  return -2;
}

extern "C" int tm_plan_watchdog(void* opaque, std::uint64_t out[12]) {
#if !TILEMEGA_WATCHDOG
  (void)opaque, (void)out;
  return 0;
#else
  auto* s = static_cast<tilemega::codegen::mx::MxPlanStateV2*>(opaque);
  if (!s || !out || !s->watchdog || !s->watchdog->fired) return 0;
  auto const* words = reinterpret_cast<unsigned long long const*>(s->watchdog);
  for (int i = 0; i < 12; ++i) out[i] = words[i];
  return 1;
#endif
}

#if TILEMEGA_TRACE_STAGE
/// <dir>/mx_stage_trace.csv: slot,stage,cta,start_ns,work_end_ns,release_ns.
extern "C" int tm_plan_dump_serving_trace(void* opaque, char const* directory) {
  using namespace tm_mx_runtime_v2;
  auto* s = static_cast<MxPlanStateV2*>(opaque);
  if (!s || !directory || !s->trace.data || cudaDeviceSynchronize() != cudaSuccess) return -1;
  std::size_t const n = std::size_t(s->trace.slots) * R::kStages * P::kGrid * 3;
  std::vector<unsigned long long> host(n);
  if (cudaMemcpy(host.data(), s->trace.data, n * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  std::string const path = std::string(directory) + "/mx_stage_trace.csv";
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return -1;
  std::fprintf(f, "slot,stage,cta,start_ns,work_end_ns,release_ns\n");
  for (unsigned slot = 0; slot < s->trace.slots; ++slot)
    for (int stage = 0; stage < R::kStages; ++stage)
      for (int cta = 0; cta < P::kGrid; ++cta) {
        auto const* v = host.data() + ((std::size_t(slot) * R::kStages + stage) * P::kGrid + cta) * 3;
        if (v[0] || v[1] || v[2]) std::fprintf(f, "%u,%d,%d,%llu,%llu,%llu\n", slot, stage, cta, v[0], v[1], v[2]);
      }
  std::fclose(f);
  return 0;
}
#endif

extern "C" void tm_plan_destroy(void* opaque) {
  tm_mx_runtime_v2::Free(static_cast<tilemega::codegen::mx::MxPlanStateV2*>(opaque));
}
