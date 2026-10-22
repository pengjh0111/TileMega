// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/DecodeRuntime.cuh
//
// The serving C ABI (ServingAbi.h) for an MX-4 static decode plan (MX4
// prompt §3.8). The generated source defines, before including this file,
//   namespace tm_mx_plan {
//     struct Plan { ... };                 // DecodeProgram.cuh's plan type
//     inline constexpr MxBufferRow kBuffers[] = {...};   // external buffers
//   }
// Every buffer of the table is external (weights, KV, tokens, RoPE): Python
// binds them by name; the program's scratch is internal to the plan, made
// and cleared here. Only L1 exists: one launch per step or a device loop of
// steps (launch_steps); there is no L2 and no paging.
#pragma once

#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/tasks/mx/DecodeProgram.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TILEMEGA_WATCHDOG
#define TILEMEGA_WATCHDOG 1
#endif

namespace tilemega::codegen::mx {

template <class Body, class Shape>
constexpr std::size_t MxGemmPartialFloats(int grid) {
  if constexpr (Body::kPart == backend::mx::MxPart::kTileRange) {
    return 0;
  } else {
    int const tiles = Shape::n / Body::kTN;
    return std::size_t(tiles) * backend::mx::MxMaxContributors(tiles, grid) * Shape::b * (Body::kTN + 1);
  }
}
template <class Body, class Shape>
constexpr int MxGemmSmemBytes() {
  return backend::mx::MxGemmSmem(Shape::b, Body::kTN).bytes;
}
constexpr std::size_t MxMax(std::size_t a, std::size_t b) { return a > b ? a : b; }

template <class P>
struct MxRuntime {
  static constexpr int kGrid = P::kGrid, kStages = 5 * P::kLayers + 2;
  static constexpr int kSmem = int(MxMax(
      MxMax(MxMax(MxGemmSmemBytes<typename P::Qkv, typename P::QkvShape>(),
                  MxGemmSmemBytes<typename P::O, typename P::OShape>()),
            MxMax(MxGemmSmemBytes<typename P::GateUp, typename P::GateUpShape>(),
                  MxGemmSmemBytes<typename P::Down, typename P::DownShape>())),
      MxMax(MxMax(MxGemmSmemBytes<typename P::Head, typename P::HeadShape>(), P::Attn::kSharedBytes),
            65536 / (P::kGrid / 104 + 1) + 16)));
  static constexpr std::size_t kPartial = MxMax(
      MxMax(MxGemmPartialFloats<typename P::Qkv, typename P::QkvShape>(kGrid),
            MxGemmPartialFloats<typename P::O, typename P::OShape>(kGrid)),
      MxMax(MxGemmPartialFloats<typename P::GateUp, typename P::GateUpShape>(kGrid),
            MxGemmPartialFloats<typename P::Down, typename P::DownShape>(kGrid)));
};

struct MxPlanState {
  void* binding = nullptr;  // P::Binding, host copy passed by value
  std::vector<void*> owned;
  int* pasts = nullptr;
  std::uint32_t steps = 0;
  unsigned long long* step_ns = nullptr;
  std::uint64_t next_iteration = 0;
  WatchdogRecord* watchdog = nullptr;
  MxTrace trace{nullptr, 0};
  int device = 0;
};

}  // namespace tilemega::codegen::mx

namespace tm_mx_runtime {
using namespace tilemega::codegen::mx;
using P = tm_mx_plan::Plan;
using R = MxRuntime<P>;
inline constexpr std::uint32_t kBufferCount = sizeof(tm_mx_plan::kBuffers) / sizeof(tm_mx_plan::kBuffers[0]);

inline void* Alloc(MxPlanState* s, std::size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes ? bytes : 16) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, bytes ? bytes : 16);
  s->owned.push_back(p);
  return p;
}
inline void Free(MxPlanState* s) {
  if (!s) return;
  for (void* p : s->owned) cudaFree(p);
  if (s->pasts) cudaFree(s->pasts);
  if (s->step_ns) cudaFree(s->step_ns);
  if (s->watchdog) cudaFreeHost(s->watchdog);
  delete static_cast<P::Binding*>(s->binding);
  delete s;
}
}  // namespace tm_mx_runtime

extern "C" int tm_plan_query(tm_plan_info* out) {
  if (!out) return -1;
  *out = {};
  out->abi_version = TM_SERVING_ABI_VERSION;
  out->phase = TM_SERVING_DECODE;
  out->batch_lo = out->batch_hi = tm_mx_runtime::P::kBatch;
  out->seq = 1;
  out->past_lo = tm_mx_runtime::P::kPastLo;
  out->past_hi = tm_mx_runtime::P::kPastHi;
  out->capacity = tm_mx_runtime::P::kCapacity;
  out->grid = tm_mx_runtime::P::kGrid;
  out->residency = tm_mx_runtime::P::kGrid / 104;
  out->modes = TM_SERVING_L1;
  out->buffer_count = tm_mx_runtime::kBufferCount;
  return 0;
}

extern "C" int tm_plan_buffer(std::uint32_t index, tm_buffer_info* out) {
  if (!out || index >= tm_mx_runtime::kBufferCount) return -1;
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

extern "C" std::uint64_t tm_plan_shared_bytes() { return tm_mx_runtime::R::kSmem; }

extern "C" void* tm_plan_create(int batch, void* const* external, int device) {
  using namespace tm_mx_runtime;
  if (batch != P::kBatch || !external) return nullptr;
  if (cudaSetDevice(device) != cudaSuccess) return nullptr;
  auto* s = new MxPlanState;
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
      case kSlotTokens: b->tokens = static_cast<int*>(p); break;
      case kSlotRopeCos: b->rope_cos = e; break;
      case kSlotRopeSin: b->rope_sin = e; break;
      case kSlotEmbed: b->embed = e; break;
      case kSlotHead: b->w_head = e; break;
      case kSlotQkv: layer.w_qkv = e; break;
      case kSlotO: layer.w_o = e; break;
      case kSlotGateUp: layer.w_gu = e; break;
      case kSlotDown: layer.w_down = e; break;
      case kSlotQNorm: layer.q_norm = e; break;
      case kSlotKNorm: layer.k_norm = e; break;
      case kSlotKCache: layer.k_cache = e; break;
      case kSlotVtCache: layer.vt_cache = e; break;
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
      Alloc(s, B * P::AttnShape::hkv * ((P::kCapacity + 31) / 32) * P::kQ * (P::kHeadDim + 2) * 4));
  sc.head_value = static_cast<float*>(Alloc(s, B * P::kGrid * 4));
  sc.head_index = static_cast<int*>(Alloc(s, B * P::kGrid * 4));
  sc.t_qkv = static_cast<unsigned*>(Alloc(s, (P::QkvShape::n / P::Qkv::kTN) * 4));
  sc.t_o = static_cast<unsigned*>(Alloc(s, (P::OShape::n / P::O::kTN) * 4));
  sc.t_gu = static_cast<unsigned*>(Alloc(s, (P::GateUpShape::n / P::GateUp::kTN) * 4));
  sc.t_down = static_cast<unsigned*>(Alloc(s, (P::DownShape::n / P::Down::kTN) * 4));
  sc.t_attn = static_cast<unsigned*>(Alloc(s, B * P::AttnShape::hkv * 4));
  sc.t_head = static_cast<unsigned*>(Alloc(s, 64));
  sc.barrier = static_cast<unsigned long long*>(Alloc(s, std::size_t(R::kStages) * 64));
  for (void* p : s->owned)
    if (!p) {
      Free(s);
      return nullptr;
    }
#if TILEMEGA_WATCHDOG
  {
    unsigned long long ms = 10000;
    if (auto const env = std::getenv("TILEMEGA_WATCHDOG_MS")) ms = std::strtoull(env, nullptr, 10);
    if (ms && cudaHostAlloc(&s->watchdog, sizeof(tilemega::codegen::WatchdogRecord), cudaHostAllocMapped) == cudaSuccess) {
      *s->watchdog = {};
      cudaHostGetDevicePointer(reinterpret_cast<void**>(&sc.watchdog), s->watchdog, 0);
      sc.watchdog_ns = ms * 1000000ull;
    }
  }
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
  auto kernel = mx_decode_kernel<P>;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, R::kSmem) != cudaSuccess) {
    Free(s);
    return nullptr;
  }
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 512, R::kSmem);
  if (blocks != P::kGrid / 104) {
    std::fprintf(stderr, "tm_plan_create: occupancy %d, the program needs %d CTAs per AP\n", blocks, P::kGrid / 104);
    Free(s);
    return nullptr;
  }
  std::printf("E2E_LOOP_MODES 1\nE2E_PDL 0\n");
  return s;
}

extern "C" int tm_plan_set_steps(void* opaque, std::int32_t const* past, std::uint32_t count) {
  using namespace tm_mx_runtime;
  auto* s = static_cast<MxPlanState*>(opaque);
  if (!s || !past || !count || s->pasts) return -1;
  for (std::uint32_t i = 0; i < count; ++i)
    if (past[i] < P::kPastLo || past[i] > P::kPastHi) return -1;
  if (cudaMalloc(&s->pasts, count * 4) != cudaSuccess) return -1;
  if (cudaMemcpy(s->pasts, past, count * 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
  if (cudaMalloc(&s->step_ns, (count + 1) * 8) != cudaSuccess) return -1;
  cudaMemset(s->step_ns, 0, (count + 1) * 8);
  s->steps = count;
  return 0;
}

namespace tm_mx_runtime {
inline int Launch(MxPlanState* s, std::uint32_t first, std::uint32_t steps, std::uint64_t iteration,
                  unsigned long long* step_ns, void* stream) {
  if (iteration != s->next_iteration) return -2;
  mx_decode_kernel<P><<<P::kGrid, 512, R::kSmem, static_cast<cudaStream_t>(stream)>>>(
      *static_cast<P::Binding const*>(s->binding), s->pasts, first, steps, iteration, step_ns, s->trace);
  if (cudaGetLastError() != cudaSuccess) return 1;
  s->next_iteration += steps;
  return 0;
}
}  // namespace tm_mx_runtime

extern "C" int tm_plan_launch(void* opaque, std::uint32_t step, std::uint32_t mode, std::uint64_t iteration,
                              void* stream) {
  using namespace tm_mx_runtime;
  auto* s = static_cast<MxPlanState*>(opaque);
  if (!s || mode != TM_SERVING_L1 || step >= s->steps) return -1;
  return Launch(s, step, 1, iteration, nullptr, stream);
}

extern "C" int tm_plan_launch_steps(void* opaque, std::uint32_t first_step, std::uint32_t steps, std::uint32_t mode,
                                    std::uint64_t base_iteration, void* stream) {
  using namespace tm_mx_runtime;
  auto* s = static_cast<MxPlanState*>(opaque);
  if (!s) return -1;
  if (mode != TM_SERVING_L1) return -3;
  if (!steps || first_step + steps > s->steps) return -1;
  return Launch(s, first_step, steps, base_iteration, s->step_ns + first_step, stream);
}

extern "C" unsigned tm_plan_loop_modes(void* opaque) { return opaque ? TM_SERVING_L1 : 0; }
extern "C" unsigned tm_plan_pdl_enabled(void*) { return 0; }
extern "C" std::uint64_t tm_plan_execution_smem(void* opaque, unsigned mode) {
  return opaque && mode == TM_SERVING_L1 ? tm_mx_runtime::R::kSmem : 0;
}

extern "C" int tm_plan_read_step_ns(void* opaque, std::uint32_t first, std::uint32_t count, std::uint64_t* host) {
  auto* s = static_cast<tilemega::codegen::mx::MxPlanState*>(opaque);
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
  auto* s = static_cast<tilemega::codegen::mx::MxPlanState*>(opaque);
  if (!s || !out || !s->watchdog || !s->watchdog->fired) return 0;
  auto const* words = reinterpret_cast<unsigned long long const*>(s->watchdog);
  for (int i = 0; i < 12; ++i) out[i] = words[i];
  return 1;
#endif
}

#if TILEMEGA_TRACE_STAGE
/// <dir>/mx_stage_trace.csv: slot,stage,cta,start_ns,work_end_ns,release_ns.
extern "C" int tm_plan_dump_serving_trace(void* opaque, char const* directory) {
  using namespace tm_mx_runtime;
  auto* s = static_cast<MxPlanState*>(opaque);
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
  tm_mx_runtime::Free(static_cast<tilemega::codegen::mx::MxPlanState*>(opaque));
}
