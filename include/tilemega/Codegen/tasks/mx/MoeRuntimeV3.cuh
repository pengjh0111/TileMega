// SPDX-License-Identifier: BSD-3-Clause
// Native MoE decode and 64-token prompt serving ABI. External metadata is
// generated from ModelPlan; all dispatch/intermediate buffers are owned here.
#pragma once

#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/tasks/mx/MoeProgramV3.cuh>

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

template <int TN, class Shape>
constexpr std::size_t MxGemmPartialFloatsV3(int grid, bool stream_k) {
  if (!stream_k) return 0;
  int const tiles = Shape::n / TN;
  // This is the slot stride used by the production GEMM body.
  return std::size_t(tiles) * backend::mx::MxMaxContributors(tiles, grid) * Shape::b * (TN + 1);
}
constexpr std::size_t MxMaxV3(std::size_t a, std::size_t b) { return a > b ? a : b; }

template <class P>
struct MxMoeRuntimeV3 {
  static constexpr int kGrid = P::kGrid, kStages = 6 * P::kLayers + 1;
  static constexpr int kGenerationOffset = MoeControlV3<P>::kGenerationOffset;
  static constexpr int kSmem = MoeControlV3<P>::kSharedBytes;
  static_assert(P::kThreads == 512);
  static_assert(P::kGrid == 104 && kSmem <= 65536);
  static_assert(P::kBodySmemBytes >= 32784 && P::kBodySmemBytes % 16 == 0);
  static_assert(kSmem == MoeControlV3<P>::kSharedBytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::Qkv::kTN, P::Qkv::kWN).bytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::O::kTN, P::O::kWN).bytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::GateUp::kTN, P::GateUp::kWN).bytes);
  static_assert(P::kBodySmemBytes >= backend::mx::MxGemmSmemV2(P::kBatch, P::Down::kTN, P::Down::kWN).bytes);
  static_assert(P::kBodySmemBytes >= P::Head::SmemBytes(P::kBatch));
  static_assert(P::kBodySmemBytes >= P::Attn::kSharedBytes);
  static constexpr std::size_t kPartial = MxMaxV3(
      MxGemmPartialFloatsV3<P::Qkv::kTN,typename P::QkvShape>(kGrid,P::kQkvStreamK),
      MxGemmPartialFloatsV3<P::O::kTN,typename P::OShape>(kGrid,P::kOStreamK));
};

struct MxMoePlanStateV3 {
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
};

}  // namespace tilemega::codegen::mx

namespace tm_mx_moe_runtime_v3 {
using namespace tilemega::codegen::mx;
using P = tm_mx_plan::Plan;
using R = MxMoeRuntimeV3<P>;
inline constexpr std::uint32_t kBufferCount = sizeof(tm_mx_plan::kBuffers) / sizeof(tm_mx_plan::kBuffers[0]);

inline void* Alloc(MxMoePlanStateV3* s, std::size_t bytes) {
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
inline void Free(MxMoePlanStateV3* s) {
  if (!s) return;
  for (void* p : s->owned) cudaFree(p);
  if (s->pasts) cudaFree(s->pasts);
  if (s->step_ns) cudaFree(s->step_ns);
  if (s->watchdog) cudaFreeHost(s->watchdog);
  delete static_cast<P::Binding*>(s->binding);
  delete s;
}
}  // namespace tm_mx_moe_runtime_v3

extern "C" int tm_plan_query(tm_plan_info* out) {
  if (!out) return -1;
  *out = {};
  out->abi_version = TM_SERVING_ABI_VERSION;
  out->phase = tm_mx_moe_runtime_v3::P::kPrompt ? TM_SERVING_PREFILL : TM_SERVING_DECODE;
  out->batch_lo = out->batch_hi = tm_mx_moe_runtime_v3::P::kBatch;
  out->seq = tm_mx_moe_runtime_v3::P::kPrompt ? 64 : 1;
  out->past_lo = tm_mx_moe_runtime_v3::P::kPastLo;
  out->past_hi = tm_mx_moe_runtime_v3::P::kPastHi;
  out->capacity = tm_mx_moe_runtime_v3::P::kCapacity;
  out->grid = tm_mx_moe_runtime_v3::P::kGrid;
  out->residency = tm_mx_moe_runtime_v3::P::kGrid / 104;
  out->modes = TM_SERVING_L1;
  out->buffer_count = tm_mx_moe_runtime_v3::kBufferCount;
  return 0;
}

extern "C" int tm_plan_buffer(std::uint32_t index, tm_buffer_info* out) {
  if (!out || index >= tm_mx_moe_runtime_v3::kBufferCount) return -1;
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

extern "C" std::uint64_t tm_plan_shared_bytes() { return tm_mx_moe_runtime_v3::R::kSmem; }

extern "C" void* tm_plan_create(int batch, void* const* external, int device) {
  using namespace tm_mx_moe_runtime_v3;
  if (batch != P::kBatch || !external) return nullptr;
  if (cudaSetDevice(device) != cudaSuccess) return nullptr;
  auto* s = new MxMoePlanStateV3;
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
      case 0: break;  // Original raw norm weights remain in the external table.
      case 1: b->tokens=static_cast<int*>(p);break;
      case 2: b->embed=e;break;
      case 3: b->rope_cos=e;break;
      case 4: b->rope_sin=e;break;
      case 5: b->w_head=e;break;
      case 6: layer.w_qkv=e;break;
      case 7: layer.w_o=e;break;
      case 8: layer.w_router=e;break;
      case 9: layer.w_gu=e;break;
      case 10: layer.w_down=e;break;
      case 11: layer.q_norm=e;break;
      case 12: layer.k_norm=e;break;
      case 13: layer.k_cache=e;break;
      case 14: layer.vt_cache=e;break;
      default:Free(s);return nullptr;
    }
  }
  auto& sc = b->s;
  std::size_t const B = P::kBatch;
  sc.x = static_cast<tilemega::Bf16*>(Alloc(s, B * P::kHidden * 2));
  sc.qkv = static_cast<tilemega::Bf16*>(Alloc(s, B * P::QkvShape::n * 2));
  sc.ctx = static_cast<tilemega::Bf16*>(Alloc(s, B * P::OShape::k * 2));
  sc.h = static_cast<tilemega::Bf16*>(Alloc(s, P::kExperts * B * P::DownShape::k * 2));
  sc.gemm_partial = static_cast<float*>(Alloc(s, R::kPartial * 4));
  if constexpr(P::kAttentionVersion==2||P::kAttentionSplit>1) {
    constexpr int parts=P::kAttentionVersion==2?P::Attn::kSlots:P::kAttentionSplit;
    sc.attn_partial=static_cast<float*>(Alloc(s,B*P::AttnShape::hkv*parts*P::kQ*(P::kHeadDim+2)*4));
  }
  sc.head_partial = static_cast<tilemega::backend::mx::MxHeadPartialV2*>(
      Alloc(s, B * P::kGrid * sizeof(tilemega::backend::mx::MxHeadPartialV2)));
  sc.t_qkv = static_cast<unsigned*>(Alloc(s, (P::QkvShape::n / P::Qkv::kTN) * 4));
  sc.t_o = static_cast<unsigned*>(Alloc(s, (P::OShape::n / P::O::kTN) * 4));
  sc.router_logits=static_cast<tilemega::Bf16*>(Alloc(s,B*P::kExperts*2));
  sc.expert_partials=static_cast<tilemega::Bf16*>(Alloc(s,B*8*P::kHidden*2));
  sc.combine_tickets=(MxMoeGlobalTicketV2*)Alloc(s,(P::kHidden/16)*4);
#if TILEMEGA_MX_DIAG
  sc.route_debug=static_cast<unsigned*>(Alloc(s,64*P::kLayers*B*8*4));
  sc.route_input=static_cast<tilemega::Bf16*>(Alloc(s,64*P::kLayers*B*P::kHidden*2));
  sc.route_logits=static_cast<tilemega::Bf16*>(Alloc(s,64*P::kLayers*B*P::kExperts*2));
  sc.route_weights=Alloc(s,64*P::kLayers*B*8*(P::kRoutingFp32?4:2));
#endif
  if constexpr(P::kAttentionVersion==2||P::kAttentionSplit>1)
    sc.t_attn=static_cast<unsigned*>(Alloc(s,B*P::AttnShape::hkv*4));
  sc.t_head = static_cast<unsigned*>(Alloc(s, 64));
  sc.barrier = static_cast<unsigned long long*>(Alloc(s, std::size_t(R::kStages) * 64));
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
  auto kernel = mx_moe_kernel_v3<P>;
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
  using namespace tm_mx_moe_runtime_v3;
  auto* s = static_cast<MxMoePlanStateV3*>(opaque);
  if (!s || !past || !count || s->pasts) return -1;
  if (P::kPrompt && (count!=1 || past[0]!=0)) return -1;
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

namespace tm_mx_moe_runtime_v3 {
inline int Launch(MxMoePlanStateV3* s, std::uint32_t first, std::uint32_t steps, std::uint64_t iteration,
                  unsigned long long* step_ns, void* stream) {
  if (P::kPrompt && (first!=0 || steps!=1 || iteration>(UINT64_MAX/P::kGrid-64)/64)) return -2;
  if (steps > UINT32_MAX / R::kStages || iteration != s->next_iteration || iteration > UINT64_MAX / P::kGrid ||
      steps > UINT64_MAX / P::kGrid - iteration) return -2;
  mx_moe_kernel_v3<P><<<P::kGrid, P::kThreads, R::kSmem, static_cast<cudaStream_t>(stream)>>>(
      *static_cast<P::Binding const*>(s->binding), StageLaunchV2{s->pasts, first, steps, iteration, step_ns, s->trace});
  if (cudaGetLastError() != cudaSuccess) return 1;
  s->next_iteration += steps;
  return 0;
}
}  // namespace tm_mx_moe_runtime_v3

extern "C" int tm_plan_launch(void* opaque, std::uint32_t step, std::uint32_t mode, std::uint64_t iteration,
                              void* stream) {
  using namespace tm_mx_moe_runtime_v3;
  auto* s = static_cast<MxMoePlanStateV3*>(opaque);
  if (!s || mode != TM_SERVING_L1 || step >= s->steps) return -1;
  return Launch(s, step, 1, iteration, nullptr, stream);
}

extern "C" int tm_plan_launch_steps(void* opaque, std::uint32_t first_step, std::uint32_t steps, std::uint32_t mode,
                                    std::uint64_t base_iteration, void* stream) {
  using namespace tm_mx_moe_runtime_v3;
  auto* s = static_cast<MxMoePlanStateV3*>(opaque);
  if (!s) return -1;
  if (mode != TM_SERVING_L1) return -3;
  if (!steps || first_step >= s->steps || steps > s->steps - first_step) return -1;
  return Launch(s, first_step, steps, base_iteration, s->step_ns ? s->step_ns + first_step : nullptr, stream);
}

extern "C" unsigned tm_plan_loop_modes(void* opaque) { return opaque ? TM_SERVING_L1 : 0; }
extern "C" unsigned tm_plan_pdl_enabled(void*) { return 0; }
extern "C" std::uint64_t tm_plan_execution_smem(void* opaque, unsigned mode) {
  return opaque && mode == TM_SERVING_L1 ? tm_mx_moe_runtime_v3::R::kSmem : 0;
}

extern "C" int tm_plan_read_step_ns(void* opaque, std::uint32_t first, std::uint32_t count, std::uint64_t* host) {
  auto* s = static_cast<tilemega::codegen::mx::MxMoePlanStateV3*>(opaque);
  if (!s || !s->step_ns || !host || first > s->steps || count > s->steps - first + 1) return -1;
  return int(cudaMemcpy(host, s->step_ns + first, std::size_t(count) * 8, cudaMemcpyDeviceToHost));
}

extern "C" int tm_plan_read_buffer(void* opaque,std::uint32_t index,void* host,std::uint64_t bytes) {
#if TILEMEGA_MX_DIAG
  using namespace tm_mx_moe_runtime_v3;
  auto* state=static_cast<MxMoePlanStateV3*>(opaque);
  if(!state||!host||index<kBufferCount||index>kBufferCount+3)return -1;
  auto* binding=static_cast<P::Binding*>(state->binding);
  void const* source[]={binding->s.route_debug,binding->s.route_input,binding->s.route_logits,binding->s.route_weights};
  std::uint64_t const positions=64ull*P::kLayers*P::kBatch;
  std::uint64_t const sizes[]={positions*8*4,positions*P::kHidden*2,positions*P::kExperts*2,
      positions*8*(P::kRoutingFp32?4:2)};
  unsigned const slot=index-kBufferCount;
  if(bytes!=sizes[slot])return -1;
  return int(cudaMemcpy(host,source[slot],bytes,cudaMemcpyDeviceToHost));
#else
  (void)opaque;(void)index;(void)host;(void)bytes;return -2;
#endif
}

extern "C" int tm_plan_read_buffer_named(void* opaque,char const* name,void* host,std::uint64_t bytes) {
  if(!name)return -1;
#if TILEMEGA_MX_DIAG
  if(!std::strcmp(name,"mx.router_logits")) {
    using namespace tm_mx_moe_runtime_v3;
    if(!opaque||!host||bytes!=std::uint64_t(P::kBatch)*P::kExperts*2)return -1;
    auto* state=static_cast<MxMoePlanStateV3*>(opaque);
    auto* binding=static_cast<P::Binding*>(state->binding);
    return int(cudaMemcpy(host,binding->s.router_logits,bytes,cudaMemcpyDeviceToHost));
  }
#endif
  char const* names[]={"mx.route_debug","mx.route_input","mx.route_logits","mx.route_weights"};
  for(unsigned i=0;i<4;++i)
    if(!std::strcmp(name,names[i]))
      return tm_plan_read_buffer(opaque,tm_mx_moe_runtime_v3::kBufferCount+i,host,bytes);
  return -1;
}

extern "C" int tm_plan_watchdog(void* opaque, std::uint64_t out[12]) {
#if !TILEMEGA_WATCHDOG
  (void)opaque, (void)out;
  return 0;
#else
  auto* s = static_cast<tilemega::codegen::mx::MxMoePlanStateV3*>(opaque);
  if (!s || !out || !s->watchdog || !s->watchdog->fired) return 0;
  auto const* words = reinterpret_cast<unsigned long long const*>(s->watchdog);
  for (int i = 0; i < 12; ++i) out[i] = words[i];
  return 1;
#endif
}

#if TILEMEGA_TRACE_STAGE
/// <dir>/mx_stage_trace.csv: slot,stage,cta,start_ns,work_end_ns,release_ns.
extern "C" int tm_plan_dump_serving_trace(void* opaque, char const* directory) {
  using namespace tm_mx_moe_runtime_v3;
  auto* s = static_cast<MxMoePlanStateV3*>(opaque);
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
  tm_mx_moe_runtime_v3::Free(static_cast<tilemega::codegen::mx::MxMoePlanStateV3*>(opaque));
}
