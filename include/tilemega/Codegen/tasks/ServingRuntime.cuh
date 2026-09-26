// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#ifndef TILEMEGA_SERVING_BATCH_LO
#define TILEMEGA_SERVING_BATCH_LO 1
#endif
#ifndef TILEMEGA_SERVING_BATCH_HI
#define TILEMEGA_SERVING_BATCH_HI TILEMEGA_SERVING_BATCH_LO
#endif
#ifndef TILEMEGA_SERVING_PAST_LO
#define TILEMEGA_SERVING_PAST_LO 0
#endif
#ifndef TILEMEGA_SERVING_PAST_HI
#define TILEMEGA_SERVING_PAST_HI TILEMEGA_SERVING_PAST_LO
#endif

namespace tilemega::codegen::serving {

struct Plan {
  harness::DeviceModel model;
  Params* ring = nullptr;
  std::uint32_t steps = 0;
  int grid = 0;
  bool pdl = false;
  executor::TensorMap* tensor_maps = nullptr;
  // L1 grid-barrier rows and L2 task-event rows are disjoint. Ticket equality
  // requires a gap-free sequence for each mode, even when launches alternate.
  std::uint64_t next_iteration[2] = {0, 0};
};

inline int Count(ModelSpec const& spec, RuntimeVariantDesc const& variant,
                 StageDesc const& stage, ModelDims dims) {
  switch (stage.kind) {
    case TaskKind::kGemm:
    case TaskKind::kGemmCombine: {
      auto const& gemm = spec.gemms[stage.gemm];
      auto const& geometry = variant.gemms[stage.gemm];
      int rows = stage.batch_rows ? dims.batch : dims.tokens();
      int tiles = CeilDiv(rows, geometry.tile_m) *
                  CeilDiv(gemm.n, geometry.tile_n);
      return stage.kind == TaskKind::kGemm ? tiles * geometry.split_k : tiles;
    }
    case TaskKind::kEmbedding:
      return dims.tokens();
    case TaskKind::kRMSNorm:
      return stage.batch_rows ? dims.batch : dims.tokens();
    case TaskKind::kFusedAttention:
      return dims.batch * int(stage.extent) *
          CeilDiv(int(stage.group) * dims.seq,
                           stage.attention_query_rows) *
          CeilDiv(dims.capacity, stage.attention_kv_block);
    case TaskKind::kAttentionMerge:
      return dims.batch * int(stage.extent);
    case TaskKind::kArgmaxReduce:
      return dims.batch;
    default:
      return -1;
  }
}

inline bool StructureInvariant(ModelSpec const& spec,
                               RuntimeVariantDesc const& variant,
                               ModelDims lo, ModelDims hi) {
  std::vector<int> lower_counts(spec.stage_count);
  std::vector<int> upper_counts(spec.stage_count);
  for (std::uint32_t stage = 0; stage < spec.stage_count; ++stage) {
    int lower = Count(spec, variant, spec.stages[stage], lo);
    int upper = Count(spec, variant, spec.stages[stage], hi);
    if (lower < 0 || lower != upper) return false;
    lower_counts[stage] = lower;
    upper_counts[stage] = upper;
  }
  // A StageDependency is a literal window in the generated table. If both
  // endpoint task cardinalities agree, its window is identical at every past.
  for (std::uint32_t edge = 0; edge < variant.dependency_count; ++edge) {
    auto const& dep = variant.dependencies[edge];
    if (dep.producer >= lower_counts.size() ||
        dep.consumer >= lower_counts.size())
      return false;
    for (int task : {0, lower_counts[dep.consumer] - 1}) {
      if (task < 0) continue;
      auto lower = RuntimeDependencyBounds(
          task, lower_counts[dep.producer],
          dep.map == StageDependency::Map::kAll,
          dep.div, dep.scale, dep.offset, dep.count);
      auto upper = RuntimeDependencyBounds(
          task, upper_counts[dep.producer],
          dep.map == StageDependency::Map::kAll,
          dep.div, dep.scale, dep.offset, dep.count);
      if (lower.first != upper.first || lower.past != upper.past)
        return false;
    }
  }
  return true;
}

inline void CreateTensorMaps(Plan& plan,ModelSpec const& spec,TargetSpec const& target) {
#if TILEMEGA_PAGED
  if(!target.caps.tma || TILEMEGA_ARCH_PATH_SM80)return;
  std::vector<executor::TensorMap> maps(spec.buffer_count);
  std::vector<bool> encoded(spec.buffer_count,false);
  auto encode=[&](unsigned id,executor::TensorMapShape shape) {
    if(encoded.at(id))return;
    auto code=executor::EncodeTensorMap(maps[id],plan.model.buffers.at(id),shape);
    if(code!=CUDA_SUCCESS)throw std::runtime_error("serving tensor map encoding failed: "+std::to_string(int(code)));
    encoded[id]=true;
  };
  for(unsigned i=0;i<spec.gemm_count;++i) {
    auto const& g=spec.gemms[i];auto const& tile=spec.runtime_variants[0].gemms[i];
    encode(g.b,{std::uint64_t(g.k),std::uint64_t(g.n),std::uint64_t(g.k)*2,
        64,std::uint32_t(std::min(int(tile.tile_n),TILEMEGA_PAGE_BYTES/128))});
  }
  for(unsigned i=0;i<spec.stage_count;++i) {
    auto const& s=spec.stages[i];if(s.kind!=TaskKind::kFusedAttention)continue;
    auto dims=plan.model.params.dims;
    executor::TensorMapShape shape{std::uint64_t(s.width),
        std::uint64_t(dims.batch)*s.extent*dims.capacity,std::uint64_t(s.width)*2,64,16};
    encode(s.operand[1],shape);encode(s.operand[2],shape);
  }
  TILEMEGA_CUDA_CHECK(cudaMalloc(&plan.tensor_maps,maps.size()*sizeof(maps[0])));
  TILEMEGA_CUDA_CHECK(cudaMemcpy(plan.tensor_maps,maps.data(),maps.size()*sizeof(maps[0]),cudaMemcpyHostToDevice));
  plan.model.params.serving_tensor_maps=plan.tensor_maps;
#endif
}

inline void Destroy(Plan* plan) {
  if (!plan) return;
  if (plan->ring) cudaFree(plan->ring);
  if (plan->tensor_maps) cudaFree(plan->tensor_maps);
  auto& model = plan->model;
#if TILEMEGA_TRACE_V2
  if (model.device_task_trace_v2) cudaFree(model.device_task_trace_v2);
  if (model.device_event_publish) cudaFree(model.device_event_publish);
#endif
  for (std::size_t i = 0; i < model.buffers.size(); ++i)
    if (i < model.owned_buffers.size() && model.owned_buffers[i])
      cudaFree(model.buffers[i]);
  for (void* pointer : {
           static_cast<void*>(model.device_buffers),
           static_cast<void*>(model.device_gemms),
           static_cast<void*>(model.device_stages),
           static_cast<void*>(model.device_dependencies),
           static_cast<void*>(model.device_dependency_offsets),
           static_cast<void*>(model.device_schedule),
           static_cast<void*>(model.device_schedule_offsets),
           static_cast<void*>(model.device_task_waits),
           static_cast<void*>(model.device_event_offsets),
           static_cast<void*>(model.device_event_flags),
           static_cast<void*>(model.device_event_fanin),
           static_cast<void*>(model.device_shard_arrivals),
           static_cast<void*>(model.device_shard_targets),
           static_cast<void*>(model.device_shard_local_offsets),
           static_cast<void*>(model.device_cluster_shard_offsets),
           static_cast<void*>(model.device_cluster_shard_indices),
           static_cast<void*>(model.device_params),
           static_cast<void*>(model.events)})
    if (pointer) cudaFree(pointer);
  delete plan;
}

}  // namespace tilemega::codegen::serving

extern "C" int tm_plan_query(tm_plan_info* output) {
  if (!output) return -1;
  auto target = tilemega::TargetSpec::Probe();
  int grid=kModel.runtime_variants[0].plan.eft_grid
      ? int(kModel.runtime_variants[0].plan.eft_grid):target.res.num_sms;
  *output = {TM_SERVING_ABI_VERSION,
             TILEMEGA_SERVING_SEQ == 1 ? TM_SERVING_DECODE : TM_SERVING_PREFILL,
             TILEMEGA_SERVING_BATCH_LO, TILEMEGA_SERVING_BATCH_HI,
             TILEMEGA_SERVING_SEQ, TILEMEGA_SERVING_PAST_LO,
             TILEMEGA_SERVING_PAST_HI, kModel.dims.capacity,
             grid, (grid+target.res.num_sms-1)/target.res.num_sms,
             TM_SERVING_L1 | TM_SERVING_L2,
             kModel.buffer_count};
  return 0;
}

extern "C" int tm_plan_buffer(std::uint32_t index,
                                      tm_buffer_info* output) {
  if (!output || index >= kModel.buffer_count) return -1;
  auto const& desc = kModel.buffers[index];
  if (desc.per_past || desc.per_total) return -2;
  *output = {desc.external_name ? desc.external_name : desc.name,
             desc.role, desc.dtype,
             desc.constant + std::uint64_t(desc.per_seq) * TILEMEGA_SERVING_SEQ,
             desc.per_batch, desc.pack_json};
  return 0;
}

extern "C" void* tm_plan_create(int batch, void* const* external,
                                        int device) {
  using namespace tilemega::codegen;
  if (!external || batch < TILEMEGA_SERVING_BATCH_LO ||
      batch > TILEMEGA_SERVING_BATCH_HI) return nullptr;
  if (cudaSetDevice(device) != cudaSuccess) return nullptr;
  try {
    auto target = tilemega::TargetSpec::Probe();
    ModelDims dims = kModel.dims;
    dims.batch = batch;
    dims.seq = TILEMEGA_SERVING_SEQ;
    dims.past = TILEMEGA_SERVING_PAST_LO;
    dims.total = dims.seq + dims.past;
    if (!serving::StructureInvariant(kModel, kModel.runtime_variants[0],
          dims, ModelDims{dims.seq, TILEMEGA_SERVING_PAST_HI,
                          dims.seq + TILEMEGA_SERVING_PAST_HI,
                          batch, dims.capacity}))
      return nullptr;
    std::unique_ptr<serving::Plan, void (*)(serving::Plan*)> plan(
        new serving::Plan, serving::Destroy);
    int const grid = kModel.runtime_variants[0].plan.eft_grid
        ? int(kModel.runtime_variants[0].plan.eft_grid):target.res.num_sms;
    std::size_t const smem = kServingSharedBytes;
    if (smem > target.res.max_dynamic_smem_per_cta) return nullptr;
    if (smem > 48 * 1024) {
      if (cudaFuncSetAttribute(tilemega_l1_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
          cudaFuncSetAttribute(tilemega_l2_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
        return nullptr;
    }
    int l1 = target.ActiveBlocksPerSM(
        reinterpret_cast<void const*>(tilemega_l1_kernel),
        kServingThreads, smem);
    int l2 = target.ActiveBlocksPerSM(
        reinterpret_cast<void const*>(tilemega_l2_kernel),
        kServingThreads, smem);
    if (grid > target.res.num_sms * std::min(l1, l2)) return nullptr;
#if TILEMEGA_PAGED
    if(grid>target.res.num_sms)return nullptr;
#endif
    plan->model = harness::Create(
        kModel, kModel.runtime_variants[0], 0, dims, "", grid,
        std::min(l1, l2), target, smem, external);
    serving::CreateTensorMaps(*plan,kModel,target);
    harness::PrepareEvents(plan->model, grid);
    if (cudaMemset(plan->model.events, 0,
                   plan->model.event_count * sizeof(EventCounter)) != cudaSuccess)
      return nullptr;
    plan->grid = grid;
    plan->pdl = TILEMEGA_PAGED && TILEMEGA_SERVING_SEQ==1 && TILEMEGA_PDL &&
        !TILEMEGA_ARCH_PATH_SM80 && target.caps.pdl;
    return plan.release();
  } catch (std::exception const& error) {
    std::fprintf(stderr, "tm_plan_create: %s\n", error.what());
    return nullptr;
  }
}

extern "C" int tm_plan_set_steps(void* opaque,
                                         std::int32_t const* past,
                                         std::uint32_t count) {
  using namespace tilemega::codegen;
  auto* plan = static_cast<serving::Plan*>(opaque);
  if (!plan || !past || !count || plan->ring) return -1;
  std::vector<Params> host(count, plan->model.params);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (past[i] < TILEMEGA_SERVING_PAST_LO ||
        past[i] > TILEMEGA_SERVING_PAST_HI) return -2;
    host[i].dims.past = past[i];
    host[i].dims.total = past[i] + host[i].dims.seq;
  }
  if (cudaMalloc(&plan->ring, count * sizeof(Params)) != cudaSuccess)
    return -3;
  if (cudaMemcpy(plan->ring, host.data(), count * sizeof(Params),
                 cudaMemcpyHostToDevice) != cudaSuccess) return -4;
  plan->steps = count;
  return 0;
}

extern "C" int tm_plan_launch(void* opaque, std::uint32_t step,
                                      std::uint32_t mode,
                                      std::uint64_t iteration,
                                      void* stream) {
  using namespace tilemega::codegen;
  auto* plan = static_cast<serving::Plan*>(opaque);
  if (!plan || !plan->ring || step >= plan->steps ||
      (mode != TM_SERVING_L1 && mode != TM_SERVING_L2)) return -1;
  std::uint32_t mode_index = mode == TM_SERVING_L1 ? 0 : 1;
  if (iteration != plan->next_iteration[mode_index]) return -2;
  auto* params = plan->ring + step;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  cudaError_t status;
  if (mode == TM_SERVING_L1)
    status=executor::LaunchServing(tilemega_l1_kernel,plan->grid,kServingThreads,
        kServingSharedBytes,cuda_stream,plan->pdl,params,plan->model.events,iteration);
  else
    status=executor::LaunchServing(tilemega_l2_kernel,plan->grid,kServingThreads,
        plan->model.l2_smem_bytes,cuda_stream,plan->pdl,params,plan->model.events,iteration);
  if (status != cudaSuccess) return int(status);
  ++plan->next_iteration[mode_index];
  return 0;
}

#if TILEMEGA_TRACE_V2
// Diagnostic-only entry point. A trace build is separate from the measured
// serving library; dumping synchronizes only after its selected launch.
extern "C" int tm_plan_dump_trace_v2(void* opaque, float step_ms) {
  auto* plan = static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if (!plan || !plan->model.trace_v2_enabled) return -1;
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  tilemega::codegen::harness::DumpTraceV2(
      plan->model, "serving", plan->grid, 0.0f, step_ms,
      "serving_decode_last_diagnostic_launch");
  return 0;
}
#endif

extern "C" void tm_plan_destroy(void* opaque) {
  tilemega::codegen::serving::Destroy(
      static_cast<tilemega::codegen::serving::Plan*>(opaque));
}
