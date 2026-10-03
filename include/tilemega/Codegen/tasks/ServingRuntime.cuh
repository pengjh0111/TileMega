// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>

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
  std::uint64_t* step_ns = nullptr;
  std::uint32_t steps = 0;
  int grid = 0;
  bool pdl = false;
  unsigned loop_modes = 0;
  executor::TensorMap* tensor_maps = nullptr;
  LagDependency* lag_dependencies = nullptr;
  PageTraceRecord* page_trace = nullptr;
#if TILEMEGA_TRACE_STAGE
  StageTraceRecord* stage_trace=nullptr;
#endif
#if TILEMEGA_TRACE_STEP
  StepTraceRecord* step_trace=nullptr;
#endif
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP
  unsigned trace_launches=0;
#endif
  WatchdogRecord* watchdog = nullptr;
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
          dep.map == StageDependency::Map::kAll ||
              dep.map == StageDependency::Map::kPhase,
          dep.div, dep.scale, dep.offset, dep.count);
      auto upper = RuntimeDependencyBounds(
          task, upper_counts[dep.producer],
          dep.map == StageDependency::Map::kAll ||
              dep.map == StageDependency::Map::kPhase,
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
  if(plan->page_trace) {
    if(auto path=std::getenv("TILEMEGA_PAGE_TRACE_OUT")) {
      cudaDeviceSynchronize();
      std::vector<PageTraceRecord> rows(std::size_t(plan->grid)*plan->steps);
      if(cudaMemcpy(rows.data(),plan->page_trace,rows.size()*sizeof(rows[0]),
                    cudaMemcpyDeviceToHost)==cudaSuccess)
        if(auto* out=std::fopen(path,"w")) {
          std::fprintf(out,"step\tworker\tkernel_begin_ns\tkernel_end_ns\tdependency_wait_ns\tpage_full_ns\tfull_and_wait_ns\tdependency_episodes\tpage_full_episodes\n");
          for(std::size_t i=0;i<rows.size();++i)
            std::fprintf(out,"%zu\t%zu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n",i/std::size_t(plan->grid),i%std::size_t(plan->grid),
                rows[i].kernel_begin_ns,rows[i].kernel_end_ns,
                rows[i].dependency_wait_ns,rows[i].page_full_ns,rows[i].full_and_wait_ns,
                rows[i].dependency_episodes,rows[i].page_full_episodes);
          std::fclose(out);
        }
    }
    cudaFree(plan->page_trace);
  }
#if TILEMEGA_TRACE_STAGE
  if(plan->stage_trace)cudaFree(plan->stage_trace);
#endif
#if TILEMEGA_TRACE_STEP
  if(plan->step_trace)cudaFree(plan->step_trace);
#endif
  if (plan->ring) cudaFree(plan->ring);
  if (plan->step_ns) cudaFree(plan->step_ns);
  if (plan->tensor_maps) cudaFree(plan->tensor_maps);
  if (plan->lag_dependencies) cudaFree(plan->lag_dependencies);
  if (plan->watchdog) cudaFreeHost(plan->watchdog);
  auto& model = plan->model;
  if(model.params.serving_handoff_tickets)cudaFree(model.params.serving_handoff_tickets);
  if(model.params.serving_no_producer)cudaFree(const_cast<std::uint8_t*>(model.params.serving_no_producer));
#if TILEMEGA_TRACE_V2
  if (model.device_reducer_trace) cudaFree(model.device_reducer_trace);
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
#if TILEMEGA_PAGED
      if (cudaFuncSetAttribute(tilemega_loop_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
        return nullptr;
#elif TILEMEGA_SERVING_SEQ==1
      if(cudaFuncSetAttribute(tilemega_l1_loop_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,smem)!=cudaSuccess)
        return nullptr;
#endif
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
    {
      auto const env=std::getenv("TILEMEGA_WATCHDOG_MS");
      unsigned long long ms=env?std::strtoull(env,nullptr,10):10000ull;
      if(ms) {
        TILEMEGA_CUDA_CHECK(cudaHostAlloc(&plan->watchdog,sizeof(WatchdogRecord),
            cudaHostAllocMapped));
        *plan->watchdog={};
        TILEMEGA_CUDA_CHECK(cudaHostGetDevicePointer(
            &plan->model.params.serving_watchdog,plan->watchdog,0));
        plan->model.params.serving_watchdog_ns=ms*1000000ull;
      }
    }
#if TILEMEGA_PAGED
    if(!plan->model.lag_dependencies.empty()) {
      TILEMEGA_CUDA_CHECK(cudaMalloc(&plan->lag_dependencies,
          plan->model.lag_dependencies.size()*sizeof(LagDependency)));
      TILEMEGA_CUDA_CHECK(cudaMemcpy(plan->lag_dependencies,plan->model.lag_dependencies.data(),
          plan->model.lag_dependencies.size()*sizeof(LagDependency),cudaMemcpyHostToDevice));
      plan->model.params.lag_dependencies=plan->lag_dependencies;
      plan->model.params.lag_dependency_count=plan->model.lag_dependencies.size();
    }
#endif
    bool has_handoff=false;
    std::vector<unsigned> reduce_users(plan->model.stages.size(),0);
    for(std::size_t i=0;i<plan->model.stages.size();++i) {
      auto const& stage=plan->model.stages[i];
      if(stage.handoff_reduce_stage==kNoOperand)continue;
      if(!TILEMEGA_PAGED || TILEMEGA_SERVING_SEQ!=1 ||
         stage.handoff_reduce_stage<=i ||
         stage.handoff_reduce_stage>=plan->model.stages.size())
        throw std::invalid_argument("last-arriver requires a later decode reducer in a paged plan");
      auto const& reduce=plan->model.stages[stage.handoff_reduce_stage];
      if(!reduce.handoff_elided ||
         !((stage.kind==TaskKind::kGemm && reduce.kind==TaskKind::kGemmCombine &&
            stage.gemm==reduce.gemm) ||
           (stage.kind==TaskKind::kGemm && reduce.kind==TaskKind::kArgmaxReduce) ||
           (stage.kind==TaskKind::kFusedAttention && reduce.kind==TaskKind::kAttentionMerge)))
        throw std::invalid_argument("last-arriver stage pair is not a complete GEMM or attention reduction");
      ++reduce_users[stage.handoff_reduce_stage];
      has_handoff=true;
    }
    for(std::size_t i=0;i<reduce_users.size();++i)
      if(reduce_users[i]>1)
        throw std::invalid_argument("a reducer cannot have multiple last-arriver owners");
    if(has_handoff) {
      std::uint32_t stride=1;
      for(std::uint32_t i=0;i<kModel.stage_count;++i) {
        auto const& stage=kModel.stages[i];
        stride=std::max(stride,std::uint32_t(serving::Count(
            kModel,kModel.runtime_variants[0],stage,dims)));
      }
      auto count=std::size_t(plan->model.stages.size())*stride;
      if(count>std::numeric_limits<std::size_t>::max()/sizeof(unsigned))
        throw std::overflow_error("handoff ticket allocation overflow");
      TILEMEGA_CUDA_CHECK(cudaMalloc(&plan->model.params.serving_handoff_tickets,
          count*sizeof(unsigned)));
      TILEMEGA_CUDA_CHECK(cudaMemset(plan->model.params.serving_handoff_tickets,0,
          count*sizeof(unsigned)));
      plan->model.params.serving_handoff_ticket_stride=stride;
    }
    serving::CreateTensorMaps(*plan,kModel,target);
#if TILEMEGA_L2_PREFETCH
    std::uint8_t* frontier=nullptr;
    TILEMEGA_CUDA_CHECK(cudaMalloc(&frontier,sizeof(kServingFrontier)));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(frontier,kServingFrontier,sizeof(kServingFrontier),cudaMemcpyHostToDevice));
    plan->model.params.serving_no_producer=frontier;
#endif
    harness::PrepareEvents(plan->model, grid);
    if (cudaMemset(plan->model.events, 0,
                   plan->model.event_count * sizeof(EventCounter)) != cudaSuccess)
      return nullptr;
    plan->grid = grid;
#if TILEMEGA_PAGED
    plan->loop_modes=2;
#elif TILEMEGA_SERVING_SEQ==1
    int loop_resident=target.ActiveBlocksPerSM(
        reinterpret_cast<void const*>(tilemega_l1_loop_kernel),kServingThreads,smem);
    if(grid<=target.res.num_sms*loop_resident)plan->loop_modes=1;
#endif
    std::fprintf(stderr,"E2E_LOOP_MODES available=%u\n",plan->loop_modes);
    plan->pdl = TILEMEGA_SERVING_SEQ==1 && TILEMEGA_PDL &&
        !TILEMEGA_ARCH_PATH_SM80 && target.caps.pdl;
    std::fprintf(stderr,"E2E_PDL enabled=%d caps=%d\n",int(plan->pdl),int(target.caps.pdl));
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
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP
  auto capacity_env=std::getenv("TILEMEGA_TRACE_LAUNCHES");
  plan->trace_launches=capacity_env?unsigned(std::strtoul(capacity_env,nullptr,10)):64;
  if(!plan->trace_launches || plan->trace_launches>4096)return -7;
#if TILEMEGA_TRACE_STAGE
  std::size_t stage_bytes=std::size_t(plan->trace_launches)*plan->grid*
      plan->model.stages.size()*sizeof(StageTraceRecord);
  if(cudaMalloc(&plan->stage_trace,stage_bytes)!=cudaSuccess)return -5;
  if(cudaMemset(plan->stage_trace,0,stage_bytes)!=cudaSuccess)return -6;
  for(auto& p:host)p.serving_stage_trace=plan->stage_trace;
#endif
#if TILEMEGA_TRACE_STEP
  std::size_t step_bytes=std::size_t(plan->trace_launches)*plan->grid*sizeof(StepTraceRecord);
  if(cudaMalloc(&plan->step_trace,step_bytes)!=cudaSuccess)return -5;
  if(cudaMemset(plan->step_trace,0,step_bytes)!=cudaSuccess)return -6;
  for(auto& p:host)p.serving_step_trace=plan->step_trace;
#endif
  for(auto& p:host)p.serving_trace_launches=plan->trace_launches;
#endif
#if TILEMEGA_PAGED && TILEMEGA_PAGE_TRACE
  if(std::getenv("TILEMEGA_PAGE_TRACE_OUT")) {
    std::size_t bytes=std::size_t(count)*plan->grid*sizeof(PageTraceRecord);
    if(cudaMalloc(&plan->page_trace,bytes)!=cudaSuccess)return -5;
    if(cudaMemset(plan->page_trace,0,bytes)!=cudaSuccess)return -6;
    for(std::uint32_t i=0;i<count;++i)
      host[i].serving_page_trace=plan->page_trace+std::size_t(i)*plan->grid;
  }
#endif
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
  if (cudaMalloc(&plan->step_ns, (std::size_t(count)+1)*sizeof(std::uint64_t))
      != cudaSuccess) return -5;
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

extern "C" unsigned tm_plan_loop_modes(void* opaque) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  return plan?plan->loop_modes:0;
}

extern "C" int tm_plan_launch_steps(void* opaque, std::uint32_t first_step,
    std::uint32_t steps, std::uint32_t mode, std::uint64_t base_iteration,
    void* stream) {
  using namespace tilemega::codegen;
  auto* plan=static_cast<serving::Plan*>(opaque);
  if(!plan || !plan->ring || !steps || first_step>plan->steps ||
     steps>plan->steps-first_step ||
     (mode!=TM_SERVING_L1 && mode!=TM_SERVING_L2))return -1;
  if(!(plan->loop_modes&mode))return -3;
  unsigned index=mode==TM_SERVING_L1?0:1;
  if(base_iteration!=plan->next_iteration[index])return -2;
  auto cuda_stream=static_cast<cudaStream_t>(stream);
  cudaError_t status;
#if TILEMEGA_PAGED
  status=executor::LaunchServing(tilemega_loop_kernel,plan->grid,
        kServingThreads,plan->model.l2_smem_bytes,cuda_stream,plan->pdl,
        static_cast<Params const*>(plan->ring+first_step),steps,
        plan->model.events,base_iteration,
        reinterpret_cast<unsigned long long*>(plan->step_ns+first_step));
#elif TILEMEGA_SERVING_SEQ==1
  status=executor::LaunchServing(tilemega_l1_loop_kernel,plan->grid,
        kServingThreads,kServingSharedBytes,cuda_stream,plan->pdl,
        static_cast<Params const*>(plan->ring+first_step),steps,
        plan->model.events,base_iteration,
        reinterpret_cast<unsigned long long*>(plan->step_ns+first_step));
#else
  return -3;
#endif
#if TILEMEGA_PAGED || TILEMEGA_SERVING_SEQ==1
  if(status!=cudaSuccess)return int(status);
  plan->next_iteration[index]+=steps;
  return 0;
#endif
}

extern "C" int tm_plan_read_step_ns(void* opaque,std::uint32_t first,
    std::uint32_t count,std::uint64_t* host) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if(!plan || !plan->step_ns || !host || first>plan->steps ||
     count>plan->steps-first+1)return -1;
  return int(cudaMemcpy(host,plan->step_ns+first,
      std::size_t(count)*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
}

extern "C" int tm_plan_watchdog(void* opaque,std::uint64_t out[12]) {
#if !TILEMEGA_WATCHDOG
  return 0;
#else
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if(!plan || !out || !plan->watchdog || !plan->watchdog->fired)return 0;
  auto const* words=reinterpret_cast<unsigned long long const*>(plan->watchdog);
  for(int i=0;i<12;++i)out[i]=words[i];
  return 1;
#endif
}

#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP
extern "C" int tm_plan_dump_serving_trace(void* opaque,char const* directory) {
  using namespace tilemega::codegen;
  auto* plan=static_cast<serving::Plan*>(opaque);
  if(!plan || !directory || cudaDeviceSynchronize()!=cudaSuccess)return -1;
  auto dump=[&](char const* name,auto* device,std::size_t count,char const* header) {
    using Row=std::remove_pointer_t<decltype(device)>;
    std::vector<Row> host(count);
    if(cudaMemcpy(host.data(),device,count*sizeof(Row),cudaMemcpyDeviceToHost)!=cudaSuccess)
      return false;
    auto path=std::string(directory)+"/"+name;
    auto* out=std::fopen(path.c_str(),"w");if(!out)return false;
    std::fprintf(out,"%s\n",header);
    for(std::size_t i=0;i<count;++i) {
      auto* words=reinterpret_cast<unsigned long long const*>(&host[i]);
      // A zero begin stamp is an unused launch/stage slot.
      if(!words[2])continue;
      std::fprintf(out,"%zu",i%plan->grid);
      for(unsigned j=0;j<sizeof(Row)/sizeof(*words);++j)
        std::fprintf(out,"\t%llu",words[j]);
      std::fprintf(out,"\n");
    }
    std::fclose(out);return true;
  };
#if TILEMEGA_TRACE_STAGE
  if(!dump("stage_trace.tsv",plan->stage_trace,std::size_t(plan->trace_launches)*
      plan->model.stages.size()*plan->grid,
      "worker\titeration\tpast\tt_begin\tt_tasks_end\tt_release\ttasks\tsmid\tstage"))return -2;
#endif
#if TILEMEGA_TRACE_STEP
  if(!dump("step_trace.tsv",plan->step_trace,std::size_t(plan->trace_launches)*plan->grid,
      "worker\titeration\tpast\tkernel_begin\tkernel_end\tfirst_task\tlast_task\ttoken_lag\tkv_lag\tlast_barrier_wait\tstage_has_tasks"))return -2;
#endif
  std::size_t gemm_count=0;
  for(auto const& stage:plan->model.stages)
    if(stage.kind==TaskKind::kGemm || stage.kind==TaskKind::kGemmCombine)
      gemm_count=std::max(gemm_count,std::size_t(stage.gemm)+1);
  std::vector<GemmInvocation> gemms(gemm_count);
  if(gemm_count && cudaMemcpy(gemms.data(),plan->model.device_gemms,
      gemm_count*sizeof(gemms[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)return -3;
  auto* out=std::fopen((std::string(directory)+"/runtime_stages.tsv").c_str(),"w");
  if(!out)return -4;
  std::fprintf(out,"stage\tkind\tkind_name\tname\tweight_bytes\textent\tgroup\twidth\tkv_block\telided\treducer\n");
  for(unsigned i=0;i<plan->model.stages.size();++i) {
    auto const& stage=plan->model.stages[i];unsigned long long bytes=0;
    char const* name="";
    if(stage.kind==TaskKind::kGemm || stage.kind==TaskKind::kGemmCombine) {
      auto const& inv=gemms[stage.gemm];auto [m,n,k,l]=inv.problem;
      if(stage.kind==TaskKind::kGemm)bytes=2ull*n*inv.serving_k_total_full;
      if(inv.serving_weight_buffer<plan->model.spec->buffer_count)
        name=plan->model.spec->buffers[inv.serving_weight_buffer].name;
    }
    char const* kind_name="unknown";switch(stage.kind) {case TaskKind::kGemm:kind_name="kGemm";break;case TaskKind::kRMSNorm:kind_name="kRMSNorm";break;case TaskKind::kRoPE:kind_name="kRoPE";break;case TaskKind::kKVAppend:kind_name="kKVAppend";break;case TaskKind::kElementwise:kind_name="kElementwise";break;case TaskKind::kAttention:kind_name="kAttention";break;case TaskKind::kGemmCombine:kind_name="kGemmCombine";break;case TaskKind::kGemmAdd:kind_name="kGemmAdd";break;case TaskKind::kGemmRMSNorm:kind_name="kGemmRMSNorm";break;case TaskKind::kRoPEKVAppend:kind_name="kRoPEKVAppend";break;case TaskKind::kAdd:kind_name="kAdd";break;case TaskKind::kEmbedding:kind_name="kEmbedding";break;case TaskKind::kQKNorm:kind_name="kQKNorm";break;case TaskKind::kFusedAttention:kind_name="kFusedAttention";break;case TaskKind::kAttentionMerge:kind_name="kAttentionMerge";break;case TaskKind::kArgmaxReduce:kind_name="kArgmaxReduce";break;}
    std::fprintf(out,"%u\t%u\t%s\t%s\t%llu\t%u\t%u\t%u\t%u\t%u\t%u\n",i,
        unsigned(stage.kind),kind_name,name,bytes,stage.extent,stage.group,stage.width,
        unsigned(stage.attention_kv_block),unsigned(stage.handoff_elided),
        unsigned(stage.handoff_reduce_stage));
  }
  std::fclose(out);return 0;
}
#endif

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
