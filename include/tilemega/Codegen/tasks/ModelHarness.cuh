// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.2 (only TaskBodies are handwritten), §8 (sync and launch).
//
// The model-independent half of the runtime.  It contains no dimension, no
// stage sequence, no operator name and no layer count: a model reaches it only
// as the generated `ModelSpec` tables.  Adding a model means emitting new
// tables, never editing this file.
#pragma once

#include <cuda_runtime.h>

#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/ElementwiseTaskBody.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/KVAppendTaskBody.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/RoPETaskBody.h>
#include <tilemega/Target/TargetSpec.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define TILEMEGA_CUDA_CHECK(expr) do { cudaError_t e = (expr); \
  if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

namespace tilemega::codegen {

#ifndef TILEMEGA_GENERATED_WAIT_global
#define TILEMEGA_GENERATED_WAIT_global(ev, need) do { \
  while (atomicAdd((ev), 0ull) < (need)) __nanosleep(64); \
} while (0)
#endif
#ifndef TILEMEGA_GENERATED_NOTIFY_global
#define TILEMEGA_GENERATED_NOTIFY_global(ev, value) atomicExch((ev), (value))
#endif
#ifndef TILEMEGA_GENERATED_RESIDENT_GRID
#define TILEMEGA_GENERATED_RESIDENT_GRID(target, function, block_size, dynamic_smem) \
  ((target).res.num_sms * (target).ActiveBlocksPerSM( \
      reinterpret_cast<void const*>(function), (block_size), (dynamic_smem)))
#endif

inline constexpr int kHarnessThreads = kGemmThreads;

/// §8.6: one explicit union covering every family the dispatch can reach.
union TaskSmem {
  float rms[kHarnessThreads];
  float attention[kHarnessThreads];
  float pointwise[1];
  GemmMainloop::SharedStorage gemm;
};
inline constexpr std::size_t kExpectedTaskSmem =
    sizeof(GemmMainloop::SharedStorage) > sizeof(float) * kHarnessThreads
        ? sizeof(GemmMainloop::SharedStorage)
        : sizeof(float) * kHarnessThreads;
static_assert(sizeof(TaskSmem) == kExpectedTaskSmem,
              "one explicit union must equal max_i(TaskBody::SharedStorage)");

using HarnessArch = cutlass::arch::Sm80;
using T_Gemm = GemmStageTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Norm = RMSNormTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_RoPE = RoPETaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_KV = KVAppendTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Elementwise = ElementwiseTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Attention = AttentionTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
static_assert(T_Gemm::kLegal && T_Norm::kLegal && T_RoPE::kLegal &&
              T_KV::kLegal && T_Elementwise::kLegal && T_Attention::kLegal,
              "every dispatched TaskBody must be legal at this granularity");

/// The dispatch is over the TaskBody families, which are a property of the
/// library, not of any model.  A model that needs no attention simply never
/// emits those stage kinds.
__device__ inline void RunStage(Params const& p, std::uint32_t index,
                                TaskSmem& smem) {
  StageDesc const& stage = p.stages[index];
  switch (stage.kind) {
    case TaskKind::kGemm: T_Gemm{}(p, stage, smem); break;
    case TaskKind::kRMSNorm: T_Norm{}(p, stage, smem); break;
    case TaskKind::kRoPE: T_RoPE{}(p, stage, smem); break;
    case TaskKind::kKVAppend: T_KV{}(p, stage, smem); break;
    case TaskKind::kElementwise: T_Elementwise{}(p, stage, smem); break;
    case TaskKind::kAttention: T_Attention{}(p, stage, smem); break;
  }
}

__host__ __device__ inline int CeilDiv(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

/// Cardinality of image(C_kappa) along the launch axis.  CTAs beyond this
/// bound participate only in control flow and own no producer event.
__device__ inline int ActiveBlocks(Params const& p, StageDesc const& stage) {
  switch (stage.kind) {
    case TaskKind::kGemm:
      return static_cast<GemmInvocation const*>(p.gemms)[stage.gemm].tiles_n;
    case TaskKind::kRMSNorm: return p.dims.seq;
    case TaskKind::kRoPE:
      return CeilDiv(p.dims.seq * static_cast<int>(stage.extent) *
                         static_cast<int>(stage.width / 2),
                     kHarnessThreads);
    case TaskKind::kKVAppend: {
      int appended = p.dims.seq * static_cast<int>(stage.extent) * stage.width;
      int retained = p.dims.past * static_cast<int>(stage.extent) * stage.width;
      return CeilDiv(appended > retained ? appended : retained, kHarnessThreads);
    }
    case TaskKind::kElementwise:
      return CeilDiv(p.dims.seq * static_cast<int>(stage.extent),
                     kHarnessThreads);
    case TaskKind::kAttention:
      return p.dims.seq * static_cast<int>(stage.extent);
  }
  return 0;
}

/// §8 wait sequence over tile-addressed events synthesized from CG couplings.
__device__ inline void WaitDependencies(Params const& p, EventCounter* events,
                                        std::uint32_t consumer, bool active) {
  if (threadIdx.x == 0 && active) {
    for (std::uint32_t edge = 0; edge < p.dependency_count; ++edge) {
      StageDependency dependency = p.dependencies[edge];
      if (dependency.consumer != consumer) continue;
      int producers = ActiveBlocks(p, p.stages[dependency.producer]);
      int begin = dependency.map == StageDependency::Map::kIdentity
                      ? static_cast<int>(blockIdx.x) : 0;
      int end = dependency.map == StageDependency::Map::kIdentity
                    ? begin + 1 : producers;
      if (begin >= producers) continue;
      for (int block = begin; block < end; ++block) {
        auto* epoch = &events[static_cast<std::size_t>(dependency.producer) *
                              gridDim.x + block].epoch;
        TILEMEGA_GENERATED_WAIT_global(epoch, 1ull);
      }
    }
  }
  __syncthreads();
  __threadfence();
}

/// §8.5 CTA-cooperative release and monotonic publication of one tile.
__device__ inline void NotifyTile(EventCounter* events, std::uint32_t producer,
                                  bool active) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && active)
    TILEMEGA_GENERATED_NOTIFY_global(
        &events[static_cast<std::size_t>(producer) * gridDim.x + blockIdx.x].epoch,
        1ull);
  __syncthreads();
}

/// §8.1/§8.2/§8.3: single-thread polling with backoff on a monotonic counter,
/// release fence before CTA convergence (F-1), acquire fence after.
__device__ inline void GridBarrier(EventCounter* events, std::uint32_t stage) {
  constexpr unsigned long long iteration_num = 1ull;
  unsigned long long needed =
      static_cast<unsigned long long>(gridDim.x) * iteration_num;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long ticket = atomicAdd(&events[stage].arrivals, 1ull);
    if (ticket + 1 == needed) {
      __threadfence();
      TILEMEGA_GENERATED_NOTIFY_global(&events[stage].epoch, iteration_num);
    } else {
      TILEMEGA_GENERATED_WAIT_global(&events[stage].epoch, iteration_num);
    }
  }
  __syncthreads();
  __threadfence();
}

__global__ __launch_bounds__(kHarnessThreads, 1)
void tilemega_stage_kernel(Params const* params, std::uint32_t stage) {
  extern __shared__ unsigned char bytes[];
  RunStage(*params, stage, *reinterpret_cast<TaskSmem*>(bytes));
}

/// The L1 megakernel.  The stage loop is a run-time loop over the generated
/// table: its trip count is data, so one compiled kernel serves every model.
__global__ __launch_bounds__(kHarnessThreads, 1)
void tilemega_l1_kernel(Params const* params, EventCounter* events) {
  extern __shared__ unsigned char bytes[];
  auto& smem = *reinterpret_cast<TaskSmem*>(bytes);
  for (std::uint32_t stage = 0; stage < params->stage_count; ++stage) {
    RunStage(*params, stage, smem);
    GridBarrier(events, stage);
  }
}

/// L2 uses the generated coupling DAG and one event per producer tile.  It has
/// no per-stage grid arrival counter; conservative relaxed C edges may wait on
/// all active producer tiles, but unrelated stages do not acquire each other.
__global__ __launch_bounds__(kHarnessThreads, 1)
void tilemega_l2_kernel(Params const* params, EventCounter* events) {
  extern __shared__ unsigned char bytes[];
  auto& smem = *reinterpret_cast<TaskSmem*>(bytes);
  for (std::uint32_t stage = 0; stage < params->stage_count; ++stage) {
    bool active = static_cast<int>(blockIdx.x) <
                  ActiveBlocks(*params, params->stages[stage]);
    WaitDependencies(*params, events, stage, active);
    RunStage(*params, stage, smem);
    NotifyTile(events, stage, active);
  }
}

namespace harness {

inline std::vector<float> Load(std::string const& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<float> value(count);
  input.read(reinterpret_cast<char*>(value.data()), count * sizeof(float));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
    std::fprintf(stderr, "wrong fixture size: %s\n", path.c_str());
    std::exit(2);
  }
  return value;
}

struct DeviceModel {
  ModelSpec const* spec = nullptr;
  std::vector<float*> buffers;
  std::vector<std::vector<float>> host_sources;  ///< per buffer, empty if scratch
  float** device_buffers = nullptr;
  GemmInvocation* device_gemms = nullptr;
  StageDesc* device_stages = nullptr;
  StageDependency* device_dependencies = nullptr;
  Params params{};
  Params* device_params = nullptr;
  EventCounter* events = nullptr;
  std::size_t event_count = 0;
};

/// Bind the symbolic dimensions.  The generated tables never carry a token
/// count; it arrives with the workload, here from the fixture manifest.
inline ModelDims BindDims(ModelDims dims, std::string const& dir) {
  std::ifstream input(dir + "/manifest.json");
  if (!input) {
    std::fprintf(stderr, "cannot open %s/manifest.json\n", dir.c_str());
    std::exit(2);
  }
  std::string text((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  auto field = [&](char const* name) {
    std::string key = std::string("\"") + name + "\"";
    std::size_t at = text.find(key);
    if (at == std::string::npos) {
      std::fprintf(stderr, "manifest has no %s\n", name);
      std::exit(2);
    }
    at = text.find(':', at) + 1;
    return std::atoi(text.c_str() + at);
  };
  dims.seq = field("seq");
  dims.past = field("past");
  dims.total = dims.seq + dims.past;
  return dims;
}

inline DeviceModel Create(ModelSpec const& spec, ModelDims const& dims,
                          std::string const& dir) {
  DeviceModel model;
  model.spec = &spec;
  model.host_sources.resize(spec.buffer_count);
  for (std::uint32_t i = 0; i < spec.buffer_count; ++i) {
    BufferDesc const& desc = spec.buffers[i];
    std::size_t elements = desc.Elements(dims);
    float* pointer = nullptr;
    TILEMEGA_CUDA_CHECK(cudaMalloc(&pointer, elements * sizeof(float)));
    if (desc.file != nullptr) {
      model.host_sources[i] = Load(dir + "/" + desc.file, elements);
      TILEMEGA_CUDA_CHECK(cudaMemcpy(pointer, model.host_sources[i].data(),
                                     elements * sizeof(float),
                                     cudaMemcpyHostToDevice));
    } else {
      TILEMEGA_CUDA_CHECK(cudaMemset(pointer, 0, elements * sizeof(float)));
    }
    model.buffers.push_back(pointer);
  }

  std::vector<GemmInvocation> gemms(spec.gemm_count);
  for (std::uint32_t i = 0; i < spec.gemm_count; ++i) {
    GemmDesc const& desc = spec.gemms[i];
    int m = dims.seq;
    GemmProblem problem{m, desc.n, desc.k, 1};
    auto stride_a = cutlass::make_cute_packed_stride(
        typename GemmMainloop::StrideA{}, cute::make_shape(m, desc.k, 1));
    auto stride_b = cutlass::make_cute_packed_stride(
        typename GemmMainloop::StrideB{}, cute::make_shape(desc.n, desc.k, 1));
    auto stride_c = cutlass::make_cute_packed_stride(
        typename GemmEpilogue::StrideC{}, cute::make_shape(m, desc.n, 1));
    auto stride_d = cutlass::make_cute_packed_stride(
        typename GemmEpilogue::StrideD{}, cute::make_shape(m, desc.n, 1));
    typename GemmMainloop::Arguments main_args{
        model.buffers[desc.a], stride_a, model.buffers[desc.b], stride_b};
    typename GemmEpilogue::Arguments epilogue_args{
        {1.0f, desc.beta}, model.buffers[desc.c], stride_c,
        model.buffers[desc.d], stride_d};
    gemms[i].problem = problem;
    gemms[i].mainloop =
        GemmMainloop::to_underlying_arguments(problem, main_args, nullptr);
    gemms[i].epilogue =
        GemmEpilogue::to_underlying_arguments(problem, epilogue_args, nullptr);
    gemms[i].tiles_n = (desc.n + kGemmTileN - 1) / kGemmTileN;
  }

  auto upload = [](void const* host, std::size_t bytes) {
    void* device = nullptr;
    TILEMEGA_CUDA_CHECK(cudaMalloc(&device, bytes));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    return device;
  };
  model.device_buffers = static_cast<float**>(
      upload(model.buffers.data(), model.buffers.size() * sizeof(float*)));
  model.device_gemms = static_cast<GemmInvocation*>(
      upload(gemms.data(), gemms.size() * sizeof(GemmInvocation)));
  model.device_stages = static_cast<StageDesc*>(upload(
      spec.stages, spec.stage_count * sizeof(StageDesc)));
  if (spec.dependency_count)
    model.device_dependencies = static_cast<StageDependency*>(upload(
        spec.dependencies, spec.dependency_count * sizeof(StageDependency)));

  model.params.dims = dims;
  model.params.buffers = model.device_buffers;
  model.params.gemms = model.device_gemms;
  model.params.stages = model.device_stages;
  model.params.stage_count = spec.stage_count;
  model.params.dependencies = model.device_dependencies;
  model.params.dependency_count = spec.dependency_count;
  TILEMEGA_CUDA_CHECK(cudaMalloc(&model.device_params, sizeof(Params)));
  TILEMEGA_CUDA_CHECK(cudaMemcpy(model.device_params, &model.params,
                                 sizeof(Params), cudaMemcpyHostToDevice));
  return model;
}

inline void PrepareEvents(DeviceModel& model, int grid) {
  if (model.events) TILEMEGA_CUDA_CHECK(cudaFree(model.events));
  model.event_count = static_cast<std::size_t>(model.spec->stage_count) * grid;
  TILEMEGA_CUDA_CHECK(
      cudaMalloc(&model.events, sizeof(EventCounter) * model.event_count));
}

/// Restore every buffer to its pre-run contents so two launches see the same
/// input.  File-backed buffers are re-uploaded, scratch is zeroed.
inline void Reset(DeviceModel& model) {
  ModelSpec const& spec = *model.spec;
  for (std::uint32_t i = 0; i < spec.buffer_count; ++i) {
    std::size_t bytes = spec.buffers[i].Elements(model.params.dims) * sizeof(float);
    if (spec.buffers[i].file != nullptr)
      TILEMEGA_CUDA_CHECK(cudaMemcpy(model.buffers[i],
                                     model.host_sources[i].data(), bytes,
                                     cudaMemcpyHostToDevice));
    else
      TILEMEGA_CUDA_CHECK(cudaMemset(model.buffers[i], 0, bytes));
  }
  if (model.events)
    TILEMEGA_CUDA_CHECK(cudaMemset(model.events, 0,
                                   sizeof(EventCounter) * model.event_count));
}

inline std::vector<std::vector<float>> Download(DeviceModel const& model) {
  ModelSpec const& spec = *model.spec;
  std::vector<std::vector<float>> output(spec.output_count);
  for (std::uint32_t i = 0; i < spec.output_count; ++i) {
    output[i].resize(
        spec.buffers[spec.outputs[i].buffer].Elements(model.params.dims));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(output[i].data(),
                                   model.buffers[spec.outputs[i].buffer],
                                   output[i].size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
  }
  return output;
}

struct Difference {
  std::size_t mismatch = 0;
  float max_abs = 0;
  float max_rel = 0;
};

inline Difference Compare(std::vector<std::vector<float>> const& actual,
                          std::vector<std::vector<float>> const& expected) {
  Difference result;
  for (std::size_t tensor = 0; tensor < actual.size(); ++tensor)
    for (std::size_t i = 0; i < actual[tensor].size(); ++i) {
      float delta = std::fabs(actual[tensor][i] - expected[tensor][i]);
      float relative = delta / std::max(std::fabs(expected[tensor][i]), 1.0e-6f);
      if (delta > 3.0e-5f + 3.0e-5f * std::fabs(expected[tensor][i]))
        ++result.mismatch;
      result.max_abs = std::max(result.max_abs, delta);
      result.max_rel = std::max(result.max_rel, relative);
    }
  return result;
}

inline unsigned long long BitHash(
    std::vector<std::vector<float>> const& values) {
  unsigned long long hash = 1469598103934665603ull;
  for (auto const& tensor : values) {
    auto const* bytes = reinterpret_cast<unsigned char const*>(tensor.data());
    for (std::size_t i = 0; i < tensor.size() * sizeof(float); ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

inline float LaunchL05(DeviceModel& model, int grid) {
  cudaEvent_t start, stop;
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&start));
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&stop));
  TILEMEGA_CUDA_CHECK(cudaEventRecord(start));
  for (std::uint32_t stage = 0; stage < model.params.stage_count; ++stage)
    tilemega_stage_kernel<<<grid, kHarnessThreads, sizeof(TaskSmem)>>>(
        model.device_params, stage);
  TILEMEGA_CUDA_CHECK(cudaEventRecord(stop));
  TILEMEGA_CUDA_CHECK(cudaEventSynchronize(stop));
  TILEMEGA_CUDA_CHECK(cudaGetLastError());
  float ms = 0;
  TILEMEGA_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

inline float LaunchL1(DeviceModel& model, int grid) {
  cudaEvent_t start, stop;
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&start));
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&stop));
  TILEMEGA_CUDA_CHECK(cudaEventRecord(start));
  tilemega_l1_kernel<<<grid, kHarnessThreads, sizeof(TaskSmem)>>>(
      model.device_params, model.events);
  TILEMEGA_CUDA_CHECK(cudaEventRecord(stop));
  TILEMEGA_CUDA_CHECK(cudaEventSynchronize(stop));
  TILEMEGA_CUDA_CHECK(cudaGetLastError());
  float ms = 0;
  TILEMEGA_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

inline float LaunchL2(DeviceModel& model, int grid) {
  cudaEvent_t start, stop;
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&start));
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&stop));
  TILEMEGA_CUDA_CHECK(cudaEventRecord(start));
  tilemega_l2_kernel<<<grid, kHarnessThreads, sizeof(TaskSmem)>>>(
      model.device_params, model.events);
  TILEMEGA_CUDA_CHECK(cudaEventRecord(stop));
  TILEMEGA_CUDA_CHECK(cudaEventSynchronize(stop));
  TILEMEGA_CUDA_CHECK(cudaGetLastError());
  float ms = 0;
  TILEMEGA_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

}  // namespace harness

/// The whole generated `main` is this call: everything model specific is in
/// `spec`, which the code generator emitted from the CG.
inline int RunModel(ModelSpec const& spec, char const* fixture_dir) {
  using namespace harness;
  ModelDims dims = BindDims(spec.dims, fixture_dir);
  DeviceModel model = Create(spec, dims, fixture_dir);
  std::vector<std::vector<float>> reference(spec.output_count);
  for (std::uint32_t i = 0; i < spec.output_count; ++i)
    reference[i] = Load(std::string(fixture_dir) + "/" + spec.outputs[i].file,
                        spec.buffers[spec.outputs[i].buffer].Elements(dims));

  auto target = tilemega::TargetSpec::Probe();
  TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(
      tilemega_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(TaskSmem)));
  TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(
      tilemega_l1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(TaskSmem)));
  TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(
      tilemega_l2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(TaskSmem)));
  int blocks_per_sm = target.ActiveBlocksPerSM(
      reinterpret_cast<void const*>(tilemega_l1_kernel), kHarnessThreads,
      sizeof(TaskSmem));
  int grid = TILEMEGA_GENERATED_RESIDENT_GRID(target, tilemega_l1_kernel,
                                              kHarnessThreads, sizeof(TaskSmem));
  PrepareEvents(model, grid);

  Reset(model);
  float l05_ms = LaunchL05(model, grid);
  auto l05 = Download(model);
  Reset(model);
  float l1_ms = LaunchL1(model, grid);
  auto l1 = Download(model);
  Reset(model);
  float l2_ms = LaunchL2(model, grid);
  auto l2 = Download(model);
  Difference l05_l0 = Compare(l05, reference);
  Difference l1_l05 = Compare(l1, l05);
  Difference l2_l1 = Compare(l2, l1);
  std::printf("E2E_RESOURCE block=%d reg=ptxas smem=%zu ctas_per_sm=%d "
              "num_sms=%d grid=%d resident_formula=ctas_per_sm*num_sms\n",
              kHarnessThreads, sizeof(TaskSmem), blocks_per_sm,
              target.res.num_sms, grid);
  std::printf("E2E_TIME l05_ms=%.6f l1_ms=%.6f ratio=%.6f l2_ms=%.6f "
              "l2_over_l1=%.6f\n", l05_ms, l1_ms, l1_ms / l05_ms, l2_ms,
              l2_ms / l1_ms);
  std::printf("E2E_DIFF l05_vs_l0_mismatch=%zu max_abs=%.8g max_rel=%.8g "
              "l1_vs_l05_mismatch=%zu max_abs=%.8g max_rel=%.8g "
              "l2_vs_l1_mismatch=%zu max_abs=%.8g max_rel=%.8g\n",
              l05_l0.mismatch, l05_l0.max_abs, l05_l0.max_rel,
              l1_l05.mismatch, l1_l05.max_abs, l1_l05.max_rel,
              l2_l1.mismatch, l2_l1.max_abs, l2_l1.max_rel);
  std::printf("E2E_HASH l05=%016llx l1=%016llx l2=%016llx\n", BitHash(l05),
              BitHash(l1), BitHash(l2));
  bool pass = l05_l0.mismatch == 0 && l1_l05.mismatch == 0 &&
              l2_l1.mismatch == 0;
  std::printf("RESULT status=%s\n", pass ? "PASS" : "MISMATCH");
  return pass ? 0 : 1;
}

}  // namespace tilemega::codegen
