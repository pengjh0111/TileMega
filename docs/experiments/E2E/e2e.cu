// L0.5/L1 correctness ladder for the V-H two-layer Llama ExportedProgram.
#include <cuda_runtime.h>

#include <tilemega/Target/TargetSpec.h>

#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <unit/gemm/device/default_gemm_configuration.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

namespace {
constexpr int kLayers = 2;
constexpr int kSeq = 4;
constexpr int kPast = 3;
constexpr int kTotal = kSeq + kPast;
constexpr int kHidden = 512;
constexpr int kIntermediate = 1024;
constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 128;
constexpr int kStagesPerLayer = 12;
constexpr int kStages = kLayers * kStagesPerLayer;

// The same direct-collective adapter family validated by V-B. FP32 SIMT is
// selected here so the E2E numerical oracle remains the FP32 ExportedProgram;
// CUTLASS still selects MainloopSm80CpAsync<3> for this configuration.
using GemmConfig = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    float, cutlass::layout::RowMajor,
    // CUTLASS's logical B tensor is (N,K); ColumnMajor here gives (K,1)
    // logical strides and therefore consumes PyTorch's contiguous [N,K]
    // weight storage without a host transpose.
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float>;
using GemmMainloop = GemmConfig::CollectiveMainloop;
using GemmEpilogue = GemmConfig::CollectiveEpilogue;
using GemmProblem = cute::Shape<int, int, int, int>;
constexpr int kThreads = GemmConfig::ThreadCount;
constexpr int kGemmTileN = cute::size<1>(typename GemmMainloop::TileShape{});

struct GemmInvocation {
  GemmProblem problem;
  GemmMainloop::Params mainloop;
  GemmEpilogue::Params epilogue;
  int tiles_n;
};

struct LayerWeights {
  float const* input_norm;
  float const* post_norm;
  float const* q;
  float const* k;
  float const* v;
  float const* o;
  float const* gate;
  float const* up;
  float const* down;
  float const* inv_freq;
};

struct Params {
  float* hidden[kLayers + 1];
  float* norm;
  float* q;
  float* k;
  float* v;
  float* q_rot;
  float* k_rot;
  float* context;
  float* gate;
  float* up;
  float const* past_k[kLayers];
  float const* past_v[kLayers];
  float* full_k[kLayers];
  float* full_v[kLayers];
  LayerWeights weights[kLayers];
  GemmInvocation gemms[kLayers * 7];
};

union TaskSmem {
  float rms[kThreads];
  float attention[kThreads];
  float pointwise[1];
  GemmMainloop::SharedStorage gemm;
  float rope[1];
};

struct alignas(128) EventCounter {
  unsigned long long arrivals;
  unsigned long long epoch;
  unsigned char padding[112];
};
static_assert(sizeof(EventCounter) == 128, "event cache-line padding");

__device__ void RmsNorm(float const* input, float const* weight, float* output,
                        TaskSmem& storage) {
  int token = static_cast<int>(blockIdx.x);
  bool active = token < kSeq;
  float local = 0.0f;
  if (active) {
    for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
      float value = input[token * kHidden + d];
      local += value * value;
    }
  }
  storage.rms[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset; offset /= 2) {
    if (threadIdx.x < offset)
      storage.rms[threadIdx.x] += storage.rms[threadIdx.x + offset];
    __syncthreads();
  }
  float scale = rsqrtf(storage.rms[0] / kHidden + 1.0e-6f);
  if (active) {
    for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
      output[token * kHidden + d] =
          input[token * kHidden + d] * scale * weight[d];
    }
  }
  __syncthreads();
}

__device__ void Linear(GemmInvocation const& invocation, TaskSmem& storage) {
  using namespace cute;
  int tile_n = static_cast<int>(blockIdx.x);
  if (tile_n >= invocation.tiles_n) return;
  constexpr auto tile_shape = typename GemmMainloop::TileShape{};
  auto [M, N, K, L] = invocation.problem;
  Tensor matrix_a = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_A),
                                make_shape(M, K, L), invocation.mainloop.dA);
  Tensor matrix_b = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_B),
                                make_shape(N, K, L), invocation.mainloop.dB);
  auto block_coord = make_coord(0, tile_n, _, 0);
  Tensor gA = local_tile(matrix_a(_, _, 0), tile_shape,
                         take<0, 3>(block_coord), Step<_1, X, _1>{});
  Tensor gB = local_tile(matrix_b(_, _, 0), tile_shape,
                         take<0, 3>(block_coord), Step<X, _1, _1>{});
  auto residue = make_tuple(M, N - size<0>(gB) * tile_n,
                            K - size<1>(gA) * size<2>(gA));
  typename GemmMainloop::TiledMma tiled_mma;
  Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(tile_shape));
  clear(accum);
  auto k_iter = make_coord_iterator(shape<2>(gA));
  char* shared = reinterpret_cast<char*>(&storage.gemm);
  GemmMainloop mainloop;
  mainloop(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
           static_cast<int>(threadIdx.x), shared);
  GemmEpilogue epilogue(invocation.epilogue);
  epilogue(invocation.problem, tile_shape, make_coord(0, tile_n, 0, 0),
           accum, tiled_mma, residue, static_cast<int>(threadIdx.x),
           shared);
  __syncthreads();
}

__device__ void RopeAndCache(Params const& p, int layer) {
  auto const& w = p.weights[layer];
  int q_pairs = kSeq * kHeads * (kHeadDim / 2);
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < q_pairs;
       index += gridDim.x * blockDim.x) {
    int half = index % (kHeadDim / 2);
    int head_token = index / (kHeadDim / 2);
    int token = head_token / kHeads;
    int base = head_token * kHeadDim;
    float angle = (kPast + token) * w.inv_freq[half];
    float c = cosf(angle), s = sinf(angle);
    float a = p.q[base + half], b = p.q[base + half + kHeadDim / 2];
    p.q_rot[base + half] = a * c - b * s;
    p.q_rot[base + half + kHeadDim / 2] = b * c + a * s;
  }
  int k_pairs = kSeq * kKvHeads * (kHeadDim / 2);
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < k_pairs;
       index += gridDim.x * blockDim.x) {
    int half = index % (kHeadDim / 2);
    int kv_token = index / (kHeadDim / 2);
    int token = kv_token / kKvHeads;
    int kv = kv_token % kKvHeads;
    int source = (token * kKvHeads + kv) * kHeadDim;
    float angle = (kPast + token) * w.inv_freq[half];
    float c = cosf(angle), s = sinf(angle);
    float a = p.k[source + half], b = p.k[source + half + kHeadDim / 2];
    float ra = a * c - b * s, rb = b * c + a * s;
    int target = (kv * kTotal + kPast + token) * kHeadDim;
    p.k_rot[source + half] = ra;
    p.k_rot[source + half + kHeadDim / 2] = rb;
    p.full_k[layer][target + half] = ra;
    p.full_k[layer][target + half + kHeadDim / 2] = rb;
  }
  int past_elements = kKvHeads * kPast * kHeadDim;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < past_elements;
       index += gridDim.x * blockDim.x) {
    int d = index % kHeadDim;
    int temp = index / kHeadDim;
    int pos = temp % kPast;
    int kv = temp / kPast;
    p.full_k[layer][(kv * kTotal + pos) * kHeadDim + d] = p.past_k[layer][index];
    p.full_v[layer][(kv * kTotal + pos) * kHeadDim + d] = p.past_v[layer][index];
  }
  int current_elements = kSeq * kKvHeads * kHeadDim;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < current_elements; index += gridDim.x * blockDim.x) {
    int d = index % kHeadDim;
    int temp = index / kHeadDim;
    int kv = temp % kKvHeads;
    int token = temp / kKvHeads;
    p.full_v[layer][(kv * kTotal + kPast + token) * kHeadDim + d] = p.v[index];
  }
}

__device__ void Attention(Params const& p, int layer, TaskSmem& storage) {
  int query = static_cast<int>(blockIdx.x);
  bool active = query < kSeq * kHeads;
  int token = query / kHeads;
  int head = query % kHeads;
  int kv = head / (kHeads / kKvHeads);
  if (active && threadIdx.x < kTotal) {
    int key_pos = threadIdx.x;
    float score = -INFINITY;
    if (key_pos <= kPast + token) {
      score = 0.0f;
      int qbase = (token * kHeads + head) * kHeadDim;
      int kbase = (kv * kTotal + key_pos) * kHeadDim;
      for (int d = 0; d < kHeadDim; ++d)
        score = fmaf(p.q_rot[qbase + d], p.full_k[layer][kbase + d], score);
      score /= sqrtf(static_cast<float>(kHeadDim));
    }
    storage.attention[threadIdx.x] = score;
  }
  __syncthreads();
  if (active && threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int j = 0; j < kTotal; ++j) maximum = fmaxf(maximum, storage.attention[j]);
    float sum = 0.0f;
    for (int j = 0; j < kTotal; ++j) {
      float value = expf(storage.attention[j] - maximum);
      storage.attention[j] = value;
      sum += value;
    }
    for (int j = 0; j < kTotal; ++j) storage.attention[j] /= sum;
  }
  __syncthreads();
  if (active) {
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
      float value = 0.0f;
      for (int j = 0; j < kTotal; ++j) {
        int vbase = (kv * kTotal + j) * kHeadDim;
        value = fmaf(storage.attention[j], p.full_v[layer][vbase + d], value);
      }
      p.context[(token * kHeads + head) * kHeadDim + d] = value;
    }
  }
  __syncthreads();
}

__device__ void SiluMultiply(float* gate, float const* up) {
  int count = kSeq * kIntermediate;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    float x = gate[i];
    gate[i] = (x / (1.0f + expf(-x))) * up[i];
  }
}

__device__ void RunStage(int stage, Params const& p, TaskSmem& storage) {
  int layer = stage / kStagesPerLayer;
  int gemm = layer * 7;
  switch (stage % kStagesPerLayer) {
    case 0: RmsNorm(p.hidden[layer], p.weights[layer].input_norm, p.norm, storage); break;
    case 1: Linear(p.gemms[gemm + 0], storage); break;
    case 2: Linear(p.gemms[gemm + 1], storage); break;
    case 3: Linear(p.gemms[gemm + 2], storage); break;
    case 4: RopeAndCache(p, layer); break;
    case 5: Attention(p, layer, storage); break;
    case 6: Linear(p.gemms[gemm + 3], storage); break;
    case 7: RmsNorm(p.hidden[layer + 1], p.weights[layer].post_norm, p.norm, storage); break;
    case 8: Linear(p.gemms[gemm + 4], storage); break;
    case 9: Linear(p.gemms[gemm + 5], storage); break;
    case 10: SiluMultiply(p.gate, p.up); break;
    case 11: Linear(p.gemms[gemm + 6], storage); break;
  }
}

__device__ void GridBarrier(EventCounter* events, int stage) {
  // Every writer performs the release fence before CTA convergence (F-1).
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long ticket = atomicAdd(&events[stage].arrivals, 1ull);
    if (ticket + 1 == static_cast<unsigned long long>(gridDim.x)) {
      __threadfence();
      atomicExch(&events[stage].epoch, 1ull);
    } else {
      while (atomicAdd(&events[stage].epoch, 0ull) < 1ull) __nanosleep(64);
    }
  }
  __syncthreads();
  __threadfence();
}

__global__ __launch_bounds__(kThreads, 1)
void stage_kernel(int stage, Params const* params) {
  extern __shared__ unsigned char bytes[];
  auto& storage = *reinterpret_cast<TaskSmem*>(bytes);
  RunStage(stage, *params, storage);
}

__global__ __launch_bounds__(kThreads, 1)
void l1_kernel(Params const* params, EventCounter* events) {
  extern __shared__ unsigned char bytes[];
  auto& storage = *reinterpret_cast<TaskSmem*>(bytes);
  // Fixed model schedule: unroll so stage dispatch is compile-time in L1.
  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    RunStage(stage, *params, storage);
    GridBarrier(events, stage);
  }
}

std::vector<float> Load(std::string const& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
  std::vector<float> value(count);
  input.read(reinterpret_cast<char*>(value.data()), count * sizeof(float));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
    std::fprintf(stderr, "wrong fixture size: %s\n", path.c_str()); std::exit(2);
  }
  return value;
}

float* Upload(std::vector<float> const& host) {
  float* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  return device;
}

float* Allocate(std::size_t count) {
  float* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, count * sizeof(float)));
  CUDA_CHECK(cudaMemset(device, 0, count * sizeof(float)));
  return device;
}

struct HostFixture {
  std::string dir;
  std::vector<float> input;
  std::vector<float> cache[4];
  std::vector<float> reference[5];
};

struct DeviceModel {
  Params params{};
  Params* device_params = nullptr;
  EventCounter* events = nullptr;
  std::vector<void*> allocations;
};

std::string Join(std::string const& dir, std::string const& file) {
  return dir + "/" + file;
}

DeviceModel CreateDeviceModel(HostFixture const& fixture) {
  DeviceModel model;
  auto own_upload = [&](std::vector<float> const& value) {
    float* pointer = Upload(value); model.allocations.push_back(pointer); return pointer;
  };
  auto own_alloc = [&](std::size_t count) {
    float* pointer = Allocate(count); model.allocations.push_back(pointer); return pointer;
  };
  model.params.hidden[0] = own_upload(fixture.input);
  model.params.hidden[1] = own_alloc(kSeq * kHidden);
  model.params.hidden[2] = own_alloc(kSeq * kHidden);
  model.params.norm = own_alloc(kSeq * kHidden);
  model.params.q = own_alloc(kSeq * kHidden);
  model.params.k = own_alloc(kSeq * kKvHeads * kHeadDim);
  model.params.v = own_alloc(kSeq * kKvHeads * kHeadDim);
  model.params.q_rot = own_alloc(kSeq * kHidden);
  model.params.k_rot = own_alloc(kSeq * kKvHeads * kHeadDim);
  model.params.context = own_alloc(kSeq * kHidden);
  model.params.gate = own_alloc(kSeq * kIntermediate);
  model.params.up = own_alloc(kSeq * kIntermediate);
  for (int layer = 0; layer < kLayers; ++layer) {
    model.params.past_k[layer] = own_upload(fixture.cache[layer * 2]);
    model.params.past_v[layer] = own_upload(fixture.cache[layer * 2 + 1]);
    model.params.full_k[layer] = own_alloc(kKvHeads * kTotal * kHeadDim);
    model.params.full_v[layer] = own_alloc(kKvHeads * kTotal * kHeadDim);
    std::string prefix = "state_layers_" + std::to_string(layer) + "_";
    auto weight = [&](std::string const& name, std::size_t count) {
      return own_upload(Load(Join(fixture.dir, prefix + name + ".bin"), count));
    };
    auto& w = model.params.weights[layer];
    w.input_norm = weight("input_norm_weight", kHidden);
    w.post_norm = weight("post_norm_weight", kHidden);
    w.q = weight("q_proj_weight", kHidden * kHidden);
    w.k = weight("k_proj_weight", kKvHeads * kHeadDim * kHidden);
    w.v = weight("v_proj_weight", kKvHeads * kHeadDim * kHidden);
    w.o = weight("o_proj_weight", kHidden * kHidden);
    w.gate = weight("gate_proj_weight", kIntermediate * kHidden);
    w.up = weight("up_proj_weight", kIntermediate * kHidden);
    w.down = weight("down_proj_weight", kHidden * kIntermediate);
    w.inv_freq = weight("inv_freq", kHeadDim / 2);

    auto make_gemm = [&](int slot, float const* a, float const* b,
                         float const* c, float* d, int n, int k,
                         float beta) {
      GemmProblem problem{kSeq, n, k, 1};
      auto stride_a = cutlass::make_cute_packed_stride(
          typename GemmMainloop::StrideA{}, cute::make_shape(kSeq, k, 1));
      auto stride_b = cutlass::make_cute_packed_stride(
          typename GemmMainloop::StrideB{}, cute::make_shape(n, k, 1));
      auto stride_c = cutlass::make_cute_packed_stride(
          typename GemmEpilogue::StrideC{}, cute::make_shape(kSeq, n, 1));
      auto stride_d = cutlass::make_cute_packed_stride(
          typename GemmEpilogue::StrideD{}, cute::make_shape(kSeq, n, 1));
      typename GemmMainloop::Arguments main_args{a, stride_a, b, stride_b};
      typename GemmEpilogue::Arguments epilogue_args{
          {1.0f, beta}, c, stride_c, d, stride_d};
      auto& invocation = model.params.gemms[layer * 7 + slot];
      invocation.problem = problem;
      invocation.mainloop =
          GemmMainloop::to_underlying_arguments(problem, main_args, nullptr);
      invocation.epilogue =
          GemmEpilogue::to_underlying_arguments(problem, epilogue_args, nullptr);
      invocation.tiles_n = (n + kGemmTileN - 1) / kGemmTileN;
    };
    make_gemm(0, model.params.norm, w.q, model.params.q,
              model.params.q, kHidden, kHidden, 0.0f);
    make_gemm(1, model.params.norm, w.k, model.params.k,
              model.params.k, kKvHeads * kHeadDim, kHidden, 0.0f);
    make_gemm(2, model.params.norm, w.v, model.params.v,
              model.params.v, kKvHeads * kHeadDim, kHidden, 0.0f);
    make_gemm(3, model.params.context, w.o, model.params.hidden[layer],
              model.params.hidden[layer + 1], kHidden, kHidden, 1.0f);
    make_gemm(4, model.params.norm, w.gate, model.params.gate,
              model.params.gate, kIntermediate, kHidden, 0.0f);
    make_gemm(5, model.params.norm, w.up, model.params.up,
              model.params.up, kIntermediate, kHidden, 0.0f);
    make_gemm(6, model.params.gate, w.down, model.params.hidden[layer + 1],
              model.params.hidden[layer + 1], kHidden, kIntermediate, 1.0f);
  }
  CUDA_CHECK(cudaMalloc(&model.events, sizeof(EventCounter) * kStages));
  CUDA_CHECK(cudaMalloc(&model.device_params, sizeof(Params)));
  CUDA_CHECK(cudaMemcpy(model.device_params, &model.params, sizeof(Params),
                        cudaMemcpyHostToDevice));
  return model;
}

void Reset(DeviceModel& model, HostFixture const& fixture) {
  CUDA_CHECK(cudaMemcpy(model.params.hidden[0], fixture.input.data(),
                        fixture.input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(model.params.hidden[1], 0, kSeq * kHidden * sizeof(float)));
  CUDA_CHECK(cudaMemset(model.params.hidden[2], 0, kSeq * kHidden * sizeof(float)));
  CUDA_CHECK(cudaMemset(model.events, 0, sizeof(EventCounter) * kStages));
}

float LaunchL05(DeviceModel& model, int grid) {
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int stage = 0; stage < kStages; ++stage)
    stage_kernel<<<grid, kThreads, sizeof(TaskSmem)>>>(stage, model.device_params);
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError()); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start); cudaEventDestroy(stop); return ms;
}

float LaunchL1(DeviceModel& model, int grid) {
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  l1_kernel<<<grid, kThreads, sizeof(TaskSmem)>>>(model.device_params, model.events);
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError()); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start); cudaEventDestroy(stop); return ms;
}

std::vector<std::vector<float>> Download(DeviceModel const& model) {
  std::vector<std::vector<float>> output(5);
  output[0].resize(kSeq * kHidden);
  CUDA_CHECK(cudaMemcpy(output[0].data(), model.params.hidden[2], output[0].size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int layer = 0; layer < kLayers; ++layer) {
    output[1 + layer * 2].resize(kKvHeads * kTotal * kHeadDim);
    output[2 + layer * 2].resize(kKvHeads * kTotal * kHeadDim);
    CUDA_CHECK(cudaMemcpy(output[1 + layer * 2].data(), model.params.full_k[layer], output[1 + layer * 2].size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output[2 + layer * 2].data(), model.params.full_v[layer], output[2 + layer * 2].size() * sizeof(float), cudaMemcpyDeviceToHost));
  }
  return output;
}

struct Difference { std::size_t mismatch = 0; float max_abs = 0; float max_rel = 0; };
Difference Compare(std::vector<std::vector<float>> const& actual,
                   std::vector<std::vector<float>> const& expected) {
  Difference result;
  for (std::size_t tensor = 0; tensor < actual.size(); ++tensor) {
    for (std::size_t i = 0; i < actual[tensor].size(); ++i) {
      float delta = std::fabs(actual[tensor][i] - expected[tensor][i]);
      float relative = delta / std::max(std::fabs(expected[tensor][i]), 1.0e-6f);
      if (delta > 3.0e-5f + 3.0e-5f * std::fabs(expected[tensor][i])) ++result.mismatch;
      result.max_abs = std::max(result.max_abs, delta);
      result.max_rel = std::max(result.max_rel, relative);
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: e2e FIXTURE_DIR\n"); return 2; }
  HostFixture fixture; fixture.dir = argv[1];
  fixture.input = Load(Join(fixture.dir, "input_hidden.bin"), kSeq * kHidden);
  for (int i = 0; i < 4; ++i)
    fixture.cache[i] = Load(Join(fixture.dir, "input_cache_" + std::to_string(i) + ".bin"), kKvHeads * kPast * kHeadDim);
  fixture.reference[0] = Load(Join(fixture.dir, "reference_hidden.bin"), kSeq * kHidden);
  for (int layer = 0; layer < kLayers; ++layer) {
    fixture.reference[1 + layer * 2] = Load(Join(fixture.dir, "reference_k" + std::to_string(layer) + ".bin"), kKvHeads * kTotal * kHeadDim);
    fixture.reference[2 + layer * 2] = Load(Join(fixture.dir, "reference_v" + std::to_string(layer) + ".bin"), kKvHeads * kTotal * kHeadDim);
  }

  DeviceModel model = CreateDeviceModel(fixture);
  auto target = tilemega::TargetSpec::Probe();
  CUDA_CHECK(cudaFuncSetAttribute(stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(TaskSmem)));
  CUDA_CHECK(cudaFuncSetAttribute(l1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(TaskSmem)));
  int blocks_per_sm = target.ActiveBlocksPerSM(reinterpret_cast<void const*>(l1_kernel), kThreads, sizeof(TaskSmem));
  int grid = blocks_per_sm * target.res.num_sms;

  Reset(model, fixture); float l05_ms = LaunchL05(model, grid); auto l05 = Download(model);
  Reset(model, fixture); float l1_ms = LaunchL1(model, grid); auto l1 = Download(model);
  Difference l05_l0 = Compare(l05, std::vector<std::vector<float>>(std::begin(fixture.reference), std::end(fixture.reference)));
  Difference l1_l05 = Compare(l1, l05);
  std::printf("E2E_RESOURCE block=%d reg=ptxas smem=%zu ctas_per_sm=%d num_sms=%d grid=%d resident_formula=ctas_per_sm*num_sms\n",
              kThreads, sizeof(TaskSmem), blocks_per_sm, target.res.num_sms, grid);
  std::printf("E2E_TIME l05_ms=%.6f l1_ms=%.6f ratio=%.6f\n", l05_ms, l1_ms, l1_ms / l05_ms);
  std::printf("E2E_DIFF l05_vs_l0_mismatch=%zu max_abs=%.8g max_rel=%.8g l1_vs_l05_mismatch=%zu max_abs=%.8g max_rel=%.8g\n",
              l05_l0.mismatch, l05_l0.max_abs, l05_l0.max_rel,
              l1_l05.mismatch, l1_l05.max_abs, l1_l05.max_rel);
  bool pass = l05_l0.mismatch == 0 && l1_l05.mismatch == 0;
  std::printf("RESULT status=%s\n", pass ? "PASS" : "MISMATCH");
  return pass ? 0 : 1;
}
