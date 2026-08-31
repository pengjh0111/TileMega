// V-B: invoke CUTLASS's SM80 cp.async collective directly from a custom
// persistent kernel. GemmUniversal is instantiated only as the comparison.
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

// CUTLASS main currently exposes the SM80 CollectiveMma construction through
// this porting configuration helper, not CollectiveBuilder (V-B finding).
#include <unit/gemm/device/default_gemm_configuration.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float>;
using Mainloop = Config::CollectiveMainloop;
using Epilogue = Config::CollectiveEpilogue;
using ProblemShape = cute::Shape<int, int, int, int>;
using BaselineKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Mainloop, Epilogue>;
using Baseline = cutlass::gemm::device::GemmUniversalAdapter<BaselineKernel>;

struct PersistentParams {
  ProblemShape problem;
  Mainloop::Params mainloop;
  Epilogue::Params epilogue;
  int tiles_m;
  int tiles_n;
  int* next_tile;
};

__global__ __launch_bounds__(128)
void collective_persistent(PersistentParams params) {
  using namespace cute;
  extern __shared__ char smem[];
  __shared__ int tile_id;
  constexpr auto tile_shape = typename Mainloop::TileShape{};
  constexpr int kTileM = size<0>(tile_shape);
  constexpr int kTileN = size<1>(tile_shape);

  auto [M, N, K, L] = params.problem;
  Tensor matrix_a = make_tensor(make_gmem_ptr(params.mainloop.ptr_A),
                                make_shape(M, K, L), params.mainloop.dA);
  Tensor matrix_b = make_tensor(make_gmem_ptr(params.mainloop.ptr_B),
                                make_shape(N, K, L), params.mainloop.dB);

  while (true) {
    if (threadIdx.x == 0) tile_id = atomicAdd(params.next_tile, 1);
    __syncthreads();
    int tile = tile_id;
    if (tile >= params.tiles_m * params.tiles_n) return;
    int m_coord = tile / params.tiles_n;
    int n_coord = tile % params.tiles_n;
    auto block_coord = make_coord(m_coord, n_coord, _, 0);

    Tensor gA = local_tile(matrix_a(_, _, 0), tile_shape,
                           take<0, 3>(block_coord), Step<_1, X, _1>{});
    Tensor gB = local_tile(matrix_b(_, _, 0), tile_shape,
                           take<0, 3>(block_coord), Step<X, _1, _1>{});
    int m_residue = M - size<0>(gA) * m_coord;
    int n_residue = N - size<0>(gB) * n_coord;
    int k_residue = K - size<1>(gA) * size<2>(gA);
    auto residue = make_tuple(m_residue, n_residue, k_residue);

    typename Mainloop::TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(tile_shape));
    clear(accum);
    auto k_iter = make_coord_iterator(shape<2>(gA));
    int k_tiles = size<2>(gA);

    Mainloop collective;
    collective(accum, gA, gB, accum, k_iter, k_tiles, residue,
               static_cast<int>(threadIdx.x), smem);

    Epilogue epilogue(params.epilogue);
    epilogue(params.problem, tile_shape, make_coord(m_coord, n_coord, 0, 0),
             accum, tiled_mma, residue, static_cast<int>(threadIdx.x), smem);
    __syncthreads();
  }
}

static float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main(int argc, char** argv) {
  int M = 2048, N = 2048, K = 2048, repeats = 20;
  if (argc > 1) M = N = K = std::atoi(argv[1]);
  if (argc > 2) repeats = std::atoi(argv[2]);
  if (M % 128 || N % 128 || K % 32) {
    std::fprintf(stderr, "dimensions must be multiples of 128,128,32\n");
    return 2;
  }

  cutlass::half_t *a = nullptr, *b = nullptr;
  float *c = nullptr, *d_custom = nullptr, *d_baseline = nullptr;
  int* counter = nullptr;
  CUDA_CHECK(cudaMalloc(&a, sizeof(*a) * static_cast<size_t>(M) * K));
  CUDA_CHECK(cudaMalloc(&b, sizeof(*b) * static_cast<size_t>(K) * N));
  CUDA_CHECK(cudaMalloc(&c, sizeof(*c) * static_cast<size_t>(M) * N));
  CUDA_CHECK(cudaMalloc(&d_custom, sizeof(*d_custom) * static_cast<size_t>(M) * N));
  CUDA_CHECK(cudaMalloc(&d_baseline, sizeof(*d_baseline) * static_cast<size_t>(M) * N));
  CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
  CUDA_CHECK(cudaMemset(a, 0x3c, sizeof(*a) * static_cast<size_t>(M) * K));
  CUDA_CHECK(cudaMemset(b, 0x38, sizeof(*b) * static_cast<size_t>(K) * N));
  CUDA_CHECK(cudaMemset(c, 0, sizeof(*c) * static_cast<size_t>(M) * N));

  ProblemShape problem{M, N, K, 1};
  auto stride_a = cutlass::make_cute_packed_stride(
      typename Mainloop::StrideA{}, cute::make_shape(M, K, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      typename Mainloop::StrideB{}, cute::make_shape(N, K, 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      typename Epilogue::StrideC{}, cute::make_shape(M, N, 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      typename Epilogue::StrideD{}, cute::make_shape(M, N, 1));

  typename Mainloop::Arguments main_args{a, stride_a, b, stride_b};
  typename Epilogue::Arguments custom_epi{{1.0f, 0.0f}, c, stride_c,
                                           d_custom, stride_d};
  PersistentParams params{problem, Mainloop::to_underlying_arguments(problem, main_args, nullptr),
                          Epilogue::to_underlying_arguments(problem, custom_epi, nullptr),
                          M / 128, N / 128, counter};

  typename Baseline::Arguments baseline_args{
      cutlass::gemm::GemmUniversalMode::kGemm, problem,
      {a, stride_a, b, stride_b},
      {{1.0f, 0.0f}, c, stride_c, d_baseline, stride_d}};
  Baseline baseline;
  auto status = baseline.initialize(baseline_args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "baseline initialize failed: %d\n", static_cast<int>(status));
    return 2;
  }

  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  int blocks_per_sm = 0;
  constexpr int smem_bytes = sizeof(Mainloop::SharedStorage);
  CUDA_CHECK(cudaFuncSetAttribute(collective_persistent,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, collective_persistent, 128, smem_bytes));
  int grid = std::min(params.tiles_m * params.tiles_n, sms * blocks_per_sm);

  auto launch_custom = [&] {
    CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(int)));
    collective_persistent<<<grid, 128, smem_bytes>>>(params);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_baseline = [&] {
    if (baseline.run() != cutlass::Status::kSuccess) std::exit(2);
  };
  launch_custom(); launch_baseline(); CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> custom_ms, baseline_ms;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start)); launch_custom(); CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); custom_ms.push_back(ms);
    CUDA_CHECK(cudaEventRecord(start)); launch_baseline(); CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); baseline_ms.push_back(ms);
  }

  std::vector<float> host_custom(static_cast<size_t>(M) * N);
  std::vector<float> host_baseline(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(host_custom.data(), d_custom, sizeof(float) * host_custom.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), d_baseline, sizeof(float) * host_baseline.size(), cudaMemcpyDeviceToHost));
  size_t mismatch = 0;
  float max_abs = 0;
  for (size_t i = 0; i < host_custom.size(); ++i) {
    float diff = std::fabs(host_custom[i] - host_baseline[i]);
    if (diff > 1e-3f) ++mismatch;
    max_abs = std::max(max_abs, diff);
  }
  float custom = median(custom_ms), base = median(baseline_ms);
  std::printf("V_B M=%d N=%d K=%d repeats=%d grid=%d ctas_per_sm=%d smem=%d "
              "custom_ms=%.6f baseline_ms=%.6f ratio=%.6f mismatch=%zu max_abs=%.6g\n",
              M, N, K, repeats, grid, blocks_per_sm, smem_bytes,
              custom, base, custom / base, mismatch, max_abs);
  return mismatch ? 1 : 0;
}
