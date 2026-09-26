// SPDX-License-Identifier: BSD-3-Clause
// Isolated BF16 mainloop comparison; run only while the GPU is otherwise idle.
#include <tilemega/Backend/CutlassGemmCandidate.h>
#include <tilemega/Backend/ServingGemm.h>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Element = cutlass::bfloat16_t;
#ifndef TILEMEGA_BENCH_ARCH_ID
#error "Set TILEMEGA_BENCH_ARCH_ID from the target compute capability"
#endif
using Arch = tilemega::arch::ArchFromId<TILEMEGA_BENCH_ARCH_ID>::type;

__global__ void FillWeights(Element* data, std::size_t elements) {
  auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  std::uint32_t bits = std::uint32_t(index) * 0x9e3779b9u;
  bits ^= bits >> 16;
  bits *= 0x85ebca6bu;
  bits ^= bits >> 13;
  reinterpret_cast<std::uint16_t*>(data)[index] =
      std::uint16_t(0x3c00u | (bits & 0x03ffu));
}

template <class Mainloop, int TileM, int TileN, int TileK>
__global__ void Probe(Element const* a, Element const* b, int rows,
                      int reduction, int source_slot, float* output) {
  using namespace cute;
  extern __shared__ char shared[];
  b += (source_slot + int(blockIdx.x)) * TileN * reduction;
  auto a_tensor = make_tensor(make_gmem_ptr(a),
      make_shape(rows, reduction, 1),
      make_stride(int64_t(reduction), _1{}, int64_t(rows) * reduction));
  auto b_tensor = make_tensor(make_gmem_ptr(b),
      make_shape(TileN, reduction, 1),
      make_stride(int64_t(reduction), _1{}, int64_t(TileN) * reduction));
  constexpr auto tile = typename Mainloop::TileShape{};
  auto coordinate = make_coord(0, 0, _, 0);
  auto global_a = local_tile(a_tensor(_, _, 0), tile,
                             take<0, 3>(coordinate), Step<_1, X, _1>{});
  auto global_b = local_tile(b_tensor(_, _, 0), tile,
                             take<0, 3>(coordinate), Step<X, _1, _1>{});
  auto residue = make_tuple(rows, TileN,
                             reduction - size<1>(global_a) * size<2>(global_a));
  typename Mainloop::TiledMma mma;
  auto accum = partition_fragment_C(mma, take<0, 2>(tile));
  clear(accum);
  auto iterator = make_coord_iterator(shape<2>(global_a));
  Mainloop{}(accum, global_a, global_b, accum, iterator,
             size<2>(global_a), residue, int(threadIdx.x), shared);
  if (threadIdx.x == 0)
    output[blockIdx.x] = float(accum(0));
}

template <class Mainloop, int TileM, int TileN, int TileK>
void Measure(char const* kind, int rows, int reduction, int ctas,
             Element const* a, Element const* b, float* output,
             int shared_bytes, int slots) {
  if (shared_bytes > 48 * 1024) {
    auto status = cudaFuncSetAttribute(Probe<Mainloop, TileM, TileN, TileK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (status != cudaSuccess) std::exit(3);
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::vector<float> micros;
  for (int pass = 0; pass < slots + 8; ++pass) {
    // Rotate disjoint CTA windows so the weight footprint exceeds device L2.
    int offset = (pass * ctas) % slots;
    cudaEventRecord(start);
    Probe<Mainloop, TileM, TileN, TileK><<<ctas, 128, shared_bytes>>>(
        a, b, rows, reduction, offset, output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "%s failed: %s\n", kind, cudaGetErrorString(status));
      std::exit(4);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (pass >= 8) micros.push_back(milliseconds * 1000);
  }
  std::sort(micros.begin(), micros.end());
  double median_us = (micros[(micros.size() - 1) / 2] +
                      micros[micros.size() / 2]) / 2;
  double bytes = double(ctas) * TileN * reduction * sizeof(Element);
  double gbps = bytes / (median_us * 1000);
  std::printf("%s\t%d\t%d\t%d\t%.6f\t%.6f\n",
              kind, rows, reduction, ctas, median_us, gbps);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

}  // namespace

int main() {
  int max_ctas = 0;
  if (cudaDeviceGetAttribute(&max_ctas, cudaDevAttrMultiProcessorCount, 0) !=
      cudaSuccess || max_ctas < 1) return 2;
  int const slots = 4 * max_ctas;
  constexpr int max_n = 128;
  constexpr int max_k = 8192;
  Element *a = nullptr, *b = nullptr;
  float* output = nullptr;
  if (cudaMalloc(&a, 16 * max_k * sizeof(Element)) != cudaSuccess ||
      cudaMalloc(&b, slots * max_n * max_k * sizeof(Element)) != cudaSuccess ||
      cudaMalloc(&output, max_ctas * sizeof(float)) != cudaSuccess)
    return 2;
  cudaMemset(a, 0, 16 * max_k * sizeof(Element));
  std::size_t const weight_elements = std::size_t(slots) * max_n * max_k;
  FillWeights<<<(weight_elements + 255) / 256, 256>>>(b, weight_elements);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  using Old = tilemega::backend::TypedGemmCandidate<true, 32, 128, 64, 2,
                                                     Arch>::Mainloop;
  using New = tilemega::backend::ServingGemmConfig<Arch, 16, 128, 64, 2>::Mainloop;
  constexpr int old_shared = sizeof(typename Old::SharedStorage);
  constexpr int new_shared = sizeof(typename New::SharedStorage);
  std::puts("collective\tM\tK\tctas\tmedian_us\tweight_gbps");
  for (int rows : {1, 16})
    for (int reduction : {2048, 8192})
      for (int ctas : {1, max_ctas}) {
        Measure<Old, 32, 128, 64>("legacy", rows, reduction, ctas,
                                   a, b, output, old_shared, slots);
        Measure<New, 16, 128, 64>("serving", rows, reduction, ctas,
                                   a, b, output, new_shared, slots);
      }
  cudaFree(a);
  cudaFree(b);
  cudaFree(output);
}
