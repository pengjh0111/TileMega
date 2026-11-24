// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#define Sm90DenseMark Sm90DenseOriginalMark
#define Sm90DenseMarkFirst Sm90DenseOriginalMarkFirst
#include <tilemega/Backend/Sm90DenseMark.cuh>
#undef Sm90DenseMark
#undef Sm90DenseMarkFirst

#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tilemega::sq2 {
using Stamp = unsigned long long;
using BF16 = cutlass::bfloat16_t;

struct Trace {
  static constexpr int kMaxTasks = 32;
  Stamp entry, body_begin, body_end;
  Stamp task_complete[kMaxTasks];
  unsigned count;
};
}  // namespace tilemega::sq2

namespace tilemega::backend {
// Bind the existing epilogue marker to a calibration-only ledger. The
// production header and the measurement binary use their original bodies.
__device__ __forceinline__ void Sm90DenseMark(int index) {
  if (index != 2) return;
  auto* trace = reinterpret_cast<sq2::Trace*>(Sm90DenseMarks::Base());
  if (trace) {
    auto n = trace->count++;
    if (n < sq2::Trace::kMaxTasks) trace->task_complete[n] = Sm90DenseMarks::Now();
  }
}
__device__ __forceinline__ void Sm90DenseMarkFirst(int) {}
}  // namespace tilemega::backend

#include <tilemega/Backend/Sm90DenseGemm.h>

namespace tilemega::sq2 {
inline void Check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

template<class T>
class Device {
 public:
  explicit Device(std::size_t count) { Check(cudaMalloc(&ptr_, count * sizeof(T))); }
  ~Device() { cudaFree(ptr_); }
  Device(Device const&) = delete;
  Device& operator=(Device const&) = delete;
  T* get() const { return ptr_; }
 private:
  T* ptr_ = nullptr;
};

__global__ void Fill(BF16* data, std::size_t count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    unsigned bits = unsigned(i) * 2654435761u + 17u;
    data[i] = BF16(float(int(bits >> 24) - 128) / 4096.0f);
  }
}

__global__ void Touch(unsigned const* data, std::size_t count, unsigned* sink) {
  unsigned sum = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x)
    sum += data[i];
  if (sum) atomicXor(sink, sum);
}

struct Buffers {
  explicit Buffers(cudaDeviceProp const& prop)
      : eviction_words((std::size_t(prop.l2CacheSize) * 3 + 3) / 4),
        eviction(eviction_words), sink(1) {
    Check(cudaMemset(eviction.get(), 0x15, eviction_words * 4));
    Check(cudaMemset(sink.get(), 0, 4));
    Check(cudaEventCreate(&start));
    Check(cudaEventCreate(&end));
  }
  ~Buffers() { cudaEventDestroy(start); cudaEventDestroy(end); }
  void ColdWeightsWarmActivation(BF16 const* activation, unsigned elements) {
    Touch<<<264, 256>>>(eviction.get(), eviction_words, sink.get());
    Touch<<<1, 256>>>(reinterpret_cast<unsigned const*>(activation), elements / 2, sink.get());
  }
  std::size_t eviction_words;
  Device<unsigned> eviction, sink;
  cudaEvent_t start, end;
};

template<class G, int Residency>
__global__ __launch_bounds__(G::kThreads, Residency)
void GemmProbe(backend::Sm90DenseOperands operand, Trace* traces) {
  extern __shared__ __align__(1024) char shared[];
  auto* trace = traces + blockIdx.x;
  if (threadIdx.x == 0) {
    trace->entry = backend::Sm90DenseMarks::Now();
    trace->count = 0;
    *reinterpret_cast<Stamp**>(shared + backend::Sm90DenseOperandHeader::kTraceSlot) =
        reinterpret_cast<Stamp*>(trace);
  }
  auto* ring = reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  ring->Initialize();
  if (threadIdx.x == 0) trace->body_begin = backend::Sm90DenseMarks::Now();
  G::Run(operand, shared, ring, backend::Sm90DenseNoTail{}, false);
  if (threadIdx.x == 0) trace->body_end = backend::Sm90DenseMarks::Now();
  ring->Invalidate();
}

inline double Median(std::vector<Stamp>& values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  return (double(values[(values.size() - 1) / 2]) + double(values[values.size() / 2])) / 2;
}

template<int N, int K, int TN, int TK, int Slots, int WN, int WK, int Reg,
         int Residency, backend::ServingEpilogueOp Op, int U, bool Norm>
void GemmCase(Buffers& buffers, float epsilon, int shared, char const* id) {
  using G = backend::Sm90DenseGemm<1, N, K, TN, TK, Slots, WN, WK, Op, U, Norm, Reg>;
  constexpr int grid = 132 * Residency, tasks = N / TN;
  static_assert((tasks + grid - 1) / grid <= Trace::kMaxTasks);
  if (shared < G::kSharedBytes) throw std::invalid_argument("shared request smaller than GEMM storage");
  auto kernel = GemmProbe<G, Residency>;
  Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  int resident = 0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, G::kThreads, shared));
  if (resident < Residency) throw std::runtime_error("probe cannot realize requested grid occupancy");
  Device<BF16> w(std::size_t(N) * K), x(K), residual(N), y(N);
  Device<CUtensorMap> map(1);
  Device<Trace> stamps(grid);
  unsigned points = 264;
  Fill<<<points, 256>>>(w.get(), std::size_t(N) * K);
  Fill<<<points, 256>>>(x.get(), K);
  Fill<<<points, 256>>>(residual.get(), N);
  CUtensorMap tensor{};
  codegen::executor::EncodeTiledMap<3>(tensor, x.get(), {{64, 1, K / 64},
      {2ULL * K, 128}, {64, G::kRows, TK / 64}});
  Check(cudaMemcpy(map.get(), &tensor, sizeof tensor, cudaMemcpyHostToDevice));
  backend::Sm90DenseOperands operand{map.get(), w.get(), residual.get(), y.get(), nullptr, epsilon};
  std::vector<Trace> rows(grid);
  for (int repeat = 0; repeat < 17; ++repeat) {
    buffers.ColdWeightsWarmActivation(x.get(), K);
    Check(cudaEventRecord(buffers.start));
    kernel<<<grid, G::kThreads, shared>>>(operand, stamps.get());
    Check(cudaGetLastError());
    Check(cudaEventRecord(buffers.end));
    Check(cudaMemcpy(rows.data(), stamps.get(), rows.size() * sizeof(Trace), cudaMemcpyDeviceToHost));
    float event_ms = 0;
    Check(cudaEventElapsedTime(&event_ms, buffers.start, buffers.end));
    Stamp entry = ~0ULL, body = ~0ULL, end = 0;
    std::vector<Stamp> heads, body_heads, steady;
    std::vector<double> bandwidth;
    for (int cta = 0; cta < grid; ++cta) {
      auto const& row = rows[cta];
      unsigned expected = tasks / grid + unsigned(cta < tasks % grid);
      if (row.count != expected || row.body_end < row.body_begin || row.body_begin < row.entry)
        throw std::runtime_error("invalid probe trace");
      if (!expected) continue;
      entry = std::min(entry, row.entry);
      body = std::min(body, row.body_begin);
      end = std::max(end, row.body_end);
      if (row.task_complete[0] < row.body_begin || row.task_complete[expected - 1] > row.body_end)
        throw std::runtime_error("epilogue trace falls outside the body");
      heads.push_back(row.task_complete[0] - row.entry);
      body_heads.push_back(row.task_complete[0] - row.body_begin);
      for (unsigned t = 1; t < expected; ++t) {
        if (row.task_complete[t] < row.task_complete[t - 1])
          throw std::runtime_error("epilogue marks are out of order");
        steady.push_back(row.task_complete[t] - row.task_complete[t - 1]);
      }
      if (row.body_end > row.body_begin)
        bandwidth.push_back(2.0 * expected * TN * K / (row.body_end - row.body_begin));
    }
    std::sort(bandwidth.begin(), bandwidth.end());
    if (repeat >= 5) {
      std::printf("%s,%d,%d,%d,%.6f,%.6f,%.6f,%llu,%llu,%.6f,%.6f,%zu\n", id, repeat,
          tasks, grid, Median(heads), Median(body_heads), Median(steady), end - entry, end - body,
          event_ms * 1e6, bandwidth.at(bandwidth.size() / 2), buffers.eviction_words * 4);
      std::fflush(stdout);
    }
  }
}
}  // namespace tilemega::sq2
