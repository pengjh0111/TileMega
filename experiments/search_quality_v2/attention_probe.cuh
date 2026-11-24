// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gemm_probe.cuh"
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>

namespace tilemega::sq2 {
using Marks = backend::Sm90DenseMarks;
using Async = codegen::executor::Async<arch::Sm90>;

template<class A, int InboxWriters>
__global__ __launch_bounds__(A::kThreads, 1)
void AttentionProbe(backend::Sm90DenseAttentionOperands p, Trace* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto& trace = stamps[blockIdx.x];
  auto* ring = reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  if (threadIdx.x == 0) {
    trace.entry = Marks::Now();
    *reinterpret_cast<Stamp**>(shared + backend::Sm90DenseOperandHeader::kTraceSlot) = nullptr;
  }
  ring->Initialize();
  if constexpr (InboxWriters) {
    auto& cluster = backend::Sm90DenseClusterState::Of(shared);
    if (threadIdx.x == 0) {
      Async::Init(&cluster.inbox, InboxWriters);
      cluster.inbox_generation = 0;
      Async::InitFence();
    }
    __syncthreads();
    Async::ClusterSync();
  }
  if (threadIdx.x == 0) trace.body_begin = Marks::Now();
  A::Run(p, shared, ring, backend::Sm90DenseNoTail{}, false);
  if (threadIdx.x == 0) trace.body_end = Marks::Now();
  if constexpr (InboxWriters) Async::ClusterSync();
  ring->Invalidate();
}

template<class A, int Hkv, int D, int Q, int Chunks>
__global__ __launch_bounds__(A::kThreads, 1)
void MergeProbe(backend::Sm90DenseAttentionOperands p, Trace* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto* weight = reinterpret_cast<float*>(shared + 1024);
  auto& trace = stamps[blockIdx.x];
  if (threadIdx.x == 0) trace.body_begin = Marks::Now();
  if (threadIdx.x < 128) {
    for (int bg = blockIdx.x; bg < Hkv; bg += gridDim.x) {
      for (int i = threadIdx.x; i < Chunks * Q; i += 128)
        weight[i] = p.lse[std::size_t(bg) * Chunks * Q + i];
      codegen::executor::ComputeSync();
      A::Combine(p, weight, bg, [&] (int chunk, int i) {
        auto offset = (std::size_t(bg) * Chunks + chunk) * Q * D + i;
        return *reinterpret_cast<float4 const*>(p.partial + offset);
      });
      codegen::executor::ComputeSync();
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) trace.body_end = Marks::Now();
}

inline Stamp Makespan(Trace const* traces, int grid, bool body) {
  Stamp begin = ~0ULL, end = 0;
  for (int cta = 0; cta < grid; ++cta) {
    auto const& trace = traces[cta];
    auto entered = body ? trace.body_begin : trace.entry;
    if (!entered || trace.body_end < entered) throw std::runtime_error("invalid attention probe trace");
    begin = std::min(begin, entered);
    end = std::max(end, trace.body_end);
  }
  return end - begin;
}

template<int Hkv, int D, int Q, int Norm, int Chunks, int Keys, int Slots,
         int NewRow, bool Dsmem, bool StageMerge>
void AttentionCase(Buffers& buffers, float epsilon, int shared, char const* id) {
  constexpr int cap = 1088, grid = 132;
  using A = backend::Sm90DenseAttentionBody<1, Hkv, D, Q, Norm, Chunks, Keys,
                                           Slots, NewRow, cap, Dsmem, StageMerge>;
  using MergeBody = backend::Sm90DenseAttentionBody<1, Hkv, D, Q, 0, Chunks, 64, 2, 0, cap>;
  constexpr int n_qkv = Hkv * (Q + 2) * D;
  constexpr std::size_t cache_elements = std::size_t(D) * cap * Hkv;
  Device<BF16> qkv(n_qkv), key(cache_elements), val(cache_elements), y(Hkv * Q * D);
  Device<BF16> cos(cap * D), sin(cap * D), qnorm(D), knorm(D);
  Device<float> partial(Hkv * Chunks * Q * D), lse(Hkv * Chunks * Q);
  Device<std::uint64_t> arrivals(Hkv);
  Device<Trace> stamps(grid), merge_stamps(grid);
  Device<CUtensorMap> maps(2);
  Fill<<<1, 256>>>(qkv.get(), n_qkv);
  Fill<<<grid, 256>>>(key.get(), cache_elements);
  Fill<<<grid, 256>>>(val.get(), cache_elements);
  Fill<<<grid, 256>>>(cos.get(), cap * D);
  Fill<<<grid, 256>>>(sin.get(), cap * D);
  Fill<<<1, 256>>>(qnorm.get(), D);
  Fill<<<1, 256>>>(knorm.get(), D);
  CUtensorMap descriptors[2]{};
  codegen::executor::TiledMapShape<4> layout{
      {D, cap, Hkv, 1}, {2ULL * D, 2ULL * D * cap, 2ULL * cache_elements}, {64, Keys, 1, 1}};
  codegen::executor::EncodeTiledMap<4>(descriptors[0], key.get(), layout);
  codegen::executor::EncodeTiledMap<4>(descriptors[1], val.get(), layout);
  Check(cudaMemcpy(maps.get(), descriptors, sizeof descriptors, cudaMemcpyHostToDevice));
  backend::Sm90DenseAttentionOperands p{maps.get(), maps.get() + 1, qkv.get(), key.get(), val.get(),
      y.get(), cos.get(), sin.get(), qnorm.get(), knorm.get(), partial.get(), lse.get(), arrivals.get(),
      575, epsilon, 0};
  if (shared < A::kSharedBytes) throw std::runtime_error("attention probe storage exceeds opt-in");
  auto kernel = AttentionProbe<A, Dsmem ? Chunks - 1 : 0>;
  Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  if constexpr (StageMerge)
    Check(cudaFuncSetAttribute(MergeProbe<MergeBody, Hkv, D, Q, Chunks>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  int active = 0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, A::kThreads, shared));
  if (active != 1) throw std::runtime_error("attention probe must place one CTA per SM");
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(grid);
  launch.blockDim = dim3(A::kThreads);
  launch.dynamicSmemBytes = shared;
  cudaLaunchAttribute cluster{};
  if constexpr (Dsmem) {
    cluster.id = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim.x = Chunks;
    cluster.val.clusterDim.y = 1;
    cluster.val.clusterDim.z = 1;
    launch.attrs = &cluster;
    launch.numAttrs = 1;
  }
  std::vector<Trace> host(grid), host_merge(grid);
  for (int past : {64, 320, 575, 831, 1086}) {
    p.past = past;
    for (int repeat = 0; repeat < 17; ++repeat) {
      Check(cudaMemset(arrivals.get(), 0, Hkv * sizeof(std::uint64_t)));
      buffers.ColdWeightsWarmActivation(qkv.get(), n_qkv);
      Check(cudaEventRecord(buffers.start));
      if constexpr (Dsmem) Check(cudaLaunchKernelEx(&launch, kernel, p, stamps.get()));
      else {
        kernel<<<grid, A::kThreads, shared>>>(p, stamps.get());
        Check(cudaGetLastError());
      }
      if constexpr (StageMerge) {
        MergeProbe<MergeBody, Hkv, D, Q, Chunks><<<grid, A::kThreads, shared>>>(p, merge_stamps.get());
        Check(cudaGetLastError());
      }
      Check(cudaEventRecord(buffers.end));
      Check(cudaMemcpy(host.data(), stamps.get(), grid * sizeof(Trace), cudaMemcpyDeviceToHost));
      float event_ms = 0;
      Check(cudaEventElapsedTime(&event_ms, buffers.start, buffers.end));
      Stamp merge_ns = 0;
      if constexpr (StageMerge) {
        Check(cudaMemcpy(host_merge.data(), merge_stamps.get(), grid * sizeof(Trace), cudaMemcpyDeviceToHost));
        merge_ns = Makespan(host_merge.data(), grid, true);
      }
      if (repeat >= 5) {
        Stamp body = Makespan(host.data(), grid, true);
        std::printf("%s,%d,%d,%d,%llu,%llu,%llu,%llu,%.6f,%zu\n", id, past, repeat, grid,
            Makespan(host.data(), grid, false), body, merge_ns, body + merge_ns,
            event_ms * 1e6, buffers.eviction_words * 4);
        std::fflush(stdout);
      }
    }
  }
}
}  // namespace tilemega::sq2
