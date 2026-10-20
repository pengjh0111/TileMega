// SPDX-License-Identifier: BSD-3-Clause
// MX3-2 item 1: does a cross-block software pipeline take effect on this
// device, and at what depth? MX-1 §1.2 records that MACA inserts a
// wait-for-all before the first use of a loaded value when the author writes
// no explicit wait, which would make a register multi-buffer collapse back to
// the serial form. No MetaX disassembler ships with this install (checked:
// /opt/maca/bin, /opt/maca/mxgpu_llvm/bin have llvm-dis and llvm-nm but no
// objdump; `cucc -device-bin` emits an ELF whose machine id decodes as
// Synopsys ARCv2.3, which system objdump refuses), so the question is settled
// behaviourally instead: a pipeline that does not take effect shows neither a
// speedup over depth 1 nor any sensitivity to depth.
//
// The loop mirrors ServingGemmMma's shape: stream B tiles for a 16xNxK GEMM
// out of DRAM and accumulate, with nothing else competing.
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cstddef>

using tilemega::maca::GlobalTimerNs;
using tilemega::maca::AsyncCopy16;
using tilemega::maca::WaitAsyncCopies;

#define CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
  std::printf("%s: %s\n", #c, cudaGetErrorString(e)); return 2; } } while(0)

/// Serial: issue every load for a block, consume them, then move on. This is
/// the shape ServingGemmMma::Run has today (loads into registers, then MMA).
template <int kLoads>
__global__ void SerialKernel(uint4 const* source, int blocks, int stride,
                             unsigned long long* out, float* sink) {
  uint4 acc = {0, 0, 0, 0};
  uint4 const* at = source + (std::size_t(blockIdx.x) * blocks * kLoads * stride)
                   + threadIdx.x % 64;
  unsigned long long begin = GlobalTimerNs();
  for (int b = 0; b < blocks; ++b) {
    uint4 v[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) v[i] = at[(b * kLoads + i) * stride];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      acc.x += v[i].x; acc.y += v[i].y; acc.z += v[i].z; acc.w += v[i].w;
    }
  }
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = end - begin;
    *sink = float(acc.x + acc.y + acc.z + acc.w);
  }
}

/// Pipelined: keep `kDepth` blocks of loads in flight in registers, waiting
/// only for the oldest with an explicit arrive_gvmcnt. If the compiler
/// inserts its own wait-for-all before the first use, this degenerates to the
/// serial form and the depth sweep will be flat.
template <int kLoads, int kDepth>
__global__ void PipelinedKernel(uint4 const* source, int blocks, int stride,
                                unsigned long long* out, float* sink) {
  uint4 acc = {0, 0, 0, 0};
  uint4 const* at = source + (std::size_t(blockIdx.x) * blocks * kLoads * stride)
                   + threadIdx.x % 64;
  __shared__ uint4 ring[kDepth][kLoads][64];
  unsigned long long begin = GlobalTimerNs();
  // Prime the pipeline: kDepth-1 blocks in flight before the first consume.
#pragma unroll
  for (int d = 0; d < kDepth - 1; ++d)
#pragma unroll
    for (int i = 0; i < kLoads; ++i)
      AsyncCopy16(&ring[d][i][threadIdx.x % 64], &at[(d * kLoads + i) * stride]);
  for (int b = 0; b < blocks; ++b) {
    int const slot = b % kDepth;
    int const ahead = (b + kDepth - 1) % kDepth;
    if (b + kDepth - 1 < blocks)
#pragma unroll
      for (int i = 0; i < kLoads; ++i)
        AsyncCopy16(&ring[ahead][i][threadIdx.x % 64],
                    &at[((b + kDepth - 1) * kLoads + i) * stride]);
    // Leave the newest (kDepth-1)*kLoads copies outstanding; the slot we are
    // about to read has retired, because retirement is in order.
    WaitAsyncCopies<(kDepth - 1) * kLoads>();
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      uint4 v = ring[slot][i][threadIdx.x % 64];
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  }
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = end - begin;
    *sink = float(acc.x + acc.y + acc.z + acc.w);
  }
}

/// Variant A: multi-buffer in registers. Loads go straight to registers and
/// an explicit arrive_gvmcnt leaves the newest (kDepth-1)*kLoads outstanding,
/// so only the oldest block is waited for. If MACA inserts its own
/// wait-for-all before the first use, this matches the serial form exactly.
///
/// The ring index must be a compile-time constant: indexing a register array
/// with `b % kDepth` puts the whole buffer in local memory and costs 5-7x
/// (measured before this was fixed), so the block loop is unrolled by kDepth
/// and each copy owns one slot.
template <int kLoads, int kDepth>
__global__ void RegisterPipelineKernel(uint4 const* source, int blocks, int stride,
                                       unsigned long long* out, float* sink) {
  uint4 acc = {0, 0, 0, 0};
  uint4 const* at = source + (std::size_t(blockIdx.x) * blocks * kLoads * stride)
                   + threadIdx.x % 64;
  uint4 buffer[kDepth][kLoads];
  unsigned long long begin = GlobalTimerNs();
#pragma unroll
  for (int d = 0; d < kDepth - 1; ++d)
#pragma unroll
    for (int i = 0; i < kLoads; ++i)
      buffer[d][i] = at[(d * kLoads + i) * stride];
  int const groups = blocks / kDepth;
  for (int g = 0; g < groups; ++g) {
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      int const block = g * kDepth + d;
      int const ahead = block + kDepth - 1;
      int const slot = (d + kDepth - 1) % kDepth;
      if (ahead < blocks)
#pragma unroll
        for (int i = 0; i < kLoads; ++i)
          buffer[slot][i] = at[(ahead * kLoads + i) * stride];
      WaitAsyncCopies<(kDepth - 1) * kLoads>();
#pragma unroll
      for (int i = 0; i < kLoads; ++i) {
        acc.x += buffer[d][i].x; acc.y += buffer[d][i].y;
        acc.z += buffer[d][i].z; acc.w += buffer[d][i].w;
      }
    }
  }
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = end - begin;
    *sink = float(acc.x + acc.y + acc.z + acc.w);
  }
}

namespace mx = tilemega::backend::mx;

/// With real MMA between the loads. A pure streaming loop may already issue
/// loads densely enough to hide latency; once the MMA pipe is occupied the
/// balance can change, so the verdict needs this case too. Two Mma per load
/// matches ServingGemmMma, which issues Lo and Hi halves per 32-K chunk.
template <int kLoads, int kDepth, int kMmaPerLoad>
__global__ void MmaPipelineKernel(uint4 const* source, int blocks, int stride,
                                  unsigned long long* out, float* sink) {
  mx::MmaAccum acc = {0.f, 0.f, 0.f, 0.f};
  uint4 const* at = source + (std::size_t(blockIdx.x) * blocks * kLoads * stride)
                   + threadIdx.x % 64;
  uint4 buffer[kDepth][kLoads];
  unsigned long long begin = GlobalTimerNs();
#pragma unroll
  for (int d = 0; d < kDepth - 1; ++d)
#pragma unroll
    for (int i = 0; i < kLoads; ++i)
      buffer[d][i] = at[(d * kLoads + i) * stride];
  int const groups = blocks / kDepth;
  for (int g = 0; g < groups; ++g) {
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      int const ahead = g * kDepth + d + kDepth - 1;
      int const slot = (d + kDepth - 1) % kDepth;
      if (ahead < blocks)
#pragma unroll
        for (int i = 0; i < kLoads; ++i)
          buffer[slot][i] = at[(ahead * kLoads + i) * stride];
      if (kDepth > 1) WaitAsyncCopies<(kDepth - 1) * kLoads>();
#pragma unroll
      for (int i = 0; i < kLoads; ++i)
#pragma unroll
        for (int m = 0; m < kMmaPerLoad; ++m)
          acc = mx::Mma(mx::Lo(buffer[d][i]), mx::Hi(buffer[d][i]), acc);
    }
  }
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = end - begin;
    *sink = acc[0] + acc[1] + acc[2] + acc[3];
  }
}

static double Median(std::vector<unsigned long long> v, int blocks, int loads) {
  std::sort(v.begin(), v.end());
  return double(v[v.size() / 2]) / (double(blocks) * loads);
}

int main(int argc, char** argv) {
  int ctas = argc > 1 ? std::atoi(argv[1]) : 104;
  int blocks = argc > 2 ? std::atoi(argv[2]) : 256;
  int const loads = 2;            // kLoadsPerChunk for a 16x32 tile
  int const stride = 64;          // 1 KiB apart: distinct cache lines
  std::size_t elements = std::size_t(ctas) * blocks * loads * stride + 64;
  uint4* source;
  unsigned long long* out;
  float* sink;
  CHECK(cudaMalloc(&source, elements * sizeof(uint4)));
  CHECK(cudaMemset(source, 1, elements * sizeof(uint4)));
  CHECK(cudaMalloc(&out, 512 * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&sink, sizeof(float)));
  std::vector<unsigned long long> host(512);
  auto run = [&](char const* name, auto launch) {
    CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
    launch();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(host.data(), out, ctas * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> got(host.begin(), host.begin() + ctas);
    std::printf("%-30s %9.2f ns/load\n", name, Median(got, blocks, loads));
    return 0;
  };
  std::printf("%d CTAs x 128 threads, %d blocks of %d loads, %.1f MiB touched\n",
              ctas, blocks, loads,
              double(elements * sizeof(uint4)) / (1 << 20));
  run("serial (depth 1)", [&] { SerialKernel<2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("A registers depth 2", [&] { RegisterPipelineKernel<2, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("A registers depth 3", [&] { RegisterPipelineKernel<2, 3><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("A registers depth 4", [&] { RegisterPipelineKernel<2, 4><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("A registers depth 6", [&] { RegisterPipelineKernel<2, 6><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("A registers depth 8", [&] { RegisterPipelineKernel<2, 8><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("B smem ring depth 2", [&] { PipelinedKernel<2, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("B smem ring depth 3", [&] { PipelinedKernel<2, 3><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("B smem ring depth 4", [&] { PipelinedKernel<2, 4><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("MMA serial (depth 1)", [&] { MmaPipelineKernel<2, 1, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("MMA registers depth 2", [&] { MmaPipelineKernel<2, 2, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("MMA registers depth 3", [&] { MmaPipelineKernel<2, 3, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("MMA registers depth 4", [&] { MmaPipelineKernel<2, 4, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  run("MMA registers depth 6", [&] { MmaPipelineKernel<2, 6, 2><<<ctas, 128>>>(source, blocks, stride, out, sink); });
  return 0;
}
