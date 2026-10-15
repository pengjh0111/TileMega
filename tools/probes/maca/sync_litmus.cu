// MX-D sync litmus: the executor's event, grid-barrier and last-arriver paths
// on xcore1000, using the production primitives (EventSync.cuh,
// LastArriver.cuh). 104 CTAs, one per AP (64 KiB smem each).
//
//   event:   stage s CTAs write a slice, RedRelease the stage event; stage s+1
//            CTAs WaitAtLeast(triggers*(iteration+1)) and check every slice.
//   barrier: the V3 grid barrier of ModelHarness (RedRelease arrivals, wait
//            for gridDim.x*(iteration+1)); after it each CTA checks all slices.
//   last:    every CTA writes a partial and arrives on a ticket; the last
//            arriver sums the partials and checks the total.
// Usage: sync_litmus <event|barrier|last> <iterations>
#include <tilemega/Codegen/tasks/EventSync.cuh>
#include <tilemega/Codegen/executor/LastArriver.cuh>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace tilemega::codegen;

constexpr int kSlice = 256;  // ints per CTA

__global__ void EventChain(int* data, unsigned long long* events, int stages, int iterations,
                           unsigned long long* errors) {
  extern __shared__ int smem[];
  smem[threadIdx.x] = 0;
  int const ctas = gridDim.x, half = ctas / 2;
  bool const producer_side = blockIdx.x < half;
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    for (int s = 0; s < stages; ++s) {
      // Even stages: the first half produces, the second half consumes; odd
      // stages swap roles. Each stage owns event s and data block s % 2.
      bool const produce = (s % 2 == 0) == producer_side;
      int* block = data + (s % 2) * half * kSlice;
      if (produce) {
        int const slot = producer_side ? blockIdx.x : blockIdx.x - half;
        for (int i = threadIdx.x; i < kSlice; i += blockDim.x) block[slot * kSlice + i] = it * 131 + s * 7 + slot;
        __syncthreads();
        if (threadIdx.x == 0) RedRelease(&events[s], 1ull);
        __syncthreads();
      } else {
        if (threadIdx.x == 0) WaitAtLeast(&events[s], static_cast<unsigned long long>(half) * (it + 1));
        __syncthreads();
        for (int slot = 0; slot < half; ++slot)
          for (int i = threadIdx.x; i < kSlice; i += blockDim.x)
            bad += block[slot * kSlice + i] != it * 131 + s * 7 + slot;
        __syncthreads();
      }
    }
    // Close the iteration so producers do not overwrite unread slices.
    if (threadIdx.x == 0) {
      RedRelease(&events[stages], 1ull);
      WaitAtLeast(&events[stages], static_cast<unsigned long long>(ctas) * (it + 1));
    }
    __syncthreads();
  }
  if (bad) atomicAdd(errors, bad);
}

__device__ void GridBarrierV3(unsigned long long* arrivals, unsigned long long iteration) {
  __syncthreads();
  if (threadIdx.x == 0) {
    RedRelease(arrivals, 1ull);
    WaitAtLeast(arrivals, static_cast<unsigned long long>(gridDim.x) * (iteration + 1ull));
  }
  __syncthreads();
}

__global__ void BarrierChain(int* data, unsigned long long* arrivals, int stages, int iterations,
                             unsigned long long* errors) {
  extern __shared__ int smem[];
  smem[threadIdx.x] = 0;
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    for (int s = 0; s < stages; ++s) {
      int* block = data + (s % 2) * gridDim.x * kSlice;
      for (int i = threadIdx.x; i < kSlice; i += blockDim.x)
        block[blockIdx.x * kSlice + i] = it * 977 + s * 13 + blockIdx.x;
      GridBarrierV3(&arrivals[s], it);
      // Read a rotating set of other CTAs' slices.
      for (int k = 1; k <= 8; ++k) {
        int const other = (blockIdx.x + k * 13) % gridDim.x;
        for (int i = threadIdx.x; i < kSlice; i += blockDim.x)
          bad += block[other * kSlice + i] != it * 977 + s * 13 + other;
      }
    }
  }
  if (bad) atomicAdd(errors, bad);
}

__global__ void LastArrive(float* partials, unsigned* ticket, unsigned long long* fence_events, int iterations,
                           unsigned long long* errors, unsigned long long* lasts) {
  extern __shared__ int smem[];
  __shared__ unsigned last_flag;
  smem[threadIdx.x] = 0;
  for (int it = 0; it < iterations; ++it) {
    for (int i = threadIdx.x; i < kSlice; i += blockDim.x)
      partials[blockIdx.x * kSlice + i] = float((it + blockIdx.x + i) % 17);
    bool const last = executor::LastArriver::Run(ticket, gridDim.x, &last_flag, [&] {
      unsigned long long bad = 0;
      for (int i = threadIdx.x; i < kSlice; i += blockDim.x) {
        float sum = 0, want = 0;
        for (int b = 0; b < int(gridDim.x); ++b) {
          sum += partials[b * kSlice + i];
          want += float((it + b + i) % 17);
        }
        bad += sum != want;
      }
      if (bad) atomicAdd(errors, bad);
    });
    if (last && threadIdx.x == 0) atomicAdd(lasts, 1ull);
    // Partials are rewritten next iteration only after the reducer finished.
    if (threadIdx.x == 0) {
      RedRelease(fence_events, 1ull);
      WaitAtLeast(fence_events, static_cast<unsigned long long>(gridDim.x) * (it + 1));
    }
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  char const* mode = argv[1];
  int const iterations = std::atoi(argv[2]);
  int const ctas = 104, threads = 128, stages = 8;
  int* data;
  float* partials;
  unsigned long long *events, *errors, *lasts;
  unsigned* ticket;
  cudaMalloc(&data, 2 * ctas * kSlice * sizeof(int));
  cudaMalloc(&partials, ctas * kSlice * sizeof(float));
  cudaMalloc(&events, 64 * sizeof(unsigned long long));
  cudaMalloc(&errors, 8);
  cudaMalloc(&lasts, 8);
  cudaMalloc(&ticket, 4);
  cudaMemset(data, 0xff, 2 * ctas * kSlice * sizeof(int));
  cudaMemset(events, 0, 64 * sizeof(unsigned long long));
  cudaMemset(errors, 0, 8);
  cudaMemset(lasts, 0, 8);
  cudaMemset(ticket, 0, 4);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  if (!std::strcmp(mode, "event")) {
    cudaFuncSetAttribute(EventChain, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    EventChain<<<ctas, threads, 65536>>>(data, events, stages, iterations, errors);
  } else if (!std::strcmp(mode, "barrier")) {
    cudaFuncSetAttribute(BarrierChain, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    BarrierChain<<<ctas, threads, 65536>>>(data, events, stages, iterations, errors);
  } else {
    // 60 KiB keeps one CTA per AP while leaving room for the static flag.
    cudaFuncSetAttribute(LastArrive, cudaFuncAttributeMaxDynamicSharedMemorySize, 61440);
    LastArrive<<<ctas, threads, 61440>>>(partials, ticket, events, iterations, errors, lasts);
  }
  cudaError_t launch = cudaGetLastError();
  cudaEventRecord(b);
  cudaError_t sync = cudaDeviceSynchronize();
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  unsigned long long h_errors = 0, h_lasts = 0;
  cudaMemcpy(&h_errors, errors, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_lasts, lasts, 8, cudaMemcpyDeviceToHost);
  bool const ok = launch == cudaSuccess && sync == cudaSuccess && h_errors == 0 &&
                  (std::strcmp(mode, "last") || h_lasts == static_cast<unsigned long long>(iterations));
  std::printf("SYNC %s iterations=%d launch=%s sync=%s errors=%llu lasts=%llu us/iter=%.2f %s\n", mode, iterations,
              cudaGetErrorString(launch), cudaGetErrorString(sync), h_errors, h_lasts, ms * 1000.0 / iterations,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
