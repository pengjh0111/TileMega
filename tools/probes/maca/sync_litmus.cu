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
//
// MX-4 (512 threads, dynamic shared memory 32 KiB + 16 B, one CTA per AP):
//   barrier_prefetch: the decode program's barrier (MX4 prompt §3.3): arrive,
//            every thread prefetches into registers, wait. The prefetch reads
//            other CTAs' slices written one stage earlier and checks them
//            after the wait; then each CTA checks this stage's slices.
//   last512: LastArriver::Run at 512 threads, flag in dynamic shared memory,
//            on a stream-K split (§3.5): each CTA's segment of a tile writes
//            partial[tile][ordinal] and arrives on the tile's ticket; the
//            last of its n(t) contributors sums them in order and checks.
//   many:    LastArriver::RunMany as decode attention uses it: 8 items per
//            CTA (one per wave, item j on CTA j mod G), C consecutive items
//            share a ticket (C cycles 2, 4, 8, 13); the CTA completing a
//            ticket sums its C partials in order and checks them.
// Usage: sync_litmus <event|barrier|last|barrier_prefetch|last512|many> <iterations>
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/tasks/EventSync.cuh>
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <tilemega/Codegen/executor/LastArriverV2.cuh>
#include <tilemega/Codegen/tasks/mx/StageSyncV2.cuh>

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

__device__ int Expected(long q, int cta, int i) { return int(q * 977 + cta * 13 + i); }

__global__ void BarrierPrefetch(int* data, unsigned long long* arrivals, int stages, int iterations,
                                unsigned long long* errors) {
  int const ctas = gridDim.x, t = threadIdx.x;
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    for (int s = 0; s < stages; ++s) {
      long const q = long(it) * stages + s;
      int* block = data + (q % 3) * ctas * kSlice;
      int const* previous = data + ((q + 2) % 3) * ctas * kSlice;
      for (int i = t; i < kSlice; i += blockDim.x) block[blockIdx.x * kSlice + i] = Expected(q, blockIdx.x, i);
      __syncthreads();
      if (t == 0) RedRelease(&arrivals[s], 1ull);
      // Thread t prefetches 16 B of another CTA's slice from stage q-1.
      int const from = (blockIdx.x + 1 + (t / 64) * 13) % ctas, index = (t % 64) * 4;
      uint4 const prefetched = tilemega::backend::LoadGlobal16Gvm(previous + from * kSlice + index);
      if (t == 0) WaitAtLeast(&arrivals[s], static_cast<unsigned long long>(ctas) * (it + 1));
      __syncthreads();
      if (q > 0) {
        bad += int(prefetched.x) != Expected(q - 1, from, index);
        bad += int(prefetched.y) != Expected(q - 1, from, index + 1);
        bad += int(prefetched.z) != Expected(q - 1, from, index + 2);
        bad += int(prefetched.w) != Expected(q - 1, from, index + 3);
      }
      int const other = (blockIdx.x + (t / 64 + 1) * 7) % ctas;
      for (int i = t % 64; i < kSlice; i += 64) bad += block[other * kSlice + i] != Expected(q, other, i);
    }
  }
  if (bad) atomicAdd(errors, bad);
}

constexpr int kTiles = 40, kChunks = 13, kMaxContributors = 8;
// Stream-K: CTA c owns units [cU/G, (c+1)U/G); the owner of unit u.
__device__ int Owner(long u, int ctas, long units) { return int(((u + 1) * ctas - 1) / units); }
__device__ float Part(int it, int tile, int k, int i) { return float((it + tile * 3 + k + i) % 17); }

__global__ void LastArrive512(float* partials, unsigned* tickets, unsigned long long* fence_events, int iterations,
                              unsigned long long* errors, unsigned long long* lasts) {
  extern __shared__ int smem[];
  unsigned* const last_flag = reinterpret_cast<unsigned*>(smem);
  int const ctas = gridDim.x, c = blockIdx.x;
  long const units = long(kTiles) * kChunks, begin = c * units / ctas, end = (c + 1) * units / ctas;
  for (int it = 0; it < iterations; ++it) {
    for (long u = begin; u < end;) {
      int const tile = int(u / kChunks);
      long const k0 = u - long(tile) * kChunks, k1 = min(end - long(tile) * kChunks, long(kChunks));
      int const first = Owner(long(tile) * kChunks, ctas, units);
      int const producers = Owner(long(tile) * kChunks + kChunks - 1, ctas, units) - first + 1;
      float* own = partials + (long(tile) * kMaxContributors + (c - first)) * kSlice;
      for (int i = threadIdx.x; i < kSlice; i += blockDim.x) {
        float sum = 0;
        for (long k = k0; k < k1; ++k) sum += Part(it, tile, int(k), i);
        own[i] = sum;
      }
      bool const last = executor::LastArriver::Run(&tickets[tile], unsigned(producers), last_flag, [&] {
        unsigned long long bad = 0;
        float const* all = partials + long(tile) * kMaxContributors * kSlice;
        for (int i = threadIdx.x; i < kSlice; i += blockDim.x) {
          float sum = 0, want = 0;
          for (int o = 0; o < producers; ++o) sum += all[o * kSlice + i];
          for (int k = 0; k < kChunks; ++k) want += Part(it, tile, k, i);
          bad += sum != want;
        }
        if (bad) atomicAdd(errors, bad);
      });
      if (last && threadIdx.x == 0) atomicAdd(lasts, 1ull);
      u = long(tile + 1) * kChunks;
    }
    // Partials are rewritten next iteration only after every reducer finished.
    __syncthreads();
    if (threadIdx.x == 0) {
      RedRelease(fence_events, 1ull);
      WaitAtLeast(fence_events, static_cast<unsigned long long>(ctas) * (it + 1));
    }
    __syncthreads();
  }
}

__device__ int ManyProducers(int it) {
  int const c[4] = {2, 4, 8, 13};
  return c[it % 4];
}
__device__ float ManyPart(int it, int item, int i) { return float((it * 7 + item * 3 + i) % 23); }

__global__ void Many(float* partials, unsigned* tickets, unsigned long long* fence_events, int iterations,
                     unsigned long long* errors, unsigned long long* lasts) {
  extern __shared__ int smem[];
  unsigned* const flags = reinterpret_cast<unsigned*>(smem);
  int const ctas = gridDim.x, wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  constexpr int kItemSlice = 64;
  for (int it = 0; it < iterations; ++it) {
    int const producers = ManyProducers(it);
    int const item = blockIdx.x + ctas * wave;
    partials[item * kItemSlice + lane] = ManyPart(it, item, lane);
    auto at = [&](int i) { return int(blockIdx.x) + ctas * i; };
    executor::LastArriver::RunMany(
        8, flags, [&](int i) { return tickets + at(i) / producers; }, [&](int) { return unsigned(producers); },
        [&](int i) {
          int const ticket = at(i) / producers;
          unsigned long long bad = 0;
          for (int index = threadIdx.x; index < kItemSlice; index += blockDim.x) {
            float sum = 0, want = 0;
            for (int c = 0; c < producers; ++c) {
              sum += partials[(ticket * producers + c) * kItemSlice + index];
              want += ManyPart(it, ticket * producers + c, index);
            }
            bad += sum != want;
          }
          if (bad) atomicAdd(errors, bad);
          if (threadIdx.x == 0) atomicAdd(lasts, 1ull);
        });
    __syncthreads();
    if (threadIdx.x == 0) {
      RedRelease(fence_events, 1ull);
      WaitAtLeast(fence_events, static_cast<unsigned long long>(ctas) * (it + 1));
    }
    __syncthreads();
  }
}

unsigned long long ManyMerges(int items, int iterations) {
  int const c[4] = {2, 4, 8, 13};
  unsigned long long total = 0;
  for (int it = 0; it < iterations; ++it) total += items / c[it % 4];
  return total;
}

using V2Sync = mx::StageSyncV2<576>;
using V2Last = executor::LastArriverV2<V2Sync>;
constexpr int kV2Slice = 512;

__device__ __forceinline__ void V2Issue(uint4 (&pending)[8], uint4 const* cold, unsigned it) {
  if (V2Sync::IsCompute()) {
    unsigned const wave = threadIdx.x / 64, lane = threadIdx.x % 64;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      unsigned const index = ((((it * 104 + blockIdx.x) * 8 + wave) * 8 + n) * 64 + lane)
                             & ((256u << 20) / 16 - 1);
      pending[n] = tilemega::backend::LoadGlobal16Gvm(cold + index);
    }
  }
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ void V2Consume(uint4 const (&pending)[8]) {
  unsigned sum = 0;
  if (V2Sync::IsCompute()) {
#pragma unroll
    for (int n = 0; n < 8; ++n) sum += pending[n].x + pending[n].y + pending[n].z + pending[n].w;
  }
  asm volatile("" :: "r"(sum) : "memory");
}

__global__ void __launch_bounds__(576)
BarrierV2(int* data, unsigned long long* events, int iterations, unsigned long long* errors,
          uint4 const* cold) {
  extern __shared__ unsigned shared[];
  unsigned* const generation = shared;
  if (V2Sync::IsRole()) *generation = 0;
  V2Sync::CtaSync();
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    if (V2Sync::IsCompute())
      data[blockIdx.x * kV2Slice + threadIdx.x] = Expected(it, blockIdx.x, threadIdx.x);
    V2Sync::CtaSync();
    uint4 pending[8];
    V2Issue(pending, cold, unsigned(it));
    V2Sync::Barrier(events, 104ull * (it + 1), generation, 2 * it + 1, [] {});
    if (V2Sync::IsCompute()) {
      for (int k = 1; k <= 8; ++k) {
        int const other = (blockIdx.x + k * 13) % 104;
        using Global = __attribute__((address_space(1))) int const;
        bad += *((Global*)data + other * kV2Slice + threadIdx.x) != Expected(it, other, threadIdx.x);
      }
    }
    V2Consume(pending);
    V2Sync::CtaSync();
    V2Sync::Barrier(events + 8, 104ull * (it + 1), generation, 2 * it + 2, [] {});
  }
  if (bad) atomicAdd(errors, bad);
}

__global__ void __launch_bounds__(576)
LastV2(float* partials, unsigned* ticket, unsigned long long* events, int iterations,
       unsigned long long* errors, unsigned long long* lasts) {
  extern __shared__ unsigned shared[];
  unsigned* const generation = shared;
  unsigned* const last_flag = shared + 1;
  if (V2Sync::IsRole()) *generation = 0;
  V2Sync::CtaSync();
  for (int it = 0; it < iterations; ++it) {
    if (V2Sync::IsCompute())
      partials[blockIdx.x * kV2Slice + threadIdx.x] = float((it + blockIdx.x + threadIdx.x) % 17);
    bool const last = V2Last::Run(ticket, 104, last_flag, [&] {
      unsigned long long bad = 0;
      if (V2Sync::IsCompute()) {
        float sum = 0, want = 0;
        for (int c = 0; c < 104; ++c) {
          sum += tilemega::backend::LoadFloatGvm(partials + c * kV2Slice + threadIdx.x);
          want += float((it + c + threadIdx.x) % 17);
        }
        bad += sum != want;
      }
      if (bad) atomicAdd(errors, bad);
    });
    if (last && V2Sync::IsRole()) atomicAdd(lasts, 1ull);
    V2Sync::CtaSync();
    V2Sync::Barrier(events, 104ull * (it + 1), generation, it + 1, [] {});
  }
}

__global__ void __launch_bounds__(576)
LdsReleaseV2(int iterations, unsigned long long* errors, uint4 const* cold) {
  extern __shared__ unsigned shared[];
  unsigned* const generation = shared;
  volatile unsigned* const ready = shared + 16;
  unsigned* const data = shared + 32;
  if (threadIdx.x < 8) ready[threadIdx.x] = 0;
  if (V2Sync::IsRole()) *generation = 0;
  V2Sync::CtaSync();
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    uint4 pending[8];
    V2Issue(pending, cold, unsigned(it));
    if (V2Sync::IsCompute()) {
      data[threadIdx.x] = unsigned(Expected(it, blockIdx.x, threadIdx.x));
      __builtin_mxc_arrive_bsmcnt(0);
      if (threadIdx.x % 64 == 0) ready[threadIdx.x / 64] = it + 1;
    }
    if (V2Sync::IsRole()) {
      for (int w = 0; w < 8; ++w) while (ready[w] < unsigned(it + 1)) {}
      V2Sync::Publish(generation, it + 1);
    }
    V2Sync::Await(generation, it + 1);
    if (V2Sync::IsCompute()) {
      for (int w = 1; w < 8; ++w) {
        unsigned const other = (threadIdx.x + w * 64) % 512;
        bad += data[other] != unsigned(Expected(it, blockIdx.x, other));
      }
    }
    V2Consume(pending);
    V2Sync::CtaSync();
  }
  if (bad) atomicAdd(errors, bad);
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  char const* mode = argv[1];
  int const iterations = std::atoi(argv[2]);
  int const ctas = 104, threads = 128, stages = 8;
  bool const v2 = !std::strcmp(mode, "barrier_v2") || !std::strcmp(mode, "la_v2") ||
                  !std::strcmp(mode, "lds_release");
  // The MX-4 program's request at G = 104: floor(65536 / 2) + 16 B.
  int const kSmem512 = 65536 / 2 + 16;
  int* data;
  float* partials;
  unsigned long long *events, *errors, *lasts;
  unsigned* ticket;
  uint4* cold = nullptr;
  if (v2) {
    cudaMalloc(&cold, 256u << 20);
    cudaMemset(cold, 0x5a, 256u << 20);
    std::printf("MX5_SHAPE ctas=104 threads=576 smem=%d\n", kSmem512);
  }
  int const data_slice = v2 ? kV2Slice : kSlice;
  cudaMalloc(&data, 3 * ctas * data_slice * sizeof(int));
  cudaMalloc(&partials, max(kTiles * kMaxContributors * kSlice, ctas * 8 * 64) * sizeof(float));
  cudaMalloc(&events, 64 * sizeof(unsigned long long));
  cudaMalloc(&errors, 8);
  cudaMalloc(&lasts, 8);
  cudaMalloc(&ticket, ctas * 8 * 4);
  cudaMemset(data, 0xff, 3 * ctas * data_slice * sizeof(int));
  cudaMemset(events, 0, 64 * sizeof(unsigned long long));
  cudaMemset(errors, 0, 8);
  cudaMemset(lasts, 0, 8);
  cudaMemset(ticket, 0, ctas * 8 * 4);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  if (!std::strcmp(mode, "barrier_v2")) {
    BarrierV2<<<ctas, 576, kSmem512>>>(data, events, iterations, errors, cold);
  } else if (!std::strcmp(mode, "la_v2")) {
    LastV2<<<ctas, 576, kSmem512>>>(partials, ticket, events, iterations, errors, lasts);
  } else if (!std::strcmp(mode, "lds_release")) {
    LdsReleaseV2<<<ctas, 576, kSmem512>>>(iterations, errors, cold);
  } else if (!std::strcmp(mode, "event")) {
    cudaFuncSetAttribute(EventChain, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    EventChain<<<ctas, threads, 65536>>>(data, events, stages, iterations, errors);
  } else if (!std::strcmp(mode, "barrier")) {
    cudaFuncSetAttribute(BarrierChain, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    BarrierChain<<<ctas, threads, 65536>>>(data, events, stages, iterations, errors);
  } else if (!std::strcmp(mode, "barrier_prefetch")) {
    cudaFuncSetAttribute(BarrierPrefetch, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem512);
    BarrierPrefetch<<<ctas, 512, kSmem512>>>(data, events, stages, iterations, errors);
  } else if (!std::strcmp(mode, "last512")) {
    cudaFuncSetAttribute(LastArrive512, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem512);
    LastArrive512<<<ctas, 512, kSmem512>>>(partials, ticket, events, iterations, errors, lasts);
  } else if (!std::strcmp(mode, "many")) {
    cudaFuncSetAttribute(Many, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem512);
    Many<<<ctas, 512, kSmem512>>>(partials, ticket, events, iterations, errors, lasts);
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
                  (std::strcmp(mode, "last") || h_lasts == static_cast<unsigned long long>(iterations)) &&
                  (std::strcmp(mode, "last512") || h_lasts == static_cast<unsigned long long>(kTiles) * iterations) &&
                  (std::strcmp(mode, "la_v2") || h_lasts == static_cast<unsigned long long>(iterations)) &&
                  (std::strcmp(mode, "many") || h_lasts == ManyMerges(ctas * 8, iterations));
  std::printf("SYNC %s iterations=%d launch=%s sync=%s errors=%llu lasts=%llu us/iter=%.2f %s\n", mode, iterations,
              cudaGetErrorString(launch), cudaGetErrorString(sync), h_errors, h_lasts, ms * 1000.0 / iterations,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
