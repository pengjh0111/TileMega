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
#include <tilemega/Codegen/tasks/mx/StageSyncV3.cuh>
#include <tilemega/Backend/Mx/MoeCombineV3.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

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

using V3Sync = mx::StageSyncV3<512>;

__device__ __forceinline__ void V3Issue(uint4 (&pending)[8], uint4 const* cold, unsigned it) {
  unsigned const wave = threadIdx.x / 64, lane = threadIdx.x % 64;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    unsigned const index = ((((it * 104 + blockIdx.x) * 8 + wave) * 8 + n) * 64 + lane)
                           & ((256u << 20) / 16 - 1);
    pending[n] = tilemega::backend::LoadGlobal16Gvm(cold + index);
  }
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ void V3Consume(uint4 const (&pending)[8]) {
  unsigned sum = 0;
#pragma unroll
  for (int n = 0; n < 8; ++n) sum += pending[n].x + pending[n].y + pending[n].z + pending[n].w;
  asm volatile("" :: "r"(sum) : "memory");
}

__global__ void __launch_bounds__(512)
BarrierV3EarlyArrive(int* data, unsigned long long* events, int iterations,
                    unsigned long long* errors, unsigned long long* participants, uint4 const* cold) {
  extern __shared__ unsigned shared[];
  unsigned* const generation = shared;
  if (V3Sync::IsRole()) *generation = 0;
  V3Sync::CtaSync();
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    data[blockIdx.x * 512 + threadIdx.x] = Expected(it, blockIdx.x, threadIdx.x);
    V3Sync::CtaSync();
    V3Sync::Arrive(events);
    uint4 pending[8];
    V3Issue(pending, cold, unsigned(it));
    V3Sync::Wait(events, 104ull * (it + 1), generation, 2 * it + 1, [] {}, {});
    using Global = __attribute__((address_space(1))) int const;
    for (int other = 0; other < 104; ++other)
      bad += *((Global*)data + other * 512 + threadIdx.x) != Expected(it, other, threadIdx.x);
    V3Consume(pending);
    V3Sync::CtaSync();
    // Close the reader phase before any CTA overwrites its slot.
    V3Sync::Arrive(events + 8);
    V3Sync::Wait(events + 8, 104ull * (it + 1), generation, 2 * it + 2, [] {}, {});
    if (V3Sync::IsRole()) atomicAdd(participants, 1ull);
  }
  if (bad) atomicAdd(errors, bad);
}

__global__ void __launch_bounds__(512)
LdsSyncInflight(int iterations, unsigned long long* errors,
                unsigned long long* participants, uint4 const* cold) {
  extern __shared__ unsigned shared[];
  using Lds = __attribute__((address_space(3))) unsigned;
  auto* data = (Lds*)shared;
  unsigned long long bad = 0;
  for (int it = 0; it < iterations; ++it) {
    uint4 pending[8];
    V3Issue(pending, cold, unsigned(it));
    data[threadIdx.x] = unsigned(Expected(it, blockIdx.x, threadIdx.x));
    V3Sync::LdsSync();
#pragma unroll
    for (int w = 1; w < 8; ++w) {
      unsigned const other = (threadIdx.x + w * 64) % 512;
      bad += data[other] != unsigned(Expected(it, blockIdx.x, other));
    }
    V3Sync::LdsSync();
    data[threadIdx.x] = ~unsigned(Expected(it, blockIdx.x, threadIdx.x));
    V3Sync::LdsSync();
#pragma unroll
    for (int w = 1; w < 8; ++w) {
      unsigned const other = (threadIdx.x + w * 64) % 512;
      bad += data[other] != ~unsigned(Expected(it, blockIdx.x, other));
    }
    V3Sync::LdsSync();
    V3Consume(pending);
    if (V3Sync::IsRole()) atomicAdd(participants, 1ull);
  }
  if (bad) atomicAdd(errors, bad);
}

namespace owner_combine {
using namespace tilemega;
using namespace tilemega::backend::mx;
constexpr unsigned kB=16,kH=2048,kTickets=kH/16;
TM_MX_HD unsigned Hash(unsigned x) {
  x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);
}
TM_MX_HD float Weight(unsigned seed,unsigned it,unsigned b,unsigned rank) {
  return float(1+(Hash(seed+it*131+b*17+rank)&15))/32.f;
}
TM_MX_HD float Value(unsigned seed,unsigned it,unsigned b,unsigned rank,unsigned column) {
  return float(int(Hash(seed+it*977+b*131+rank*37+column)&127)-64)/16.f;
}
TM_MX_HD float Residual(unsigned seed,unsigned it,unsigned b,unsigned column) {
  return float(int(Hash(seed+it*313+b*59+column)&63)-32)/8.f;
}
template<unsigned TN,MxCombineModeV3 Mode>
__device__ __forceinline__ void Produce(MxMoeCombineOperandsV3<false> const& p,unsigned active,
    unsigned it,unsigned seed,char* storage) {
  using Combine=MxMoeCombineTileV3<TN,false,Mode>;
  constexpr unsigned tiles=kH/TN;
  unsigned const begin=blockIdx.x*(active*tiles)/104,end=(blockIdx.x+1)*(active*tiles)/104;
  if(V3Sync::IsRole()) {
    unsigned const total=kH/Combine::kTile;
    p.shared->first=(blockIdx.x*total+103)/104;
    p.shared->end=((blockIdx.x+1)*total+103)/104;
  }
  unsigned const salt=Hash(seed+it),factor=(salt>>8)|1u;
  for(unsigned unit=begin;unit<end;++unit) {
    unsigned const expert=unit/tiles,tile=unit%tiles;
    unsigned first=0,rows=kB*8;
    if constexpr(Mode==MxCombineModeV3::kUnitOwner) {
      while(((first*factor+salt)&(active-1))!=expert)++first;
      rows=kB*8/active;
      // WN1 Finish rendezvous precedes stores; expert rows are compact in a unit.
      V3Sync::LdsSync();
    }
    for(unsigned item=threadIdx.x;item<rows*TN/8;item+=512) {
      unsigned const local_row=item/(TN/8);
      unsigned const row=Mode==MxCombineModeV3::kUnitOwner?first+local_row*active:local_row;
      unsigned const b=row/8,rank=row%8,column=tile*TN+item%(TN/8)*8;
      if(((row*factor+salt)&(active-1))==expert) {
        alignas(16) Bf16 values[8];
        for(unsigned j=0;j<8;++j)values[j]=Bf16(Value(seed,it,b,rank,column+j));
        MxStore16V2(const_cast<Bf16*>(p.partials)+row*kH+column,*reinterpret_cast<uint4*>(values));
      }
    }
    if constexpr(Mode==MxCombineModeV3::kUnitOwner)
      if(threadIdx.x==0)Combine::UnitOwner(p,tile);
  }
  Combine::Complete(p,begin,end,active,kB,kH,storage);
}
template<MxCombineModeV3 Mode>
__global__ void __launch_bounds__(512) Kernel(Bf16* partial,Bf16* x,unsigned* tickets,
    unsigned long long* events,unsigned long long* errors,unsigned long long* participants,
    unsigned long long* case_errors,Bf16 const* reference,int iterations,unsigned seed) {
  extern __shared__ __align__(16) char storage[];
  auto* table=(MxSharedRouteTableV3<false>*)(storage+4096);
  auto* state=(MxSharedCombineStateV3*)(storage+4096+sizeof(MxRouteTableV3));
  auto* generation=reinterpret_cast<unsigned*>(storage+2048);
  if(V3Sync::IsRole()){*generation=0;state->generation=0;}
  V3Sync::CtaSync();
  unsigned long long bad=0;
  for(int it=0;it<iterations;++it) {
    unsigned const mode=unsigned(it)&3,active=mode==0?1:mode==1?8:mode==2?64:128;
    if(threadIdx.x<kB*8) {
      unsigned const b=threadIdx.x/8,rank=threadIdx.x%8;
      table->weight[b][rank]=Bf16(Weight(seed,it,b,rank)).raw();
    }
    for(unsigned item=blockIdx.x*512+threadIdx.x;item<kB*kH/8;item+=104*512) {
      unsigned const b=item/(kH/8),column=item%(kH/8)*8;
      alignas(16) Bf16 values[8];
      for(unsigned j=0;j<8;++j)values[j]=Bf16(Residual(seed,it,b,column+j));
      MxStore16V2(x+b*kH+column,*reinterpret_cast<uint4*>(values));
    }
    V3Sync::CtaSync();V3Sync::Arrive(events);
    V3Sync::Wait(events,104ull*(it+1),generation,3*it+1,[]{},{});
    MxMoeCombineOperandsV3<false> p{partial,table,x,x,(MxMoeGlobalTicketV2*)tickets,state,unsigned(it)+1,{}};
    switch((unsigned(it)>>2)&3) {
      case 0:Produce<16,Mode>(p,active,it,seed,storage);break;
      case 1:Produce<32,Mode>(p,active,it,seed,storage);break;
      case 2:Produce<64,Mode>(p,active,it,seed,storage);break;
      case 3:Produce<128,Mode>(p,active,it,seed,storage);break;
    }
    V3Sync::Arrive(events+8);
    V3Sync::Wait(events+8,104ull*(it+1),generation,3*it+2,[]{},{});
    unsigned long long current_bad=0;
    for(unsigned item=blockIdx.x*512+threadIdx.x;item<kB*kH/8;item+=104*512) {
      uint4 const got=tilemega::backend::LoadGlobal16Gvm(x+item*8);
      uint4 const want=tilemega::backend::LoadGlobal16Gvm(reference+std::size_t(it)*kB*kH+item*8);
      __builtin_mxc_arrive_gvmcnt(0);
      current_bad+=got.x!=want.x;current_bad+=got.y!=want.y;
      current_bad+=got.z!=want.z;current_bad+=got.w!=want.w;
    }
    bad+=current_bad;
    if(current_bad)atomicAdd(case_errors+(((unsigned(it)>>2)&3)*4+(unsigned(it)&3)),current_bad);
    V3Sync::CtaSync();V3Sync::Arrive(events+16);
    V3Sync::Wait(events+16,104ull*(it+1),generation,3*it+3,[]{},{});
    if(V3Sync::IsRole())atomicAdd(participants,1ull);
  }
  if(bad)atomicAdd(errors,bad);
}
template<MxCombineModeV3 Mode>
int Run(int iterations,unsigned seed,char const* label) {
  if(iterations<2000)return 2;
  constexpr unsigned bytes=32784;
  auto kernel=Kernel<Mode>;
  cudaFuncAttributes attr{};int resident=0;
  auto set=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  auto query=cudaFuncGetAttributes(&attr,kernel);
  auto occupancy=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,bytes);
  std::printf("SYNC_SHAPE ctas=104 threads=512 smem=%u seed=%u n_active=1,8,64,128 TN=16,32,64,128 synthetic_n_active1_case=1\n",bytes,seed);
  std::printf("SYNC_RESOURCES set=%d query=%d occupancy=%d resident=%d MT=%d local=%zu static=%zu\n",
      int(set),int(query),int(occupancy),resident,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes);
  if(set||query||occupancy||resident!=1||attr.numRegs>255||attr.localSizeBytes||attr.sharedSizeBytes)return 1;
  std::vector<Bf16> expected(std::size_t(iterations)*kB*kH);
  for(int it=0;it<iterations;++it)
    for(unsigned b=0;b<kB;++b)
      for(unsigned col=0;col<kH;++col) {
        float sum=0;
        for(unsigned r=0;r<8;++r)sum+=float(Bf16(Weight(seed,it,b,r)*Value(seed,it,b,r,col)));
        expected[(std::size_t(it)*kB+b)*kH+col]=Bf16(float(Bf16(sum))+Residual(seed,it,b,col));
      }
  Bf16 *partial,*x,*reference;unsigned* tickets;
  unsigned long long *events,*errors,*participants,*case_errors;
  if(cudaMalloc(&partial,kB*8*kH*2)||cudaMalloc(&x,kB*kH*2)||
      cudaMalloc(&reference,expected.size()*2)||cudaMalloc(&tickets,kTickets*4)||
      cudaMalloc(&events,32*8)||cudaMalloc(&errors,8)||cudaMalloc(&participants,8)||cudaMalloc(&case_errors,16*8))return 1;
  if(cudaMemcpy(reference,expected.data(),expected.size()*2,cudaMemcpyHostToDevice)||
      cudaMemset(tickets,0,kTickets*4)||cudaMemset(events,0,32*8)||cudaMemset(errors,0,8)||
      cudaMemset(participants,0,8)||cudaMemset(case_errors,0,16*8))return 1;
  kernel<<<104,512,bytes>>>(partial,x,tickets,events,errors,participants,case_errors,reference,iterations,seed);
  auto launch=cudaGetLastError();auto sync=cudaDeviceSynchronize();
  unsigned long long bad=0,joined=0;unsigned final_tickets[kTickets];
  auto read_errors=cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost);
  auto read_participants=cudaMemcpy(&joined,participants,8,cudaMemcpyDeviceToHost);
  auto read_tickets=cudaMemcpy(final_tickets,tickets,kTickets*4,cudaMemcpyDeviceToHost);
  for(unsigned v:final_tickets)bad+=v!=0;
  unsigned long long cases[16]={};
  auto read_cases=cudaMemcpy(cases,case_errors,16*8,cudaMemcpyDeviceToHost);
  for(unsigned i=0;i<16;++i)
    std::printf("SYNC_CASE TN=%u n_active=%u errors=%llu\n",16u<<(i/4),i%4==0?1:i%4==1?8:i%4==2?64:128,cases[i]);
  bool const pass=!launch&&!sync&&!read_errors&&!read_participants&&!read_tickets&&
      !read_cases&&!bad&&joined==104ull*iterations;
  std::printf("SYNC %s iterations=%d launch=%s sync=%s errors=%llu participants=%llu host_x_bitwise=1 %s\n",
      label,iterations,cudaGetErrorString(launch),cudaGetErrorString(sync),bad,joined,pass?"PASS":"FAIL");
  return pass?0:1;
}
}  // namespace owner_combine

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  char const* mode = argv[1];
  int const iterations = std::atoi(argv[2]);
  unsigned const seed=argc>3?unsigned(std::strtoul(argv[3],nullptr,10)):unsigned(getpid());
  using tilemega::backend::mx::MxCombineModeV3;
  if(!std::strcmp(mode,"moe_owner_combine"))
    return owner_combine::Run<MxCombineModeV3::kOwner16>(iterations,seed,mode);
  if(!std::strcmp(mode,"moe_owner_tile"))
    return owner_combine::Run<MxCombineModeV3::kOwnerTile>(iterations,seed,mode);
  if(!std::strcmp(mode,"moe_unit_owner"))
    return owner_combine::Run<MxCombineModeV3::kUnitOwner>(iterations,seed,mode);
  int const ctas = 104, threads = 128, stages = 8;
  bool const v2 = !std::strcmp(mode, "barrier_v2") || !std::strcmp(mode, "la_v2") ||
                  !std::strcmp(mode, "lds_release");
  bool const early = !std::strcmp(mode, "barrier_v3_early_arrive");
  bool const lds_inflight = !std::strcmp(mode, "lds_sync_inflight");
  bool const v3 = early || lds_inflight;
  if (v3 && iterations < 2000) return 2;
  // The MX-4 program's request at G = 104: floor(65536 / 2) + 16 B.
  int const kSmem512 = 65536 / 2 + 16;
  int* data;
  float* partials;
  unsigned long long *events, *errors, *lasts;
  unsigned* ticket;
  uint4* cold = nullptr;
  if (v2 || v3) {
    cudaMalloc(&cold, 256u << 20);
    cudaMemset(cold, 0x5a, 256u << 20);
    std::printf("SYNC_SHAPE ctas=104 threads=%d smem=%d cold_bytes=%u inflight_bytes_per_wave=8192\n",
                v3 ? 512 : 576, kSmem512, 256u << 20);
  }
  int const data_slice = v2 || v3 ? kV2Slice : kSlice;
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
  if (v3) {
    void const* kernel = early ? (void const*)BarrierV3EarlyArrive : (void const*)LdsSyncInflight;
    cudaFuncAttributes attr{};
    int resident = 0;
    cudaError_t set = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem512);
    cudaError_t query = cudaFuncGetAttributes(&attr, kernel);
    cudaError_t occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, kSmem512);
    std::printf("SYNC_RESOURCES set=%d query=%d occupancy=%d resident=%d MT=%d local=%zu static=%zu\n",
                int(set), int(query), int(occupancy), resident, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes);
    if (set || query || occupancy || resident != 1 || attr.numRegs > 255 || attr.localSizeBytes || attr.sharedSizeBytes) return 1;
  }
  cudaEventRecord(a);
  if (early) {
    BarrierV3EarlyArrive<<<ctas, 512, kSmem512>>>(data, events, iterations, errors, lasts, cold);
  } else if (lds_inflight) {
    LdsSyncInflight<<<ctas, 512, kSmem512>>>(iterations, errors, lasts, cold);
  } else if (!std::strcmp(mode, "barrier_v2")) {
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
                  (!v3 || h_lasts == 104ull * iterations) &&
                  (std::strcmp(mode, "last") || h_lasts == static_cast<unsigned long long>(iterations)) &&
                  (std::strcmp(mode, "last512") || h_lasts == static_cast<unsigned long long>(kTiles) * iterations) &&
                  (std::strcmp(mode, "la_v2") || h_lasts == static_cast<unsigned long long>(iterations)) &&
                  (std::strcmp(mode, "many") || h_lasts == ManyMerges(ctas * 8, iterations));
  std::printf("SYNC %s iterations=%d launch=%s sync=%s errors=%llu lasts=%llu us/iter=%.2f %s\n", mode, iterations,
              cudaGetErrorString(launch), cudaGetErrorString(sync), h_errors, h_lasts, ms * 1000.0 / iterations,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
