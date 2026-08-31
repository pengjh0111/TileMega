// SPDX-License-Identifier: BSD-3-Clause
//
// V-A : cross-block event synchronisation for the TileMega megakernel.
//
// This experiment decides whether the core mechanism of the whole project is
// sound on this backend: can one CTA publish a tile to global memory, signal a
// monotonic event counter, and have a *different* CTA in the same persistent
// launch observe the data correctly -- including when the consumer performs a
// cross-warp reduction over what it read?
//
// Rules implemented verbatim from skeleton §8.1 - §8.5:
//   §8.1  spin must be single-threaded, collective barrier at a non-divergent point
//   §8.2  monotonic counter: needed = num_triggers * iteration_num, never reset
//   §8.3  spin must back off (__nanosleep)
//   §8.4  event counters padded to 128B against false sharing
//   §8.5  release order: data write -> __threadfence() -> atomicExch
//
// The binary is one program with orthogonal switches so that every cell of the
// V-A matrix is a single-variable change from its neighbour:
//
//   --variant  elementwise | reduce      what the consumer computes
//   --deps     circular | shared | exclusive
//   --sync     correct | no_barrier | no_fence | none | allthread |
//              correct_hostile | no_fence_hostile | barrier_in_spin
//   --fill     alt | const               alternating-fill verifier on/off
//
// A-1 = --variant elementwise --deps circular --fill alt
// A-2 = --variant reduce      --deps circular --fill const
// A-3 = --variant reduce      --deps circular --fill alt
// A-4 = --variant reduce      --deps shared    --fill alt
// A-5 = --variant reduce      --deps exclusive --fill alt
//
// The non-`correct` sync modes are NEGATIVE CONTROLS.  A test that has never
// been observed to fail is not evidence; these modes prove the checker can
// actually detect a broken synchronisation.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "alternating_fill.h"

using tilemega::testing::FillMode;
using tilemega::testing::FillValue;
using tilemega::testing::PickNonce;

using u64 = unsigned long long;

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[cuda] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,  \
                   cudaGetErrorString(e_));                                    \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// §8.4 event buffer layout: one counter per cache line.
// ---------------------------------------------------------------------------
struct alignas(128) EventCounter {
  u64 v;
  char pad[128 - sizeof(u64)];
};

enum Variant : int { kElementwise = 0, kReduce = 1 };
enum Deps : int { kCircular = 0, kShared = 1, kExclusive = 2, kBackward = 3 };
enum SyncMode : int {
  kCorrect = 0,     // §8.1 compliant
  kNoBarrier = 1,   // tid0 polls, no __syncthreads() -> other threads race ahead
  kNoFence = 2,     // barriers kept, both __threadfence() removed
  kNone = 3,        // no wait at all -- pure race, checker sanity
  kAllThread = 4,   // every thread polls (contention/control experiment)
  kCorrectHostile = 5,  // reused data, no spin backoff, fences retained
  kNoFenceHostile = 6,  // same hostile layout, both fences removed
  kBarrierInSpin = 7    // deliberately illegal collective inside divergent spin
};

struct Config {
  int variant = kReduce;
  int deps = kCircular;
  int sync = kCorrect;
  FillMode fill = FillMode::kAlternating;
  int grid = 0;      // 0 => num_sms (resolved from device, never hard-coded)
  int block = 256;
  int tile = 1024;   // floats per producer tile
  int fanin = 4;     // producers each consumer waits on
  int iters = 4;     // rounds -> exercises the monotonic counter §8.2
  uint32_t nonce = 0;
  bool verbose = false;
  bool reuse_data = false;
  bool no_backoff = false;
  bool tile_was_set = false;
};

// Producer index that consumer `b`, slot `k`, reads from.
// - circular : b+1+k (mod G).  Backwards in launch order => genuinely needs
//              co-residency of the whole grid, which is the megakernel case.
// - shared   : k.  Every consumer reads the SAME producers (many readers).
// - exclusive: b+1 (mod G), fanin forced to 1.  One reader per producer.
// - backward : b+G/2 (mod G), fanin forced to 1.  This non-streaming
//              wait-for graph is the V-J co-residency negative control.
__host__ __device__ __forceinline__ int producer_of(int deps, int b, int k, int G) {
  switch (deps) {
    case kShared:    return k % G;
    case kExclusive: return (b + 1) % G;
    case kBackward:  return (b + G / 2) % G;
    default:         return (b + 1 + k) % G;
  }
}

__host__ __device__ __forceinline__ int fanin_of(int deps, int fanin) {
  return (deps == kExclusive || deps == kBackward) ? 1 : fanin;
}

// ---------------------------------------------------------------------------
// The kernel.  One persistent launch; every block is first a producer and then
// a consumer, for `iters` rounds.
// ---------------------------------------------------------------------------
template <int VARIANT, int SYNC>
__global__ __launch_bounds__(256) void event_kernel(float* __restrict__ data,
                             float* __restrict__ out,
                             EventCounter* __restrict__ ev,
                             EventCounter* __restrict__ consumed,
                             int G, int tile, int fanin, int deps, int iters,
                             int fill_mode, uint32_t nonce, int reuse_data,
                             int no_backoff) {
  extern __shared__ float smem[];
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const FillMode fm = static_cast<FillMode>(fill_mode);
  const int fin = fanin_of(deps, fanin);

  for (int iter = 0; iter < iters; ++iter) {
    // -------------------------------------------------- producer
    if (reuse_data && iter > 0) {
      if (tid == 0) {
        const u64 need_consumed = (u64)iter * fin;
        while (atomicAdd(&consumed[b].v, 0ull) < need_consumed) {}
      }
      __syncthreads();
    }
    const size_t data_round = reuse_data ? 0 : (size_t)iter * G * tile;
    float* mine = data + data_round + (size_t)b * tile;
    for (int i = tid; i < tile; i += nthreads) {
      mine[i] = FillValue(fm, nonce, iter, b, i);
    }

    if (SYNC != kNoFence && SYNC != kNoFenceHostile) {
      // Every thread releases its own writes, THEN we join.  The snippet in
      // skeleton §8.5 shows only `write; fence; atomicExch` which is correct
      // for a single producing thread; with a whole CTA producing, the fence
      // must precede the barrier so that the signalling thread's atomic is
      // ordered after *all* threads' writes.  See docs/FINDINGS.md F-1.
      __threadfence();
    }
    __syncthreads();
    if (SYNC != kNone && tid == 0) {
      // §8.2 monotonic: never reset, value == round number.
      atomicExch(&ev[b].v, (u64)(iter + 1));
    }

    // -------------------------------------------------- consumer
    const u64 need = (u64)(iter + 1);

    if (SYNC == kCorrect || SYNC == kNoBarrier || SYNC == kNoFence ||
        SYNC == kCorrectHostile || SYNC == kNoFenceHostile) {
      // §8.1: single-thread poll, §8.3: backed-off spin.
      if (tid == 0) {
        for (int k = 0; k < fin; ++k) {
          int p = producer_of(deps, b, k, G);
          while (atomicAdd(&ev[p].v, 0ull) < need) {
            if (!no_backoff) __nanosleep(64);
          }
        }
      }
      if (SYNC != kNoBarrier) {
        __syncthreads();  // §8.1 collective sync at a non-divergent point
      }
      if (SYNC != kNoFence && SYNC != kNoFenceHostile) {
        __threadfence();  // acquire
      }
    }
    // kNone: no wait at all.  kAllThread handled inside the read loop below.

    if (VARIANT == kElementwise) {
      // ---- A-1: no cross-warp communication at all.
      float* dst = out + (size_t)iter * G * tile + (size_t)b * tile;
      for (int i = tid; i < tile; i += nthreads) {
        float acc = 0.f;
        for (int k = 0; k < fin; ++k) {
          int p = producer_of(deps, b, k, G);
          acc += data[data_round + (size_t)p * tile + i];
        }
        dst[i] = acc;
      }
    } else {
      // ---- A-2..A-5: cross-warp reduction.  This is the case the task calls
      // out as mandatory: it exercises collective barriers after event waits.
      float acc = 0.f;
      for (int k = 0; k < fin; ++k) {
        int p = producer_of(deps, b, k, G);
        if (SYNC == kAllThread || SYNC == kBarrierInSpin) {
          // allthread is a contention control. barrier_in_spin is the true
          // negative control: threads execute a collective a different number
          // of times because they observe the event at different moments.
          while (atomicAdd(&ev[p].v, 0ull) < need) {
            if (SYNC == kBarrierInSpin) __syncthreads();
            if (!no_backoff) __nanosleep(64);
          }
        }
        const float* src = data + data_round + (size_t)p * tile;
        for (int i = tid; i < tile; i += nthreads) {
          acc += src[i];
        }
        if (SYNC == kAllThread) {
          __syncthreads();  // convergence after independently finishing spins
        }
      }

      // warp reduce
      for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
      }
      // cross-warp reduce through shared memory
      const int lane = tid & 31;
      const int warp = tid >> 5;
      const int nwarps = (nthreads + 31) / 32;
      if (lane == 0) smem[warp] = acc;
      __syncthreads();
      if (warp == 0) {
        float w = (lane < nwarps) ? smem[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
          w += __shfl_down_sync(0xffffffffu, w, off);
        }
        if (lane == 0) out[(size_t)iter * G + b] = w;
      }
      __syncthreads();
    }

    if (reuse_data) {
      // Prevent iteration i+1 from overwriting a producer tile until every
      // circular consumer has completed all reads from iteration i.
      if (tid == 0) {
        for (int k = 0; k < fin; ++k) {
          int p = producer_of(deps, b, k, G);
          atomicAdd(&consumed[p].v, 1ull);
        }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side expected values.  Integer-valued fills keep this exact in fp32.
// ---------------------------------------------------------------------------
static bool verify(Config const& c, int G, std::vector<float> const& out,
                   std::string* why) {
  const int fin = fanin_of(c.deps, c.fanin);
  char buf[256];
  if (c.variant == kElementwise) {
    for (int iter = 0; iter < c.iters; ++iter) {
      for (int b = 0; b < G; ++b) {
        for (int i = 0; i < c.tile; ++i) {
          double e = 0.0;
          for (int k = 0; k < fin; ++k) {
            e += FillValue(c.fill, c.nonce, iter, producer_of(c.deps, b, k, G), i);
          }
          float got = out[(size_t)iter * G * c.tile + (size_t)b * c.tile + i];
          if (got != (float)e) {
            std::snprintf(buf, sizeof buf,
                          "iter=%d block=%d elem=%d expected=%.1f got=%.1f",
                          iter, b, i, e, got);
            *why = buf;
            return false;
          }
        }
      }
    }
  } else {
    for (int iter = 0; iter < c.iters; ++iter) {
      for (int b = 0; b < G; ++b) {
        double e = 0.0;
        for (int k = 0; k < fin; ++k) {
          int p = producer_of(c.deps, b, k, G);
          for (int i = 0; i < c.tile; ++i) {
            e += FillValue(c.fill, c.nonce, iter, p, i);
          }
        }
        float got = out[(size_t)iter * G + b];
        if (got != (float)e) {
          std::snprintf(buf, sizeof buf,
                        "iter=%d block=%d expected=%.1f got=%.1f delta=%.1f",
                        iter, b, e, got, (double)got - e);
          *why = buf;
          return false;
        }
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
static int parse_enum(const char* v, std::vector<const char*> const& names) {
  for (size_t i = 0; i < names.size(); ++i)
    if (std::strcmp(v, names[i]) == 0) return (int)i;
  std::fprintf(stderr, "bad enum value '%s'\n", v);
  std::exit(3);
}

int main(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(3); }
      return argv[++i];
    };
    std::string a = argv[i];
    if (a == "--variant") c.variant = parse_enum(next(), {"elementwise", "reduce"});
    else if (a == "--deps") c.deps = parse_enum(next(), {"circular", "shared", "exclusive", "backward"});
    else if (a == "--sync") c.sync = parse_enum(next(), {"correct", "no_barrier", "no_fence", "none", "allthread", "correct_hostile", "no_fence_hostile", "barrier_in_spin"});
    else if (a == "--fill") c.fill = (std::strcmp(next(), "const") == 0) ? FillMode::kConstant : FillMode::kAlternating;
    else if (a == "--grid") c.grid = std::atoi(next());
    else if (a == "--block") c.block = std::atoi(next());
    else if (a == "--tile") { c.tile = std::atoi(next()); c.tile_was_set = true; }
    else if (a == "--fanin") c.fanin = std::atoi(next());
    else if (a == "--iters") c.iters = std::atoi(next());
    else if (a == "--nonce") c.nonce = (uint32_t)std::strtoul(next(), nullptr, 0);
    else if (a == "--reuse-data") c.reuse_data = true;
    else if (a == "--no-backoff") c.no_backoff = true;
    else if (a == "-v" || a == "--verbose") c.verbose = true;
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 3; }
  }
  if (c.sync == kCorrectHostile || c.sync == kNoFenceHostile) {
    c.reuse_data = true;
    c.no_backoff = true;
    if (!c.tile_was_set) c.tile = 8192;
  }
  if (c.sync == kBarrierInSpin && c.variant != kReduce) {
    std::fprintf(stderr, "barrier_in_spin requires --variant reduce\n");
    return 3;
  }
  if (c.block != 256) {
    std::fprintf(stderr, "this experiment is compiled with __launch_bounds__(256) and requires --block 256\n");
    return 3;
  }
  if (!c.nonce) c.nonce = PickNonce();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // Grid defaults to the SM count *read from the device* -- never a literal.
  const int G = c.grid > 0 ? c.grid : prop.multiProcessorCount;
  if (G < 2) { std::fprintf(stderr, "need grid >= 2\n"); return 3; }

  const int nwarps = (c.block + 31) / 32;
  const size_t smem_bytes = (size_t)nwarps * sizeof(float);

  // Resolve the kernel instantiation.
  void* kfn = nullptr;
#define PICK(V, S) \
  if (c.variant == (V) && c.sync == (S)) kfn = (void*)event_kernel<V, S>;
  PICK(kElementwise, kCorrect) PICK(kElementwise, kNoBarrier) PICK(kElementwise, kNoFence)
  PICK(kElementwise, kNone) PICK(kElementwise, kAllThread)
  PICK(kElementwise, kCorrectHostile) PICK(kElementwise, kNoFenceHostile)
  PICK(kElementwise, kBarrierInSpin)
  PICK(kReduce, kCorrect) PICK(kReduce, kNoBarrier) PICK(kReduce, kNoFence)
  PICK(kReduce, kNone) PICK(kReduce, kAllThread)
  PICK(kReduce, kCorrectHostile) PICK(kReduce, kNoFenceHostile)
  PICK(kReduce, kBarrierInSpin)
#undef PICK

  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kfn, c.block, smem_bytes));
  const int resident_cap = occ * prop.multiProcessorCount;

  const size_t data_elems = (c.reuse_data ? 1u : (size_t)c.iters) * G * c.tile;
  const size_t out_elems =
      (c.variant == kElementwise) ? data_elems : (size_t)c.iters * G;

  float *d_data = nullptr, *d_out = nullptr;
  EventCounter* d_ev = nullptr;
  EventCounter* d_consumed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, data_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ev, (size_t)G * sizeof(EventCounter)));
  CUDA_CHECK(cudaMalloc(&d_consumed, (size_t)G * sizeof(EventCounter)));
  CUDA_CHECK(cudaMemset(d_ev, 0, (size_t)G * sizeof(EventCounter)));
  CUDA_CHECK(cudaMemset(d_consumed, 0, (size_t)G * sizeof(EventCounter)));
  // NOTE: d_data and d_out are deliberately NOT zeroed.  Leaving whatever the
  // previous process wrote there is the whole point of the alternating fill.

  if (c.verbose) {
    std::printf("[cfg] gpu=%s sm=%d.%d nsm=%d grid=%d block=%d occ=%d cap=%d "
                "tile=%d fanin=%d iters=%d reuse=%d no_backoff=%d nonce=0x%08x\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, G,
                c.block, occ, resident_cap, c.tile, fanin_of(c.deps, c.fanin),
                c.iters, (int)c.reuse_data, (int)c.no_backoff, c.nonce);
  }
  // Machine-readable line the harness greps for.
  std::printf("OCCUPANCY blocks_per_sm=%d num_sms=%d resident_cap=%d grid=%d "
              "co_resident=%s\n",
              occ, prop.multiProcessorCount, resident_cap, G,
              (G <= resident_cap) ? "yes" : "NO");
  std::fflush(stdout);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

#define LAUNCH(V, S)                                                           \
  if (c.variant == (V) && c.sync == (S))                                       \
    event_kernel<V, S><<<G, c.block, smem_bytes>>>(                            \
        d_data, d_out, d_ev, d_consumed, G, c.tile, c.fanin, c.deps, c.iters,  \
        (int)c.fill, c.nonce, (int)c.reuse_data, (int)c.no_backoff);
  LAUNCH(kElementwise, kCorrect) LAUNCH(kElementwise, kNoBarrier) LAUNCH(kElementwise, kNoFence)
  LAUNCH(kElementwise, kNone) LAUNCH(kElementwise, kAllThread)
  LAUNCH(kElementwise, kCorrectHostile) LAUNCH(kElementwise, kNoFenceHostile)
  LAUNCH(kElementwise, kBarrierInSpin)
  LAUNCH(kReduce, kCorrect) LAUNCH(kReduce, kNoBarrier) LAUNCH(kReduce, kNoFence)
  LAUNCH(kReduce, kNone) LAUNCH(kReduce, kAllThread)
  LAUNCH(kReduce, kCorrectHostile) LAUNCH(kReduce, kNoFenceHostile)
  LAUNCH(kReduce, kBarrierInSpin)
#undef LAUNCH

  CUDA_CHECK(cudaEventRecord(stop));

  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {
    std::printf("RESULT status=launch_error detail=%s nonce=0x%08x\n",
                cudaGetErrorString(le), c.nonce);
    return 2;
  }
  cudaError_t se = cudaEventSynchronize(stop);
  if (se != cudaSuccess) {
    std::printf("RESULT status=exec_error detail=%s nonce=0x%08x\n",
                cudaGetErrorString(se), c.nonce);
    return 2;
  }
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  std::printf("KERNEL_MS %.6f\n", kernel_ms);

  std::vector<float> h_out(out_elems);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::string why;
  bool ok = verify(c, G, h_out, &why);
  std::printf("RESULT status=%s nonce=0x%08x%s%s\n", ok ? "pass" : "MISMATCH",
              c.nonce, ok ? "" : " detail=", ok ? "" : why.c_str());
  cudaFree(d_data); cudaFree(d_out); cudaFree(d_ev);
  cudaFree(d_consumed); cudaEventDestroy(start); cudaEventDestroy(stop);
  return ok ? 0 : 1;
}
