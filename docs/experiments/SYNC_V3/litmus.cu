// SPDX-License-Identifier: BSD-3-Clause
// R4 release litmus, derived from SYNC_V2 without changing its artifacts.
// Two fixed stresses separate cache visibility (skew=0, acquire=0) from
// publication-before-writers-finish (skew=2000000, acquire=1). The latter
// delays nonpublisher warps on odd CTAs identically in every arm. Its negative
// control removes ONLY the release barrier; consumer barriers stay present.
// The fill, address reuse, monotone epochs and expected values are unchanged.
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// §8.4: one counter per cache line.
struct alignas(128) EventCounter {
  u64 v;
  char pad[128 - sizeof(u64)];
};

enum Release : int {
  kPerWriter = 0,
  kThread0Fence = 1,
  kNoBarrier = 2,
  kNoFence = 3
};

struct Config {
  int release = kPerWriter;
  int grid = 128;
  int block = 256;
  int tile = 1024;
  int fanin = 4;
  int iters = 8;
  uint32_t nonce = 0;
  int no_backoff = 1;  // hostile by default: F-3 stressors are not monotonic
  int acquire_fence = 1;
  unsigned long long writer_skew_cycles = 0;
  bool verbose = false;
};

__host__ __device__ __forceinline__ int producer_of(int b, int k, int G) {
  return (b + 1 + k) % G;  // circular: needs whole-grid co-residency
}

template <int R>
__global__ __launch_bounds__(256) void litmus_kernel(
    float* __restrict__ data, float* __restrict__ out,
    EventCounter* __restrict__ ev, EventCounter* __restrict__ consumed, int G,
    int tile, int fanin, int iters, uint32_t nonce, int no_backoff,
    int acquire_fence, unsigned long long writer_skew_cycles) {
  extern __shared__ float smem[];
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  for (int iter = 0; iter < iters; ++iter) {
    // Address reuse: one buffer for every round, so a consumer that reads early
    // sees the *previous* round's values rather than uninitialised memory.
    // Hold off until last round's readers are done (F-3).
    if (iter > 0) {
      if (tid == 0) {
        const u64 need_consumed = (u64)iter * fanin;
        while (atomicAdd(&consumed[b].v, 0ull) < need_consumed) {
        }
      }
      __syncthreads();
    }

    // CTA-cooperative write: the whole block produces the tile, which is the
    // case where a lone thread-0 fence could fail to cover the other writers.
    if ((b & 1) && tid >= 32 && writer_skew_cycles) {
      const unsigned long long start = clock64();
      while (clock64() - start < writer_skew_cycles) {}
    }
    float* mine = data + (size_t)b * tile;
    for (int i = tid; i < tile; i += nthreads) {
      mine[i] = FillValue(FillMode::kAlternating, nonce, iter, b, i);
    }

    // ---- release: the only thing that varies across arms.
    if (R == kPerWriter || R == kNoBarrier) __threadfence();
    if (R != kNoBarrier) __syncthreads();
    if (R == kThread0Fence && tid == 0) __threadfence();
    if (tid == 0) atomicExch(&ev[b].v, (u64)(iter + 1));  // §8.2 monotone

    // ---- acquire: identical in every arm.
    const u64 need = (u64)(iter + 1);
    if (tid == 0) {
      for (int k = 0; k < fanin; ++k) {
        int p = producer_of(b, k, G);
        while (atomicAdd(&ev[p].v, 0ull) < need) {
          if (!no_backoff) __nanosleep(64);
        }
      }
    }
    __syncthreads();
    // V_A drops this fence in its `no_fence` arm too (event_sync.cu:186).
    // Kept in every arm it invalidates the L1 the stale read needs (F-10),
    // which is the other reason that first scan was vacuous.
    if (acquire_fence) __threadfence();

    float acc = 0.f;
    for (int k = 0; k < fanin; ++k) {
      const float* src = data + (size_t)producer_of(b, k, G) * tile;
      for (int i = tid; i < tile; i += nthreads) acc += src[i];
    }
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    const int lane = tid & 31, warp = tid >> 5;
    const int nwarps = (nthreads + 31) / 32;
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
      float w = (lane < nwarps) ? smem[lane] : 0.f;
      for (int off = 16; off > 0; off >>= 1)
        w += __shfl_down_sync(0xffffffffu, w, off);
      if (lane == 0) out[(size_t)iter * G + b] = w;
    }
    __syncthreads();

    if (tid == 0) {
      for (int k = 0; k < fanin; ++k)
        atomicAdd(&consumed[producer_of(b, k, G)].v, 1ull);
    }
    __syncthreads();
  }
}

// Integer-valued fills in [0,255] keep this exact in fp32, so a mismatch is a
// synchronisation bug and never a rounding difference.
static bool verify(Config const& c, int G, std::vector<float> const& out,
                   std::string* why) {
  char buf[256];
  // Sum each producer's tile once.  The naive nesting recomputes it `fanin`
  // times, which at grid 256 / tile 16384 is 134M hashes of host time per
  // process and dominates the run.  Values are in [0,255] and fanin*tile
  // <= 65536, so every sum is below 2^24 and stays exact in both double and
  // the float compare.
  std::vector<double> tile_sum((size_t)c.iters * G, 0.0);
  for (int iter = 0; iter < c.iters; ++iter) {
    for (int p = 0; p < G; ++p) {
      double s = 0.0;
      for (int i = 0; i < c.tile; ++i)
        s += FillValue(FillMode::kAlternating, c.nonce, iter, p, i);
      tile_sum[(size_t)iter * G + p] = s;
    }
  }
  for (int iter = 0; iter < c.iters; ++iter) {
    for (int b = 0; b < G; ++b) {
      double e = 0.0;
      for (int k = 0; k < c.fanin; ++k)
        e += tile_sum[(size_t)iter * G + producer_of(b, k, G)];
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
  return true;
}

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
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[i]);
        std::exit(3);
      }
      return argv[++i];
    };
    std::string a = argv[i];
    if (a == "--release")
      c.release = parse_enum(
          next(), {"per_writer", "thread0_fence", "no_barrier", "no_fence"});
    else if (a == "--grid") c.grid = std::atoi(next());
    else if (a == "--tile") c.tile = std::atoi(next());
    else if (a == "--writer-skew-cycles") c.writer_skew_cycles = std::strtoull(next(), nullptr, 10);
    else if (a == "--fanin") c.fanin = std::atoi(next());
    else if (a == "--iters") c.iters = std::atoi(next());
    else if (a == "--nonce") c.nonce = (uint32_t)std::strtoul(next(), nullptr, 0);
    else if (a == "--backoff") c.no_backoff = 0;
    else if (a == "--no-acquire-fence") c.acquire_fence = 0;
    else if (a == "-v" || a == "--verbose") c.verbose = true;
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 3; }
  }
  if (!c.nonce) c.nonce = PickNonce();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int G = c.grid;
  if (G < 2) { std::fprintf(stderr, "need grid >= 2\n"); return 3; }
  if (c.fanin >= G) { std::fprintf(stderr, "need fanin < grid\n"); return 3; }

  const int nwarps = (c.block + 31) / 32;
  const size_t smem_bytes = (size_t)nwarps * sizeof(float);

  void* kfn = nullptr;
#define PICK(R) if (c.release == (R)) kfn = (void*)litmus_kernel<R>;
  PICK(kPerWriter) PICK(kThread0Fence) PICK(kNoBarrier) PICK(kNoFence)
#undef PICK

  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kfn, c.block,
                                                           smem_bytes));
  const int resident_cap = occ * prop.multiProcessorCount;

  if (G > resident_cap) { std::fprintf(stderr, "grid exceeds resident cap\n"); return 3; }

  const size_t data_elems = (size_t)G * c.tile;  // reuse: one round of storage
  const size_t out_elems = (size_t)c.iters * G;

  float *d_data = nullptr, *d_out = nullptr;
  EventCounter *d_ev = nullptr, *d_consumed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, data_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ev, (size_t)G * sizeof(EventCounter)));
  CUDA_CHECK(cudaMalloc(&d_consumed, (size_t)G * sizeof(EventCounter)));
  CUDA_CHECK(cudaMemset(d_ev, 0, (size_t)G * sizeof(EventCounter)));
  CUDA_CHECK(cudaMemset(d_consumed, 0, (size_t)G * sizeof(EventCounter)));
  // d_data and d_out stay unzeroed on purpose: last process's bytes are what
  // the alternating fill exists to distinguish from this one's answer.

  if (c.verbose) {
    std::printf("[cfg] gpu=%s nsm=%d grid=%d block=%d occ=%d cap=%d tile=%d "
                "fanin=%d iters=%d no_backoff=%d acquire_fence=%d "
                "nonce=0x%08x\n",
                prop.name, prop.multiProcessorCount, G, c.block, occ,
                resident_cap, c.tile, c.fanin, c.iters, c.no_backoff,
                c.acquire_fence, c.nonce);
  }
  std::printf("OCCUPANCY blocks_per_sm=%d num_sms=%d resident_cap=%d grid=%d "
              "co_resident=%s\n",
              occ, prop.multiProcessorCount, resident_cap, G,
              (G <= resident_cap) ? "yes" : "NO");
  std::fflush(stdout);

#define LAUNCH(R)                                                              \
  if (c.release == (R))                                                        \
    litmus_kernel<R><<<G, c.block, smem_bytes>>>(                              \
        d_data, d_out, d_ev, d_consumed, G, c.tile, c.fanin, c.iters, c.nonce, \
        c.no_backoff, c.acquire_fence, c.writer_skew_cycles);
  LAUNCH(kPerWriter) LAUNCH(kThread0Fence) LAUNCH(kNoBarrier) LAUNCH(kNoFence)
#undef LAUNCH

  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {
    std::printf("RESULT status=launch_error detail=%s nonce=0x%08x\n",
                cudaGetErrorString(le), c.nonce);
    return 2;
  }
  cudaError_t se = cudaDeviceSynchronize();
  if (se != cudaSuccess) {
    std::printf("RESULT status=exec_error detail=%s nonce=0x%08x\n",
                cudaGetErrorString(se), c.nonce);
    return 2;
  }

  std::vector<float> h_out(out_elems);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::string why;
  bool ok = verify(c, G, h_out, &why);
  std::printf("RESULT status=%s nonce=0x%08x%s%s\n", ok ? "pass" : "MISMATCH",
              c.nonce, ok ? "" : " detail=", ok ? "" : why.c_str());
  cudaFree(d_data); cudaFree(d_out); cudaFree(d_ev); cudaFree(d_consumed);
  return ok ? 0 : 1;
}
