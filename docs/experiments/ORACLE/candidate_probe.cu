// SPDX-License-Identifier: BSD-3-Clause
// Tier-1 of §4.1: enumerate the GEMM implementation candidates and answer
// legality from CUTLASS's constexpr traits alone.  This translation unit
// declares no __global__ function, so nothing here is compiled to a kernel.
#include <tilemega/Backend/CutlassGemmCandidate.h>

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

constexpr int kTileM[] = {16, 32, 64, 128, 256};
constexpr int kTileN[] = {16, 32, 64, 128, 256};
constexpr int kTileK[] = {8, 16, 32};
constexpr int kStages[] = {2, 3, 4, 5};

int smem_budget = 101376;  // sm_89 opt-in maximum; overridden by argv[1]

template <int M, int N, int K, int S>
void report() {
  using C = tilemega::backend::GemmCandidate<M, N, K, S>;
  auto info = C::Traits();
  bool fits = info.shape_legal && info.smem_bytes > 0 &&
              info.smem_bytes <= smem_budget;
  std::printf(
      "CANDIDATE m=%d n=%d k=%d stages=%d shape_legal=%d smem=%d threads=%d "
      "cluster=%dx%dx%d align=%d/%d arch=sm%d fits=%d\n",
      info.tile_m, info.tile_n, info.tile_k, info.stages, (int)info.shape_legal,
      info.smem_bytes, info.threads, info.cluster.m, info.cluster.n,
      info.cluster.k, info.alignment.a, info.alignment.b, info.arch_sm,
      (int)fits);
}

template <int Mi, int Ni, int Ki, int... Si>
void walk_stages(std::integer_sequence<int, Si...>) {
  (report<kTileM[Mi], kTileN[Ni], kTileK[Ki], kStages[Si]>(), ...);
}

template <int Mi, int Ni, int... Ki>
void walk_k(std::integer_sequence<int, Ki...>) {
  (walk_stages<Mi, Ni, Ki>(
       std::make_integer_sequence<int, sizeof(kStages) / sizeof(int)>{}),
   ...);
}

template <int Mi, int... Ni>
void walk_n(std::integer_sequence<int, Ni...>) {
  (walk_k<Mi, Ni>(
       std::make_integer_sequence<int, sizeof(kTileK) / sizeof(int)>{}),
   ...);
}

template <int... Mi>
void walk_m(std::integer_sequence<int, Mi...>) {
  (walk_n<Mi>(std::make_integer_sequence<int, sizeof(kTileN) / sizeof(int)>{}),
   ...);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) smem_budget = std::atoi(argv[1]);
  std::printf("BUDGET smem=%d\n", smem_budget);
  walk_m(std::make_integer_sequence<int, sizeof(kTileM) / sizeof(int)>{});
  return 0;
}
