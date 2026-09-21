// SPDX-License-Identifier: BSD-3-Clause
// R8 BE-5 / B-a: does the release rule still hold when the producer is one
// role of a CTA rather than the whole CTA?
//
// Shape follows F-1: the producing threads write a tile cooperatively, each
// writer fences, the producers converge on a barrier, and one thread then
// publishes the flag the consumer polls. What changes is the granularity of
// that convergence -- a named barrier over the producing role, not
// `__syncthreads()` over the CTA, because a CTA barrier is unreachable once
// the other role is running different code.
//
// F-10 is why the tile sweep goes small: a large tile hides a missing fence,
// so the mix has to include tiles at and below 4096 floats.
//
// Arms:
//   roles      producer fences per writer, named barrier over the role, then
//              the flag -- the rule under test
//   nofence    the same without the per-writer fence  (negative control)
//   nobarrier  the same without the role barrier      (negative control)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#ifndef TILEMEGA_LITMUS_ROLE_THREADS
#define TILEMEGA_LITMUS_ROLE_THREADS 64
#endif

namespace {

__device__ __forceinline__ void RoleBarrier(int name, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(name), "r"(threads) : "memory");
}

enum Arm { kRoles = 0, kNoFence = 1, kNoBarrier = 2 };

/// Role 0 of CTA `b` produces tile `b`; role 1 of CTA `b` consumes the tile
/// of CTA `b+1`. Both roles live in one CTA -- the situation warp
/// specialization creates -- but the publication under test crosses CTAs,
/// which is where the megakernel's own events live and where a release
/// sequence is load-bearing. A within-CTA version of this test passes with
/// every fence removed, because the two roles share an L1; that is F-3's
/// warning, and it is why this is not that test.
template <int Arm_>
__global__ void Litmus(float* tiles, unsigned int* flags, unsigned int* acks,
                       int elements, int iterations, unsigned int* mismatches) {
  int const role_threads = TILEMEGA_LITMUS_ROLE_THREADS;
  int const tid = static_cast<int>(threadIdx.x);
  bool const producer = tid < role_threads;
  int const peer = (static_cast<int>(blockIdx.x) + 1) % static_cast<int>(gridDim.x);
  // Two slots, so a payload is never overwritten while its flag still names
  // it, and the same two addresses are reused on every iteration (F-3, F-10).
  size_t const stride = static_cast<size_t>(elements);
  float* mine = tiles + static_cast<size_t>(blockIdx.x) * 2 * stride;
  float const* theirs = tiles + static_cast<size_t>(peer) * 2 * stride;
  unsigned int* my_flag = flags + blockIdx.x;
  unsigned int* my_ack = acks + blockIdx.x;
  unsigned int const* peer_flag = flags + peer;
  unsigned int* peer_ack = acks + peer;

  for (int iteration = 1; iteration <= iterations; ++iteration) {
    unsigned int const value = static_cast<unsigned int>(iteration);
    if (producer) {
      // Wait until the consumer has validated everything but the slot we are
      // about to rewrite; otherwise the test would race itself.
      // Every producing thread waits for the acknowledgement itself. A
      // barrier here would supply exactly the convergence the `nobarrier`
      // arm is supposed to be missing, and the control could never fail.
      for (long long spin = 0; spin < (1ll << 24); ++spin) {
        if (atomicAdd(my_ack, 0u) + 1u >= value) break;
      }
      float* slot = mine + static_cast<size_t>(iteration & 1) * stride;
      // The signalling thread owns one element and the other writers share
      // the rest. That is the hazard F-1 names -- the thread that publishes
      // finishes first -- and without it the barrier control cannot fail,
      // because the poll's round trip is longer than the tail of an evenly
      // split write.
      if (tid == 0) {
        slot[0] = static_cast<float>(value);
      } else {
        // Threads 1..role_threads-1 cover elements 1..elements-1 between
        // them, so every element still has exactly one writer. Each store
        // depends on a load, which is what a real body's epilogue does and
        // what makes the tail of the write long enough to matter: a loop of
        // independent stores retires faster than the flag propagates, and
        // then no ordering rule can be observed at all.
        float const* previous =
            mine + static_cast<size_t>((iteration + 1) & 1) * stride;
        for (int i = tid; i < elements; i += role_threads - 1)
          slot[i] = static_cast<float>(value) + 0.5f * static_cast<float>(i & 7) +
                    0.0f * previous[i];
      }
      // Device scope: the consumer runs on another SM, so the release has to
      // reach the L2 rather than this CTA's L1.
      if (Arm_ != kNoFence) __threadfence();
      if (Arm_ != kNoBarrier) RoleBarrier(1, role_threads);
      if (tid == 0) {
        if (Arm_ != kNoFence) __threadfence();
        atomicExch(my_flag, value);
      }
    } else {
      unsigned int seen = 0;
      for (long long spin = 0; spin < (1ll << 24); ++spin) {
        seen = atomicAdd(const_cast<unsigned int*>(peer_flag), 0u);
        if (seen >= value) break;
      }
      if (seen < value) {
        atomicAdd(mismatches, 1u);
      } else {
        float const* slot =
            theirs + static_cast<size_t>(seen & 1u) * stride;
        for (int i = tid - role_threads; i < elements; i += role_threads) {
          float const want =
              i == 0 ? static_cast<float>(seen)
                     : static_cast<float>(seen) + 0.5f * static_cast<float>(i & 7);
          if (slot[i] != want) { atomicAdd(mismatches, 1u); break; }
        }
      }
      RoleBarrier(2, role_threads);
      if (tid == role_threads) atomicExch(peer_ack, seen);
    }
    __syncthreads();  // one iteration boundary, both roles participate
  }
}

}  // namespace

int main(int argc, char** argv) {
  int grid = argc > 1 ? std::atoi(argv[1]) : 128;
  int elements = argc > 2 ? std::atoi(argv[2]) : 1024;
  int iterations = argc > 3 ? std::atoi(argv[3]) : 64;
  char const* arm = argc > 4 ? argv[4] : "roles";

  float* tiles = nullptr;
  unsigned int* flags = nullptr;
  unsigned int* acks = nullptr;
  unsigned int* mismatches = nullptr;
  cudaMalloc(&tiles, sizeof(float) * static_cast<size_t>(grid) * 2 * elements);
  cudaMalloc(&flags, sizeof(unsigned int) * grid);
  cudaMalloc(&acks, sizeof(unsigned int) * grid);
  cudaMalloc(&mismatches, sizeof(unsigned int));
  cudaMemset(flags, 0, sizeof(unsigned int) * grid);
  cudaMemset(acks, 0, sizeof(unsigned int) * grid);
  cudaMemset(mismatches, 0, sizeof(unsigned int));

  int const threads = 2 * TILEMEGA_LITMUS_ROLE_THREADS;
  if (std::strcmp(arm, "roles") == 0)
    Litmus<kRoles><<<grid, threads>>>(tiles, flags, acks, elements, iterations, mismatches);
  else if (std::strcmp(arm, "nofence") == 0)
    Litmus<kNoFence><<<grid, threads>>>(tiles, flags, acks, elements, iterations, mismatches);
  else if (std::strcmp(arm, "nobarrier") == 0)
    Litmus<kNoBarrier><<<grid, threads>>>(tiles, flags, acks, elements, iterations, mismatches);
  else { std::fprintf(stderr, "unknown arm %s\n", arm); return 2; }

  cudaError_t const launch = cudaDeviceSynchronize();
  unsigned int host = 0;
  cudaMemcpy(&host, mismatches, sizeof(host), cudaMemcpyDeviceToHost);
  std::printf("LITMUS arm=%s grid=%d elements=%d iterations=%d role_threads=%d "
              "mismatches=%u status=%s\n",
              arm, grid, elements, iterations, TILEMEGA_LITMUS_ROLE_THREADS,
              host, cudaGetErrorString(launch));
  if (launch != cudaSuccess) return 3;
  return host == 0 ? 0 : 1;
}
