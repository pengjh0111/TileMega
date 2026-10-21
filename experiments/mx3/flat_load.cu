// SPDX-License-Identifier: BSD-3-Clause
// MX3-1 primitive: latency of a dependent load through a generic pointer
// (flat, LDX on xcore1000) against the same load through the global address
// space (LDG). The megakernel reads its operand pointers out of the plan's
// parameter block, so the compiler cannot prove they address global memory
// and emits flat loads; scripts/mx_mir_report.py shows them in every body
// except the GEMM mainloop.
//
// One wave chases a single cycle through a ring (stride 4099 elements, so
// consecutive loads touch different lines); the pointer is selected at run
// time between a shared array and the ring, which is what makes it generic.
// Prints, per ring size and mode, the median and p99 over `reps` launches.
//
// Usage: flat_load [reps]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Box { unsigned const* ring; };

template <bool Global>
__global__ void Chase(Box const* box, int steps, unsigned* sink, unsigned long long* ns) {
  __shared__ unsigned scratch[64];
  scratch[threadIdx.x] = 0;
  unsigned const* p = steps < 0 ? scratch : box->ring;
  unsigned i = 0;
  __syncthreads();
  unsigned long long const t0 = __builtin_mxc_get_realtime();
  for (int s = 0; s < steps; ++s) {
    if constexpr (Global) i = ((__attribute__((address_space(1))) unsigned const*)p)[i];
    else i = p[i];
  }
  unsigned long long const t1 = __builtin_mxc_get_realtime();
  if (threadIdx.x == 0) {
    sink[0] = i;
    ns[0] = (t1 - t0) * 10;  // 100 MHz realtime counter
  }
}

int main(int argc, char** argv) {
  int const reps = argc > 1 ? std::atoi(argv[1]) : 21;
  int const steps = 4000;
  unsigned *sink;
  unsigned long long* ns;
  Box* box;
  cudaMalloc(&sink, 64);
  cudaMalloc(&ns, 64);
  cudaMalloc(&box, sizeof(Box));
  std::printf("ring_kib,mode,median_ns,p99_ns\n");
  for (int kib : {4, 256, 16384}) {
    int const n = kib * 256;
    std::vector<unsigned> host(n);
    for (int k = 0; k < n; ++k) host[k] = unsigned((k + 4099) % n);
    unsigned* ring;
    cudaMalloc(&ring, std::size_t(n) * 4);
    cudaMemcpy(ring, host.data(), std::size_t(n) * 4, cudaMemcpyHostToDevice);
    Box h{ring};
    cudaMemcpy(box, &h, sizeof(Box), cudaMemcpyHostToDevice);
    std::vector<double> flat, global;
    for (int r = 0; r < reps + 1; ++r) {  // first launch of each mode is warm-up
      for (int mode = 0; mode < 2; ++mode) {
        if (mode) Chase<true><<<1, 64>>>(box, steps, sink, ns);
        else Chase<false><<<1, 64>>>(box, steps, sink, ns);
        unsigned long long t = 0;
        cudaMemcpy(&t, ns, 8, cudaMemcpyDeviceToHost);
        if (r) (mode ? global : flat).push_back(double(t) / steps);
      }
    }
    for (int mode = 0; mode < 2; ++mode) {
      auto& v = mode ? global : flat;
      std::sort(v.begin(), v.end());
      std::printf("%d,%s,%.1f,%.1f\n", kib, mode ? "global" : "flat", v[v.size() / 2],
                  v[std::min(v.size() - 1, v.size() * 99 / 100)]);
    }
    cudaFree(ring);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
