// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef TILEMEGA_MIN_BLOCKS_PER_SM
#define TILEMEGA_MIN_BLOCKS_PER_SM 1
#endif
#ifndef TILEMEGA_COLD_START_TIMING
#define TILEMEGA_COLD_START_TIMING 0
#endif
static_assert(TILEMEGA_MIN_BLOCKS_PER_SM > 0, "positive launch residency required");

namespace tilemega::codegen::benchmark {
inline void Check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "benchmark: %s\n", cudaGetErrorString(error));
    std::exit(2);
  }
}
inline int Count(char const* name, int fallback, int minimum) {
  char const* value = std::getenv(name);
  if (!value) return fallback;
  char* end = nullptr;
  errno = 0;
  long const n = std::strtol(value, &end, 10);
  if (errno || end == value || *end || n < minimum || n > INT_MAX) {
    std::fprintf(stderr, "%s must be an integer >= %d\n", name, minimum);
    std::exit(2);
  }
  return static_cast<int>(n);
}
struct Settings {
  int warmup = TILEMEGA_COLD_START_TIMING ? 0 : Count("TILEMEGA_WARMUP", 5, 0);
  int repeat = TILEMEGA_COLD_START_TIMING ? 1 : Count("TILEMEGA_REPEAT", 11, 1);
};
template <class Launch>
float Time(Launch launch, bool timed) {
  if (!timed) {
    launch();
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    return 0;
  }
  cudaEvent_t start, stop;
  Check(cudaEventCreate(&start)); Check(cudaEventCreate(&stop));
  Check(cudaEventRecord(start));
  launch();
  Check(cudaEventRecord(stop)); Check(cudaEventSynchronize(stop));
  Check(cudaGetLastError());
  float ms = 0;
  Check(cudaEventElapsedTime(&ms, start, stop));
  Check(cudaEventDestroy(start)); Check(cudaEventDestroy(stop));
  return ms;
}
// Each sample is an independent forward with identical inputs. Reset is
// outside timing. This must NOT wrap the monotonic iteration/ABA test.
template <class Reset, class Launch>
float Forward(Settings const& settings, Reset reset, Launch launch) {
  for (int i = 0; i < settings.warmup; ++i) {
    reset(); launch(false);
  }
  std::vector<float> samples;
  samples.reserve(settings.repeat);
  for (int i = 0; i < settings.repeat; ++i) {
    reset(); samples.push_back(launch(true));
  }
  std::sort(samples.begin(), samples.end());
  std::size_t const middle = samples.size() / 2;
  return samples.size() % 2 ? samples[middle]
      : (samples[middle - 1] + samples[middle]) * 0.5f;
}
}  // namespace tilemega::codegen::benchmark
