// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeDependencies.h>
#include <tilemega/Codegen/executor/CountedDependency.cuh>
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace tilemega::codegen;
using executor::CountedDependency;
using executor::LastArriver;
void Check(cudaError_t error) { if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::abort(); } }
__host__ __device__ bool Selected(int producer, int consumer, int consumers) {
  return consumer + 1 < consumers && (producer + 2 * consumer) % 3 != 1;
}
__host__ __device__ int Value(int producer, int lane, int epoch) {
  return (producer + 1) * 101 + lane * 3 + epoch * 17;
}
__global__ void TableTasks(RuntimeDependencyTableView table, int producers, int consumers,
    int* values, unsigned long long* events, int* output, int epoch) {
  int lane = threadIdx.x, task = blockIdx.x;
  if (task < producers) {
    if (lane == 0) __nanosleep(unsigned((task * 17 + epoch) % 11) * 100);
    values[task * 128 + lane] = Value(task, lane, epoch);
    CountedDependency::Publish(events + task, 1);
  } else {
    int consumer = task - producers;
    bool valid = VisitDependencyTable(table, consumer, producers, [&](RuntimeWindowBounds bounds) {
      for (int p = bounds.first; p < bounds.past; ++p)
        CountedDependency::Wait(events + p, epoch + 1);
    });
    if (!valid) asm volatile("trap;");
    int sum = 0;
    for (int p = 0; p < producers; ++p)
      if (Selected(p, consumer, consumers)) sum += values[p * 128 + lane];
    output[consumer * 128 + lane] = sum;
  }
}
__global__ void CountedTasks(int producers, int* values, unsigned long long* counters,
    int* output, unsigned total, unsigned epoch) {
  int task = blockIdx.x, lane = threadIdx.x;
  if (task < producers) {
    unsigned target = (task + epoch) % 4;
    unsigned weight = unsigned(task / 4 % 4 + 1);
    values[task * 128 + lane] = Value(task, lane, epoch);
    CountedDependency::Publish(counters + target, weight);
  } else {
    int target = task - producers;
    std::uint64_t expected;
    if (!CountedDependencyTarget(total, epoch, &expected)) asm volatile("trap;");
    CountedDependency::Wait(counters + target, expected);
    int sum = 0;
    for (int p = 0; p < producers; ++p)
      if ((p + epoch) % 4 == target) sum += values[p * 128 + lane] * (p / 4 % 4 + 1);
    output[target * 128 + lane] = sum;
  }
}
__global__ void WeightedTasks(int producers, int* values, unsigned* tickets, int* output,
    unsigned* callbacks, unsigned total, unsigned epoch) {
  int task = blockIdx.x, lane = threadIdx.x;
  unsigned target = (task + epoch) % 4;
  unsigned weight = task < producers ? unsigned(task / 4 % 4 + 1) : 0;
  __shared__ unsigned last;
  if (task < producers) values[task * 128 + lane] = Value(task, lane, epoch);
  LastArriver::RunWeighted(tickets + target, total, weight, &last, [&] {
    int sum = 0;
    for (int p = 0; p < producers; ++p)
      if ((p + epoch) % 4 == target) sum += values[p * 128 + lane] * (p / 4 % 4 + 1);
    output[target * 128 + lane] = sum;
    if (lane == 0) atomicAdd(callbacks + target, 1);
  });
}
int Oracle(int producers, int target, int lane, int epoch) {
  int result = 0;
  for (int p = 0; p < producers; ++p)
    if ((p + epoch) % 4 == target) result += Value(p, lane, epoch) * (p / 4 % 4 + 1);
  return result;
}
}
int main() {
  int device, sms; Check(cudaGetDevice(&device));
  Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int producers = 4 * (std::min(sms, 24) / 4), consumers = 7;
  assert(producers >= 4);
  for (auto kernel : {TableTasks}) {
    int blocks; Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 128, 0));
    assert(producers + consumers <= blocks * sms);
  }
  int *values, *output; unsigned long long* counters; unsigned *tickets, *callbacks;
  Check(cudaMallocManaged(&values, producers * 128 * sizeof(int)));
  Check(cudaMallocManaged(&output, consumers * 128 * sizeof(int)));
  Check(cudaMallocManaged(&counters, producers * sizeof(unsigned long long)));
  Check(cudaMallocManaged(&tickets, 4 * sizeof(unsigned)));
  Check(cudaMallocManaged(&callbacks, 4 * sizeof(unsigned)));
  std::vector<std::vector<RuntimeDependencyInterval>> rows(consumers);
  unsigned stride = 1;
  for (int c = 0; c < consumers; ++c) {
    for (int p = 0; p < producers;) {
      if (!Selected(p, c, consumers)) { ++p; continue; }
      int begin = p;
      while (p < producers && Selected(p, c, consumers)) ++p;
      rows[c].push_back({unsigned(begin), unsigned(p - begin)});
    }
    stride = std::max(stride, unsigned(rows[c].size()));
  }
  RuntimeDependencyInterval* intervals;
  Check(cudaMallocManaged(&intervals, consumers * stride * sizeof(*intervals)));
  for (int c = 0; c < consumers; ++c) for (unsigned i = 0; i < stride; ++i)
    intervals[c * stride + i] = i < rows[c].size() ? rows[c][i] : RuntimeDependencyInterval{};
  RuntimeDependencyTableView table{intervals, unsigned(consumers), stride};
  bool visited = false;
  assert(!VisitDependencyTable(table, consumers, producers, [&](auto) { visited = true; }) && !visited);
  Check(cudaMemset(counters, 0, producers * sizeof(*counters)));
  for (int epoch = 0; epoch < 32; ++epoch) {
    Check(cudaMemset(values, 0xA5, producers * 128 * sizeof(int)));
    TableTasks<<<producers + consumers, 128>>>(table, producers, consumers, values, counters, output, epoch);
    Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
    for (int c = 0; c < consumers; ++c) for (int lane = 0; lane < 128; ++lane) {
      int expected = 0;
      for (int p = 0; p < producers; ++p) if (Selected(p, c, consumers)) expected += Value(p, lane, epoch);
      assert(output[c * 128 + lane] == expected);
    }
  }
  unsigned total = 0;
  for (int p = 0; p < producers; p += 4) total += p / 4 % 4 + 1;
  Check(cudaMemset(counters, 0, producers * sizeof(*counters)));
  Check(cudaMemset(tickets, 0, 4 * sizeof(*tickets)));
  Check(cudaMemset(callbacks, 0, 4 * sizeof(*callbacks)));
  for (int epoch = 0; epoch < 32; ++epoch) {
    Check(cudaMemset(values, 0xA5, producers * 128 * sizeof(int)));
    CountedTasks<<<producers + 4, 128>>>(producers, values, counters, output, total, epoch);
    Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
    for (int c = 0; c < 4; ++c) {
      assert(counters[c] == total * (epoch + 1));
      for (int lane = 0; lane < 128; ++lane) assert(output[c * 128 + lane] == Oracle(producers, c, lane, epoch));
    }
    Check(cudaMemset(values, 0xA5, producers * 128 * sizeof(int)));
    WeightedTasks<<<producers + 4, 128>>>(producers, values, tickets, output, callbacks, total, epoch);
    Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
    for (int c = 0; c < 4; ++c) {
      assert(tickets[c] == 0 && callbacks[c] == unsigned(epoch + 1));
      for (int lane = 0; lane < 128; ++lane) assert(output[c * 128 + lane] == Oracle(producers, c, lane, epoch));
    }
  }
  std::uint64_t target;
  assert(!CountedDependencyTarget(0, 0, &target));
  assert(!CountedDependencyTarget(8, UINT64_MAX, &target));
  assert(!CountedDependencyTarget(8, UINT64_MAX / 8, &target));
  for (void* pointer : {static_cast<void*>(values), static_cast<void*>(output), static_cast<void*>(counters),
                        static_cast<void*>(tickets), static_cast<void*>(callbacks), static_cast<void*>(intervals)})
    Check(cudaFree(pointer));
  std::puts("dependency primitives: table, counted epochs, weighted LA, empty contribution passed");
}
