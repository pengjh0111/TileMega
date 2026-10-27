// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeAttentionV2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
constexpr int kGrid = 104, kWarp = 64, kComputeThreads = 512;
constexpr int kSharedBytes = 32784;
constexpr std::size_t kColdBytes = std::size_t(256) << 20;
using Attention = tilemega::backend::mx::MxAttentionV2<128, 2, 1, false, 0>;

__host__ __device__ __forceinline__ unsigned Mix(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
__host__ __device__ __forceinline__ uint4 ExpectedCold(unsigned index) {
  return uint4{Mix(index ^ 0x12345678u), Mix(index ^ 0x89abcdefu),
               Mix(index ^ 0x31415926u), Mix(index ^ 0x27182818u)};
}
__device__ __forceinline__ unsigned ExpectedLds(unsigned iteration, unsigned cta, unsigned thread) {
  return (iteration + 1u) * 0x9e3779b9u ^ (cta + 1u) * 0x85ebca6bu ^ thread * 0xc2b2ae35u;
}
struct Result { unsigned raw_errors, cold_errors, checksum; };

template <int Threads, int Loads>
__global__ void __launch_bounds__(Threads)
LocalLdsLitmus(uint4 const* cold, unsigned copies, unsigned vectors_per_copy,
               unsigned iterations, unsigned* visited, Result* result) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* const shared = reinterpret_cast<volatile unsigned*>(storage);
  unsigned const thread = threadIdx.x, wave = thread / kWarp, lane = thread % kWarp;
  Result local{};
  unsigned visit_count = 0;
#pragma unroll 1
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    uint4 pending[Loads ? Loads : 1];
    unsigned const base = (iteration % copies) * vectors_per_copy;
    if (thread < kComputeThreads) {
#pragma unroll
      for (int n = 0; n < Loads; ++n) {
        unsigned const at = base + ((blockIdx.x * 8u + wave) * Loads + n) * kWarp + lane;
        pending[n] = tilemega::backend::LoadGlobal16Gvm(cold + at);
      }
      shared[thread] = ExpectedLds(iteration, blockIdx.x, thread);
    }
    // Both publication and reuse exercise the actual Attention helper.
    Attention::PublishWaveResults();
    if (thread < kComputeThreads) {
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        unsigned const other = unsigned(w) * kWarp + lane;
        local.raw_errors += shared[other] != ExpectedLds(iteration, blockIdx.x, other);
      }
#pragma unroll
      for (int n = 0; n < Loads; ++n) {
        uint4 value = pending[n];
        // Keep consumption after the helper, without claiming loads remain in flight.
        asm volatile("" : "+r"(value.x), "+r"(value.y), "+r"(value.z), "+r"(value.w) :: "memory");
        unsigned const at = base + ((blockIdx.x * 8u + wave) * Loads + n) * kWarp + lane;
        uint4 const expected = ExpectedCold(at);
        local.cold_errors += (value.x != expected.x) + (value.y != expected.y) +
                             (value.z != expected.z) + (value.w != expected.w);
        local.checksum += value.x ^ value.y ^ value.z ^ value.w;
      }
    }
    Attention::PublishWaveResults();
    if (thread == 0) ++visit_count;
  }
  result[blockIdx.x * Threads + thread] = local;
  if (thread == 0) visited[blockIdx.x] = visit_count;
}

void Check(cudaError_t error, char const* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(expression) Check((expression), #expression)

struct DeviceBuffer {
  void* data = nullptr;
  explicit DeviceBuffer(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&data, bytes)); }
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};
std::string JsonString(std::string const& value) {
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '\\' || c == '"') escaped += '\\';
    if (c == '\n') escaped += "\\n";
    else if (c == '\r') escaped += "\\r";
    else escaped += c;
  }
  return escaped + '"';
}

template <int Threads, int Loads>
bool Run(unsigned iterations, cudaDeviceProp const& properties, int device) {
  cudaFuncAttributes attributes{};
  int occupancy = -1;
  int attributes_status = -1, set_shared_status = -1, occupancy_status = -1;
  bool gate = false, attempted = false, completed = false, pass = false;
  unsigned visited_ctas = 0, visited_min = 0, visited_max = 0;
  unsigned long long raw_errors = 0, cold_errors = 0, checksum = 0;
  unsigned const vectors_per_copy = kGrid * 8u * kWarp * Loads;
  std::size_t const copy_bytes = std::size_t(vectors_per_copy) * sizeof(uint4);
  unsigned copies = 1;
  if constexpr (Loads > 0) copies = unsigned((kColdBytes + copy_bytes - 1) / copy_bytes);
  std::size_t const allocation = Loads ? copy_bytes * copies : kColdBytes;
  std::string status, error;
  try {
    auto api = cudaFuncGetAttributes(&attributes, LocalLdsLitmus<Threads, Loads>);
    attributes_status = int(api); Check(api, "cudaFuncGetAttributes");
    if (attributes.numRegs <= 0 || attributes.numRegs > (Threads == 576 ? 227 : 255) ||
        attributes.localSizeBytes != 0 || attributes.sharedSizeBytes != 0 ||
        attributes.maxThreadsPerBlock != Threads) {
      status = "resource_rejected_no_launch";
    } else {
      api = cudaFuncSetAttribute(LocalLdsLitmus<Threads, Loads>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
      set_shared_status = int(api); Check(api, "cudaFuncSetAttribute");
      api = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, LocalLdsLitmus<Threads, Loads>, Threads, kSharedBytes);
      occupancy_status = int(api); Check(api, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
      if (occupancy != 1) status = "occupancy_rejected_no_launch";
      else {
        gate = true;
        std::vector<uint4> input(allocation / sizeof(uint4));
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = ExpectedCold(unsigned(i));
        DeviceBuffer cold(allocation), visits(sizeof(unsigned) * kGrid),
            results(sizeof(Result) * kGrid * Threads);
        CUDA_CHECK(cudaMemcpy(cold.data, input.data(), allocation, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(visits.data, 0, sizeof(unsigned) * kGrid));
        CUDA_CHECK(cudaMemset(results.data, 0xff, sizeof(Result) * kGrid * Threads));
        attempted = true;
        LocalLdsLitmus<Threads, Loads><<<kGrid, Threads, kSharedBytes>>>(
            static_cast<uint4 const*>(cold.data), copies, vectors_per_copy, iterations,
            static_cast<unsigned*>(visits.data), static_cast<Result*>(results.data));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned> host_visits(kGrid);
        std::vector<Result> host_results(kGrid * Threads);
        CUDA_CHECK(cudaMemcpy(host_visits.data(), visits.data, sizeof(unsigned) * kGrid,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_results.data(), results.data, sizeof(Result) * kGrid * Threads,
                              cudaMemcpyDeviceToHost));
        completed = true;
        for (unsigned value : host_visits) visited_ctas += value == iterations;
        visited_min = *std::min_element(host_visits.begin(), host_visits.end());
        visited_max = *std::max_element(host_visits.begin(), host_visits.end());
        for (Result const& r : host_results) {
          raw_errors += r.raw_errors; cold_errors += r.cold_errors; checksum += r.checksum;
        }
        pass = visited_ctas == kGrid && raw_errors == 0 && cold_errors == 0;
        status = pass ? "pass" : "correctness_failed";
      }
    }
  } catch (std::exception const& e) { status = "runtime_error"; error = e.what(); }
  std::cout << "{\"record_type\":\"attention_local_lds_litmus\",\"pid\":" << getpid()
            << ",\"threads\":" << Threads << ",\"grid\":" << kGrid
            << ",\"ap_count\":" << properties.multiProcessorCount << ",\"warp_size\":" << properties.warpSize
            << ",\"cuda_device\":" << device << ",\"physical_gpu\":" << JsonString(std::getenv("TILEMEGA_DEVICE_INDEX"))
            << ",\"compute_waves\":8,\"loads_16B_per_compute_lane\":" << Loads
            << ",\"bytes_per_compute_wave\":" << Loads * 16 * kWarp
            << ",\"iterations\":" << iterations << ",\"copies\":" << copies
            << ",\"allocated_cold_bytes\":" << allocation
            << ",\"active_cold_working_set_bytes\":" << (Loads ? allocation : 0)
            << ",\"dynamic_shared_bytes\":" << kSharedBytes
            << ",\"mtreg\":" << attributes.numRegs << ",\"frame_bytes\":" << attributes.localSizeBytes
            << ",\"static_shared_bytes\":" << attributes.sharedSizeBytes
            << ",\"attribute_max_threads\":" << attributes.maxThreadsPerBlock
            << ",\"attributes_returncode\":" << attributes_status
            << ",\"set_dynamic_shared_returncode\":" << set_shared_status
            << ",\"occupancy_returncode\":" << occupancy_status
            << ",\"resident_ctas\":" << occupancy << ",\"resource_gate_pass\":" << (gate ? "true" : "false")
            << ",\"launch_attempted\":" << (attempted ? "true" : "false")
            << ",\"completed\":" << (completed ? "true" : "false")
            << ",\"visited_ctas\":" << visited_ctas << ",\"visited_min\":" << visited_min
            << ",\"visited_max\":" << visited_max << ",\"raw_errors\":" << raw_errors
            << ",\"cold_errors\":" << cold_errors << ",\"checksum\":" << checksum
            << ",\"status\":" << JsonString(status) << ",\"error\":" << JsonString(error)
            << ",\"in_flight_loading_claim\":false,\"TPOT_claim\":false}" << std::endl;
  return pass;
}
}  // namespace

int main(int argc, char** argv) {
  unsigned iterations = 2000;
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "attention_local_lds_litmus [--iterations 2000..1000000]\n";
    return 0;
  }
  if (argc != 1) {
    if (argc != 3 || std::string(argv[1]) != "--iterations") return 2;
    char* end = nullptr; unsigned long const parsed = std::strtoul(argv[2], &end, 10);
    if (!end || *end || parsed < 2000 || parsed > 1000000) return 2;
    iterations = unsigned(parsed);
  }
  char const *guard = std::getenv("TILEMEGA_GPU_LOCK_HELD"),
             *gpu = std::getenv("TILEMEGA_DEVICE_INDEX"),
             *cuda = std::getenv("CUDA_VISIBLE_DEVICES"), *maca = std::getenv("MACA_VISIBLE_DEVICES");
  if (!guard || std::string(guard) != "1" || !gpu || !cuda || !maca ||
      std::string(gpu) != cuda || std::string(gpu) != maca ||
      (std::string(gpu) != "4" && std::string(gpu) != "5" && std::string(gpu) != "6" && std::string(gpu) != "7")) {
    std::cerr << "requires the unchanged single-card GPU guard\n";
    return 2;
  }
  int device = -1; cudaDeviceProp properties{};
  try {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.warpSize != kWarp || properties.multiProcessorCount != kGrid)
      throw std::runtime_error("requires actual AP104/warp64 target");
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 2; }
  bool pass = true;
  pass = Run<512, 0>(iterations, properties, device) && pass;
  pass = Run<512, 2>(iterations, properties, device) && pass;
  pass = Run<512, 8>(iterations, properties, device) && pass;
  pass = Run<576, 0>(iterations, properties, device) && pass;
  pass = Run<576, 2>(iterations, properties, device) && pass;
  pass = Run<576, 8>(iterations, properties, device) && pass;
  return pass ? 0 : 1;
}
