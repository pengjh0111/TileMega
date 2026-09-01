// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/TargetSpec.h>

#include <tilemega/Target/ArchDispatch.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace tilemega {
namespace {

void CheckCuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::string ReadFile(std::string const& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open target config: " + path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string JsonString(std::string const& text, char const* key) {
  std::smatch match;
  std::regex pattern(std::string("\\\"") + key +
                     "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  if (!std::regex_search(text, match, pattern))
    throw std::runtime_error(std::string("missing JSON string: ") + key);
  return match[1].str();
}

double JsonNumber(std::string const& text, char const* key) {
  std::smatch match;
  std::regex pattern(std::string("\\\"") + key +
                     "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
  if (!std::regex_search(text, match, pattern))
    throw std::runtime_error(std::string("missing JSON number: ") + key);
  return std::stod(match[1].str());
}

bool JsonBool(std::string const& text, char const* key) {
  std::smatch match;
  std::regex pattern(std::string("\\\"") + key +
                     "\\\"\\s*:\\s*(true|false)");
  if (!std::regex_search(text, match, pattern))
    throw std::runtime_error(std::string("missing JSON boolean: ") + key);
  return match[1].str() == "true";
}

void ApplyKnownCaps(TargetSpec& spec) {
  auto caps = arch::RuntimeCapsForTag(spec.arch_tag);
  if (std::string_view(caps.collective) == "unsupported") {
    throw std::runtime_error("unsupported TileMega target: " + spec.arch_tag);
  }
  spec.caps.cluster = caps.cluster;
  spec.caps.tma = caps.tma;
  spec.caps.warp_specialized = caps.warp_specialized;
  spec.caps.tcgen05 = caps.tcgen05;
  spec.caps.cp_async = caps.cp_async;
  spec.caps.mbarrier = caps.mbarrier;
  spec.res.max_cluster_size = caps.max_cluster_size;
}

}  // namespace

TargetSpec TargetSpec::Probe(int device_ordinal) {
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device_ordinal),
            "cudaGetDeviceProperties");

  TargetSpec spec;
  spec.sm_major = properties.major;
  spec.sm_minor = properties.minor;
  spec.arch_tag = "sm_" + std::to_string(spec.sm_major) +
                  std::to_string(spec.sm_minor);
  ApplyKnownCaps(spec);
  spec.res.num_sms = properties.multiProcessorCount;
  spec.res.max_smem_per_sm =
      static_cast<int>(properties.sharedMemPerMultiprocessor);
  spec.res.max_dynamic_smem_per_cta =
      static_cast<int>(properties.sharedMemPerBlockOptin);
  spec.res.regs_per_sm = properties.regsPerMultiprocessor;
  spec.res.max_threads_per_sm = properties.maxThreadsPerMultiProcessor;
  spec.res.warp_size = properties.warpSize;
  return spec;
}

TargetSpec TargetSpec::FromJson(std::string const& path) {
  std::string text = ReadFile(path);
  TargetSpec spec;
  spec.arch_tag = JsonString(text, "arch_tag");
  spec.sm_major = static_cast<int>(JsonNumber(text, "sm_major"));
  spec.sm_minor = static_cast<int>(JsonNumber(text, "sm_minor"));
  spec.caps.cluster = JsonBool(text, "cluster");
  spec.caps.tma = JsonBool(text, "tma");
  spec.caps.warp_specialized = JsonBool(text, "warp_specialized");
  spec.caps.tcgen05 = JsonBool(text, "tcgen05");
  spec.caps.cp_async = JsonBool(text, "cp_async");
  spec.caps.mbarrier = JsonBool(text, "mbarrier");
  spec.res.num_sms = static_cast<int>(JsonNumber(text, "num_sms"));
  spec.res.max_smem_per_sm =
      static_cast<int>(JsonNumber(text, "max_smem_per_sm"));
  spec.res.max_dynamic_smem_per_cta =
      static_cast<int>(JsonNumber(text, "max_dynamic_smem_per_cta"));
  spec.res.regs_per_sm = static_cast<int>(JsonNumber(text, "regs_per_sm"));
  spec.res.max_cluster_size =
      static_cast<int>(JsonNumber(text, "max_cluster_size"));
  spec.res.max_threads_per_sm =
      static_cast<int>(JsonNumber(text, "max_threads_per_sm"));
  spec.res.warp_size = static_cast<int>(JsonNumber(text, "warp_size"));
  spec.calib.atomic_latency_ns = JsonNumber(text, "atomic_latency_ns");
  spec.calib.cluster_sync_latency_ns =
      JsonNumber(text, "cluster_sync_latency_ns");
  spec.calib.named_barrier_ns = JsonNumber(text, "named_barrier_ns");
  spec.calib.hbm_bandwidth_gbps = JsonNumber(text, "hbm_bandwidth_gbps");
  spec.calib.calibrated = JsonBool(text, "calibrated");
  return spec;
}

std::string TargetSpec::ToJson() const {
  std::ostringstream out;
  out << std::boolalpha << std::fixed << std::setprecision(3);
  out << "{\n"
      << "  \"arch_tag\": \"" << arch_tag << "\",\n"
      << "  \"sm_major\": " << sm_major << ",\n"
      << "  \"sm_minor\": " << sm_minor << ",\n"
      << "  \"caps\": {\n"
      << "    \"cluster\": " << caps.cluster << ",\n"
      << "    \"tma\": " << caps.tma << ",\n"
      << "    \"warp_specialized\": " << caps.warp_specialized << ",\n"
      << "    \"tcgen05\": " << caps.tcgen05 << ",\n"
      << "    \"cp_async\": " << caps.cp_async << ",\n"
      << "    \"mbarrier\": " << caps.mbarrier << "\n"
      << "  },\n"
      << "  \"resources\": {\n"
      << "    \"num_sms\": " << res.num_sms << ",\n"
      << "    \"max_smem_per_sm\": " << res.max_smem_per_sm << ",\n"
      << "    \"max_dynamic_smem_per_cta\": "
      << res.max_dynamic_smem_per_cta << ",\n"
      << "    \"regs_per_sm\": " << res.regs_per_sm << ",\n"
      << "    \"max_cluster_size\": " << res.max_cluster_size << ",\n"
      << "    \"max_threads_per_sm\": " << res.max_threads_per_sm << ",\n"
      << "    \"warp_size\": " << res.warp_size << "\n"
      << "  },\n"
      << "  \"calibration\": {\n"
      << "    \"atomic_latency_ns\": " << calib.atomic_latency_ns << ",\n"
      << "    \"cluster_sync_latency_ns\": "
      << calib.cluster_sync_latency_ns << ",\n"
      << "    \"named_barrier_ns\": " << calib.named_barrier_ns << ",\n"
      << "    \"hbm_bandwidth_gbps\": " << calib.hbm_bandwidth_gbps << ",\n"
      << "    \"calibrated\": " << calib.calibrated << "\n"
      << "  }\n"
      << "}\n";
  return out.str();
}

void TargetSpec::ToJson(std::string const& path) const {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write target config: " + path);
  output << ToJson();
  if (!output) throw std::runtime_error("failed to write target config: " + path);
}

int TargetSpec::ComputeStages(int smem_budget_bytes, int bytes_per_stage,
                              int fixed_overhead_bytes, int max_stages) {
  if (smem_budget_bytes < 0 || bytes_per_stage <= 0 ||
      fixed_overhead_bytes < 0 || max_stages < 0) {
    throw std::invalid_argument("invalid shared-memory stage parameters");
  }
  int available = smem_budget_bytes - fixed_overhead_bytes;
  if (available <= 0) return 0;
  return std::min(max_stages, available / bytes_per_stage);
}

int TargetSpec::ActiveBlocksPerSM(void const* kernel, int block_size,
                                  std::size_t dynamic_smem_bytes) const {
  if (kernel == nullptr || block_size <= 0) {
    throw std::invalid_argument("invalid occupancy query");
  }
  int blocks = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, kernel, block_size, dynamic_smem_bytes),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  return blocks;
}

int TargetSpec::ResidentGridLimit(void const* kernel, int block_size,
                                  std::size_t dynamic_smem_bytes) const {
  return res.num_sms *
         ActiveBlocksPerSM(kernel, block_size, dynamic_smem_bytes);
}

std::string TargetSpec::Summary() const {
  std::ostringstream out;
  out << arch_tag << ": SMs=" << res.num_sms
      << " smem/SM=" << res.max_smem_per_sm
      << " smem/CTA=" << res.max_dynamic_smem_per_cta
      << " regs/SM=" << res.regs_per_sm
      << " cluster=" << std::boolalpha << caps.cluster
      << " TMA=" << caps.tma << " tcgen05=" << caps.tcgen05;
  return out.str();
}

}  // namespace tilemega
