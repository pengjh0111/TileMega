// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/TargetSpec.h>

#include <tilemega/Support/Json.h>
#include <tilemega/Target/ArchDispatch.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
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

std::vector<double> NumberArray(json::Value const& value, char const* what) {
  std::vector<double> out;
  for (auto const& item : value.AsArray(what)) out.push_back(item.AsNumber(what));
  return out;
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
  spec.caps.l1_5 = caps.l1_5;
  spec.caps.net = caps.net;
  spec.caps.cp_async = caps.cp_async;
  spec.caps.mbarrier = caps.mbarrier;
  spec.res.max_cluster_size = caps.max_cluster_size;
}

void ParseCalibration(json::Value const& cal, TargetSpec::Calib& out) {
  auto number = [&](char const* key) { return cal.At(key).AsNumber(key); };
  out.calibrated = cal.At("calibrated").AsBool("calibrated");
  json::Value const& pipes = cal.At("pipelines");
  auto pipe = [&](char const* key) { return pipes.At(key).AsNumber(key); };
  out.tc_fp16_gflops = pipe("tc_fp16_gflops");
  if (json::Value const* value = pipes.Find("tc_bf16_gflops"))
    out.tc_bf16_gflops = value->AsNumber("tc_bf16_gflops");
  out.cuda_fp32_gflops = pipe("cuda_fp32_gflops");
  out.cuda_int32_gops = pipe("cuda_int32_gops");
  out.sfu_exp2_gops = pipe("sfu_exp2_gops");
  out.sfu_rsqrt_gops = pipe("sfu_rsqrt_gops");
  out.smem_gbps = pipe("smem_gbps");
  out.smem_conflict_slope = pipe("smem_conflict_slope");
  out.l1_latency_ns = pipe("l1_latency_ns");
  out.l2_latency_ns = pipe("l2_latency_ns");
  out.dram_latency_ns = pipe("dram_latency_ns");
  out.l2_gbps = pipe("l2_gbps");
  out.l2_knee_bytes = pipe("l2_knee_bytes");
  out.dram_gbps = pipe("dram_gbps");
  out.l2_curve_bytes = NumberArray(pipes.At("l2_curve_bytes"), "l2_curve_bytes");
  out.l2_curve_gbps = NumberArray(pipes.At("l2_curve_gbps"), "l2_curve_gbps");
  out.smem_occupancy_ctas =
      NumberArray(pipes.At("smem_occupancy_ctas"), "smem_occupancy_ctas");
  out.smem_occupancy_gbps =
      NumberArray(pipes.At("smem_occupancy_gbps"), "smem_occupancy_gbps");

  json::Value const& sync = cal.At("sync");
  auto sync_number = [&](char const* key) { return sync.At(key).AsNumber(key); };
  out.atomic_uncontended_ns = sync_number("atomic_uncontended_ns");
  out.atomic_contention_ctas =
      NumberArray(sync.At("atomic_contention_ctas"), "atomic_contention_ctas");
  out.atomic_contention_ns =
      NumberArray(sync.At("atomic_contention_ns"), "atomic_contention_ns");
  out.threadfence_ns = sync_number("threadfence_ns");
  out.syncthreads_ns = sync_number("syncthreads_ns");
  out.named_barrier_ns = sync_number("named_barrier_ns");
  out.cluster_sync_ns = sync_number("cluster_sync_ns");
  out.cluster_sync_calibrated =
      sync.At("cluster_sync_calibrated").AsBool("cluster_sync_calibrated");
  out.grid_barrier_ctas =
      NumberArray(sync.At("grid_barrier_ctas"), "grid_barrier_ctas");
  out.grid_barrier_ns = NumberArray(sync.At("grid_barrier_ns"), "grid_barrier_ns");

  for (auto const& item : cal.At("streamk").AsArray("streamk")) {
    TargetSpec::StreamKPoint point;
    point.tile_m = static_cast<int>(item.At("tile_m").AsNumber("tile_m"));
    point.tile_n = static_cast<int>(item.At("tile_n").AsNumber("tile_n"));
    point.tile_k = static_cast<int>(item.At("tile_k").AsNumber("tile_k"));
    point.stages = static_cast<int>(item.At("stages").AsNumber("stages"));
    point.a_ns = item.At("a_ns").AsNumber("a_ns");
    point.b_ns = item.At("b_ns").AsNumber("b_ns");
    point.c_ns = item.At("c_ns").AsNumber("c_ns");
    point.d_ns = item.At("d_ns").AsNumber("d_ns");
    point.fit_r2 = item.At("fit_r2").AsNumber("fit_r2");
    point.ac_r2 = item.At("ac_r2").AsNumber("ac_r2");
    point.occ_per_sm = NumberArray(item.At("occ_per_sm"), "occ_per_sm");
    point.occ_a_ns = NumberArray(item.At("occ_a_ns"), "occ_a_ns");
    point.occ_c_ns = NumberArray(item.At("occ_c_ns"), "occ_c_ns");
    out.streamk.push_back(std::move(point));
  }
  out.combine_fixed_ns = cal.At("combine_fixed_ns").AsNumber("combine_fixed_ns");
  if (json::Value const* resolved = cal.Find("combine_fixed_resolved"))
    out.combine_fixed_resolved = resolved->AsBool("combine_fixed_resolved");
  out.combine_d_dram_ns =
      cal.At("combine_d_dram_ns").AsNumber("combine_d_dram_ns");
  if (auto const* value=cal.Find("fp32_partial_combine")) {
    auto& fit=out.fp32_partial_combine;
    fit.reason=value->At("reason").AsString("partial combine reason");
    fit.method=value->At("method").AsString("partial combine method");
    auto read=[&](char const* key,std::optional<double>& dst) {
      auto const& number=value->At(key);
      if (!number.IsNull()) {
        double rate=number.AsNumber(key);
        if (!std::isfinite(rate) || rate<0 || fit.reason!="measured")
          throw std::invalid_argument("invalid measured FP32-partial combine rate");
        dst=rate;
      }
    };
    read("fixed_ns",fit.fixed_ns); read("base_ns",fit.base_ns);
    read("d_l2_ns",fit.d_l2_ns); read("d_dram_ns",fit.d_dram_ns);
    if (fit.reason=="measured") {
      if (!fit.fixed_ns || !fit.base_ns || !fit.d_l2_ns || !fit.d_dram_ns || fit.method.empty())
        throw std::invalid_argument("incomplete FP32-partial combine calibration");
    } else if (fit.reason!="not_calibrated" && fit.reason!="capability_absent")
      throw std::invalid_argument("missing FP32-partial combine rate needs an explicit reason");
  }
  out.interference_ratio = number("interference_ratio");
  if (json::Value const* value = cal.Find("device"))
    out.device = value->AsString("device");
  if (json::Value const* value = cal.Find("measured_at"))
    out.measured_at = value->AsString("measured_at");
  if (json::Value const* value = cal.Find("wall_seconds"))
    out.wall_seconds = value->AsNumber("wall_seconds");
  if (json::Value const* records = cal.Find("measurements")) {
    for (auto const& item : records->AsArray("measurements")) {
      TargetSpec::Measurement record;
      record.name = item.At("name").AsString("name");
      record.value = item.At("value").AsNumber("value");
      record.unit = item.At("unit").AsString("unit");
      record.samples = static_cast<int>(item.At("samples").AsNumber("samples"));
      record.rel_stddev = item.At("rel_stddev").AsNumber("rel_stddev");
      record.method = item.At("method").AsString("method");
      out.measurements.push_back(std::move(record));
    }
  }
}

json::Value CalibrationJson(TargetSpec::Calib const& calib) {
  auto const& combine=calib.fp32_partial_combine;
  auto optional=[](std::optional<double> const& v) { return v ? json::Value(*v) : json::Value(); };
  json::Value partial_combine(json::Object{
      {"reason",combine.reason},{"method",combine.method},
      {"fixed_ns",optional(combine.fixed_ns)},{"base_ns",optional(combine.base_ns)},
      {"d_l2_ns",optional(combine.d_l2_ns)},{"d_dram_ns",optional(combine.d_dram_ns)}});
  json::Value pipelines(json::Object{
      {"tc_fp16_gflops", calib.tc_fp16_gflops},
      {"tc_bf16_gflops", calib.tc_bf16_gflops},
      {"cuda_fp32_gflops", calib.cuda_fp32_gflops},
      {"cuda_int32_gops", calib.cuda_int32_gops},
      {"sfu_exp2_gops", calib.sfu_exp2_gops},
      {"sfu_rsqrt_gops", calib.sfu_rsqrt_gops},
      {"smem_gbps", calib.smem_gbps},
      {"smem_conflict_slope", calib.smem_conflict_slope},
      {"l1_latency_ns", calib.l1_latency_ns},
      {"l2_latency_ns", calib.l2_latency_ns},
      {"dram_latency_ns", calib.dram_latency_ns},
      {"l2_gbps", calib.l2_gbps},
      {"l2_knee_bytes", calib.l2_knee_bytes},
      {"dram_gbps", calib.dram_gbps},
      {"l2_curve_bytes", json::Numbers(calib.l2_curve_bytes)},
      {"l2_curve_gbps", json::Numbers(calib.l2_curve_gbps)},
      {"smem_occupancy_ctas", json::Numbers(calib.smem_occupancy_ctas)},
      {"smem_occupancy_gbps", json::Numbers(calib.smem_occupancy_gbps)}});
  json::Value sync(json::Object{
      {"atomic_uncontended_ns", calib.atomic_uncontended_ns},
      {"atomic_contention_ctas", json::Numbers(calib.atomic_contention_ctas)},
      {"atomic_contention_ns", json::Numbers(calib.atomic_contention_ns)},
      {"threadfence_ns", calib.threadfence_ns},
      {"syncthreads_ns", calib.syncthreads_ns},
      {"named_barrier_ns", calib.named_barrier_ns},
      {"cluster_sync_ns", calib.cluster_sync_ns},
      {"cluster_sync_calibrated", calib.cluster_sync_calibrated},
      {"grid_barrier_ctas", json::Numbers(calib.grid_barrier_ctas)},
      {"grid_barrier_ns", json::Numbers(calib.grid_barrier_ns)}});
  json::Array streamk;
  for (auto const& point : calib.streamk) {
    streamk.emplace_back(json::Object{
        {"tile_m", point.tile_m}, {"tile_n", point.tile_n},
        {"tile_k", point.tile_k}, {"stages", point.stages},
        {"a_ns", point.a_ns}, {"b_ns", point.b_ns},
        {"c_ns", point.c_ns}, {"d_ns", point.d_ns},
        {"fit_r2", point.fit_r2}, {"ac_r2", point.ac_r2},
        {"occ_per_sm", json::Numbers(point.occ_per_sm)},
        {"occ_a_ns", json::Numbers(point.occ_a_ns)},
        {"occ_c_ns", json::Numbers(point.occ_c_ns)}});
  }
  json::Array measurements;
  for (auto const& record : calib.measurements) {
    measurements.emplace_back(json::Object{
        {"name", record.name}, {"value", record.value},
        {"unit", record.unit}, {"samples", record.samples},
        {"rel_stddev", record.rel_stddev}, {"method", record.method}});
  }
  return json::Value(json::Object{
      {"calibrated", calib.calibrated}, {"device", calib.device},
      {"measured_at", calib.measured_at}, {"wall_seconds", calib.wall_seconds},
      {"pipelines", pipelines}, {"sync", sync},
      {"streamk", json::Value(streamk)},
      {"combine_fixed_ns", calib.combine_fixed_ns},
      {"combine_fixed_resolved", calib.combine_fixed_resolved},
      {"combine_d_dram_ns", calib.combine_d_dram_ns},
      {"fp32_partial_combine", partial_combine},
      {"interference_ratio", calib.interference_ratio},
      {"measurements", json::Value(measurements)}});
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
  json::Value root = json::ParseFile(path);
  TargetSpec spec;
  spec.arch_tag = root.At("arch_tag").AsString("arch_tag");
  spec.sm_major = static_cast<int>(root.At("sm_major").AsNumber("sm_major"));
  spec.sm_minor = static_cast<int>(root.At("sm_minor").AsNumber("sm_minor"));
  if (auto grouped = root.Find("event_calibration_by_dtype")) {
    auto parse = [&](char const* dtype, EventCalibration& out) {
      auto record = grouped->Find(dtype);
      if (!record) return;
      out.source = record->At("source").AsString("event source");
      if (auto v = record->Find("source_sha256")) out.source_sha256 = v->AsString("source hash");
      if (auto v = record->Find("method")) out.method = v->AsString("event fit method");
      auto rate = [&](char const* name, EventRate& dst) {
        auto const& value = record->At(name);
        dst.reason = value.At("reason").AsString("event rate reason");
        if (auto unit = value.Find("unit")) dst.unit = unit->AsString("event rate unit");
        if (!value.At("ns").IsNull()) {
          double ns = value.At("ns").AsNumber("event rate ns");
          if (!std::isfinite(ns) || ns < 0 || dst.reason != "measured" || dst.unit.empty())
            throw std::invalid_argument("invalid calibrated event rate");
          dst.ns = ns;
        } else if (dst.reason != "not_calibrated" && dst.reason != "capability_absent")
          throw std::invalid_argument("missing event rate needs an explicit reason");
      };
      rate("notify", out.notify); rate("poll", out.poll); rate("fence", out.fence);
      if (record->Find("notify_stage")) rate("notify_stage",out.notify_stage);
      if (record->Find("notify_longest_worker")) rate("notify_longest_worker",out.notify_longest_worker);
      if (record->Find("poll_stage")) rate("poll_stage",out.poll_stage);
      if (record->Find("poll_longest_worker")) rate("poll_longest_worker",out.poll_longest_worker);
    };
    parse("bf16", spec.event_bf16); parse("f32", spec.event_f32);
  }

  json::Value const& caps = root.At("caps");
  spec.caps.cluster = caps.At("cluster").AsBool("caps.cluster");
  spec.caps.tma = caps.At("tma").AsBool("caps.tma");
  spec.caps.warp_specialized =
      caps.At("warp_specialized").AsBool("caps.warp_specialized");
  spec.caps.tcgen05 = caps.At("tcgen05").AsBool("caps.tcgen05");
  spec.caps.l1_5 = caps.At("l1_5").AsBool("caps.l1_5");
  spec.caps.net = caps.At("net").AsBool("caps.net");
  spec.caps.cp_async = caps.At("cp_async").AsBool("caps.cp_async");
  spec.caps.mbarrier = caps.At("mbarrier").AsBool("caps.mbarrier");

  json::Value const& res_json = root.At("resources");
  auto res_int = [&](char const* key) {
    return static_cast<int>(res_json.At(key).AsNumber(key));
  };
  spec.res.num_sms = res_int("num_sms");
  spec.res.max_smem_per_sm = res_int("max_smem_per_sm");
  spec.res.max_dynamic_smem_per_cta = res_int("max_dynamic_smem_per_cta");
  spec.res.regs_per_sm = res_int("regs_per_sm");
  spec.res.max_cluster_size = res_int("max_cluster_size");
  spec.res.max_threads_per_sm = res_int("max_threads_per_sm");
  spec.res.warp_size = res_int("warp_size");

  ParseCalibration(root.At("calibration"), spec.calib);
  if (json::Value const* grouped = root.Find("calibration_by_dtype")) {
    if (json::Value const* bf16 = grouped->Find("bf16"))
      ParseCalibration(*bf16, spec.calib_bf16);
  }
  return spec;
}

TargetSpec::StreamKPoint const* TargetSpec::Calib::FindStreamK(
    int m, int n, int k, int stages) const {
  for (auto const& point : streamk) {
    if (point.tile_m == m && point.tile_n == n && point.tile_k == k &&
        point.stages == stages) {
      return &point;
    }
  }
  return nullptr;
}

TargetSpec::Calib const& TargetSpec::CalibrationFor(
    std::string_view dtype) const {
  if (dtype == "f32") return calib;
  if (dtype == "bf16") return calib_bf16;
  throw std::invalid_argument("unsupported calibration dtype: " +
                              std::string(dtype));
}

TargetSpec::EventCalibration const& TargetSpec::EventCalibrationFor(std::string_view dtype) const {
  if (dtype == "bf16") return event_bf16;
  if (dtype == "f32") return event_f32;
  throw std::invalid_argument("unsupported event calibration dtype");
}

TargetSpec::Calib& TargetSpec::CalibrationFor(std::string_view dtype) {
  return const_cast<Calib&>(
      static_cast<TargetSpec const&>(*this).CalibrationFor(dtype));
}

std::string TargetSpec::ToJson() const {
  json::Value root(json::Object{});
  root.Set("arch_tag", arch_tag);
  root.Set("sm_major", sm_major);
  root.Set("sm_minor", sm_minor);
  root.Set("caps", json::Object{{"cluster", caps.cluster},
                                {"tma", caps.tma},
                                {"warp_specialized", caps.warp_specialized},
                                {"tcgen05", caps.tcgen05},
                                {"l1_5", caps.l1_5},
                                {"net", caps.net},
                                {"cp_async", caps.cp_async},
                                {"mbarrier", caps.mbarrier}});
  root.Set("resources",
           json::Object{{"num_sms", res.num_sms},
                        {"max_smem_per_sm", res.max_smem_per_sm},
                        {"max_dynamic_smem_per_cta", res.max_dynamic_smem_per_cta},
                        {"regs_per_sm", res.regs_per_sm},
                        {"max_cluster_size", res.max_cluster_size},
                        {"max_threads_per_sm", res.max_threads_per_sm},
                        {"warp_size", res.warp_size}});

  root.Set("calibration", CalibrationJson(calib));
  auto event_json = [](EventCalibration const& e) {
    auto rate = [](EventRate const& r) {
      return json::Value(json::Object{{"ns", r.ns ? json::Value(*r.ns) : json::Value()},
                                     {"reason", r.reason}, {"unit", r.unit}});
    };
    return json::Value(json::Object{{"source", e.source}, {"source_sha256", e.source_sha256},
                                    {"method", e.method}, {"notify", rate(e.notify)},
                                    {"poll", rate(e.poll)}, {"fence", rate(e.fence)},
                                    {"notify_stage",rate(e.notify_stage)},
                                    {"notify_longest_worker",rate(e.notify_longest_worker)},
                                    {"poll_stage",rate(e.poll_stage)},
                                    {"poll_longest_worker",rate(e.poll_longest_worker)}});
  };
  root.Set("event_calibration_by_dtype", json::Object{
      {"bf16", event_json(event_bf16)}, {"f32", event_json(event_f32)}});
  if (calib_bf16.calibrated || !calib_bf16.measurements.empty()) {
    root.Set("calibration_by_dtype",
             json::Object{{"bf16", CalibrationJson(calib_bf16)}});
  }
  return root.Dump();
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
