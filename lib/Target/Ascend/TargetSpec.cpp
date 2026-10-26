// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Support/Json.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tilemega {
namespace {
int Integer(json::Value const& value, char const* key) {
  double n = value.At(key).AsNumber(key);
  if (!std::isfinite(n) || n < 0 || n > std::numeric_limits<int>::max() || std::floor(n) != n)
    throw std::invalid_argument(std::string("Invalid nonnegative integer: ")+key);
  return int(n);
}
ascend::ImplementationRole Role(std::string const& name) {
  if (name == "AIC_ONLY") return ascend::ImplementationRole::kAicOnly;
  if (name == "AIV_ONLY") return ascend::ImplementationRole::kAivOnly;
  if (name == "MIX_1_2") return ascend::ImplementationRole::kMix1_2;
  throw std::invalid_argument("Unsupported implementation role: "+name);
}
[[noreturn]] void Unsupported() {
  throw std::invalid_argument("CUDA occupancy and calibration are excluded from AS-1; use fixed native implementations");
}
}

TargetSpec TargetSpec::FromJson(std::string const& path) {
  auto root = json::ParseFile(path);
  TargetSpec out;
  ascend::TargetCapability t;
  t.arch_tag = root.At("arch_tag").AsString("arch_tag");
  t.product_sku = root.At("product_sku").AsString("product_sku");
  t.cann_version = root.At("cann_version").AsString("cann_version");
  t.catlass_sha = root.At("catlass_sha").AsString("catlass_sha");
  t.gm_event_evidence=root.At("evidence").At("gm_event").AsString("gm_event evidence");
  t.max_verified_mix_groups = Integer(root, "max_verified_mix_groups");
  const auto selection=root.At("selection").AsString("selection");
  const bool native_provider=selection=="native_empirical_provider" && root.Find("cost_provider_schema") &&
      root.At("cost_provider_schema").AsString("cost_provider_schema")=="tilemega.ascend.empirical_gemm.v1";
  if (Integer(root, "residency") != 1 ||
      root.At("calibration_status").AsString("calibration_status") != "not_calibrated" ||
      (selection!="fixed_legal_whitelist" && !native_provider))
    throw std::invalid_argument("Require residency=1 and an explicit native selection policy");
  auto const& caps = root.At("capabilities");
  t.aic_only_kernel = caps.At("aic_only_kernel").AsBool("aic_only_kernel");
  t.aiv_only_kernel = caps.At("aiv_only_kernel").AsBool("aiv_only_kernel");
  t.mix_1_2 = caps.At("mix_1_2").AsBool("mix_1_2");
  t.group_flag = caps.At("group_flag").AsBool("group_flag");
  t.gm_event = caps.At("gm_event_single_owner").AsBool("gm_event_single_owner");
  t.cross_group_stage_barrier = caps.At("cross_group_stage_barrier").AsBool("cross_group_stage_barrier");
  t.same_group_l0c_to_ub = caps.At("same_group_l0c_to_ub").AsBool("same_group_l0c_to_ub");
  t.same_group_ub_to_l1 = caps.At("same_group_ub_to_l1").AsBool("same_group_ub_to_l1");
  t.ssbuffer_control = caps.At("ssbuffer_control").AsBool("ssbuffer_control");
  for (auto const& item : root.At("tracks").AsArray("tracks")) {
    ascend::TrackClass track;
    track.name = item.At("name").AsString("name");
    if (track.name != "aic" && track.name != "aiv") throw std::invalid_argument("Unsupported TrackClass");
    track.role = track.name == "aic" ? ascend::TrackRole::kAic : ascend::TrackRole::kAiv;
    track.count = Integer(item, "count"); track.grouping = item.At("grouping").AsString("grouping");
    auto const& r = item.At("local_resources");
    track.local_resources = {unsigned(Integer(r,"l1")), unsigned(Integer(r,"l0a")),
      unsigned(Integer(r,"l0b")), unsigned(Integer(r,"l0c")), unsigned(Integer(r,"ub")),
      unsigned(Integer(r,"registers")), unsigned(Integer(r,"ssbuffer"))};
    for (auto const& name : item.At("available_impl_types").AsArray("available_impl_types"))
      track.available_impl_types.push_back(Role(name.AsString("implementation role")));
    t.tracks.push_back(std::move(track));
  }
  for (auto const& item : root.At("implementations").AsArray("implementations")) {
    auto id = item.At("impl_id").AsString("impl_id");
    int kind = Integer(item,"task_kind");
    if (kind > int(codegen::TaskKind::kLayoutConvert) ||
        !t.implementations.emplace(id,static_cast<codegen::TaskKind>(kind)).second)
      throw std::invalid_argument("Invalid or duplicate implementation binding");
  }
  t.Validate(); out.arch_tag = t.arch_tag;
  out.runtime_product = root.At("runtime_product").AsString("runtime_product");
  out.ascend_config_json = root.Dump(); out.ascend_capability = std::move(t);
  return out;
}
std::string TargetSpec::ToJson() const {
  if (!ascend_capability || ascend_config_json.empty()) throw std::invalid_argument("Missing native target metadata");
  auto root=json::Parse(ascend_config_json);
  root.Set("physical_device_id",physical_device_id); root.Set("logical_device_id",logical_device_id);
  return root.Dump();
}
void TargetSpec::ToJson(std::string const& path) const {
  std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write target: "+path);
  out << ToJson() << '\n';
}
std::string TargetSpec::Summary() const {
  if (!ascend_capability) throw std::invalid_argument("Missing Ascend capability");
  auto const& t=*ascend_capability;
  return arch_tag+"/"+t.product_sku+" AIC="+std::to_string(t.TrackCount(ascend::TrackRole::kAic))+
    " AIV="+std::to_string(t.TrackCount(ascend::TrackRole::kAiv))+" verified_groups="+
    std::to_string(t.max_verified_mix_groups)+" cost=not_calibrated";
}
std::string TargetSpec::NvccArch() const { Unsupported(); }
int TargetSpec::ComputeStages(int,int,int,int) { Unsupported(); }
int TargetSpec::ActiveBlocksPerSM(void const*,int,std::size_t) const { Unsupported(); }
int TargetSpec::ResidentGridLimit(void const*,int,std::size_t) const { Unsupported(); }
TargetSpec::Calib const& TargetSpec::CalibrationFor(std::string_view) const { Unsupported(); }
TargetSpec::Calib& TargetSpec::CalibrationFor(std::string_view) { Unsupported(); }
TargetSpec::EventCalibration const& TargetSpec::EventCalibrationFor(std::string_view) const { Unsupported(); }
TargetSpec::StreamKPoint const* TargetSpec::Calib::FindStreamK(int,int,int,int) const { Unsupported(); }
} // namespace tilemega
