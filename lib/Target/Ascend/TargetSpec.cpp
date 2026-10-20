// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Support/Json.h>
#include <acl/acl.h>
#include <runtime/dev.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
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
void Check(aclError e, char const* call) {
  if (e) throw std::runtime_error(std::string(call)+": "+std::to_string(e));
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
  if (Integer(root, "residency") != 1 ||
      root.At("calibration_status").AsString("calibration_status") != "not_calibrated" ||
      root.At("selection").AsString("selection") != "fixed_legal_whitelist")
    throw std::invalid_argument("AS-1 requires residency=1 and fixed uncalibrated selection");
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
TargetSpec TargetSpec::Probe(int ordinal) {
  const char* expected = std::getenv("AS1_PHYSICAL_DEVICE_ID");
  if (!expected || ordinal != 0) throw std::invalid_argument("Probe requires the AS-1 single-device wrapper and logical ID 0");
  auto target = FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
  auto config = json::Parse(target.ascend_config_json);
  bool measured = false;
  for (auto const& id : config.At("verified_physical_device_ids").AsArray("verified_physical_device_ids"))
    measured |= id.AsNumber("physical ID") == std::stoi(expected);
  if (!measured) throw std::invalid_argument("Selected card lacks this target's runtime probe evidence");
  Check(aclInit(nullptr),"aclInit");
  try {
    uint32_t count = 0, physical = ~0u;
    Check(aclrtGetDeviceCount(&count),"device count");
    if (count != 1) throw std::invalid_argument("Exactly one visible NPU is required");
    Check(rtGetDevicePhyIdByIndex(0,&physical),"physical mapping");
    if (physical != unsigned(std::stoi(expected))) throw std::invalid_argument("Physical mapping mismatch");
    Check(aclrtSetDevice(0),"set device");
    const char* sku = aclrtGetSocName();
    if (!sku || target.runtime_product != sku) throw std::invalid_argument("Runtime product differs from compiled target");
    int64_t aic=0,aiv=0,ub=0;
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_CUBE_CORE_NUM,&aic),"AIC count");
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_VECTOR_CORE_NUM,&aiv),"AIV count");
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE,&ub),"UB budget");
    auto const& c = *target.ascend_capability;
    if (aic != c.TrackCount(ascend::TrackRole::kAic) || aiv != c.TrackCount(ascend::TrackRole::kAiv) ||
        ub < int64_t(c.Budget(ascend::TrackRole::kAiv).ub)) throw std::invalid_argument("Runtime resources differ from target evidence");
    target.physical_device_id=physical; target.logical_device_id=0;
    Check(aclFinalize(),"finalize");
  } catch (...) { aclFinalize(); throw; }
  return target;
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
