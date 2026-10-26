// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Support/Json.h>
#include <acl/acl.h>
#include <runtime/dev.h>
#include <cstdlib>
#include <stdexcept>

namespace tilemega {
namespace {
void Check(aclError e, char const* call) {
  if (e) throw std::runtime_error(std::string(call)+": "+std::to_string(e));
}
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
} // namespace tilemega
