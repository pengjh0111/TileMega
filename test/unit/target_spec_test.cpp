#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Target/TargetSpec.h>

#include <cassert>
#include <string>

int main() {
  tilemega::analysis::IslContext isl_context;
  using tilemega::TargetSpec;
  auto sm80 = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_80.json");
  auto sm120 = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_120.json");
  assert(sm80.caps.cp_async && !sm80.caps.cluster);
  assert(sm120.caps.cluster && sm120.caps.tma && !sm120.caps.tcgen05);
  assert(TargetSpec::ComputeStages(100, 30, 10, 16) == 3);
  assert(TargetSpec::ComputeStages(8, 16, 0, 16) == 0);
  assert(sm120.ToJson().find("\"tcgen05\": false") != std::string::npos);
  assert(!sm80.EventCalibrationFor("bf16").notify.ns);
  assert(sm80.EventCalibrationFor("bf16").notify.reason == "not_calibrated");
  auto const& events = sm120.EventCalibrationFor("bf16");
  assert(events.notify.ns && *events.notify.ns > 0);
  assert(events.notify.unit == "ns/runtime_task_ref");
  assert(events.poll.unit == "ns/runtime_wait_entry");
  assert(!events.fence.ns && events.fence.reason == "not_calibrated");
  assert(!sm120.EventCalibrationFor("f32").notify.ns);
  assert(sm120.ToJson().find("ns/runtime_task_ref") != std::string::npos);
  static_assert(!tilemega::arch::Caps<tilemega::arch::Sm120>::kTcgen05);
  return 0;
}
