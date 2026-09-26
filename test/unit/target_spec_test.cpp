#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Target/TargetSpec.h>

#include <cassert>
#include <string>
#include <filesystem>
#include <unistd.h>

namespace tilemega::tests::target_spec_test {

int TestTargetSpec(int argc, char** argv) {
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
  auto const& partial=sm120.CalibrationFor("bf16").fp32_partial_combine;
  assert(partial.reason=="not_calibrated" && !partial.fixed_ns && !partial.d_l2_ns);
  assert(sm120.ToJson().find("fp32_partial_combine") != std::string::npos);
  // EX-E3 step 0: a target JSON written before the wait policy was calibrated
  // parses to the generated wait, and the policy survives a round trip.
  assert(sm80.CalibrationFor("bf16").wait_spin_iters == 0);
  assert(sm80.CalibrationFor("bf16").wait_backoff_ns == 64);
  assert(sm80.CalibrationFor("bf16").wait_backoff_grow == 1);
  assert(sm80.ToJson().find("wait_backoff_cap_ns") != std::string::npos);
  sm120.calibration_stamps["task_bodies"]="fixture-source-stamp";
  sm120.serving_hop_coefficients={400.0,2.0,-1.0};
  auto path=std::filesystem::temp_directory_path()/
      ("tilemega-target-"+std::to_string(getpid())+".json");
  sm120.ToJson(path.string());
  auto restored=TargetSpec::FromJson(path.string());std::filesystem::remove(path);
  assert(restored.calibration_stamps==sm120.calibration_stamps);
  assert(restored.serving_hop_coefficients==sm120.serving_hop_coefficients);
  static_assert(!tilemega::arch::Caps<tilemega::arch::Sm120>::kTcgen05);
  return 0;

  return 0;
}

}  // namespace tilemega::tests::target_spec_test
