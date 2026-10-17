#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Target/TargetSpec.h>

#include <cassert>
#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace tilemega::tests::target_spec_test {

int TestTargetSpec(int argc, char** argv) {
  tilemega::analysis::IslContext isl_context;
  using tilemega::TargetSpec;
  for (auto architecture : {80, 89, 90, 100, 120}) {
    auto file = std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_" +
                std::to_string(architecture) + ".json";
    auto explicit_policy = TargetSpec::FromJson(file);
    auto legacy_file = std::filesystem::temp_directory_path() /
        ("tilemega-legacy-policy-" + std::to_string(getpid()) + ".json");
    {
      std::ifstream source(file);
      std::ofstream old(legacy_file);
      std::string line;
      bool found = false;
      while (std::getline(source, line)) {
        if (line.find("\"wait_protocol\"") != std::string::npos) found = true;
        else old << line << '\n';
      }
      assert(found && source.eof() && old.good());
    }
    auto omitted_policy = TargetSpec::FromJson(legacy_file.string());
    std::filesystem::remove(legacy_file);
    assert(explicit_policy.ToJson() == omitted_policy.ToJson());
  }
  auto sm80 = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_80.json");
  auto sm120 = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_120.json");
  assert(sm80.caps.cp_async && !sm80.caps.cluster);
  assert(sm120.caps.cluster && sm120.caps.tma && !sm120.caps.tcgen05);
  assert(!sm80.caps.mbarrier_try_wait && !sm80.caps.mbarrier_tx && !sm80.caps.pdl);
  assert(sm120.caps.mbarrier_try_wait && sm120.caps.mbarrier_tx && sm120.caps.bulk_copy);
  assert(sm120.caps.bulk_prefetch && sm120.caps.pdl);
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
  sm120.serving_legacy_hop_coefficients={1200.0,3.0,-2.0};
  auto path=std::filesystem::temp_directory_path()/
      ("tilemega-target-"+std::to_string(getpid())+".json");
  sm120.ToJson(path.string());
  auto restored=TargetSpec::FromJson(path.string());std::filesystem::remove(path);
  assert(restored.calibration_stamps==sm120.calibration_stamps);
  assert(restored.serving_hop_coefficients==sm120.serving_hop_coefficients);
  assert(restored.serving_legacy_hop_coefficients==sm120.serving_legacy_hop_coefficients);
  auto& body=sm120.calib_bf16.task_body;
  body={};
  body.serving["embedding"]={100.0,0.01,0.0,0.02,4};
  sm120.ToJson(path.string());
  restored=TargetSpec::FromJson(path.string());
  assert(restored.calib_bf16.task_body.samples==0);
  assert(restored.calib_bf16.task_body.fixed.empty());
  assert(restored.calib_bf16.task_body.serving.at("embedding").samples==4);
  body.serving_paged["attention_decode_d64"]={100.0,20.0,0.03};
  sm120.ToJson(path.string());
  restored=TargetSpec::FromJson(path.string());
  assert(restored.calib_bf16.task_body.serving_paged.at("attention_decode_d64").iter_ns==20.0);
  auto rejected=[&] {
    sm120.ToJson(path.string());
    try {TargetSpec::FromJson(path.string());} catch(std::invalid_argument const&) {return true;}
    return false;
  };
  body.fixed={1.0,2.0,3.0};
  assert(rejected());
  body.fixed.clear();
  body.serving_paged.at("attention_decode_d64").iter_ns=-1.0;
  assert(rejected());
  std::filesystem::remove(path);
  auto xcore = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/xcore1000.json");
  assert(xcore.arch_tag == "xcore1000" && xcore.device_name == "MetaX C550");
  assert(xcore.res.num_sms == 104 && xcore.res.warp_size == 64);
  assert(xcore.res.max_dynamic_smem_per_cta == 65536);
  assert(!xcore.caps.cp_async && !xcore.caps.tma && !xcore.caps.pdl && !xcore.caps.cluster);
  static_assert(tilemega::arch::Caps<tilemega::arch::Xcore1000>::kWarpSize == 64);
  static_assert(!tilemega::arch::Caps<tilemega::arch::Xcore1000>::kPdl);

  // BF16 serving reads CalibrationFor("bf16"), which parses
  // calibration_by_dtype.bf16 -- not the top-level `calibration` block, which
  // holds the FP32 profile and is still uncalibrated here. The committed
  // target carried the device identity but not that section, so a fresh clone
  // got a zero-filled Calib while the measurements ran against a copy under
  // the ignored runs/ tree. These are the values of that copy; the point of
  // asserting them is that dropping the section again fails the build rather
  // than silently restoring zeros.
  auto const& bf16 = xcore.CalibrationFor("bf16");
  assert(bf16.dram_gbps == 1610.715799);
  assert(bf16.l2_gbps == 3484.71128);
  assert(bf16.tc_bf16_gflops == 288831.3898);
  assert(bf16.smem_gbps == 12504.51889);
  assert(bf16.syncthreads_ns == 26.98047319);
  assert(!bf16.grid_barrier_ns.empty() && bf16.grid_barrier_ns.front() == 1834.499999);
  // A zero or empty L2 curve makes CacheServiceCurve::HitFraction return 0 for
  // every footprint, which removes reuse from the cost model altogether.
  assert(bf16.l2_curve_bytes.size() == 18 && bf16.l2_curve_gbps.size() == 18);
  assert(bf16.l2_gbps > bf16.dram_gbps);
  return 0;
}

}  // namespace tilemega::tests::target_spec_test
