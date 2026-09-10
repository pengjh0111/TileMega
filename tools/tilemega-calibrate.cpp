// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
// Phase-4 (P4.1) calibration entry point: probe the device, run the §4.4
// microbenchmarks, and write configs/targets/<arch>.json.
//
// A target the local machine cannot run is never filled in from another
// target's numbers -- run this on that hardware, or leave the file
// uncalibrated.
#include <tilemega/Target/Calibration.h>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Target/TargetSpec.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
  tilemega::analysis::IslContext isl_context;
  tilemega::calib::Options options;
  std::string output;
  std::string base;
  std::string dtype = "f32";
  bool quiet = false;
  bool partial_combine_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) options.device = std::stoi(argv[++i]);
    else if (arg == "--repeats" && i + 1 < argc) options.repeats = std::stoi(argv[++i]);
    else if (arg == "--combine-graph-batch" && i + 1 < argc) options.combine_graph_batch = std::stoi(argv[++i]);
    else if (arg == "--skip-streamk") options.skip_streamk = true;
    else if (arg == "--fp32-partial-combine-only") partial_combine_only = true;
    else if (arg == "--dtype" && i + 1 < argc) dtype = argv[++i];
    else if (arg == "--base" && i + 1 < argc) base = argv[++i];
    else if (arg == "--quiet") quiet = true;
    else if (arg == "--out" && i + 1 < argc) output = argv[++i];
    else if (arg == "--help") {
      std::cout <<
          "usage: tilemega-calibrate [--device N] [--repeats N]\n"
          "                          [--dtype f32|bf16] [--base FILE]\n"
          "                          [--skip-streamk] [--quiet] [--out FILE]\n"
          "                          [--fp32-partial-combine-only]\n"
          "                          [--combine-graph-batch N]\n"
          "\n"
          "Measures the §4.4 cost-model constants on the GPU at --device and\n"
          "writes the target JSON to --out (stdout when omitted).\n"
          "--skip-streamk drops the CUTLASS GEMM fit and the interference\n"
          "ratio, and leaves the file marked calibrated=false.\n"
          "A BF16 run writes calibration_by_dtype.bf16; --base preserves\n"
          "the existing FP32 profile (and may be the same path as --out).\n";
      return 0;
    } else {
      std::cerr << "unknown or incomplete argument: " << arg << '\n';
      return 2;
    }
  }

  if (dtype != "f32" && dtype != "bf16") {
    std::cerr << "--dtype must be f32 or bf16\n";
    return 2;
  }
  options.bf16 = dtype == "bf16";
  if (options.combine_graph_batch<0 || (options.combine_graph_batch>0 && !partial_combine_only)) {
    std::cerr << "--combine-graph-batch must be nonnegative and requires --fp32-partial-combine-only\n";
    return 2;
  }
  if (partial_combine_only && (base.empty() || options.skip_streamk)) {
    std::cerr << "--fp32-partial-combine-only requires --base and excludes --skip-streamk\n";
    return 2;
  }

  try {
    auto const probed = tilemega::TargetSpec::Probe(options.device);
    auto target = base.empty() ? probed : tilemega::TargetSpec::FromJson(base);
    if (target.arch_tag != probed.arch_tag) {
      throw std::runtime_error("--base target " + target.arch_tag +
                               " does not match probed " + probed.arch_tag);
    }
    std::ostringstream discard;
    std::ostream& log =
        quiet ? static_cast<std::ostream&>(discard) : std::cerr;
    log << target.Summary() << '\n';
    if (partial_combine_only) {
      if (!target.CalibrationFor(dtype).calibrated || target.res.num_sms!=probed.res.num_sms)
        throw std::runtime_error("partial-only calibration requires the matching calibrated device profile");
      if (options.bf16 && !tilemega::arch::RuntimeCapsForTag(target.arch_tag).bf16_tensor_core)
        throw std::runtime_error("BF16 partial combine: capability_absent");
      tilemega::calib::MeasureFP32PartialCombine(target,options,log);
    } else if (options.bf16) {
      auto fp32 = std::move(target.calib);
      target.calib = {};
      tilemega::calib::Run(target, options, log);
      target.calib_bf16 = std::move(target.calib);
      target.calib = std::move(fp32);
    } else {
      target.calib = {};
      tilemega::calib::Run(target, options, log);
    }
    if (output.empty()) std::cout << target.ToJson();
    else target.ToJson(output);
    std::cerr << "dtype=" << dtype << " calibrated=" << std::boolalpha
              << target.CalibrationFor(dtype).calibrated
              << " in " << target.CalibrationFor(dtype).wall_seconds << " s\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-calibrate: " << error.what() << '\n';
    return 1;
  }
}
