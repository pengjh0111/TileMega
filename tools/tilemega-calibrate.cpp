// SPDX-License-Identifier: BSD-3-Clause
// Phase-4 (P4.1) calibration entry point: probe the device, run the §4.4
// microbenchmarks, and write configs/targets/<arch>.json.
//
// A target the local machine cannot run is never filled in from another
// target's numbers -- run this on that hardware, or leave the file
// uncalibrated.
#include <tilemega/Target/Calibration.h>
#include <tilemega/Target/TargetSpec.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
  tilemega::calib::Options options;
  std::string output;
  bool quiet = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) options.device = std::stoi(argv[++i]);
    else if (arg == "--repeats" && i + 1 < argc) options.repeats = std::stoi(argv[++i]);
    else if (arg == "--skip-streamk") options.skip_streamk = true;
    else if (arg == "--quiet") quiet = true;
    else if (arg == "--out" && i + 1 < argc) output = argv[++i];
    else if (arg == "--help") {
      std::cout <<
          "usage: tilemega-calibrate [--device N] [--repeats N]\n"
          "                          [--skip-streamk] [--quiet] [--out FILE]\n"
          "\n"
          "Measures the §4.4 cost-model constants on the GPU at --device and\n"
          "writes the target JSON to --out (stdout when omitted).\n"
          "--skip-streamk drops the CUTLASS GEMM fit and the interference\n"
          "ratio, and leaves the file marked calibrated=false.\n";
      return 0;
    } else {
      std::cerr << "unknown or incomplete argument: " << arg << '\n';
      return 2;
    }
  }

  try {
    auto target = tilemega::TargetSpec::Probe(options.device);
    std::ostringstream discard;
    std::ostream& log =
        quiet ? static_cast<std::ostream&>(discard) : std::cerr;
    log << target.Summary() << '\n';
    tilemega::calib::Run(target, options, log);
    if (output.empty()) std::cout << target.ToJson();
    else target.ToJson(output);
    std::cerr << "calibrated=" << std::boolalpha << target.calib.calibrated
              << " in " << target.calib.wall_seconds << " s\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-calibrate: " << error.what() << '\n';
    return 1;
  }
}
