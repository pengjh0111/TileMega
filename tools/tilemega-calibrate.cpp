// SPDX-License-Identifier: BSD-3-Clause
// Phase-4 calibration entry point. Today it records probed capabilities and
// resource budgets; latency/bandwidth microbenchmarks remain intentionally
// unimplemented (skeleton §4.4 and §7 Phase 4).
#include <tilemega/Target/TargetSpec.h>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  int device = 0;
  std::string output;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
    else if (arg == "--out" && i + 1 < argc) output = argv[++i];
    else if (arg == "--help") {
      std::cout << "usage: tilemega-calibrate [--device N] [--out FILE]\n";
      return 0;
    } else {
      std::cerr << "unknown or incomplete argument: " << arg << '\n';
      return 2;
    }
  }

  try {
    auto target = tilemega::TargetSpec::Probe(device);
    // TODO(P4.1): measure latency and bandwidth fields, then set calibrated.
    if (output.empty()) std::cout << target.ToJson();
    else target.ToJson(output);
    std::cerr << "Phase-4 measurements are not implemented; calibrated=false\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-calibrate: " << error.what() << '\n';
    return 1;
  }
}
