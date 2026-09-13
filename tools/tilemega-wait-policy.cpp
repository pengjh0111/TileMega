// SPDX-License-Identifier: BSD-3-Clause
//
// EX-E3 step 2 (R3 H5): print the wait policy a target was calibrated for, as
// the nvcc flags that compile it in.
//
//   tilemega-wait-policy [tag] [dtype] [repo_root]
//
// One line on stdout, meant to be spliced into an nvcc command line with
// `$(...)`; the file it read goes to stderr so the capture stays clean. The
// values live in `configs/targets/` and reach a kernel only through here, so
// no runner carries a literal and no architecture inherits another's answer --
// sm_89 wants a 64-poll spin ahead of the sleep and sm_120, whose hop is
// already at a ~400 ns floor, does not (F-145). Passing no flags at all is the
// off state: the headers default to the wait the generator has always emitted.

#include <tilemega/Target/TargetSpec.h>

#include <cstdio>
#include <exception>
#include <string>

#ifndef TILEMEGA_SOURCE_DIR
#define TILEMEGA_SOURCE_DIR "."
#endif

int main(int argc, char** argv) {
  std::string const tag = argc > 1 ? argv[1] : "sm_89";
  std::string const dtype = argc > 2 ? argv[2] : "bf16";
  std::string const root = argc > 3 ? argv[3] : TILEMEGA_SOURCE_DIR;
  std::string const path = root + "/configs/targets/" + tag + ".json";

  try {
    tilemega::TargetSpec const spec = tilemega::TargetSpec::FromJson(path);
    tilemega::TargetSpec::Calib const& calib = spec.CalibrationFor(dtype);
    std::fprintf(stderr, "wait policy from %s [%s]\n", path.c_str(), dtype.c_str());
    std::printf("-DTILEMEGA_WAIT_POLICY=1 -DTILEMEGA_WAIT_SPIN_ITERS=%d "
                "-DTILEMEGA_WAIT_BACKOFF_NS=%d -DTILEMEGA_WAIT_BACKOFF_GROW=%d "
                "-DTILEMEGA_WAIT_BACKOFF_CAP_NS=%d\n",
                calib.wait_spin_iters, calib.wait_backoff_ns,
                calib.wait_backoff_grow, calib.wait_backoff_cap_ns);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "tilemega-wait-policy: %s: %s\n", path.c_str(), e.what());
    return 2;
  }
  return 0;
}
