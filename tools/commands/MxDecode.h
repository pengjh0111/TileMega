// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- tools/commands/MxDecode.h: the MX-4 static decode compile path.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace tilemega::commands::mx_decode {

struct Request {
  std::filesystem::path input, output;
  std::string model, pin, runtime_target, selection = "predicted", measure_cmd, table;
  std::string v2_solve_table, v2_results;
  int v2_threads = 576;
  int tmin = 128;  // §6 D5: geometric-mean TPOT of the 4 dev cells, 128 < 64 < 32
  int batch = 1, capacity = 0, past_lo = 0, past_hi = 0, grid = 104;
  int watchdog = 1, barrier_prefetch = 1;
  bool trace_stage = false, allow_resources = false;
  /// Interpreted-executor options given on the command line (rejected).
  std::vector<std::string> interpreted_flags;
};

/// Builds the static program when the bridge describes a dense decoder;
/// returns -1 (nothing done) for a DM plan, which keeps the old path.
int TryCompile(Request const& request);
/// Temporary explicit-pin entry while the v2 measured-table/E2E gates remain
/// open. Remove the MX-4 default route only after those gates select v2.
int TryCompileV2(Request const& request);
/// CPU-only explicit route: write qualified solver candidate previews or consume measured pairs.
int TryPrepareV2(Request const& request);

}  // namespace tilemega::commands::mx_decode
