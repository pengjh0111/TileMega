// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.1 (the Plan's sync price), §4.4 (measured, not assumed).
//
// The measured cross-worker hop.  It lives in its own header because two
// consumers must price a hop identically or their rankings are not comparable:
// the execution simulator (EX-S1) replays a plan with it, and the EFT scheduler
// (EX-S2) chooses a plan by it.
#pragma once

#include <string>

namespace tilemega::solver {

/// `c0 + c1*log2(1 + N/R) + c2*log2(R)`.  N is the number of CTAs polling one
/// event row, R the number of rows under contention.  On sm_89 c1 and c2 are
/// zero inside one standard error (SIMULATOR/README.md); they are carried
/// rather than pinned to zero so that "contention is free" stays a measurement
/// and not a modelling assumption.
struct HopCurve {
  double c0 = 0.0;
  double c1 = 0.0;
  double c2 = 0.0;

  double Ns(int consumers, int rows) const;

  /// Read the three coefficients from a `hop_ns.tsv` as `hop_fit.py` writes it.
  static bool FromTsv(std::string const& path, HopCurve* out, std::string* error);
};

}  // namespace tilemega::solver
