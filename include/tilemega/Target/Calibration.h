// SPDX-License-Identifier: BSD-3-Clause
//
// TileMega -- Target/Calibration.h
//
// Phase-4 (P4.1) microbenchmarks that fill TargetSpec::Calib.
//
// The quantities here are exactly the ones the §4.4 cost model consumes, and
// nothing else: a rate that no term of the model reads is GPU time spent for
// a number nobody uses.  What each group feeds:
//
//   Pipelines    -> the resource vector u(o) = <t_TC, t_CUDA, t_SFU, t_SMEM,
//                   t_L2, t_DDR>; each component is work / rate.
//   Sync         -> T_sync = |image(C_kappa)| * latency(sync_kind).
//   Stream-K     -> the split-factor term, time_CTA = a + b*[peers>1]
//                   + c*iters + d*(peers-1).
//   Interference -> whether the model may treat concurrent tasks as having
//                   independent durations at all.
//
// Nothing here may be filled in by hand.  A target the local machine cannot
// run stays uncalibrated; a guessed constant makes the ranker look better and
// means nothing.
#pragma once

#include <tilemega/Target/TargetSpec.h>

#include <iosfwd>
#include <vector>

namespace tilemega::calib {

struct Options {
  int device = 0;
  /// Repeats per measurement.  The reported value is the median and the
  /// reported dispersion is stddev/mean over these repeats.
  ///
  /// 41 rather than a handful because the Stream-K intercept `a` is a few
  /// hundred nanoseconds to a few microseconds, the same order as the launch
  /// path subtracted from every point it is fitted through; at 7 repeats it
  /// scattered ~20% run to run and at 41 it scatters ~5%.  The whole run is
  /// 6 seconds either way.
  int repeats = 41;
  /// Skip the Stream-K fit, which is the only group that compiles and runs
  /// real CUTLASS GEMMs and so dominates wall time.
  bool skip_streamk = false;
  /// Tile shapes to fit Stream-K coefficients for.  Empty means the built-in
  /// set (see StreamKShapes).
  std::vector<TargetSpec::StreamKPoint> streamk_shapes;
};

/// The tile shapes MeasureStreamK fits by default: the CUTLASS SIMT f32
/// default, the two shapes the oracle sweep found optimal, and three spread
/// across the legal envelope so the cost model's extrapolation has something
/// to be checked against.
std::vector<TargetSpec::StreamKPoint> StreamKShapes();

void MeasurePipelines(TargetSpec& spec, Options const& options, std::ostream& log);
void MeasureSync(TargetSpec& spec, Options const& options, std::ostream& log);
void MeasureInterference(TargetSpec& spec, Options const& options, std::ostream& log);
void MeasureStreamK(TargetSpec& spec, Options const& options, std::ostream& log);

/// Run every group, stamp provenance, and set `calibrated`.
void Run(TargetSpec& spec, Options const& options, std::ostream& log);

}  // namespace tilemega::calib
