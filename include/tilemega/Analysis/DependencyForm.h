// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2.3 (Coarsen/Place), §2.7 (the coupling table), §5.3
// (TaskBody ownership).
//
// The generated megakernel waits per CTA, so it needs C in a form a CTA can
// evaluate with no table lookup: the wait set as a function of the consumer
// task id and the two stages' own task counts, both of which the runtime
// already computes from `TaskOwnership`. One window shape covers every exact
// case §2.7's table contains; anything else is reported as the I2 relaxation
// to the producer's whole launch axis and must be annotated as such.
#pragma once

#include <string>

#include <tilemega/Analysis/CouplingDerivation.h>

namespace tilemega::analysis {

/// Consumer task `c` waits on producer tasks
/// `[(c / div) * scale + offset, ... + count)`, intersected with the
/// producer's live range `[0, P)`. The intersection is not a widening: it is
/// the same clamp the fit is verified under, and it is what keeps a window
/// fitted at prefill sound at decode, where the producer has fewer tasks.
///
/// The five map shapes §2.7's rows exhibit are all instances:
///   identity  `c -> c`               div=1  scale=1   offset=0 count=1
///   projection `(m,n) -> m`          div=N  scale=1   offset=0 count=1
///   floordiv  `row -> row/Tm`        div=Tm scale=1   offset=0 count=1
///   interval  `(m,n) -> {(m,k)}`     div=N  scale=K   offset=0 count=K
///   whole axis (the honest one)      div=1  scale=0   offset=0 count=P
/// A multi-producer wait set (§2.7 row 11) is not a shape at all: the
/// derivation emits one edge per operand, so it is two windows.
struct WaitWindow {
  bool narrowed = false;  ///< false is kAll: every producer task
  long div = 1;
  long scale = 0;
  long offset = 0;
  long count = 1;

  bool IsIdentity() const {
    return narrowed && div == 1 && scale == 1 && offset == 0 && count == 1;
  }
  /// "all", "identity", or "window(div,scale,offset,count)".
  std::string ToString() const;
};

bool operator==(WaitWindow const& a, WaitWindow const& b);
inline bool operator!=(WaitWindow const& a, WaitWindow const& b) {
  return !(a == b);
}

WaitWindow ParseWaitWindow(std::string const& text);

/// The tightest window that reproduces `C` exactly for *every* consumer task,
/// under the row-major linearization of both task spaces -- the order a
/// `kTilePerBlock` TaskBody indexes its tiles in (GemmStageTaskBody's
/// `local / tiles_n`, AttentionChunkTaskBody's `query / heads`). `known` must
/// bind every symbol the two spaces use; a relation isl cannot enumerate under
/// it (a free parameter, an unbounded domain) yields the relaxation rather
/// than a guess.
WaitWindow FitWaitWindow(CouplingEdge const& edge, OperatorNode const& producer,
                         OperatorNode const& consumer,
                         ParamBinding const& known);

}  // namespace tilemega::analysis
