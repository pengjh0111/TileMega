// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.3 / P4.8 Place.
//
// A placement is a bijection on [0, gridDim.x): the CTA whose hardware index
// is `blockIdx.x` takes the logical index `PlacedBlock()`, and everything
// downstream follows from that one number -- which task it owns, whether it is
// active in a stage, which event group it publishes.  Relabelling all three
// together is what makes an arbitrary permutation correct: no task is dropped
// or run twice, only the CTA that runs it changes.
//
// The hardware fills a persistent grid round-robin, CTA b onto SM b % sms, so
// consecutive logical indices land on distinct SMs.  `pair` inverts that: the
// CTAs co-resident on one SM take *consecutive* logical indices.  It is the
// only permutation the measured affinity argues for
// (docs/experiments/PLACE/raw/affinity.txt): two GEMM tasks differing only in
// the N tile share the whole A panel, and consecutive task indices differ in
// N.  The other two arms exist to bound the opposite direction -- if
// scattering the map costs nothing either, placement cannot be worth anything
// on this model, and that is a measurement rather than an opinion.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

#ifndef TILEMEGA_PLACEMENT
#define TILEMEGA_PLACEMENT 0
#endif

namespace tilemega::codegen {

#if TILEMEGA_PLACEMENT == 1
/// Resident CTAs per SM, published by the harness before the first launch.
/// Only the `pair` arm needs it, and it is defined only for that arm: a device
/// global shifts every later global's address in the constant bank, and a
/// default build has to emit exactly the SASS it emitted before the knob.
/// Internal linkage because whole-program mode (-rdc=false) admits no other
/// kind of device variable in a header.
static __device__ int tilemega_blocks_per_sm = 1;
#endif

__device__ inline int PlacedBlock() {
  int const b = static_cast<int>(blockIdx.x);
#if TILEMEGA_PLACEMENT == 0
  return b;
#else
  int const g = static_cast<int>(gridDim.x);
#if TILEMEGA_PLACEMENT == 1
  int const w = tilemega_blocks_per_sm;
  int const sms = w > 0 ? g / w : 0;
  // Only an exact tiling is a bijection; a ragged grid keeps today's map
  // rather than losing or duplicating a task.
  if (w <= 1 || sms * w != g) return b;
  return (b % sms) * w + b / sms;
#elif TILEMEGA_PLACEMENT == 2
  return g - 1 - b;
#elif TILEMEGA_PLACEMENT == 3
  // Multiplying by a unit of Z/g puts adjacent indices as far apart as the
  // ring allows.  31 is a unit exactly when it does not divide g.
  return g % 31 == 0 ? b
                     : static_cast<int>((static_cast<long long>(b) * 31) % g);
#else
#error "TILEMEGA_PLACEMENT must be 0 (identity), 1 (pair), 2 (reverse) or 3 (scatter)"
#endif
#endif
}

}  // namespace tilemega::codegen
