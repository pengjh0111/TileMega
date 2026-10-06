// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_EP_PARALLEL_ARGMAX
#define TILEMEGA_EP_PARALLEL_ARGMAX 0
#endif
namespace tilemega::backend {
// The inverse RMS values are dead before argmax. Its four per-warp pairs per
// row occupy a dedicated tail after the output tile, never an output row.
constexpr int ServingEpilogueScratchBytes(int m,int n,bool parallel=false) {
  return 4*m*n+(parallel?32*m:4*m);
}
}
