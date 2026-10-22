// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/DecodeCommon.h
//
// Shared by the MX-4 decode stage bodies (DecodeGemm.h, DecodeAttention.h).
#pragma once

namespace tilemega::backend::mx {

/// An int the compiler cannot see through. Every body derives its lane and
/// CTA coordinates through this at entry: the program (and stage_bench) runs
/// the bodies inside step and layer loops, and otherwise the compiler hoists
/// each body's loop-invariant geometry and addresses out of those loops and
/// keeps them live across every other stage (spills, measured in stage_bench).
__device__ __forceinline__ int Opaque(int value) {
  asm volatile("" : "+r"(value));
  return value;
}
/// The same for a value uniform over the wave (blockIdx, the wave index).
/// An asm result counts as divergent, which made the compiler lay out every
/// branch on it as a masked chain with all arms' values live: readfirstlane
/// restores uniformity.
__device__ __forceinline__ int UniformOpaque(int value) { return __builtin_mxc_readfirstlane(Opaque(value)); }

}  // namespace tilemega::backend::mx
