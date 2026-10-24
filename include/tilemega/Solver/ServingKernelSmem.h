// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Solver/ServingKernelSmem.h
//
// MX4 prompt §5.4: the dynamic shared memory of a non-paged, non-DM serving
// kernel, one formula for both sides. The solver estimates a candidate's
// residency with it (SkeletonSearch EstimateResources); codegen emits the
// same value for the selected plan as TILEMEGA_SOLVER_SMEM_BYTES, and the
// generated source static_asserts that its TaskSmem union plus the stage
// descriptor cache is exactly that size (ModelHarness.cuh), with each term
// below asserted against the type it stands for.
#pragma once

#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Solver/BackendCostQuery.h>

#include <algorithm>

namespace tilemega::solver {

/// Threads of a non-paged serving CTA (ModelHarness.cuh kHarnessThreads).
inline constexpr int kServingHarnessThreads = 128;
/// GemmStageTaskBody.h StageDescCache (stage descriptor, GEMM invocation,
/// Params, two keys), rounded to 16 bytes: the first bytes of the kernel's
/// dynamic shared memory (§5.3).
inline constexpr int kServingStageCacheBytes = 672;
/// ServingArgmaxReduceTaskBody::SharedStorage (also the serving RMSNorm's).
inline constexpr int kServingArgmaxSmemBytes = 32;
/// TaskSmem's alignment.
inline constexpr int kServingTaskSmemAlign = 16;
/// The residencies the serving task bodies are calibrated at (MX4 prompt
/// §5.6: in place, R in {1,2,3,4}); a candidate is priced, and so searched,
/// only at one of them.
inline constexpr int kServingCalibratedResidency = 4;

/// The TaskSmem members a plan has, by the task kinds it runs (KindsUsed.h).
struct ServingSmemTerms {
  int gemm = 0;          // max over the plan's GEMM variants
  int attention = 0;     // the fused attention body's storage, 0 without one
  bool norm = false;     // kRMSNorm or kQKNorm: SIMT norm storage
  bool pointwise = false;  // kElementwise, kAdd, kKVAppend or kRoPE
  bool argmax = false;   // kRMSNorm or kArgmaxReduce (serving)
};

constexpr int ServingGemmVariantSmemBytes(int m, int n, int k, int stages, int impl) {
  return impl ? ServingGemvSmemBytes(m, n) : ServingBF16SmemBytes(m, n, k, stages);
}

/// The fused attention body's storage: decode (seq 1) the SIMT body, prefill
/// the MMA body with codegen's KV tile rule (Codegen.cpp
/// emitServingAttentionConfig: 64 when it fits in the GEMM storage).
constexpr int ServingFusedAttentionSmemBytes(int head_dim, int q_per_kv, int seq, int gemm_bytes) {
  return seq == 1 ? codegen::ServingDecodeAttentionSimtSharedBytes(head_dim, q_per_kv)
                  : codegen::ServingAttentionSharedBytes(
                        head_dim, codegen::ServingAttentionKvTile(head_dim, gemm_bytes));
}

constexpr int ServingKernelSmemBytes(ServingSmemTerms const& t) {
  int bytes = t.gemm;
  bytes = std::max(bytes, t.attention);
  if (t.norm) bytes = std::max(bytes, 4 * kServingHarnessThreads);
  if (t.pointwise) bytes = std::max(bytes, 4);
  if (t.argmax) bytes = std::max(bytes, kServingArgmaxSmemBytes);
  int const unioned = (bytes + kServingTaskSmemAlign - 1) / kServingTaskSmemAlign * kServingTaskSmemAlign;
  int const offset =
      (kServingStageCacheBytes + kServingTaskSmemAlign - 1) / kServingTaskSmemAlign * kServingTaskSmemAlign;
  return offset + unioned;
}

}  // namespace tilemega::solver
