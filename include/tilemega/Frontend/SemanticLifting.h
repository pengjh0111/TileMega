// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §1.5.1 (the analysis layer was never wired in), §2 the
// three-layer IR.
//
// FX -> L-sem. The importer used to stop at a per-call_function stage list,
// which is why CouplingDerivation -- whose input is an OperatorGraph -- had no
// production caller and CouplingMapAttr carried a constant. This lifts the
// already-recognized ModelPlan stage list into the SemanticGraph that
// Instantiate(sem, g) turns into that OperatorGraph. Nothing here re-does
// semantic recognition: every decision below reads a PlanStage kind, a
// PlanGemm field, or the buffer def-use chain.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/Semantics.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Frontend/ModelPlan.h>

namespace tilemega::frontend {

/// What a lifted operator does. Derived from the plan stage kind plus the
/// buffer def-use graph, never from an operator or parameter name.
enum class OpRole {
  kNorm,
  kQkvProjection,  ///< a projection whose result a RoPE or a KV append reads
  kProjection,
  kRoPE,
  kKVAppend,
  kAttention,
  kActivation,
  kResidualAdd,  ///< the `beta * C` half of a fused GEMM epilogue
  kGeneric,
};

std::string ToString(OpRole role);

/// How the generated TaskBody owns one task (Codegen/tasks/TaskBase.h).
/// `kElementChunk` means the CTA owns a grid-stride slice of a linearized
/// element range, so the CG's per-element task space is finer than the unit
/// the generator actually schedules; Codegen must compose the two.
enum class OwnershipKind { kTilePerBlock, kElementChunk };

std::string ToString(OwnershipKind kind);

struct LiftedOp {
  std::string name;
  OpRole role = OpRole::kGeneric;
  OwnershipKind ownership = OwnershipKind::kTilePerBlock;
  int stage = 0;  ///< index into ModelPlan::stages, or the FX stage when
                  ///< there is no plan
  int layer = 0;
  /// The FX node the plan stage was named after, so an operator-granularity
  /// task space can still be traced back to the exported graph.
  std::string fx_name;
};

struct LiftedModel {
  analysis::SemanticGraph sem;
  std::vector<LiftedOp> ops;  ///< parallel to sem.ops
  analysis::ClosedForm head_dim = analysis::ClosedForm::Constant(1);
  /// Operators lifted with GenericSemantics because no rule covered them.
  std::vector<std::string> degraded;
  bool has_plan = false;
};

/// The workload symbols the plan itself does not carry. Both must be symbols
/// the module's `tilemega.theta` binds, otherwise the derived metrics cannot
/// be evaluated by the CG verifier.
struct LiftOptions {
  std::string seq_symbol = "S";
  std::string past_symbol = "past";
};

/// Lift the recognized decoder stages. One sem op per stage, except a GEMM
/// with `beta != 0`, whose epilogue residual is its own pointwise op.
LiftedModel LiftSemantics(ModelPlan const& plan, LiftOptions const& options);

/// §0.1 degradation: one conservative task space per FX call_function, used
/// when no decoder layer was recognized. Never a placeholder -- the read set
/// is the whole operand, which I2 permits.
LiftedModel LiftGenericSemantics(std::vector<FxNodeRecord> const& tasks,
                                 std::vector<int> const& stage_of_task,
                                 LiftOptions const& options);

/// The granularity the generated megakernel actually launches at: GEMM tiles
/// from the CUTLASS variant, one token per CTA for RMSNorm, one (token, head)
/// per CTA for attention, and one element per task where the TaskBody owns a
/// grid-stride element chunk.
analysis::Granularity LaunchGranularity(LiftedModel const& model);

/// §2.7's granularity: Tm/Tn/Tkv symbolic, the QKV column tile one head wide,
/// attention split at Tkv. Used to reproduce the reference coupling table.
analysis::Granularity ReferenceGranularity(LiftedModel const& model);

}  // namespace tilemega::frontend
