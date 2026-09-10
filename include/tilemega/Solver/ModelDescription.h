// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.4 (cost model input), §5.3 (generated model tables).
//
// The host-side mirror of the tables `CouplingGraphToCUDA` emits: the same
// GEMM shapes and the same stage sequence the megakernel executes, in a form
// the solver can evaluate without a compiler or a device.
//
// The archived generated-.cu reader remains an independent measurement
// anchor. The CG reader carries semantic dimension roles and symbolic metrics;
// callers must bind dimensions before entering the existing FP64 evaluator.
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/Semantics.h>
#include <map>
#include <memory>
#include <set>
#include <tilemega/Codegen/RuntimePlan.h>
#include <optional>

#include <string>
#include <vector>

#ifndef TILEMEGA_PARAMETRIC_INPUT
#define TILEMEGA_PARAMETRIC_INPUT 1
#endif

namespace mlir { class ModuleOp; }

namespace tilemega::solver {
struct AttentionCostPlan;

/// The symbolic dimensions, bound at launch (ModelRuntime.h `ModelDims`).
struct ModelDims {
  int seq = 0;
  int past = 0;
  int total = 0;
  // Empty names preserve the historical concrete representation. Unbound
  // symbolic dimensions must be substituted before entering FP64 evaluation.
  std::string seq_parameter;
  std::string past_parameter;
  bool IsSymbolic() const { return !seq_parameter.empty() || !past_parameter.empty(); }
  static ModelDims Symbolic(std::string seq_name, int concrete_past);
};

struct ModelCouplingMetrics {
  int producer = -1, consumer = -1;
  analysis::QuasiPolynomial wait, fanout, volume, count;
  analysis::CouplingRelation relation;
  std::string producer_task, consumer_task;
};
struct ModelRuntimeEventMetrics {
  analysis::QuasiPolynomial task_refs, wait_entries, max_worker_task_refs;
  analysis::QuasiPolynomial fence_free_producers, fused_edges;
  std::vector<codegen::GemmRuntimeRecord> gemms;
  int grid=0, threads=0, kappa=0, stage_count=0;
};
struct ModelCouplingAnalysis {
  std::vector<ModelCouplingMetrics> edges;
  // Counts belong to an exact runtime variant/residency, not to every g a
  // caller might try against the same concrete model description.
  std::optional<ModelRuntimeEventMetrics> runtime;
};

/// M stays symbolic, so a GEMM contributes only N and K.  The destination
/// buffer is carried too: tier-2 alignment propagation (§P4.3) has to find who
/// reads this GEMM's output, and the generated tables name it by buffer id.
struct GemmOp {
  int n = 0;
  int k = 0;
  int in_buffer = -1;
  int out_buffer = -1;
};

/// Mirrors `TaskKind`; kGemmCombine is absent because it exists only after the
/// host-side split rewrite, which the cost model performs itself.
enum class StageKind {
  kGemm,
  kRMSNorm,
  kRoPE,
  kKVAppend,
  kElementwise,
  kAttention,
};

struct ModelStage {
  StageKind kind = StageKind::kGemm;
  int gemm = -1;  ///< index into ModelDescription::gemms, -1 when not a GEMM
  int extent = 0;
  int width = 0;
  int group = 0;
  /// Buffer ids the stage touches, in the generated order (inputs first).
  std::vector<int> operands;

  /// How many contiguous elements of a read buffer one task of this stage
  /// covers.  This is the `Tr` of §P4.3's wait inflation, and it is read off
  /// the generated table rather than assumed: RoPE and KVAppend carry it in
  /// `width` (the head dimension), the elementwise tail in `extent`.
  int ReadGranularity() const;
};

struct ModelTaskSemantics {
  analysis::SemanticOp op;
  std::map<std::string,analysis::ClosedForm> tiles;
  int stage = -1;
  bool element_chunk = false;
};

struct ModelDescription {
  std::string name;
  ScalarType dtype = ScalarType::kF32;
  ModelDims dims;
  std::shared_ptr<AttentionCostPlan const> attention_plan;
  std::vector<GemmOp> gemms;
  std::vector<ModelStage> stages;
  // Empty on archived generated inputs. Access-derived pricing requires this
  // verified CG payload and must not reconstruct it from generated numbers.
  std::vector<ModelTaskSemantics> task_semantics;
  std::set<std::string> exported_tensors;
  /// `stage_successors[i]` = the stages that depend on stage `i`, read out of
  /// the generated `kDependencies` table.  That table is already transitively
  /// reduced by the generator, so it is the DAG the megakernel actually
  /// enforces rather than a re-derivation of it (§P4.8).
  std::vector<std::vector<int>> stage_successors;
  ModelCouplingAnalysis coupling_metrics;
  analysis::ParamBinding metric_bindings;
  std::string seq_metric_parameter, past_metric_parameter;
  std::vector<std::pair<std::string, std::string>> metric_aliases;

  /// Parse the `kGemms` and `kStages` tables out of a generated .cu.  Throws
  /// std::runtime_error when either table is missing or malformed -- a silent
  /// empty model would validate a cost model against nothing.
  static ModelDescription FromGeneratedCuda(std::string const& path,
                                            ModelDims dims,
                                            std::string name);
  static ModelDescription FromCouplingGraph(mlir::ModuleOp module,
                                            ModelDims dims, std::string name);
  ModelDescription SubstituteParams(analysis::ParamBinding const& bindings) const;
  analysis::ParamBinding MetricBindings(analysis::ParamBinding const& bindings = {}) const;

  /// Bytes of parameter and activation storage the model keeps live, which is
  /// what the L2 must hold for the weight stream to stay resident (§2.2(e)).
  double LiveFootprintBytes() const;
  int RuntimeStages(int stage) const;
  int NonGemmSharedBytes() const;
};

}  // namespace tilemega::solver
