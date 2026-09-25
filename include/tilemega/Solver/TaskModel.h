// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/OpArithmetic.h>
#include <tilemega/Analysis/FusionAccess.h>
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Codegen/tasks/ScalarDataflow.h>

namespace tilemega::solver {
#ifndef TILEMEGA_SCALAR_TASK_WORK
#define TILEMEGA_SCALAR_TASK_WORK 1
#endif
analysis::OperatorGraph InstantiateModelTasks(ModelDescription const& model,
                                            std::vector<GemmConfig> const& configs);
std::vector<ModelCouplingMetrics> InstantiateModelCouplings(
    ModelDescription const& model,std::vector<GemmConfig> const& configs,
    std::optional<std::pair<int,int>> stage_pair=std::nullopt);
struct RuntimeScalarAccess {
  analysis::CouplingRelation ownership,writes;
  std::map<std::string,analysis::CouplingRelation> reads;
};
struct DerivedTaskInput {
  analysis::OperatorNode task;
  analysis::TaskWork work;
  analysis::OpArithmetic arithmetic;
  std::vector<std::string> cost_coordinates;
  std::optional<codegen::ScalarDataflow> scalar_flow;
  std::optional<RuntimeScalarAccess> scalar_access;
  // Mixed-width storage (e.g. FP32 partials and BF16 residuals) retains
  // each access cardinality before converting elements to bytes.
  std::optional<analysis::QuasiPolynomial> physical_read_bytes;
  // Populated only by the regime-A preparer, using element-level write images.
  std::optional<analysis::QuasiPolynomial> no_producer_read_bytes;
  std::optional<analysis::QuasiPolynomial> external_write_bytes;
  double stream_bytes=0, produced_live_bytes=0;
  struct ServingAttention {
    int block_count=0;
    int block_extent=0;
    int total=0;
    int kv_tile=64;
  };
  std::optional<ServingAttention> serving_attention;
  /// The operand the kind's body prefetches (`ScalarPrefetchOperand`) when it
  /// is on the read-only frontier, else -1.  Runtime-ownership tasks only.
  int prefetch_operand=-1;
};
void BindTaskDramProvenance(DerivedTaskInput& input,
    ModelTaskSemantics const& semantic,analysis::DramFloor const& floor,
    analysis::ParamBinding const& theta,bool serving=false);
/// What `PriceTaskInstances` credits to §5.3.1's Prefetch, per instance: the
/// prefetch operand priced as a local read (the model's own fused-input
/// semantics) subtracted from the task priced reading it from global, and only
/// where its footprint is whole 16-byte lines within the page the executor was
/// built with (`TILEMEGA_PREFETCH_PAGE_BYTES`), which is the executor's own
/// test.  Zero where the model's bottleneck is elsewhere.
struct PrefetchPricing {
  int page_bytes=1024;
  std::vector<double>* ns=nullptr;
};
DerivedTaskInput DeriveCombineTaskInput(ModelDescription const& model, int stage,
    GemmConfig const& config, analysis::OperatorGraph const& graph,
    int threads, bool tile_ownership, bool fp32_partials);
TaskMemoryTraffic DeriveTaskMemoryTraffic(DerivedTaskInput const& input,
    analysis::ParamBinding const& theta, analysis::ParamBinding const& coordinates,
    int read_element_bytes, int write_element_bytes,
    analysis::AccessDomain domain = analysis::AccessDomain::kPhysicalTensor);
std::vector<TaskMemoryTraffic> DeriveTaskMemoryTrafficBatch(DerivedTaskInput const& input,
    analysis::ParamBinding const& theta,std::vector<analysis::ParamBinding> const& coordinates,
    int read_element_bytes,int write_element_bytes,
    analysis::AccessDomain domain = analysis::AccessDomain::kPhysicalTensor);
std::vector<double> PriceTaskInstances(CostModel const& cost,DerivedTaskInput const& input,
    BackendTraits const& traits,Residency residency,ModelDescription const& model,int chunks,
    std::vector<analysis::ParamBinding> const& coordinates,double active_ctas_per_sm=1.0,
    PrefetchPricing const* prefetch=nullptr);
BackendTraits ModelTaskTraits(ModelDescription const& model, int stage,
                              GemmConfig const& config);
analysis::TaskAccesses DeriveModelTaskAccesses(ModelTaskSemantics const& semantic,
                                             DerivedTaskInput const& input);
struct ModelFusionCandidate {
  DerivedTaskInput producer, consumer;
  analysis::FusionAccesses accesses;
  analysis::MixedArithmetic arithmetic;
  analysis::TaskAccesses producer_accesses,consumer_accesses;
};
struct FusedTaskInput {
  std::string name;
  std::vector<ModelTaskSemantics> semantics;
  std::vector<DerivedTaskInput> phases;
  std::vector<analysis::CouplingRelation> phase_maps;
  analysis::TaskAccesses accesses;
  analysis::QuasiPolynomial task_count;
  analysis::MixedArithmetic arithmetic;
};
// Read the actual L-task payload, without pretending its original stage plan
// already describes an executable fused runtime schedule.
std::vector<FusedTaskInput> ReadFusedTaskInputs(mlir::ModuleOp module);
ModelFusionCandidate DeriveWrittenFusionCandidate(FusedTaskInput const& input,
    ModelDescription const& context,std::vector<GemmConfig> const& configs);
// L-task candidates are identified by semantic names, not runtime stage IDs.
// Runtime ownership is a later projection and cannot define fusion legality.
ModelFusionCandidate DeriveLogicalFusionCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, std::string const& producer,
    std::string const& consumer);
ModelFusionCandidate DeriveModelFusionCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, int producer_stage, int consumer_stage);
analysis::TaskWork DeriveRuntimeScalarWork(ModelDescription const& model,
    ModelTaskSemantics const& semantic,analysis::OperatorNode const& task,
    analysis::TaskWork work,int threads,RuntimeScalarAccess* accesses=nullptr);
DerivedTaskInput DeriveModelTaskInput(ModelDescription const& model,
                                    ModelTaskSemantics const& semantic,
                                    analysis::OperatorGraph const& graph,
                                    GemmConfig const* config,
                                    bool runtime_ownership=true);
}  // namespace tilemega::solver
