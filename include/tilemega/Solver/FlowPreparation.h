// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/PiecePricing.h>
#include <tilemega/Solver/StageFlowModel.h>
namespace tilemega::solver {
struct FlowPreparationCache {
  std::string target_key;
  struct OwnershipEntry {analysis::CouplingRelation map;analysis::QuasiPolynomial count;};
  std::map<std::string,OwnershipEntry> ownership;
  std::map<std::string,long> task_counts;
  std::map<std::string,analysis::CouplingRelation> projected;
  std::map<std::string,std::string> signatures;
  std::string graph_geometry;
  std::shared_ptr<analysis::OperatorGraph const> graph;
  std::map<std::string,std::string> floor_tensor_keys;
  PiecePriceCache prices;
  struct SpaceEntry {FlowSpace space;PiecePrices prices;std::string geometry_key;};
  std::map<std::string,SpaceEntry> spaces;
  std::uint64_t space_hits=0,space_misses=0;
  std::map<std::string,std::shared_ptr<std::vector<std::pair<int,int>> const>> releases;
  std::map<std::string,bool> nonprefix;
  std::map<std::string,bool> all_producer;
  // Oracle classifications are geometry-local; most edges are unchanged when
  // coordinate descent moves one operator class.
  std::map<std::string,std::pair<bool,bool>> edge_metadata; // one-to-one, all-producer
  std::uint64_t release_hits=0,release_misses=0;
  std::uint64_t incremental_space_hits=0;
  double spaces_ms=0,edges_ms=0,graph_ms=0;
  double derive_ms=0,price_ms=0,piece_map_ms=0;
};
struct PreparedFlow {
  FlowProblem flow;
  std::vector<PiecePrices> prices;
  std::vector<std::string> space_signatures,space_geometry_keys;
  std::vector<int> colocated_producer;
  std::vector<std::string> varying_spaces;
  int nonprefix_edges=0;
};
struct BoundRuntimeWindow {
  analysis::WaitWindow window;
  long offset = 0;
};
std::vector<BoundRuntimeWindow> BindRuntimeWindows(
    RuntimeProjection const& projection, int producer, int consumer,
    analysis::ParamBinding const& theta);
int RuntimeReleaseEndpoint(int cg_last, int consumer_task, int producer_count,
    std::vector<BoundRuntimeWindow> const& windows, bool force_all);
SymbolicProblem PrepareFlowStructure(SymbolicProblem const& base,std::vector<GemmConfig> const& geometry,
    int workers,int kappa,analysis::CouplingCache& cache,FlowPreparationCache* prepared=nullptr);
PreparedFlow PrepareFlow(SymbolicProblem const& problem,analysis::DramFloor const& floor,
    TargetSpec const& target,int residency,HopCurve const& hop,
    analysis::CouplingCache& coupling,FlowPreparationCache& cache,bool colocate=true,int kernel_shared_bytes=0,
    PreparedFlow const* prior=nullptr,std::vector<bool> const* reusable_stages=nullptr);
void ApplyFlowPrices(SymbolicProblem& problem,PreparedFlow const& flow,TargetSpec const& target,int residency);
std::vector<TaskPriceParts> ExpandFlowPrices(PreparedFlow const& flow);
}
