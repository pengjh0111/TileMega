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
  PiecePriceCache prices;
  struct SpaceEntry {FlowSpace space;PiecePrices prices;std::string geometry_key;};
  std::map<std::string,SpaceEntry> spaces;
  std::uint64_t space_hits=0,space_misses=0;
  std::map<std::string,std::shared_ptr<std::vector<std::pair<int,int>> const>> releases;
  std::map<std::string,bool> nonprefix;
  std::map<std::string,bool> all_producer;
  std::uint64_t release_hits=0,release_misses=0;
};
struct PreparedFlow {
  FlowProblem flow;
  std::vector<PiecePrices> prices;
  std::vector<int> colocated_producer;
  std::vector<std::string> varying_spaces;
  int nonprefix_edges=0;
};
SymbolicProblem PrepareFlowStructure(SymbolicProblem const& base,std::vector<GemmConfig> const& geometry,
    int workers,int kappa,analysis::CouplingCache& cache,FlowPreparationCache* prepared=nullptr);
PreparedFlow PrepareFlow(SymbolicProblem const& problem,analysis::DramFloor const& floor,
    TargetSpec const& target,int residency,HopCurve const& hop,
    analysis::CouplingCache& coupling,FlowPreparationCache& cache,bool colocate=true,int kernel_shared_bytes=0);
void ApplyFlowPrices(SymbolicProblem& problem,PreparedFlow const& flow,TargetSpec const& target,int residency);
std::vector<TaskPriceParts> ExpandFlowPrices(PreparedFlow const& flow);
}
