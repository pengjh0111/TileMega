// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Solver/StageFlowModel.h>
#include <map>

namespace tilemega::solver {
struct Sm90NativeCtgSpace {
  std::string name;
  int stage=0,count=0;
  frontend::OpRole role=frontend::OpRole::kGeneric;
  analysis::CouplingRelation ownership;
};
struct Sm90NativeCtgEdge {
  int producer=0,consumer=0;
  analysis::CouplingRelation relation; // physical producer -> consumer
};
struct Sm90NativeCtg {
  // Present on an importer miss; cached compositions retain the exact CTG
  // witnesses below without rebuilding an MLIR module (R6 U2).
  mlir::OwningOpRef<mlir::ModuleOp> module;
  analysis::ParamBinding binding;
  json::Value approximation; // null for exact dense domains
  std::vector<Sm90NativeCtgSpace> spaces;
  std::vector<Sm90NativeCtgEdge> edges;
};
struct Sm90NativeCtgCache {
  analysis::CouplingCache coupling;
  SolverTiming timing;
  std::map<std::string,Sm90NativeCtgSpace> spaces;
  std::map<std::string,analysis::CouplingRelation> edges;
  std::map<std::string,analysis::ParamBinding> bindings;
  std::size_t space_hits=0,space_misses=0,edge_hits=0,edge_misses=0;
  std::map<std::string,std::shared_ptr<std::vector<std::pair<int,int>> const>> releases;
  std::size_t release_hits=0,release_misses=0;
};
Sm90NativeCtg InstantiateSm90NativeCtg(frontend::ImportedSemantics const&,
    Sm90DenseCoordinates const&,int batch,mlir::MLIRContext&,Sm90NativeCtgCache&,int workers=0);
std::shared_ptr<std::vector<std::pair<int,int>> const> Sm90NativeReleaseTable(
    analysis::CouplingRelation const& producer_to_consumer,int producers,int consumers,
    int kappa,analysis::ParamBinding const&,Sm90NativeCtgCache&);
} // namespace tilemega::solver
