// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.1 torch.export ingestion (Phase 1 stub).
#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <string>
#include <vector>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Codegen/AttentionPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Solver/SolverTiming.h>
namespace tilemega::frontend {
struct ImportSummary {
  std::size_t task_spaces = 0;
  std::size_t couplings = 0;
  std::size_t stages = 0;
  std::size_t guards = 0;
  std::size_t symbolic_windows = 0;
  std::size_t fallback_windows = 0;
  /// Operators no classification rule covers. They import as one generic task
  /// space each rather than being rejected.
  std::vector<std::string> degraded;
};
struct ImportOptions {
  std::vector<GemmGranularity> gemms;
  std::vector<codegen::AttentionPlanSelection> attention;
  bool rope_tile_per_block = false;
  bool kv_tile_per_block = false;
  bool activation_tile_per_block = false;
  bool combiner_tile_per_block = false;
  bool balanced_placement = false;
  bool separate_residual_tasks = false;
};
struct ImportedSemantics {
  ExportBridge bridge;
  ModelPlan plan;
  SymbolicShape symbolic;
  LiftOptions lift_options;
  LiftedModel lifted;
};

class TorchExportImporter {
 public:
  ImportedSemantics ImportSemantics(std::string const& path,ModelPlan const& plan,
      mlir::MLIRContext& context) const;
  mlir::OwningOpRef<mlir::ModuleOp> InstantiateForGranularity(
      ImportedSemantics const& imported,mlir::MLIRContext& context,
      ImportOptions const& options,analysis::CouplingCache* cache=nullptr,
      ImportSummary* summary=nullptr,solver::SolverTiming* timing=nullptr) const;
  mlir::OwningOpRef<mlir::ModuleOp> Import(
      std::string const& stable_json_path, mlir::MLIRContext& context,
      ImportSummary* summary = nullptr,
      ImportOptions const& options = {}) const;
  mlir::OwningOpRef<mlir::ModuleOp> ImportPlan(
      std::string const& stable_json_path, ModelPlan const& plan,
      mlir::MLIRContext& context, ImportSummary* summary = nullptr,
      ImportOptions const& options = {}) const;
};
}  // namespace tilemega::frontend
