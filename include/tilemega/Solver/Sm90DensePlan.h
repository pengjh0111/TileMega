// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Solver/Sm90DenseWorkload.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Target/TargetSpec.h>

namespace tilemega::solver {
unsigned Sm90DenseFamily(frontend::ModelPlan const&,frontend::PlanGemm const&);
std::array<std::array<int,2>,5> Sm90DenseShapes(frontend::ModelPlan const&);
Sm90DenseWorkload Sm90DenseWorkloadFromPlan(frontend::ModelPlan const&,int batch);
frontend::ImportedSemantics ImportSm90Dense(std::string const& export_path,
    Sm90DenseCoordinates const&,mlir::MLIRContext&);
mlir::OwningOpRef<mlir::ModuleOp> MaterializeSm90Dense(
    frontend::ImportedSemantics const&,Sm90DenseCoordinates const&,int batch,
    TargetSpec const&,mlir::MLIRContext&,std::string const& origin);
}  // namespace tilemega::solver
