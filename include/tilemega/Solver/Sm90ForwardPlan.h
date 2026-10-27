// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Target/TargetSpec.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>
#include <vector>

namespace tilemega::solver {

// The DNN forward plan a native program is built from (prompt §10 N-1). The
// native bodies read 64-channel blocks, so the input conversion pads the
// image's channels to 64 instead of the legacy eight.
frontend::ModelPlan BuildSm90ForwardPlan(std::string const& export_path,int batch,bool fuse);

// The body each stage of that plan runs, and the shapes the forward cost
// model reads. Both come from the plan alone; nothing is fitted or guessed.
Sm90ForwardBody Sm90ForwardBodyOf(frontend::ModelPlan const&,std::size_t stage);
std::vector<Sm90ForwardShape> Sm90ForwardShapesFromPlan(frontend::ModelPlan const&);

// The plan plus its coordinates as one module the forward lowering consumes:
// the model plan with its weights repacked into the native tile pages, and
// the tmexec coordinates, batch and arena budget.
mlir::OwningOpRef<mlir::ModuleOp> MaterializeSm90Forward(frontend::ModelPlan const&,
    Sm90ForwardCoordinates const&,int batch,TargetSpec const&,mlir::MLIRContext&,
    std::string const& origin);

}  // namespace tilemega::solver
