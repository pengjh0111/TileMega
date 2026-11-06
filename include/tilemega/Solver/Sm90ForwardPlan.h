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
// image's channels to 64 instead of the legacy eight. Under the F3 skeleton
// (DNN3 prompt §4) a C<=8 image keeps its eight-channel block, an identity
// adaptive pool is an alias and an NHWC flatten reorders the Linear's weight.
frontend::ModelPlan BuildSm90ForwardPlan(std::string const& export_path,int batch,bool fuse,
    Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy);

// The body each stage of that plan runs, and the shapes the forward cost
// model reads. Both come from the plan alone; nothing is fitted or guessed.
// F3 has no GEMV body: a stage with at most sixteen rows is a swap-AB GEMM.
Sm90ForwardBody Sm90ForwardBodyOf(frontend::ModelPlan const&,std::size_t stage,
    Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy);
// The operand mode F3 assigns to a stage; always normal under legacy.
Sm90ForwardMode Sm90ForwardModeOf(frontend::ModelPlan const&,std::size_t stage,
    Sm90ForwardAbi abi);
std::vector<Sm90ForwardShape> Sm90ForwardShapesFromPlan(frontend::ModelPlan const&,
    Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy);

// The plan plus its coordinates as one module the forward lowering consumes:
// the model plan with its weights repacked into the native tile pages, and
// the tmexec coordinates, batch and arena budget. Coordinate-dependent buffer
// layouts are published back to the plan so its manifest matches allocation.
mlir::OwningOpRef<mlir::ModuleOp> MaterializeSm90Forward(frontend::ModelPlan&,
    Sm90ForwardCoordinates const&,int batch,TargetSpec const&,mlir::MLIRContext&,
    std::string const& origin);

}  // namespace tilemega::solver
