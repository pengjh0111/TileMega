// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/AscendPlan.h>
#include <string>
namespace tilemega::codegen {
std::string LowerAscend(mlir::ModuleOp,solver::SymbolicProblem const&,solver::AscendPlan const&);
std::string LowerAscendGraph(mlir::ModuleOp,solver::SymbolicProblem const&,solver::AscendPlan const&);
} // namespace tilemega::codegen
