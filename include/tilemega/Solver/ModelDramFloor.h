// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/MoeRoutingProfile.h>
namespace tilemega::solver {
analysis::DramFloor DeriveModelDramFloor(mlir::ModuleOp module,ModelDescription const& model,
    TargetSpec const& target,std::string const& fixture,
    MoeRoutingProfile const* profile=nullptr,unsigned first_profile_layer=0);
}
