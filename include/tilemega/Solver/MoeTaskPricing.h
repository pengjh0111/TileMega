// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/MoeRoutingProfile.h>
#include <tilemega/Solver/TaskModel.h>
namespace tilemega::solver {
struct MoeTaskPrice {
  TaskPriceParts parts;
  double expected_isolated_ns=0,active_probability=0,expected_rows=0;
};
// Price each observed binding row count before taking expectations. I2
// provenance remains the capacity relation; no profile substitutes an address.
MoeTaskPrice PriceMoeVirtualTask(CostModel const& cost,DerivedTaskInput const& input,
    ModelTaskSemantics const& semantic,BackendTraits const& traits,Residency residency,
    ModelDescription const& model,int chunks,analysis::DramFloor const& floor,
    MoeRoutingPoint const& routing,bool token_slot,
    analysis::ParamBinding const& coordinate);
} // namespace tilemega::solver
