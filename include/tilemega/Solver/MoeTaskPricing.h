// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/MoeRoutingProfile.h>
#include <tilemega/Solver/PiecePricing.h>
namespace tilemega::solver {
enum class MoeEmptyPricing { kRequireCalibration, kCapacitySurrogate };
struct MoeTaskPrice {
  bool inferred_empty_cost=false;
  TaskPriceParts parts;
  double expected_isolated_ns=0,active_probability=0,expected_rows=0;
};
// Price each observed binding row count before taking expectations. I2
// provenance remains the capacity relation; no profile substitutes an address.
MoeTaskPrice PriceMoeVirtualTask(CostModel const& cost,DerivedTaskInput const& input,
    ModelTaskSemantics const& semantic,BackendTraits const& traits,Residency residency,
    ModelDescription const& model,int chunks,analysis::DramFloor const& floor,
    MoeRoutingPoint const& routing,bool token_slot,
    analysis::ParamBinding const& coordinate,
    MoeEmptyPricing empty_pricing=MoeEmptyPricing::kRequireCalibration);
PiecePrices PriceMoeBoundaryPieces(CostModel const& cost,DerivedTaskInput const& input,
    ModelTaskSemantics const& semantic,BackendTraits const& traits,Residency residency,
    ModelDescription const& model,int chunks,analysis::DramFloor const& floor,
    MoeRoutingPoint const& routing,bool token_slot,PiecePrices const& capacity,
    MoeEmptyPricing empty_pricing=MoeEmptyPricing::kRequireCalibration);
} // namespace tilemega::solver
