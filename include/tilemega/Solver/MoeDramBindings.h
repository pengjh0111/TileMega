// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/MoeRoutingProfile.h>

namespace tilemega::solver {
// Frontend invariants: each token has K distinct assignments, dispatch is a
// bijection onto (token,rank), and active expert rows share one binding table.
// Internal capacity envelopes establish provenance, never physical images.
void BindMoeDramInputs(mlir::ModuleOp,ModelDescription const&,
    analysis::DramFloorOptions&,MoeRoutingProfile const* profile=nullptr,
    unsigned first_profile_layer=0);
}
