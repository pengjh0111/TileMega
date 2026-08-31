// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.4 calibrated backend cost model (Phase 4 stub).
#pragma once
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Target/TargetSpec.h>
namespace tilemega::solver {
class CostModel { public: double Evaluate(TileCandidate const&, TargetSpec const&) const; };
}  // namespace tilemega::solver
