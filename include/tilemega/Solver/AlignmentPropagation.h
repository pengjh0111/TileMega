// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.2 adjacent-candidate compatibility (Phase 3 stub).
#pragma once
#include <tilemega/Solver/CandidateGenerator.h>
namespace tilemega::solver {
class AlignmentPropagation { public: bool Compatible(TileCandidate const&, TileCandidate const&) const; };
}  // namespace tilemega::solver
