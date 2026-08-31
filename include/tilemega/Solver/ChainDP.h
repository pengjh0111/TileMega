// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.3 chain dynamic programming (Phase 3 stub).
#pragma once
#include <vector>
#include <tilemega/Solver/CandidateGenerator.h>
namespace tilemega::solver {
class ChainDP { public: std::vector<TileCandidate> Solve(std::vector<std::vector<TileCandidate>> const&) const; };
}  // namespace tilemega::solver
