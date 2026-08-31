// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.2 candidate enumeration (Phase 3 stub).
#pragma once
#include <vector>
namespace tilemega::solver {
struct TileCandidate { int m = 0; int n = 0; int k = 0; int stages = 0; };
class CandidateGenerator { public: std::vector<TileCandidate> Generate() const; };
}  // namespace tilemega::solver
