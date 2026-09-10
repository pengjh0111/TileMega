// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include <string>
#include <vector>
#include <tilemega/Analysis/QuasiPolynomial.h>

namespace tilemega::solver {
struct LaneOrderRegion {
  long begin, end;  // inclusive integer parameter domain
  int sign;        // sign(first - second), including exact ties
};
// Exact rational coefficients in ascending powers, after floor partition.
// Quadratic roots are isolated by integer square root, never by sampling.
std::vector<LaneOrderRegion> OrderQuadraticLanes(
    std::array<std::string,3> const& first,
    std::array<std::string,3> const& second, long begin, long end);
struct QuadraticEnvelopePiece {
  long begin, end;
  std::size_t choice;
  std::array<std::string,3> coefficients;
};
// Values outside a QP's explicit support are zero, as in ISL arithmetic.
// Feasibility domains are separate from these costs, never encoded as zero.
std::vector<QuadraticEnvelopePiece> QuadraticEnvelope(
    std::vector<analysis::QuasiPolynomial> const& costs,std::string const& parameter,
    long begin,long end,bool maximum);
}  // namespace tilemega::solver
