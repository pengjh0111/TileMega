// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Target/TargetSpec.h>
#include <vector>

namespace tilemega::solver {

struct Sm90DenseSearchSpace {
  std::array<std::array<int,2>,5> shapes;
  int batch,head_dim,queries;
  TargetSpec target;

  bool Legal(Sm90DenseCoordinates const&) const;
  std::vector<Sm90DenseCoordinates> Seeds() const;
  Sm90DenseCoordinates MaxPreissue(Sm90DenseCoordinates const&,bool l2) const;
  std::vector<Sm90DenseCoordinates> Neighbours(Sm90DenseCoordinates const&) const;
};

}  // namespace tilemega::solver
