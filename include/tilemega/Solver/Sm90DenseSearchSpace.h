// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Target/TargetSpec.h>
#include <vector>

namespace tilemega::solver {

// Rule-pruned native space of prompt §4 L0-2. Legal() is the buildability
// check for any coordinate (including fixed controls); the candidate
// generators enumerate only the rule-conformant subset searched by the solver.
struct Sm90DenseSearchSpace {
  std::array<std::array<int,2>,5> shapes;
  int batch,head_dim,queries;
  TargetSpec target;
  int hkv=8;

  bool Legal(Sm90DenseCoordinates const&) const;
  int Grid(int residency) const {return target.res.num_sms*residency;}
  std::vector<Sm90DenseGemm> FamilyCandidates(unsigned family,int residency) const;
  std::vector<Sm90DenseAttention> AttentionCandidates(int residency) const;
  // The structural coordinates derived from the per-family choices.
  static void Structure(Sm90DenseCoordinates&);
};

}  // namespace tilemega::solver
