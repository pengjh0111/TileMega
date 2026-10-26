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
  Sm90DenseShapeTable shapes;
  int batch,head_dim,queries;
  TargetSpec target;
  int hkv=8;
  // Decision tables DT-1 and DT-3 (target `sm90_native_decisions`).
  bool weight_stream=false,layer_loop=false;
  // MoE decode (prompt §8): families 0, 1, 5, 6, 7, 4.
  bool moe=false;
  // Expected active experts of a step, 128 (1 - (1 - 8/128)^B), which sizes
  // the grouped stages' task counts (§8 solver item 3).
  double ExpectedActive() const {return 128.0*(1-std::pow(1-8.0/128,batch));}

  bool Legal(Sm90DenseCoordinates const&) const;
  int Grid(int residency) const {return target.res.num_sms*residency;}
  std::vector<Sm90DenseGemm> FamilyCandidates(unsigned family,int residency) const;
  std::vector<Sm90DenseAttention> AttentionCandidates(int residency) const;
  // The structural coordinates derived from the per-family choices.
  static void Structure(Sm90DenseCoordinates&);
};

}  // namespace tilemega::solver
