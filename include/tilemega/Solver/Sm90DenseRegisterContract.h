// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/TargetSpec.h>

namespace tilemega::solver {
struct Sm90DenseRegisterContract {
  int threads=0,residency=0,granularity=0,subpartitions=0;
  int architecture_limit=0,callee_limit=0;
  bool Constrained() const {return callee_limit<architecture_limit;}
};
// A register-allocation upper bound for the native launch, not a proof of
// occupancy. Final linked allocation and runtime occupancy remain mandatory.
Sm90DenseRegisterContract NativeDenseRegisterContract(TargetSpec const&,int residency);
}  // namespace tilemega::solver
