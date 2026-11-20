// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/TargetSpec.h>
#include <algorithm>
#include <stdexcept>

namespace tilemega::solver {

inline int Sm90DenseSharedBudget(TargetSpec const& target,int residency) {
  if(residency!=1 && residency!=2)
    throw std::invalid_argument("native residency must be one or two");
  auto const& r=target.res;
  // For the fixed H800 target the opt-in CTA ceiling excludes the driver
  // reservation. Each resident CTA consumes that reservation separately.
  int reserved=std::max(0,r.max_smem_per_sm-r.max_dynamic_smem_per_cta);
  return std::max(0,std::min(r.max_dynamic_smem_per_cta,
      r.max_smem_per_sm/residency-reserved));
}

}  // namespace tilemega::solver
