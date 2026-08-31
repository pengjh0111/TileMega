// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.3 and §8.7 co-resident launch policy.
#pragma once
#include <tilemega/Target/TargetSpec.h>
namespace tilemega::runtime {
struct LaunchShape { int block_size = 0; int dynamic_smem_bytes = 0; int ctas_per_sm = 0; int grid_size = 0; };
inline int ResidentGridLimit(TargetSpec const& target, int ctas_per_sm) {
  return target.res.num_sms * ctas_per_sm;
}
}  // namespace tilemega::runtime
