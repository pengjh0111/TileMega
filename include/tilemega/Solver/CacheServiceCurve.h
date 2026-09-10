// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <vector>

namespace tilemega::solver {
// The measured knots are converted to time/byte before interpolation:
// interpolating GB/s then taking a reciprocal is rational, not affine.
struct CacheServiceKnot {
  double bytes;
  double ns_per_byte;
};
class CacheServiceCurve {
 public:
  CacheServiceCurve(std::vector<double> const& bytes,std::vector<double> const& gbps);
  double ServiceNsPerByte(double bytes) const;
  double HitFraction(double bytes,double l2_gbps,double dram_gbps) const;
  std::vector<CacheServiceKnot> const& knots() const { return knots_; }
 private:
  std::vector<CacheServiceKnot> knots_;
};
}  // namespace tilemega::solver
