// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CacheServiceCurve.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tilemega::solver {
CacheServiceCurve::CacheServiceCurve(std::vector<double> const& bytes,
                                   std::vector<double> const& gbps) {
  if (bytes.empty() || bytes.size()!=gbps.size())
    throw std::invalid_argument("measured cache service curve: not_calibrated");
  for (std::size_t i=0;i<bytes.size();++i) {
    if (!std::isfinite(bytes[i]) || !std::isfinite(gbps[i]) || bytes[i]<=0 || gbps[i]<=0 ||
        (i && bytes[i]<=bytes[i-1]))
      throw std::invalid_argument("invalid measured cache service knot");
    knots_.push_back({bytes[i],1.0/gbps[i]});
  }
}
double CacheServiceCurve::ServiceNsPerByte(double bytes) const {
  if (!std::isfinite(bytes) || bytes<0) throw std::invalid_argument("invalid cache working set");
  if (bytes<=knots_.front().bytes) return knots_.front().ns_per_byte;
  if (bytes>=knots_.back().bytes) return knots_.back().ns_per_byte;
  auto upper=std::lower_bound(knots_.begin(),knots_.end(),bytes,
      [](auto const& knot,double value) { return knot.bytes<value; });
  if (upper->bytes==bytes) return upper->ns_per_byte;
  auto const& lower=*(upper-1);
  double fraction=(bytes-lower.bytes)/(upper->bytes-lower.bytes);
  return lower.ns_per_byte+fraction*(upper->ns_per_byte-lower.ns_per_byte);
}
double CacheServiceCurve::HitFraction(double bytes,double l2_gbps,double dram_gbps) const {
  if (!(l2_gbps>dram_gbps && dram_gbps>0) || !std::isfinite(l2_gbps) || !std::isfinite(dram_gbps))
    throw std::invalid_argument("cache service endpoints: not_calibrated");
  double hit_time=1.0/l2_gbps,miss_time=1.0/dram_gbps;
  // Effective hit/miss mixture matching the measured service time. The
  // physical [0,1] constraint clips noisy knots beyond either endpoint;
  // those clamp crossings are additional affine segment boundaries.
  double miss=(ServiceNsPerByte(bytes)-hit_time)/(miss_time-hit_time);
  return 1.0-std::clamp(miss,0.0,1.0);
}
}  // namespace tilemega::solver
