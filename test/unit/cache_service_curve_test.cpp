// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CacheServiceCurve.h>
#include <tilemega/Solver/CostModel.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() try {
  using namespace tilemega::solver;
  CacheServiceCurve synthetic({8,16},{8,2});
  if (synthetic.ServiceNsPerByte(8)!=0.125 || synthetic.ServiceNsPerByte(16)!=0.5 ||
      synthetic.ServiceNsPerByte(12)!=0.3125 || synthetic.HitFraction(12,8,2)!=0.5 ||
      synthetic.HitFraction(0,8,2)!=1 || synthetic.HitFraction(32,8,2)!=0)
    throw std::runtime_error("affine service interpolation or physical clamp failed");
  int rejects=0;
  auto reject=[&](auto&& action) {
    bool failed=false;
    try { action(); } catch (std::invalid_argument const&) { failed=true; }
    if (!failed) throw std::runtime_error("invalid curve accepted");
    ++rejects;
  };
  reject([] { CacheServiceCurve({},{}); });
  reject([] { CacheServiceCurve({1,2},{8}); });
  reject([] { CacheServiceCurve({2,1},{8,4}); });
  reject([] { CacheServiceCurve({1,1},{8,4}); });
  reject([] { CacheServiceCurve({1},{0}); });
  reject([] { CacheServiceCurve({1},{std::numeric_limits<double>::quiet_NaN()}); });
  reject([&] { synthetic.ServiceNsPerByte(-1); });
  reject([&] { synthetic.HitFraction(8,2,8); });
  auto target=tilemega::TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json");
  int knots=0;
  for (auto dtype:{ScalarType::kBF16,ScalarType::kF32}) {
    auto const& c=target.CalibrationFor(dtype==ScalarType::kBF16 ? "bf16" : "f32");
    CacheServiceCurve curve(c.l2_curve_bytes,c.l2_curve_gbps);
    for (std::size_t i=0;i<c.l2_curve_bytes.size();++i) {
      double expected=1.0/c.l2_curve_gbps[i],actual=curve.ServiceNsPerByte(c.l2_curve_bytes[i]);
      if (std::memcmp(&expected,&actual,sizeof(double))) throw std::runtime_error("measured knot changed");
      ++knots;
    }
    CostModelOptions options; options.measured_cache_curve=true;
    CostModel cost(target,dtype,options);
    for (double bytes:c.l2_curve_bytes) {
      double expected=curve.HitFraction(bytes,c.l2_gbps,c.dram_gbps),actual=cost.CacheHitProbability(bytes);
      if (std::memcmp(&expected,&actual,sizeof(double))) throw std::runtime_error("cost model did not consume service curve");
    }
  }
  std::cout << "CACHE_SERVICE knots=" << knots << " bits_equal=1 errors=" << rejects << '\n';
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
