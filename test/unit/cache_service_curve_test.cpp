// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CacheServiceCurve.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <gmpxx.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() try {
  using namespace tilemega::solver;
  tilemega::analysis::IslContext context;
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
  reject([&] { synthetic.MissIntervals({"-1","0"},1,16,8,2); });
  reject([&] { synthetic.MissIntervals({"1/0","1"},1,16,8,2); });
  reject([&] { synthetic.MissIntervals({"0","1"},2,1,8,2); });
  int symbolic_checks=0;
  for (auto const& footprint:{std::array<std::string,2>{"0","1"},
                              std::array<std::string,2>{"64","-1"},
                              std::array<std::string,2>{"12","0"}}) {
    auto intervals=synthetic.MissIntervals(footprint,0,64,8,2);
    long next=0;
    for (auto const& interval:intervals) {
      if (interval.begin!=next) throw std::runtime_error("cache partition gap");
      for (long s=interval.begin;s<=interval.end;++s) {
        mpq_class value=mpq_class(interval.coefficients[0])+s*mpq_class(interval.coefficients[1]);
        double bytes=std::stod(footprint[0])+s*std::stod(footprint[1]);
        if (value.get_d()!=1.0-synthetic.HitFraction(bytes,8,2))
          throw std::runtime_error("symbolic cache interpolation mismatch");
        ++symbolic_checks;
      }
      next=interval.end+1;
    }
    if (next!=65) throw std::runtime_error("incomplete cache partition");
  }
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
  std::cout << "CACHE_SERVICE knots=" << knots << " bits_equal=1 errors=" << rejects
            << " symbolic_checks=" << symbolic_checks << " reference_delta=" << context.ReferenceCount() << '\n';
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
