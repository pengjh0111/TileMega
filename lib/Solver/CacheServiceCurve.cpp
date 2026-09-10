// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CacheServiceCurve.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <gmpxx.h>
#include <set>
#include <tilemega/Analysis/ISLContext.h>

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

std::vector<analysis::QuasiPolynomial::PolynomialInterval> CacheServiceCurve::MissIntervals(
    std::array<std::string,2> const& footprint,
    long begin,long end,double l2_gbps,double dram_gbps) const {
  analysis::IslReferenceAudit audit(__func__);
  // Reuse endpoint validation without using this numerical value as a sample.
  (void)HitFraction(0,l2_gbps,dram_gbps);
  auto rational=[](std::string const& text) {
    mpq_class value;
    if (mpq_set_str(value.get_mpq_t(),text.c_str(),10)!=0 || value.get_den()==0)
      throw std::invalid_argument("invalid footprint coefficient");
    value.canonicalize(); return value;
  };
  auto base=rational(footprint[0]),slope=rational(footprint[1]);
  if (base+begin*slope<0 || base+end*slope<0)
    throw std::invalid_argument("negative symbolic cache footprint");
  std::array<std::string,3> work{base.get_str(),slope.get_str(),"0"};
  // This validates the interval even for a one-knot curve.
  (void)OrderQuadraticLanes(work,{"0","0","0"},begin,end);
  std::set<long> cuts{begin,end+1};
  for (auto const& knot:knots_)
    for (auto const& region:OrderQuadraticLanes(work,{mpq_class(knot.bytes).get_str(),"0","0"},begin,end)) {
      cuts.insert(region.begin); cuts.insert(region.end+1);
    }
  mpq_class hit(1.0/l2_gbps),miss(1.0/dram_gbps),range=miss-hit;
  std::vector<analysis::QuasiPolynomial::PolynomialInterval> result;
  for (auto current=cuts.begin(),next=std::next(current);next!=cuts.end();++current,++next) {
    mpq_class bytes=base+*current*slope,constant,linear=0;
    if (bytes<=mpq_class(knots_.front().bytes)) constant=mpq_class(knots_.front().ns_per_byte);
    else if (bytes>=mpq_class(knots_.back().bytes)) constant=mpq_class(knots_.back().ns_per_byte);
    else {
      std::size_t upper=1;
      while (mpq_class(knots_[upper].bytes)<bytes) ++upper;
      auto const& low=knots_[upper-1]; auto const& high=knots_[upper];
      mpq_class rate=(mpq_class(high.ns_per_byte)-mpq_class(low.ns_per_byte))/
                    (mpq_class(high.bytes)-mpq_class(low.bytes));
      constant=mpq_class(low.ns_per_byte)+(base-mpq_class(low.bytes))*rate;
      linear=slope*rate;
    }
    constant=(constant-hit)/range; linear/=range;
    std::array<std::string,3> expression{constant.get_str(),linear.get_str(),"0"};
    std::set<long> clamp_cuts{*current,*next};
    for (auto const& endpoint:{std::string("0"),std::string("1")})
      for (auto const& region:OrderQuadraticLanes(expression,{endpoint,"0","0"},*current,*next-1)) {
        clamp_cuts.insert(region.begin); clamp_cuts.insert(region.end+1);
      }
    for (auto first=clamp_cuts.begin(),last=std::next(first);last!=clamp_cuts.end();++first,++last) {
      mpq_class value=constant+*first*linear;
      auto coefficients=value<0 ? std::array<std::string,3>{"0","0","0"} :
                        value>1 ? std::array<std::string,3>{"1","0","0"} : expression;
      if (!result.empty() && result.back().coefficients==coefficients) result.back().end=*last-1;
      else result.push_back({*first,*last-1,std::move(coefficients)});
    }
  }
  return result;
}
}  // namespace tilemega::solver
