// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/LaneIntersections.h>
#include <gmpxx.h>
#include <algorithm>
#include <tilemega/Analysis/ISLContext.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
std::vector<QuadraticEnvelopePiece> QuadraticEnvelope(
    std::vector<analysis::QuasiPolynomial> const& costs,std::string const& parameter,
    long begin,long end,bool maximum) {
  analysis::IslReferenceAudit audit(__func__);
  if (costs.empty()) throw std::invalid_argument("empty polynomial envelope");
  std::vector<std::vector<analysis::QuasiPolynomial::PolynomialInterval>> functions;
  std::set<long> cuts{begin};
  for (auto const& cost:costs) {
    auto intervals=cost.QuadraticIntervals(parameter,begin,end);
    for (auto const& interval:intervals) {
      cuts.insert(interval.begin); cuts.insert(interval.end+1);
    }
    functions.push_back(std::move(intervals));
  }
  cuts.insert(end+1);
  auto coefficients=[&](std::size_t choice,long point) {
    for (auto const& interval:functions[choice])
      if (interval.begin<=point && point<=interval.end) return interval.coefficients;
    return std::array<std::string,3>{"0","0","0"};
  };
  std::vector<long> domains(cuts.begin(),cuts.end());
  for (std::size_t domain=1;domain<domains.size();++domain) {
    long first=domains[domain-1],last=domains[domain]-1;
    for (std::size_t i=0;i<costs.size();++i) for (std::size_t j=i+1;j<costs.size();++j)
      for (auto const& order:OrderQuadraticLanes(coefficients(i,first),coefficients(j,first),first,last)) {
        cuts.insert(order.begin); cuts.insert(order.end+1);
      }
  }
  std::vector<QuadraticEnvelopePiece> result;
  for (auto current=cuts.begin(),next=std::next(current);next!=cuts.end();++current,++next) {
    std::size_t winner=0;
    auto best=coefficients(0,*current);
    for (std::size_t choice=1;choice<costs.size();++choice) {
      auto candidate=coefficients(choice,*current);
      auto order=OrderQuadraticLanes(candidate,best,*current,*next-1);
      if (order.size()!=1) throw std::logic_error("lane crossing absent from envelope partition");
      if (maximum ? order[0].sign>0 : order[0].sign<0) {
        winner=choice; best=std::move(candidate);
      }
    }
    if (!result.empty() && result.back().choice==winner && result.back().coefficients==best)
      result.back().end=*next-1;
    else result.push_back({*current,*next-1,winner,std::move(best)});
  }
  return result;
}

std::vector<LaneOrderRegion> OrderQuadraticLanes(
    std::array<std::string,3> const& first,
    std::array<std::string,3> const& second, long begin, long end) {
  if (begin>end || end==std::numeric_limits<long>::max())
    throw std::invalid_argument("invalid closed integer lane domain");
  auto rational=[](std::string const& text) {
    mpq_class value;
    if (mpq_set_str(value.get_mpq_t(),text.c_str(),10)!=0 || value.get_den()==0)
      throw std::invalid_argument("invalid rational lane coefficient");
    value.canonicalize();
    return value;
  };
  std::array<mpq_class,3> difference;
  mpz_class denominator=1;
  for (int i=0;i<3;++i) {
    difference[i]=rational(first[i])-rational(second[i]);
    mpz_lcm(denominator.get_mpz_t(),denominator.get_mpz_t(),difference[i].get_den_mpz_t());
  }
  std::array<mpz_class,3> integer;
  for (int i=0;i<3;++i)
    integer[i]=difference[i].get_num()*(denominator/difference[i].get_den());
  auto [c,b,a]=integer;
  std::set<long> cuts{begin,end+1};
  auto cut=[&](mpz_class const& value) {
    if (value>=begin && value<=end) cuts.insert(value.get_si());
  };
  auto floor_div=[](mpz_class const& numerator,mpz_class const& denominator) {
    mpz_class result;
    mpz_fdiv_q(result.get_mpz_t(),numerator.get_mpz_t(),denominator.get_mpz_t());
    return result;
  };
  auto root=[&](mpz_class const& floor) { cut(floor); cut(mpz_class(floor+1)); };
  if (a==0) {
    if (b!=0) root(floor_div(mpz_class(-c),b));
  } else {
    if (a<0) { a=-a; b=-b; c=-c; }
    mpz_class discriminant=b*b-4*a*c;
    if (discriminant>=0) {
      mpz_class square_root;
      mpz_sqrt(square_root.get_mpz_t(),discriminant.get_mpz_t());
      mpz_class divisor=2*a,minus=-b-square_root;
      auto lower=floor_div(minus,divisor);
      // A non-square negative root is strictly below the integer-radical
      // expression. Only exact divisibility crosses an integer boundary.
      if (square_root*square_root!=discriminant && minus%divisor==0) --lower;
      root(lower);
      root(floor_div(mpz_class(-b+square_root),divisor));
    }
  }
  std::vector<LaneOrderRegion> regions;
  for (auto current=cuts.begin(),next=std::next(current);next!=cuts.end();++current,++next) {
    mpz_class theta=*current;
    mpz_class value=(integer[2]*theta+integer[1])*theta+integer[0];
    int sign=mpz_sgn(value.get_mpz_t());
    // All roots are partition boundaries: this sign holds on the interval.
    // This is classification after a proof, not fitted sampling.
    if (!regions.empty() && regions.back().sign==sign) regions.back().end=*next-1;
    else regions.push_back({*current,*next-1,sign});
  }
  return regions;
}
}  // namespace tilemega::solver
