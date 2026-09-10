// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Solver/LaneIntersections.h>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() try {
  using namespace tilemega::analysis;
  using tilemega::solver::OrderQuadraticLanes;
  IslContext context;
  auto quadratic=QuasiPolynomial::FromIslText("[S] -> { 1/3 + 2*S + S^2 : 1<=S<=16 }")
      .QuadraticPieces("S");
  if (quadratic.size()!=1 || quadratic[0].coefficients!=std::array<std::string,3>{"1/3","2","1"})
    throw std::runtime_error("exact polynomial extraction failed");
  int errors=0;
  auto reject=[&](auto action) {
    auto before=context.ReferenceCount();
    bool caught=false;
    try { action(); } catch (std::invalid_argument const&) { caught=true; }
    if (!caught || context.ReferenceCount()!=before) throw std::runtime_error("unaudited lane rejection");
    ++errors;
  };
  reject([&]{QuasiPolynomial::FromIslText("[S] -> { S^3 : S>0 }").QuadraticPieces("S");});
  reject([&]{QuasiPolynomial::FromIslText("[S] -> { floor(S/2) : S>0 }").QuadraticPieces("S");});
  reject([&]{QuasiPolynomial::FromIslText("[S,P] -> { S+P }").QuadraticPieces("S");});
  reject([&]{OrderQuadraticLanes({"0","0","0"},{"0","0","0"},2,1);});
  reject([&]{OrderQuadraticLanes({"0","0","0"},{"0","0","0"},1,std::numeric_limits<long>::max());});
  reject([&]{OrderQuadraticLanes({"nan","0","0"},{"0","0","0"},1,16);});
  reject([&]{OrderQuadraticLanes({"1/0","0","0"},{"0","0","0"},1,16);});
  reject([&]{QuasiPolynomial::FromIslText("[S] -> { S }").QuadraticIntervals("S",2,1);});
  reject([&]{QuasiPolynomial::FromIslText("[S] -> { S^3 }").QuadraticIntervals("S",1,16);});
  reject([&]{QuasiPolynomial::FromIslText("[S,P] -> { S+P }").QuadraticIntervals("S",1,16);});
  auto periodic=QuasiPolynomial::FromIslText("[S] -> { floor((S+3)/4) + floor(S/7) }");
  if (QuasiPolynomial::Constant(0).QuadraticIntervals("S",1,16).size()!=0)
    throw std::runtime_error("zero polynomial support unexpectedly materialized");
  auto product=QuasiPolynomial::FromIslText("[S] -> { 2*S }").Multiply(
      QuasiPolynomial::FromIslText("[S] -> { 3*S }")).ScaleRational("1/6");
  if (!product.SemanticallyEqual(QuasiPolynomial::FromIslText("[S] -> { S^2 }"),{}))
    throw std::runtime_error("exact work/rate composition failed");
  auto intervals=periodic.QuadraticIntervals("S",1,63);
  for (long s=1;s<=63;++s) {
    int hits=0;
    for (auto const& interval:intervals) if (interval.begin<=s && s<=interval.end) {
      if (interval.coefficients[1]!="0" || interval.coefficients[2]!="0" ||
          std::stol(interval.coefficients[0])!=(s+3)/4+s/7)
        throw std::runtime_error("exact floor partition changed values");
      ++hits;
    }
    if (hits!=1) throw std::runtime_error("floor interval coverage mismatch");
  }
  long checked=0;
  for (int a=-6;a<=6;++a) for (int b=-6;b<=6;++b) for (int c=-6;c<=6;++c) {
    auto regions=OrderQuadraticLanes({std::to_string(c),std::to_string(b),std::to_string(a)},
                                    {"0","0","0"},-16,16);
    long next=-16;
    for (auto const& region:regions) {
      if (region.begin!=next) throw std::runtime_error("lane partition gap");
      for (long s=region.begin;s<=region.end;++s) {
        long value=(a*s+b)*s+c;
        if (region.sign!=((value>0)-(value<0))) throw std::runtime_error("wrong exact lane order");
        ++checked;
      }
      next=region.end+1;
    }
    if (next!=17) throw std::runtime_error("lane partition incomplete");
  }
  auto close=OrderQuadraticLanes({"-100000000000000000000000000000000000001",
                                "100000000000000000000000000000000000000","0"},
                               {"0","0","0"},0,3);
  if (close.size()!=2 || close[0].end!=1 || close[0].sign!=-1 || close[1].sign!=1)
    throw std::runtime_error("near-integer root rounded across boundary");
  auto rational=OrderQuadraticLanes({"-1/3","0","1/12"},{"0","0","0"},-3,3);
  if (rational.size()!=5 || rational[1].begin!=-2 || rational[1].end!=-2 || rational[1].sign!=0)
    throw std::runtime_error("rational root tie lost");
  auto costs=std::vector<QuasiPolynomial>{
      QuasiPolynomial::FromIslText("[S] -> { S^2 }"),
      QuasiPolynomial::FromIslText("[S] -> { 5*S + 12 }"),
      QuasiPolynomial::FromIslText("[S] -> { 7*floor((S+3)/4) }")};
  long envelope_checks=0;
  for (bool maximum:{false,true}) {
    auto envelope=tilemega::solver::QuadraticEnvelope(costs,"S",1,63,maximum);
    long next=1;
    for (auto const& piece:envelope) {
      if (piece.begin!=next) throw std::runtime_error("envelope domain gap");
      for (long s=piece.begin;s<=piece.end;++s) {
        ParamBinding theta; theta.Bind("S",s);
        std::size_t best=0;
        for (std::size_t candidate=1;candidate<costs.size();++candidate) {
          auto value=costs[candidate].Eval(theta),prior=costs[best].Eval(theta);
          if (maximum ? value>prior : value<prior) best=candidate;
        }
        if (piece.choice!=best) throw std::runtime_error("symbolic envelope disagrees with point oracle");
        ++envelope_checks;
      }
      next=piece.end+1;
    }
    if (next!=64) throw std::runtime_error("incomplete envelope domain");
  }
  std::cout << "LANE_INTERSECTIONS integer_checks=" << checked << " errors=" << errors
            << " envelope_checks=" << envelope_checks << " reference_delta=" << context.ReferenceCount() << '\n';
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
