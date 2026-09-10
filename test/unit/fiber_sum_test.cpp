// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <stdexcept>

int main() try {
  using namespace tilemega::analysis;
  IslContext isl;
  int checks=0;
  auto map=CouplingRelation::FromIslText("[S,P] -> { [c] -> [p] : S>0 and P>=0 and 0<=c<S and p=c+P }");
  auto weight=QuasiPolynomial::FromIslText("[S] -> { [p] -> p*p+S : p>=0 }");
  auto pulled=weight.SumAlong(map);
  auto scalar=QuasiPolynomial::FromIslText("[S] -> { S }").SumAlong(map);
  auto fanin=CouplingRelation::FromIslText("[S] -> { [c] -> [p] : 0<=c<S and 0<=p<=c }");
  auto sum=QuasiPolynomial::FromIslText("{ [p] -> p+1 : p>=0 }").SumAlong(fanin);
  for (long s:{1L,4L,16L}) for (long past:{0L,3L,9L}) for (long c=0;c<s;++c) {
    ParamBinding theta; theta.Bind("S",s).Bind("P",past);
    ParamBinding point; point.Bind("c",c);
    if (pulled.BindCoordinates(point).Eval(theta)!=(c+past)*(c+past)+s ||
        scalar.BindCoordinates(point).Eval(theta)!=s ||
        sum.BindCoordinates(point).Eval(theta)!=(c+1)*(c+2)/2)
      throw std::runtime_error("exact fiber sum differs from enumerated polynomial weights");
    ++checks;
  }
  int errors=0;
  for (auto text:{"{ [c] -> [p,q] : c=0 and p=0 and q=0 }","{ [c] -> [p,q,r] : c=0 and p=0 and q=0 and r=0 }"}) {
    int refs=isl.ReferenceCount(); bool caught=false;
    try { (void)weight.SumAlong(CouplingRelation::FromIslText(text)); }
    catch (std::invalid_argument const&) { caught=true; }
    if (!caught || refs!=isl.ReferenceCount()) throw std::runtime_error("fiber rank rejection failed or leaked");
    ++errors;
  }
  if (isl.ReferenceCount()) throw std::runtime_error("fiber sums retained references");
  std::cout << "FIBER_SUM checks=" << checks << " errors=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
