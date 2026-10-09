// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/OpArithmetic.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace tilemega::tests::task_arithmetic_domain_test {
int TestTaskArithmeticDomain(int,char**) {
  using namespace analysis;
  IslContext context;
  auto domain=CouplingRelation::FromIslText(
      "[T] -> { [m,n] -> [m,n] : T>=1 and 0<=m<ceil(T/4) and 0<=n<2 }");
  ArithmeticInputs inputs;inputs.dtype=ScalarType::kBF16;inputs.width=32;
  inputs.reduction=QuasiPolynomial::FromIslText(
      "[T] -> { [m,n] -> 4 : T>=1 and 0<=m and 4m+4<=T and 0<=n<2; "
      "[m,n] -> T-4m : 0<=m and 4m<T<4m+4 and 0<=n<2 }");
  auto sum=InstantiateTaskArithmetic("sum",inputs,domain);
  auto gemm=InstantiateTaskArithmetic("gemm",inputs,domain);
  auto norm=InstantiateTaskArithmetic("rmsnorm",inputs,domain);
  for(int tokens:{1,2,4,5,17,65}) {
    ParamBinding theta;theta.Bind("T",tokens);
    long additions=0,dots=0;
    for(int m=0;m<(tokens+3)/4;++m)for(int n=0;n<2;++n) {
      ParamBinding coordinates;coordinates.Bind("m",m);coordinates.Bind("n",n);
      long rows=std::min(4,tokens-4*m);
      auto evaluate=[&](ArithmeticRatio const& x) {
        return double(x.numerator.BindCoordinates(coordinates).Eval(theta))/x.denominator;
      };
      assert(evaluate(sum.flops_per_output_element)==rows-1);
      assert(evaluate(gemm.flops_per_output_element)==2*rows && gemm.flops_use_mma);
      assert(evaluate(norm.flops_per_output_element)==129.0/32);
      assert(evaluate(norm.transcendental_per_output_element)==1.0/32);
      additions+=rows-1;dots+=2*rows;
    }
    assert(sum.flops_per_output_element.numerator.SumDomain().Eval(theta)==additions);
    assert(gemm.flops_per_output_element.numerator.SumDomain().Eval(theta)==dots);
    assert(dots==4*tokens);
  }
  ArithmeticInputs scalar;scalar.reduction=QuasiPolynomial::Constant(3);
  assert(InstantiateArithmetic("sum",scalar).flops_per_output_element.Eval({})==2);
  auto five=QuasiPolynomial::Constant(5);
  assert(five.SumAlong(CouplingRelation::FromIslText("{ [] -> [] }")).Eval({})==5);
  auto broadcast=five.SumAlong(CouplingRelation::FromIslText("[B] -> { [m] -> [] : B>=1 and 0<=m<B }"));
  assert(broadcast.SemanticallyEqual(QuasiPolynomial::FromIslText(
      "[B] -> { [m] -> 5 : B>=1 and 0<=m<B }"),{}));
  assert(broadcast.SumDomain().Eval(ParamBinding{}.Bind("B",7))==35);
  for(char const* invalid:{"{ [m] -> [n] : 0<=m<3 and n=2-m }",
      "{ [m] -> [n] : 0<=m<3 and 0<=n<3 }","{ [m] -> [n,k] : n=m and k=0 }"}) {
    bool rejected=false;
    try { InstantiateTaskArithmetic("sum",scalar,CouplingRelation::FromIslText(invalid)); }
    catch(std::invalid_argument const&) { rejected=true; }
    assert(rejected);
  }
  std::cout<<"Task arithmetic: symbolic tails, constants, width ratios, legacy scalar and identity guards PASS\n";
  return 0;
}
} // namespace tilemega::tests::task_arithmetic_domain_test
