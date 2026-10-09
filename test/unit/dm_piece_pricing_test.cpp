// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PiecePricing.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace tilemega::tests::dm_piece_pricing_test {
int TestDmPiecePricing(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  auto close=[](double a,double b){assert(std::abs(a-b)<=1e-9*std::max({1.,std::abs(a),std::abs(b)}));};
  SemanticOp semantic;semantic.name="pricing_contract";semantic.exact_task_access=true;
  semantic.dtype=ScalarType::kBF16;
  semantic.domain={{"q",f(3)},{"r",f(4),f(0),IteratorType::kReduction}};
  semantic.result={"output",{{"q",f(3)}}};semantic.result_map.results={IndexResult::Dim("q")};
  semantic.task_space=semantic.result;semantic.task_map=semantic.result_map;
  TensorSpace values{"values",{{"r",f(4)}}};
  semantic.operands={{{},values,{{IndexResult::Dim("r")}}, {}}};
  auto graph=Instantiate({{semantic}},Granularity{}.Tile(semantic.name,"q",f(1)));
  solver::DerivedTaskInput input;input.task=graph.nodes.front();
  input.work=DeriveTaskWork(semantic,input.task,{});input.cost_coordinates={"q"};
  input.scalar_flow=codegen::ScalarTaskDataflow(codegen::TaskKind::kGemmCombine);
  // This synthetic cost-input contract deliberately varies arithmetic while
  // retaining identical memory counts. It tests pricing, not L-sem lowering.
  input.arithmetic.flops_per_output_element.numerator=QuasiPolynomial::FromIslText(
      "{ [q] -> 1 : q=0; [q] -> 4096 : q=1; [q] -> 3 : q=2 }");
  input.arithmetic.transcendental_per_output_element.numerator=QuasiPolynomial::FromIslText(
      "{ [q] -> 0 : 0<=q<3 }");
  input.arithmetic.runtime_implemented=true;
  solver::ModelDescription model;model.dm=true;model.dtype=solver::ScalarType::kBF16;
  model.dims={1,0,1};
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  solver::BackendTraits traits;traits.threads=128;traits.shape_legal=true;
  std::vector<ParamBinding> points(3);
  for(long q=0;q<3;++q)points[q].Bind("q",q);
  for(bool regime_a:{false,true}) {
    solver::CostModelOptions options;options.regime_a=regime_a;
    for(auto& disabled:options.disabled_lanes)disabled=true;
    options.disabled_lanes[solver::ResourceVector::kCudaCore]=false;
    solver::CostModel cost(target,model.dtype,options);
    auto prices=solver::PriceTaskInstances(cost,input,traits,{1},model,1,points);
    double sum=0,maximum=0;
    for(unsigned q=0;q<3;++q) {
      auto direct=cost.TaskInstanceNs(input,traits,{1},model,1,points[q],1);
      close(prices[q],direct);sum+=direct;maximum=std::max(maximum,direct);
    }
    assert(prices[1]>prices[0] && prices[1]>prices[2]);
    // Three tasks fit in one wave on the measured target.
    assert(target.res.num_sms>=3);
    close(cost.TaskCostNs(input,traits,{1},model,1),maximum);
    if(regime_a) {
      solver::ModelTaskSemantics op;op.op=semantic;
      solver::PiecePriceCache cache;
      auto pieces=solver::PriceBoundaryPieces(cost,input,op,traits,{1},model,1,&cache);
      assert(pieces.coordinate_varying && pieces.pieces.size()==3);
      close(pieces.total_isolated_ns,sum);
      auto again=solver::PriceBoundaryPieces(cost,input,op,traits,{1},model,1,&cache);
      close(again.total_isolated_ns,sum);assert(cache.hits==1 && cache.misses==1);
      auto side=input;side.physical_write_bytes=side.work.write_elements.Scale(6);
      for(auto const& point:points) {
        auto base=cost.PriceParts(input,traits,{1},model,1,point,1);
        auto extra=cost.PriceParts(side,traits,{1},model,1,point,1);
        // BF16 main output plus an FP32 side store changes bytes, not output FLOPs.
        close(base.compute_ns,extra.compute_ns);
      }
      (void)solver::PriceBoundaryPieces(cost,side,op,traits,{1},model,1,&cache);
      assert(cache.misses==2);
      for(unsigned change=0;change<4;++change) {
        auto altered=input;
        if(change==0)altered.arithmetic.flops_per_output_element.numerator=
            altered.arithmetic.flops_per_output_element.numerator.Scale(2);
        if(change==1)altered.arithmetic.flops_per_output_element.denominator=2;
        if(change==2)altered.arithmetic.transcendental_per_output_element.numerator=
            altered.arithmetic.transcendental_per_output_element.numerator.Add(
                QuasiPolynomial::FromIslText("{ [q] -> 16 : 0<=q<3 }"));
        if(change==3)altered.arithmetic.transcendental_per_output_element.denominator=2;
        auto misses=cache.misses;
        auto cached=solver::PriceBoundaryPieces(cost,altered,op,traits,{1},model,1,&cache);
        auto independent=solver::PriceBoundaryPieces(cost,altered,op,traits,{1},model,1);
        assert(cache.misses==misses+1);
        close(cached.total_isolated_ns,independent.total_isolated_ns);
        assert(cached.pieces.size()==independent.pieces.size());
        for(unsigned i=0;i<cached.pieces.size();++i)
          close(cached.pieces[i].parts.compute_ns,independent.pieces[i].parts.compute_ns);
        auto hits=cache.hits;
        (void)solver::PriceBoundaryPieces(cost,altered,op,traits,{1},model,1,&cache);
        assert(cache.hits==hits+1);
      }

    }
  }
  std::cout<<"DM pricing: nonuniform arithmetic, instance/stage/piece caches and typed side-output independence PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_piece_pricing_test
