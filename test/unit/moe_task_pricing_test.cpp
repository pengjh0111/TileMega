// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeTaskPricing.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <cmath>
#include <iostream>
namespace tilemega::tests::moe_task_pricing_test {
int TestMoeTaskPricing(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  auto close=[](double a,double b){assert(std::abs(a-b)<=1e-9*std::max({1.,std::abs(a),std::abs(b)}));};
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  // Synthetic coefficients isolate the probability/traffic contract. They are
  // unit-test inputs and are never serialized as a device calibration.
  auto& calibration=target.CalibrationFor("bf16").task_body;
  calibration.serving["gemm_expert_indirect"]={11,10,0.001,0,1};
  calibration.serving["gemm_expert_indirect_empty"]={7,0,0,0,1};
  solver::CostModelOptions options;options.regime_a=true;options.stage_latency=false;
  options.physical_fixed=false;
  solver::CostModel cost(target,solver::ScalarType::kBF16,options);
  unsigned checked=0,empty_checked=0;
  for(bool slot:{false,true})for(unsigned bm:{1,16,32,64,128}) {
    if(slot!=(bm==1))continue;
    solver::MoeRoutingPoint routing;routing.tokens=19;routing.experts=4;
    routing.top_k=2;routing.windows=2;
    auto capacity=slot?routing.SlotCapacity():routing.GroupCapacity(bm);
    std::vector<std::vector<unsigned>> windows;
    for(auto counts:{std::vector<unsigned>{19,19,0,0},std::vector<unsigned>{10,10,9,9}}) {
      std::vector<unsigned> rows;
      if(slot)rows.assign(38,1);
      else for(auto count:counts)while(count){auto n=std::min(bm,count);rows.push_back(n);count-=n;}
      rows.resize(capacity);windows.push_back(std::move(rows));
    }
    routing.virtual_row_histograms[bm].resize(capacity);
    for(auto const& window:windows)for(unsigned v=0;v<capacity;++v)
      if(window[v])++routing.virtual_row_histograms[bm][v][window[v]];
    for(int tm:{16,32,64,128}) {
      SemanticOp op;op.name="expert";op.kind=OperatorKind::kMatmul;
      op.dtype=analysis::ScalarType::kBF16;op.arithmetic="gemm";op.exact_task_access=true;
      op.domain={{"v",f(capacity)},{"row",f(bm)},{"n",f(19)},
          {"k",f(5),f(0),IteratorType::kReduction}};
      auto& vd=op.domain.front();vd.runtime=true;vd.capacity=f(capacity);
      vd.binding_source="bindings";vd.binding_requirement="prefix_sum";
      op.task_space={"owners",{{"v",f(capacity)},{"row",f(bm)},{"n",f(19)}}};
      op.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("n")};
      op.result={"partial",{{"t",f(19)},{"rank",f(2)},{"n",f(19)}}};
      op.result_map.results={IndexResult::DataDependent("rows",{"v","row"}),
          IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("n")};
      TensorSpace a{"hidden",{{"t",f(19)},{"k",f(5)}}};
      TensorSpace b{"experts",{{"e",f(4)},{"n",f(19)},{"k",f(5)}}};
      op.operands={{"input",a,{{IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("k")}},{}},
          {"",b,{{IndexResult::DataDependent("bindings",{"v"}),IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
      solver::ModelDescription model;model.dm=model.serving=true;
      model.dtype=solver::ScalarType::kBF16;model.dims={19,0,19};
      model.gemms={{19,5,0,1}};model.stages.resize(1);model.stages.front().gemm=0;
      model.gemm_access.resize(1);auto& access=model.gemm_access.front();
      access.b=codegen::DmBAccess::kExpertIndirect;access.block_rows=bm;access.experts=4;
      model.task_semantics={{op,{{"v",f(1)},{"row",f(tm)},{"n",f(16)}},0,false}};
      auto const& semantic=model.task_semantics.front();
      solver::GemmConfig geometry{tm,16,16,2,1};
      auto graph=solver::InstantiateModelTasks(model,{geometry});
      auto input=solver::DeriveModelTaskInput(model,semantic,graph,&geometry);
      auto traits=solver::ModelTaskTraits(model,0,geometry);
      DramFloor floor;
      floor.tensors["hidden"].element_bytes=2;
      floor.tensors["hidden"].writes=CouplingRelation::FromIslText("{ [] -> [t,k] : 0<=t<19 and 0<=k<5 }");
      floor.tensors["experts"].element_bytes=2;floor.tensors["partial"].element_bytes=2;
      floor.tensors["partial"].external_writes=CouplingRelation::FromIslText("{ [] -> [t,r,n] : 0<=t<19 and 0<=r<2 and 0<=n<19 }");
      floor.no_producer_bytes=QuasiPolynomial::Constant(4*19*5*2);
      solver::BindTaskDramProvenance(input,semantic,floor,{},true);
      double total_expected=0;
      for(unsigned v=0;v<capacity;++v)for(unsigned row=0;row<(bm+tm-1)/tm;++row)
        for(unsigned n=0;n<2;++n) {
          ParamBinding at;at.Bind("v",v).Bind("row",row).Bind("n",n);
          auto got=solver::PriceMoeVirtualTask(cost,input,semantic,traits,{1},model,1,floor,routing,slot,at);
          total_expected+=got.expected_isolated_ns;
          solver::TaskPriceParts expected;double isolated=0,active=0,live_rows=0,service=0;
          for(auto const& window:windows) {
            auto rows=std::max<long>(0,std::min<long>(tm,long(window[v])-long(row)*tm));
            solver::TaskPriceParts parts;
            if(rows) {
              auto columns=n?3:16;
              solver::TaskMemoryTraffic memory;
              memory.global_read_bytes=rows*5*2+columns*5*2;
              memory.global_write_bytes=rows*columns*2;
              memory.no_producer_read_bytes=columns*5*2;
              memory.produced_read_bytes=rows*5*2;memory.external_write_bytes=rows*columns*2;
              parts=cost.PriceParts(input,traits,{1},model,1,at,1,&memory);
              active+=0.5;live_rows+=rows*0.5;
            }else{parts.fixed_ns=7;++empty_checked;}
            expected.fixed_ns+=parts.fixed_ns*0.5;expected.compute_ns+=parts.compute_ns*0.5;
            expected.dram_bytes+=parts.dram_bytes*0.5;
            expected.no_producer_dram_bytes+=parts.no_producer_dram_bytes*0.5;
            isolated+=0.5*solver::IsolatedNs(parts,target.CalibrationFor("bf16").dram_gbps/target.res.num_sms);
            if(parts.dram_bytes)service+=0.5*parts.dram_bytes/parts.dram_rate_cap;
          }
          close(got.parts.fixed_ns,expected.fixed_ns);close(got.parts.compute_ns,expected.compute_ns);
          close(got.parts.dram_bytes,expected.dram_bytes);
          close(got.parts.no_producer_dram_bytes,expected.no_producer_dram_bytes);
          close(got.parts.dram_rate_cap,service?expected.dram_bytes/service:0);
          close(got.expected_isolated_ns,isolated);close(got.active_probability,active);close(got.expected_rows,live_rows);
          if(active==0){assert(got.parts.fixed_ns==7 && got.parts.compute_ns==0 && got.parts.dram_bytes==0);}
          ++checked;
        }
      auto capacity_prices=solver::PriceBoundaryPieces(cost,input,semantic,traits,{1},model,1);
      auto profiled=solver::PriceMoeBoundaryPieces(cost,input,semantic,traits,{1},model,1,
          floor,routing,slot,capacity_prices);
      assert(profiled.routing_profiled && !profiled.inferred_empty_cost);
      close(profiled.total_isolated_ns,total_expected);
      long covered=0;for(auto const& piece:profiled.pieces)covered+=piece.count.Eval({});
      assert(covered==long(capacity*((bm+tm-1)/tm)*2));
      if(!slot && bm==32 && tm==16) {
        auto spread=routing,constant=routing;
        spread.virtual_row_histograms[bm].assign(capacity,{});
        constant.virtual_row_histograms[bm].assign(capacity,{});
        for(auto counts:{std::vector<unsigned>{18,10,10,0},std::vector<unsigned>{10,18,10,0}})
          for(unsigned v=0;v<counts.size();++v)if(counts[v])
            ++spread.virtual_row_histograms[bm][v][counts[v]];
        for(unsigned v=0;v<3;++v)constant.virtual_row_histograms[bm][v][v<2?14:10]=2;
        ParamBinding at;at.Bind("v",0).Bind("row",1).Bind("n",0);
        // Both complete two-window populations have the same mean per block.
        // A later row subtile is active in one population and empty in the other.
        auto a=solver::PriceMoeVirtualTask(cost,input,semantic,traits,{1},model,1,floor,spread,false,at);
        auto b=solver::PriceMoeVirtualTask(cost,input,semantic,traits,{1},model,1,floor,constant,false,at);
        assert(a.active_probability==0.5 && b.active_probability==0);
        assert(a.expected_rows==1 && b.expected_rows==0);
        assert(a.expected_isolated_ns>b.expected_isolated_ns && b.expected_isolated_ns==7);
      }
      auto missing=target;missing.CalibrationFor("bf16").task_body.serving.erase("gemm_expert_indirect_empty");
      if(!slot) {
        solver::CostModel uncalibrated(missing,solver::ScalarType::kBF16,options);
        ParamBinding at;at.Bind("v",capacity-1).Bind("row",0).Bind("n",0);
        bool rejected=false;try{(void)solver::PriceMoeVirtualTask(uncalibrated,input,semantic,traits,{1},model,1,floor,routing,slot,at);}
        catch(std::invalid_argument const&){rejected=true;}assert(rejected);
        auto inferred=solver::PriceMoeVirtualTask(uncalibrated,input,semantic,traits,{1},model,
            1,floor,routing,slot,at,solver::MoeEmptyPricing::kCapacitySurrogate);
        assert(inferred.inferred_empty_cost && inferred.active_probability==0 &&
            inferred.parts.compute_ns==0 && inferred.parts.dram_bytes==0 && inferred.parts.fixed_ns>0);
      }
    }
  }
  assert(checked>500 && empty_checked>0);
  std::cout<<"MoE histogram pricing: "<<checked<<" BM/TM/slot/tail cases and "
      <<empty_checked<<" empty window contributions match independent traffic enumeration PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_task_pricing_test
