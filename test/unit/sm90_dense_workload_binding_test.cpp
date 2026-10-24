// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseWorkGraph.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <cassert>
#include <fstream>
#include <iostream>

using namespace tilemega;
using namespace tilemega::solver;
namespace {
constexpr auto absent=~std::uint32_t{0};
template<class F>void Reject(F f) {
  bool rejected=false;try {f();}catch(std::exception const&) {rejected=true;}assert(rejected);
}
void BoundGraph(frontend::ModelPlan const& plan,Sm90DenseWorkload const& work,
    Sm90DenseCoordinates const& c,TargetSpec const& target) {
  auto graph=BuildSm90DenseWorkGraph(work,c,target,575);unsigned count=0;
  assert(work.gemm_bindings && graph.stages.size()==plan.stages.size());
  for(unsigned s=0;s<plan.stages.size();++s) {
    auto const& stage=plan.stages[s];
    if(stage.kind!=frontend::PlanTaskKind::kGemm) {assert(!graph.stages[s].plan_binding);continue;}
    auto const& g=plan.gemms.at(stage.gemm);auto const& b=work.gemm_bindings->at(count++);
    assert(b.stage==int(s) && b.gemm==int(stage.gemm) && b.n==int(g.n) && b.k==int(g.k));
    assert(b.norm_ss==g.norm_ss && b.ss_out==g.ss_out);
    assert(b.binding.input_norm==(g.norm_ss!=absent) && b.binding.output_sums==(g.ss_out!=absent));
    assert(graph.stages[s].plan_binding->ToJson().Dump(0)==b.binding.ToJson().Dump(0));
    assert(Sm90DenseGemmPlanBinding::FromJson(b.ToJson()).ToJson().Dump(0)==b.ToJson().Dump(0));
  }
  assert(count==plan.gemms.size());
}
json::Value WorkJson(Sm90DenseWorkload const& work,bool bindings) {
  json::Array shapes,rows;
  for(auto const& shape:work.shapes)shapes.emplace_back(json::Array{shape[0],shape[1]});
  auto result=json::Value(json::Object{{"batch",work.batch},{"layers",work.layers},
      {"head_dim",work.head_dim},{"queries",work.queries},{"qk_norm",work.qk_norm},{"shapes",shapes}});
  if(bindings) {for(auto const& b:*work.gemm_bindings)rows.push_back(b.ToJson());result.Set("gemm_bindings",rows);}
  return result;
}
}
int main(int argc,char** argv) {
  if(argc!=9)return 2;
  Sm90DenseWorkload historical{{},1,2,128,2,true};assert(!historical.gemm_bindings);
  auto target=TargetSpec::FromJson(argv[1]);
  for(unsigned sample=0;sample<2;++sample) {
    auto bridge=frontend::ReadExportBridge(argv[2+sample*3]);
    auto coordinates=json::ParseFile(argv[3+sample*3]);
    if(coordinates.Find("plan"))coordinates=coordinates.At("plan").At("native_dense");
    auto c=Sm90DenseCoordinates::FromJson(coordinates);
    frontend::ServingOptions options;
    options.interleave_u=c.swiglu_u;options.dn_vector_sums=c.dn_vector_sums;
    options.argmax_tile_n=c.gemms[4].tile_n;options.dense_attention_chunks=c.attention.chunks;
    auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
    int batch=std::stoi(argv[4+sample*3]);auto work=Sm90DenseWorkloadFromPlan(plan,batch);
    BoundGraph(plan,work,c,target);
    assert(work.layers>1 && work.gemm_bindings->size()==unsigned(4*work.layers+1));

    // Storage-table permutation preserves actual stage references and flags.
    auto permuted=plan;std::swap(permuted.gemms[0],permuted.gemms[1]);
    for(auto& stage:permuted.stages)if(stage.kind==frontend::PlanTaskKind::kGemm) {
      if(stage.gemm==0)stage.gemm=1;else if(stage.gemm==1)stage.gemm=0;
    }
    BoundGraph(permuted,Sm90DenseWorkloadFromPlan(permuted,batch),c,target);

    auto changed=plan;auto& g=changed.gemms[changed.stages[1].gemm];
    frontend::PlanBuffer sums;sums.name="CPU_binding_metadata_fixture_only";sums.dtype="f32";
    sums.per_batch=(g.n+7)/8;changed.buffers.push_back(sums);
    g.norm_ss=absent;g.ss_out=changed.buffers.size()-1;
    auto different=Sm90DenseWorkloadFromPlan(changed,batch);
    auto const& b=different.gemm_bindings->front();assert(!b.binding.input_norm && b.binding.output_sums);
    auto before=BuildSm90DenseWorkGraph(work,c,target),after=BuildSm90DenseWorkGraph(different,c,target);
    assert(before.stages[1].ServiceDescriptor().Dump(0)!=after.stages[1].ServiceDescriptor().Dump(0));
    g.ss_out=absent;g.d=absent;auto disabled=Sm90DenseWorkloadFromPlan(changed,batch);
    assert(!disabled.gemm_bindings->front().binding.input_norm && !disabled.gemm_bindings->front().binding.output_sums);
    auto unknown=work;unknown.gemm_bindings.reset();
    auto old_graph=BuildSm90DenseWorkGraph(unknown,c,target);assert(!old_graph.stages[1].plan_binding);

    auto invalid=plan;invalid.gemms[invalid.stages[1].gemm].norm_ss=invalid.buffers.size();
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.gemms[invalid.stages[1].gemm].ss_out=invalid.buffers.size();
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.stages[1].gemm=invalid.gemms.size();
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.stages[3].gemm=invalid.stages[1].gemm;
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;std::swap(invalid.stages[1],invalid.stages[3]);
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.stages.erase(invalid.stages.begin()+1);
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;++invalid.gemms[invalid.stages[6].gemm].n;
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.stages[7].group=invalid.stages[2].group==2?3:2;
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.stages[2].operands[5]=absent;invalid.stages[2].operands[6]=0;
    Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});
    invalid=plan;invalid.serving_seq=2;Reject([&]{Sm90DenseWorkloadFromPlan(invalid,batch);});

    auto bad=work;bad.gemm_bindings->pop_back();Reject([&]{BuildSm90DenseWorkGraph(bad,c,target);});
    bad=work;std::swap(bad.gemm_bindings->at(0),bad.gemm_bindings->at(1));
    Reject([&]{BuildSm90DenseWorkGraph(bad,c,target);});
    bad=work;++bad.gemm_bindings->front().n;Reject([&]{BuildSm90DenseWorkGraph(bad,c,target);});
    bad=work;++bad.gemm_bindings->front().stage;Reject([&]{BuildSm90DenseWorkGraph(bad,c,target);});
    bad=work;bad.gemm_bindings->at(1).gemm=bad.gemm_bindings->front().gemm;
    Reject([&]{BuildSm90DenseWorkGraph(bad,c,target);});
    auto json=work.gemm_bindings->front().ToJson();json.Set("gemm",-1);
    Reject([&]{Sm90DenseGemmPlanBinding::FromJson(json);});
    json=work.gemm_bindings->front().ToJson();json.Set("norm_ss",double(absent));
    Reject([&]{Sm90DenseGemmPlanBinding::FromJson(json);});
    json=work.gemm_bindings->front().ToJson();json.Set("output_sums",1);
    Reject([&]{Sm90DenseGemmPlanBinding::FromJson(json);});
    if(!sample) {
      auto write=[&](char const* name,json::Value const& value) {std::ofstream out(std::string(argv[8])+"/"+name);out<<value.Dump(2)<<'\n';assert(out.good());};
      write("work-bound.json",WorkJson(work,true));write("work-unknown.json",WorkJson(work,false));
      auto malformed=WorkJson(work,true);
      json::Array records;for(auto const& item:*bad.gemm_bindings)records.push_back(item.ToJson());
      malformed.Set("gemm_bindings",records);write("work-invalid.json",malformed);write("coordinates.json",c.ToJson());
    }
    std::cout<<"actual exported plan layers="<<work.layers<<" gemms="<<work.gemm_bindings->size()
        <<" D="<<work.head_dim<<" Q="<<work.queries<<" qk_norm="<<work.qk_norm<<" bindings_checked\n";
  }
}
