#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskWork.h>
#include <cassert>
#include <iostream>

int main(int argc,char** argv) {
  using namespace tilemega;
  analysis::IslContext isl;
  for(unsigned ratio:{3u,4u}) {
    frontend::ModelPlan plan;plan.serving=true;plan.serving_seq=1;
    for(auto name:{"q","k","v","packed"}) {
      frontend::PlanBuffer b;b.name=name;plan.buffers.push_back(b);
    }
    frontend::PlanStage s;s.kind=frontend::PlanTaskKind::kLayoutConvert;
    s.extent=2;s.width=(ratio+2)*64;s.group=1;s.layout_group_ratio=ratio;
    s.operands.fill(UINT32_MAX);s.operands[0]=0;s.operands[1]=1;s.operands[2]=2;s.operands[3]=3;
    plan.stages.push_back(s);
    frontend::LiftOptions dims;dims.serving=true;dims.static_seq=1;dims.batch_symbol="B";
    auto lifted=frontend::LiftSemantics(plan,dims);
    auto geometry=frontend::LaunchGranularity(lifted,plan,{});
    auto instances=analysis::Instantiate(lifted.sem,geometry);
    auto const& op=lifted.sem.ops.at(0);auto const& task=instances.nodes.at(0);
    assert(op.result.axes.size()==3 && op.element_reads.size()==3);
    analysis::ParamBinding point;point.Bind("B",2);
    for(unsigned input=0;input<3;++input) {
      auto actual=analysis::ExactElementRead(op,task,op.element_reads[input],point);
      auto count=input?128:ratio*128;
      auto expected=analysis::CouplingRelation::FromIslText(
          "{ [m] -> [m,n] : 0<=m<2 and 0<=n<"+std::to_string(count)+" }");
      if(!actual.IsSubset(expected) || !expected.IsSubset(actual))
        std::cerr<<"ratio="<<ratio<<" input="<<input<<" actual="<<actual.ToString()<<" expected="<<expected.ToString()<<'\n';
      assert(actual.IsSubset(expected) && expected.IsSubset(actual));
    }
  }
  unsigned layers=0;
  if(argc==2) {
    auto bridge=frontend::ReadExportBridge(argv[1]);
    frontend::ServingOptions options;options.phase=frontend::ServingOptions::Phase::kPrefill;
    options.seq=64;options.deferred_norm=false;options.separate_qkv=true;
    auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
    frontend::ValidateDmModelPlan(plan);
    unsigned packs=0;
    for(auto const& s:plan.stages) {
      layers+=s.kind==frontend::PlanTaskKind::kFusedAttention;
      packs+=s.kind==frontend::PlanTaskKind::kLayoutConvert;
    }
    assert(layers && packs==layers && plan.gemms.size()==6*layers+1);
    auto invalid=plan;
    for(auto& s:invalid.stages)if(s.kind==frontend::PlanTaskKind::kLayoutConvert) {
      s.layout_group_ratio=0;break;
    }
    bool rejected=false;
    try {frontend::ValidateDmModelPlan(invalid);}catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);
  }
  std::cout<<"{\"status\":\"pass\",\"scope\":\"exact guarded QKV reads, GQA3/4 and distinct batch rows\",\"actual_imported_layers\":"<<layers<<",\"device_executed\":false}\n";
}
