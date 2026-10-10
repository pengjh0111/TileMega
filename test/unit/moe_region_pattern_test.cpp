// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPattern.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace tilemega::tests::moe_region_pattern_test {
int TestMoeRegionPattern(int argc,char** argv) {
  using namespace frontend;unsigned accepted=0,rejected=0;
  if(argc==2) {
    auto bridge=ReadExportBridge(argv[1]);
    auto blocks=FindDecoderMoeBlocks(bridge.nodes,bridge.inputs);
    auto count=std::count_if(bridge.nodes.begin(),bridge.nodes.end(),[](auto const& n) {
      return n.target=="tilemega.moe_experts.default";
    });
    assert(count>0 && blocks.size()==unsigned(count));
    for(auto const& block:blocks)assert(block.hidden==2048 && block.intermediate==768 &&
        block.expert_count==128 && block.top_k==8 && block.epsilon==1e-6);
    auto bad=bridge;
    auto expert=std::find_if(bad.nodes.begin(),bad.nodes.end(),[](auto const& n) {
      return n.target=="tilemega.moe_experts.default";
    });
    expert->shape[0]="2*("+expert->shape[0]+")";
    bool failed=false;try{FindDecoderMoeBlocks(bad.nodes,bad.inputs);}
    catch(std::invalid_argument const&){failed=true;}assert(failed);
    std::cout<<"Decoder MoE blocks: "<<blocks.size()<<" residual regions and negative row-count contract PASS\n";
    return 0;
  }
  for(auto spelling:{"before","core"}) {
    auto bridge=ReadExportBridge(std::string(TILEMEGA_SOURCE_DIR)+
        "/test/fixtures/moe/region_"+spelling+".json");
    auto check=[&](ExportBridge const& b) {
      auto match=MatchMoeRegion(b.nodes,b.inputs,b.outputs);
      assert(match.hidden==2048 && match.intermediate==768 && match.expert_count==128 &&
          match.top_k==8 && match.epsilon==1e-6 && match.output==b.outputs.front());++accepted;
    };
    auto reject=[&](ExportBridge const& b) {
      bool failed=false;
      try{(void)MatchMoeRegion(b.nodes,b.inputs,b.outputs);}
      catch(std::invalid_argument const& e){failed=std::string(e.what()).find("MoE region:")==0;}
      assert(failed);++rejected;
    };
    check(bridge);
    auto renamed=bridge;
    auto rename_argument=[&](auto&& self,FxArgument& a)->void {
      if(a.kind==FxArgument::Kind::kNode)a.text="renamed_"+a.text;
      for(auto& x:a.items)self(self,x);
    };
    for(auto& n:renamed.nodes) {
      n.name="renamed_"+n.name;
      for(auto& input:n.inputs)input="renamed_"+input;
      for(auto& a:n.args)rename_argument(rename_argument,a);
      for(auto& entry:n.kwargs)rename_argument(rename_argument,entry.second);
    }
    for(auto& s:renamed.inputs) {s.name="renamed_"+s.name;s.target="arbitrary."+s.target;}
    for(auto& name:renamed.outputs)name="renamed_"+name;check(renamed);
    auto mutate=[&](auto edit) {auto wrong=bridge;edit(wrong);reject(wrong);};
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="<built-in function getitem>" &&
        n.args.at(1).integer==1)n.args[1].integer=0;});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="aten.topk.default") {
      FxArgument a;a.kind=FxArgument::Kind::kBool;a.boolean=false;n.kwargs["largest"]=a;
    }});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="aten.sum.dim_IntList")n.args[2].boolean=false;});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="aten.mean.dim")n.args[1].items[0].integer=0;});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="aten.add.Tensor" && n.args.size()>1 &&
        n.args[1].kind==FxArgument::Kind::kFloat)n.args[1].real=-1e-6;});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.op=="placeholder" && n.shape.size()==3 && n.shape[1]=="1536")
        n.shape[2]="1024";});
    mutate([](auto& b) {b.outputs.push_back(b.outputs.front());});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="aten.add.Tensor") {
      FxArgument a;a.kind=FxArgument::Kind::kInt;a.integer=2;n.kwargs["alpha"]=a;
    }});
    mutate([](auto& b) {for(auto& n:b.nodes)if(n.target=="aten.topk.default") {
      n.kwargs["k"]=n.args.at(1);
    }});
    mutate([](auto& b) {FxNodeRecord n;n.op="call_function";n.name="unmatched";
      n.target="aten.sin.default";b.nodes.push_back(n);});
  }
  assert(accepted==4 && rejected==20);
  std::cout<<"MoE region structure: before/Core, renamed graphs and 20 negative contracts PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_region_pattern_test
