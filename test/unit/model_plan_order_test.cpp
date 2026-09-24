// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

int main(int argc,char** argv) {
  using namespace tilemega::frontend;
  if(argc!=3)throw std::invalid_argument("two semantically equivalent export bridges required");
  auto load=[](char const* path){auto bridge=ReadExportBridge(path);return BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);};
  auto canonical=[](ModelPlan const& plan) {
    std::map<std::string,std::string> stages;
    std::size_t previous=0;bool first=true;
    for(auto const& s:plan.stages) {
      if(!first && s.representative_index<=previous)throw std::runtime_error("non-topological semantic stages");
      first=false;previous=s.representative_index;
      std::ostringstream text;text<<int(s.kind)<<'/'<<s.extent<<'/'<<s.width<<'/'<<s.group;
      for(auto operand:s.operands)if(operand<plan.buffers.size())text<<'/'<<plan.buffers[operand].name;
      if(s.kind==PlanTaskKind::kGemm) {
        auto const& g=plan.gemms.at(s.gemm);
        text<<'/'<<g.n<<'/'<<g.k<<'/'<<g.beta;
        for(auto operand:{g.a,g.b,g.c,g.d})text<<'/'<<plan.buffers.at(operand).name;
      }
      stages.emplace(s.representative,text.str());
    }
    return stages;
  };
  auto a=load(argv[1]),b=load(argv[2]);
  if(canonical(a)!=canonical(b))throw std::runtime_error("reordered projections changed semantic buffer edges");
  std::cout<<"PROJECTION_ORDER stages="<<a.stages.size()<<" gemms="<<a.gemms.size()
           <<" topological=PASS semantic_buffer_edges_equal=PASS\n";
}
