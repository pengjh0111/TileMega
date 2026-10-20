// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <cassert>
#include <iostream>

int main(int argc,char** argv) {
  using namespace tilemega::frontend;
  if(argc!=2)return 2;
  auto bridge=ReadExportBridge(argv[1]);
  unsigned cases=0;
  for(int chunks:{1,2,4,8,16,32,64}) {
    ServingOptions options;options.dense_attention_chunks=chunks;
    options.interleave_u=8;options.dn_vector_sums=true;options.argmax_tile_n=128;
    auto plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
    LiftOptions dims;dims.serving=true;dims.batch_symbol="B";dims.past_symbol="past";dims.static_seq=1;
    auto lifted=LiftServingSemantics(plan,dims);
    unsigned layers=0;
    for(unsigned i=0;i<plan.stages.size();++i) {
      auto const& stage=plan.stages[i];
      assert(stage.kind!=PlanTaskKind::kAttentionMerge && stage.kind!=PlanTaskKind::kRMSNorm);
      if(stage.kind!=PlanTaskKind::kFusedAttention)continue;
      ++layers;assert(stage.dense_attention_chunks==chunks && stage.attention_kv_block==1088);
      auto const& arrivals=plan.buffers.at(stage.operands[10]);
      assert(arrivals.dtype=="i32" && arrivals.per_batch==2*stage.extent);
      auto const& partial=plan.buffers.at(stage.operands[8]);
      assert(partial.per_batch==stage.extent*stage.group*stage.width*chunks);
      bool output=false,edge=false;
      for(unsigned j=0;j<lifted.ops.size();++j)if(lifted.ops[j].stage==int(i)) {
        auto const& semantic=lifted.sem.ops[j];
        assert(semantic.result.name==plan.buffers.at(stage.operands[7]).name);output=true;
        for(unsigned k=0;k<lifted.ops.size();++k)if(lifted.ops[k].stage==int(i+1))
          for(auto const& input:lifted.sem.ops[k].operands)edge|=input.producer==semantic.name;
      }
      assert(output && edge);
    }
    assert(layers && plan.stages.size()==5*layers+3);
    auto prefill=options;prefill.phase=ServingOptions::Phase::kPrefill;prefill.seq=64;
    bool rejected=false;
    try {(void)BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,prefill);}
    catch(std::invalid_argument const&) {rejected=true;}
    assert(rejected);++cases;
  }
  std::cout<<"native plan cases="<<cases<<" direct attention -> o edges verified\n";
}
