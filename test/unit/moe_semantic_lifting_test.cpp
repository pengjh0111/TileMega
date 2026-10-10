// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Codegen/MoeBinding.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/CountedDependencyForm.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace tilemega::tests::moe_semantic_lifting_test {
int TestMoeSemanticLifting(int,char**) {
  using namespace frontend;using namespace analysis;using namespace codegen;
  IslContext isl;unsigned cases=0,counted=0;
  for(auto spelling:{"before","core"}) {
    auto bridge=ReadExportBridge(std::string(TILEMEGA_SOURCE_DIR)+
        "/test/fixtures/moe/region_"+spelling+".json");
    // Keep the same exported topology while making exhaustive relation proofs
    // small. Real checkpoint geometry is covered separately by model gates.
    for(auto& node:bridge.nodes)for(auto& dimension:node.shape) {
      if(dimension=="2048")dimension="64";
      else if(dimension=="768")dimension="32";
      else if(dimension=="1536")dimension="64";
      else if(dimension=="128")dimension="16";
    }
    for(unsigned tokens:{1u,17u,513u,4096u})for(bool grouped:{false,true}) {
      MoeRegionOptions options;options.tokens=tokens;options.grouped=grouped;
      options.block_rows=32;options.combine_token_tile=16;options.combine_channel_tile=32;
      auto plan=BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
      LiftOptions lift;lift.forward=true;lift.seq_symbol="T";lift.static_seq=tokens;
      auto lifted=LiftSemantics(plan,lift);auto saved=lifted.sem.Serialize();
      assert(lifted.has_plan && lifted.degraded.empty() && lifted.ops.size()==plan.stages.size());
      std::vector<GemmGranularity> geometry={{16,16,16,2,1},{16,32,16,2,1},{16,32,16,2,1}};
      auto graph=Instantiate(lifted.sem,LaunchGranularity(lifted,plan,geometry));
      ParamBinding theta;theta.Bind("T",tokens);
      assert(graph.nodes.size()==plan.stages.size());
      unsigned capacity=0;assert(MoeVirtualCapacity(tokens,8,16,grouped?32:1,grouped,&capacity));
      for(unsigned s=0;s<plan.stages.size();++s) {
        auto const& stage=plan.stages[s];auto const& task=graph.nodes[s];long expected=0;
        if(stage.kind==PlanTaskKind::kRMSNorm)expected=tokens;
        else if(stage.kind==PlanTaskKind::kGemm) {
          auto g=stage.gemm;
          expected=g==0?(tokens+15)/16:
              long(capacity)*(((grouped?32:1)+15)/16)*2;
          if(g) {
            auto const& op=lifted.sem.ops[s];
            assert(op.domain[0].runtime && op.domain[0].capacity &&
                op.domain[0].capacity->Eval(theta,{})==capacity);
            assert(op.domain[0].binding_requirement==(grouped?"prefix_sum":"tensor_values"));
            if(g==1)assert(op.arithmetic=="swiglu_gemm" && op.reduction.partial_values==2);
          }
        }else if(stage.kind==PlanTaskKind::kMoETopK) {
          expected=stage.moe.step==DmMoeStep::kPrefix || stage.moe.step==DmMoeStep::kSelectAndDispatch?
              1:(tokens+127)/128;
        }else if(stage.kind==PlanTaskKind::kMoECombine)expected=((tokens+15)/16)*2;
        assert(task.Count().Eval(theta,{})==expected);
      }
      auto narrow=geometry;narrow[0].tile_n=32;narrow[1].tile_m=32;narrow[1].tile_n=128;
      (void)Instantiate(lifted.sem,LaunchGranularity(lifted,plan,narrow));
      assert(lifted.sem.Serialize()==saved);
      if(tokens<=17) {
        auto edges=CouplingDerivation{}.Derive(graph,theta);assert(!edges.empty());
        unsigned down=plan.stages.size()-2;
        unsigned combine=plan.stages.size()-1;
        auto const& down_sem=lifted.sem.ops[down];
        auto edge=std::find_if(edges.begin(),edges.end(),[&](auto const& e) {
          return e.src.name==down_sem.name && e.dst.name==lifted.sem.ops[combine].name;
        });
        assert(edge!=edges.end());
        if(grouped) {
          auto counts=BindAlignedCountedScatterDependency(graph.nodes[down],graph.nodes[combine],
              down_sem.result.name,{0,1},plan.buffers[plan.gemms[2].access.rows].name,theta);
          assert(counts.expected.size()==((tokens+15)/16)*2);
          for(unsigned id=0;id<counts.expected.size();++id)
            assert(counts.expected[id]==std::min(16u,tokens-(id/2)*16)*8);
          ++counted;
        }else {
          assert(down_sem.result_map.results[0].kind==IndexResult::Kind::kAffine &&
              down_sem.result_map.results[1].kind==IndexResult::Kind::kAffine);
        }
      }
      ++cases;
    }
  }
  assert(cases==16 && counted==4);
  std::cout<<"MoE semantics: 16 compact before/Core slot/group graphs, invariant L-sem and four counted scatter proofs PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_semantic_lifting_test
