// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Codegen/MoeBinding.h>
#include <mlir/IR/MLIRContext.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace tilemega::tests::moe_region_plan_test {
int TestMoeRegionPlan(int argc,char** argv) {
  using namespace frontend;using namespace codegen;
  mlir::MLIRContext context;mlir::Builder builder(&context);unsigned cases=0;
  if(argc==2) {
    auto bridge=ReadExportBridge(argv[1]);
    auto blocks=FindDecoderMoeBlocks(bridge.nodes,bridge.inputs);assert(blocks.size()==48);
    for(unsigned tokens:{1u,16u,64u,1024u})for(bool grouped:{false,true}) {
      ModelPlan plan;plan.serving=true;plan.dtype="bf16";
      plan.serving_seq=tokens<64?1:64;plan.serving_capacity=1088;
      auto activation=[&](std::string name) {
        PlanBuffer b;b.name=std::move(name);b.constant=tokens*2048;
        unsigned index=plan.buffers.size();plan.buffers.push_back(b);return index;
      };
      unsigned current=activation("decoder.input");
      MoeRegionOptions options;options.tokens=tokens;options.grouped=grouped;options.block_rows=32;
      for(auto const& match:blocks) {
        auto output=activation(match.output);auto gemms=plan.gemms.size(),stages=plan.stages.size();
        AppendMoeBlock(plan,match,bridge.nodes,bridge.inputs,current,output,options);
        assert(plan.gemms.size()==gemms+3 && plan.node_buffer.at(match.input)==current &&
            plan.node_buffer.at(match.output)==output);
        for(unsigned index=stages;index<plan.stages.size();++index) {
          auto const& stage=plan.stages[index];
          if(stage.kind==PlanTaskKind::kGemm)assert(stage.gemm>=gemms && stage.gemm<gemms+3);
          if(stage.moe.step!=DmMoeStep::kNone)assert(stage.moe.router_gemm==gemms);
          if(stage.binding_producer!=kDmNoIndex)
            assert(stage.binding_producer>=stages && stage.binding_producer<index);
        }
        assert(plan.stages.back().operands[2]==current && plan.stages.back().operands[3]==output);
        current=output;
      }
      assert(plan.dm && !plan.forward && plan.serving && plan.gemms.size()==144);
      auto before=plan.stages.size(),buffers=plan.buffers.size();
      bool rejected=false;
      try{AppendMoeBlock(plan,blocks.front(),bridge.nodes,bridge.inputs,current,current,options);}
      catch(std::invalid_argument const&){rejected=true;}
      assert(rejected && plan.stages.size()==before && plan.buffers.size()==buffers);
      auto incompatible=options;incompatible.tokens=tokens==1?2:tokens/2;
      rejected=false;
      try{AppendMoeBlock(plan,blocks.front(),bridge.nodes,bridge.inputs,0,current,incompatible);}
      catch(std::invalid_argument const&){rejected=true;}
      assert(rejected && plan.stages.size()==before && plan.buffers.size()==buffers);
      ValidateDmModelPlan(plan);++cases;
    }
    std::cout<<"Decoder MoE composition: "<<cases<<" full-depth metadata plans with remapped buffers, GEMMs and bindings PASS\n";
    return 0;
  }
  for(auto spelling:{"before","core"}) {
    auto bridge=ReadExportBridge(std::string(TILEMEGA_SOURCE_DIR)+
        "/test/fixtures/moe/region_"+spelling+".json");
    for(unsigned t:{1u,2u,16u,512u,513u,1024u,4096u})
      for(bool grouped:{false,true})for(unsigned bm:{16u,32u,64u,128u}) {
        MoeRegionOptions options;options.tokens=t;options.grouped=grouped;options.block_rows=bm;
        auto plan=BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
        assert(plan.dm && plan.forward && plan.forward_token_axis && !plan.serving &&
            plan.serving_seq==int(t) && !plan.serving_capacity && plan.norm_epsilon==1e-6);
        assert(plan.gemms.size()==3 && plan.outputs.size()==1);
        assert(plan.stages.front().kind==PlanTaskKind::kRMSNorm &&
            plan.stages.back().kind==PlanTaskKind::kMoECombine);
        unsigned capacity=0;assert(MoeVirtualCapacity(t,8,128,grouped?bm:1,grouped,&capacity));
        unsigned selectors=0,histograms=0,prefixes=0,scatters=0,small=0,binding=0;
        for(unsigned id=0;id<plan.stages.size();++id) {
          auto const& s=plan.stages[id];
          if(s.kind!=PlanTaskKind::kMoETopK)continue;
          assert(s.moe.binding_capacity==capacity && s.moe.row_capacity==t*8 && s.moe.grouped==grouped);
          assert(EncodeDm(builder,s.moe)==EncodeDm(builder,DecodeDmMoeStage(EncodeDm(builder,s.moe))));
          selectors+=s.moe.step==DmMoeStep::kSelect;
          histograms+=s.moe.step==DmMoeStep::kHistogram;prefixes+=s.moe.step==DmMoeStep::kPrefix;
          scatters+=s.moe.step==DmMoeStep::kScatter;small+=s.moe.step==DmMoeStep::kSelectAndDispatch;
          binding=id;
        }
        assert(small==(t<=512) && selectors==(t>512) && scatters==(t>512));
        assert(histograms==(t>512 && grouped) && prefixes==histograms);
        for(unsigned g=1;g<3;++g) {
          auto const& gemm=plan.gemms[g];auto const& a=gemm.access;
          assert(a.b==DmBAccess::kExpertIndirect && a.block_rows==(grouped?bm:1) &&
              a.binding_blocks==capacity && a.binding_rows==t*8 && a.routing_topk==8);
          assert(plan.stages[binding+g].gemm==g && plan.stages[binding+g].binding_producer==binding);
        }
        assert(plan.gemms[1].access.a==DmAAccess::kRowGather &&
            plan.gemms[2].access.write.kind==DmWriteKind::kRowScatter);
        auto const& gate=plan.gemms[1];auto const& down=plan.gemms[2];
        assert(gate.n==1536 && gate.k==2048 && down.n==2048 && down.k==768);
        assert(plan.buffers[gate.d].constant==std::uint64_t(capacity)*(grouped?bm:1)*768);
        assert(plan.buffers[down.d].constant==std::uint64_t(t)*8*2048);
        assert(gate.chain.count==1 && gate.chain.operations[0].gate==DmGatePair::kSwiGLU &&
            gate.chain.operations[0].unit==16 && !down.chain.count);
        for(unsigned tn:{16u,32u,64u,128u,256u}) {
          MaterializeMoeRegionStorage(plan,tn);
          auto const& side=plan.gemms[0].chain.side[0];
          assert(plan.buffers[side.buffer].constant==t*((128+tn-1)/tn)*8 &&
              plan.buffers[side.auxiliary].constant==plan.buffers[side.buffer].constant);
        }
        ValidateDmModelPlan(plan);++cases;
      }
    auto rejects=[&](auto change) {
      MoeRegionOptions options;change(options);bool rejected=false;
      try{(void)BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);}
      catch(std::invalid_argument const&){rejected=true;}assert(rejected);
    };
    rejects([](auto& o){o.tokens=0;});rejects([](auto& o){o.tokens=4097;});
    rejects([](auto& o){o.grouped=true;o.block_rows=17;});
    rejects([](auto& o){o.router_tile_n=0;});rejects([](auto& o){o.combine_channel_tile=31;});
  }
  assert(cases==112);
  std::cout<<"MoE region planning: "<<cases<<" before/Core geometries and 560 router storage layouts PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_region_plan_test
