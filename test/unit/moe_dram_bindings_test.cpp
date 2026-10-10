// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/MoeDramBindings.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <filesystem>
#include <sstream>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/DnnResourceProbe.h>
#include <tilemega/Codegen/tasks/DmMoeScalarDataflow.h>
#include <fstream>
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <cmath>
#include <iostream>

namespace tilemega::tests::moe_dram_bindings_test {
int TestMoeDramBindings(int argc,char** argv) {
  using namespace frontend;using namespace solver;using namespace analysis;
  IslContext isl;
  auto path=std::string(TILEMEGA_SOURCE_DIR)+"/test/fixtures/moe/region_before.json";
  auto bridge=ReadExportBridge(path);
  for(auto& node:bridge.nodes)for(auto& d:node.shape) {
    if(d=="2048")d="64";else if(d=="768")d="32";
    else if(d=="1536")d="64";else if(d=="128")d="16";
  }
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  for(bool group:{false,true})for(unsigned bm:{16u,32u}) {
    mlir::MLIRContext context;MoeRegionOptions options;
    options.tokens=17;options.grouped=group;options.block_rows=bm;
    auto plan=BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
    ImportOptions granularity;granularity.phase_batch=1;
    granularity.gemms={{16,16,16,2,1},{16,32,16,2,1},{16,16,16,2,1}};
    auto module=TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,granularity);
    ModelDims dims{17,0,17};
    auto model=ModelDescription::FromCouplingGraph(*module,dims,"moe-floor");
    auto floor=DeriveModelDramFloor(*module,model,target,"");
    assert(floor.Evaluate(model.MetricBindings()).read_bytes==102656);
    assert(floor.Evaluate(model.MetricBindings()).write_bytes==17*64*2);
    assert(floor.Evaluate(model.MetricBindings()).flops==1705984);
    unsigned lower=0,contracts=0;
    for(auto const& [name,tensor]:floor.tensors) {
      if(tensor.expected_read_elements) {
        assert(tensor.cardinality_kind=="lower_bound" && tensor.reads.empty() && tensor.writes.empty());++lower;
      }
      if(tensor.binding_producer) {
        assert(tensor.writes.empty() && tensor.no_producer.empty() && !tensor.output);++contracts;
      }
    }
    assert(lower==2 && contracts>=4);
    auto graph=InstantiateModelTasks(model,{{16,16,16,2,1},{16,32,16,2,1},{16,16,16,2,1}});
    for(auto const& sem:model.task_semantics) {
      auto g=model.stages.at(sem.stage).gemm;
      if(g<=0 || !IsGemmStage(model.stages.at(sem.stage).kind))continue;
      GemmConfig geometry=g==1?GemmConfig{16,32,16,2,1}:GemmConfig{16,16,16,2,1};
      auto input=DeriveModelTaskInput(model,sem,graph,&geometry);
      BindTaskDramProvenance(input,sem,floor,model.MetricBindings(),true,&model);
      assert(input.physical_read_bytes && input.physical_write_bytes && input.produced_live_bytes>0);
    }
    MoeRoutingProfile profile;profile.profile_id=std::string(64,'a');profile.layers.resize(1);
    MoeRoutingPoint point;point.tokens=17;point.experts=16;point.top_k=8;point.windows=2;
    point.expected_distinct_experts=12;
    for(unsigned e=0;e<16;++e)point.tokens_per_expert.push_back(e<8?
        std::map<std::uint32_t,std::uint64_t>{{17,1},{9,1}}:
        std::map<std::uint32_t,std::uint64_t>{{0,1},{8,1}});
    for(unsigned block:{16u,32u,64u,128u}) {
      auto& histograms=point.virtual_row_histograms[block];histograms.resize(point.GroupCapacity(block));
      for(auto counts:{std::vector<unsigned>{17,17,17,17,17,17,17,17,0,0,0,0,0,0,0,0},
                      std::vector<unsigned>{9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8}}) {
        unsigned v=0;for(auto count:counts)while(count) {
          unsigned rows=std::min(block,count);++histograms.at(v++)[rows];count-=rows;
        }
      }
    }
    profile.layers[0].emplace(17,point);
    for(auto const& stage:plan.stages)if(stage.kind==PlanTaskKind::kMoETopK || stage.kind==PlanTaskKind::kMoECombine) {
      auto phases=codegen::DmMoeScalarTaskDataflow(stage.moe,17);
      auto [depth,barriers]=phases.MemoryDepthAndBarriers(128);
      assert(depth>0);
      if(stage.kind==PlanTaskKind::kMoECombine)assert(barriers==2);
      else {assert(stage.moe.step==codegen::DmMoeStep::kSelectAndDispatch);assert(barriers==(group?25:2));}
    }
    auto expected=DeriveModelDramFloor(*module,model,target,"",&profile);
    assert(expected.Evaluate(model.MetricBindings()).read_bytes==151808);
    assert(expected.Evaluate(model.MetricBindings()).flops==1705984);
    if(group && bm==16) {
      if(argc==3 && std::string(argv[1])=="--emit-probe") {
        std::ofstream out(argv[2]);out<<MoeRegionNonGemmProbeSource(plan);assert(out);
      }
      DramFloorOptions contracts;contracts.dram_gbps=1000;contracts.tc_gflops=100000;
      contracts.element_bytes=model.buffer_element_bytes;
      contracts.outputs=model.exported_tensors;contracts.infer_leaf_outputs=false;
      BindMoeDramInputs(*module,model,contracts,&profile);
      SemanticGraph semantics;for(auto const& sem:model.task_semantics)semantics.ops.push_back(sem.op);
      auto reject=[&](DramFloorOptions const& bad) {
        bool caught=false;try{(void)DeriveDramFloor(semantics,bad,model.MetricBindings());}
        catch(std::exception const&){caught=true;}assert(caught);
      };
      auto bad=contracts;auto name=bad.binding_internal.begin()->first;
      bad.binding_internal[name].writers.clear();reject(bad);
      bad=contracts;bad.binding_internal[name].source.clear();reject(bad);
      bad=contracts;bad.binding_internal[name].envelope=CouplingRelation{};reject(bad);
      bad=contracts;bad.outputs.insert(name);reject(bad);
      bad=contracts;bad.indirect_write_images[name]=bad.binding_internal[name].envelope;reject(bad);
      bad=contracts;bad.binding_internal["unused"]=bad.binding_internal[name];reject(bad);
      bad=contracts;bad.active_runtime_rows.begin()->second=QuasiPolynomial::Constant(1000000);reject(bad);
      bad=contracts;bad.active_runtime_rows["unused"]=QuasiPolynomial::Constant(1);reject(bad);
      auto invalid=profile;invalid.layers[0].at(17).tokens_per_expert[0][17]=2;
      bool caught=false;try{(void)DeriveModelDramFloor(*module,model,target,"",&invalid);}
      catch(std::invalid_argument const&){caught=true;}assert(caught);
      SkeletonSearchOptions search;search.common.placement.target=target;
      search.common.placement.target.res.num_sms=2;
      search.common.placement.dims=dims;search.moe_routing_profile=std::make_shared<MoeRoutingProfile const>(profile);search.seed={16,32,16,2,1};
      search.common.geometry_domain={search.seed};search.passes=1;search.top_m=1;
      search.search_only=false;search.common.query_residency=[](auto,int){return 1;};search.artifact_prefix=(std::filesystem::temp_directory_path()/"tilemega-moe-dram-search").string();
      search.variant_probe=[](auto const&,auto const* geometry,auto) {
        return VariantResources{32,geometry?DmServingBF16SmemBytes(geometry->tile_m,
            geometry->tile_n,geometry->tile_k,geometry->stages):2560,128,false};
      };
      auto imported=TorchExportImporter{}.ImportSemantics(path,plan,context);
      std::ostringstream evidence;
      auto solved=SolveSkeletonImported(imported,context,search,nullptr,evidence);
      assert(solved.compiled.module);
      auto attribution=(*solved.compiled.module)->getAttrOfType<mlir::StringAttr>("tilemega.moe_profile_pricing");
      assert(attribution && attribution.getValue()=="occupancy;empty=capacity_surrogate_inferred");
      bool valid=false;
      for(auto const& candidate:solved.evaluated) {
        if(!candidate.error.empty())std::cerr<<candidate.key<<": "<<candidate.error<<'\n';
        assert(candidate.error.empty() && std::isfinite(candidate.score));
        valid=true;
      }
      if(!valid)std::cerr<<evidence.str();
      assert(valid);
      std::cout<<"MoE production skeleton flow receives binding DRAM provenance PASS\n";
      std::cout<<"MoE binding contract malformed source/capacity/output/image/rows/profile rejection PASS\n";
    }
    std::cout<<"MoE floor: group="<<group<<" BM="<<bm
        <<" binding provenance, exact assignment FLOPs and rational profile mean PASS\n";
  }
  return 0;
}
}
