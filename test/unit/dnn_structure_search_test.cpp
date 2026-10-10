// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/DnnResourceProbe.h>
#include <tilemega/Solver/DnnStructureSearch.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace tilemega::tests::dnn_structure_search_test {
int TestDnnStructureSearch(int argc,char** argv) {
  using namespace frontend;using namespace solver;
  analysis::IslContext isl;mlir::MLIRContext context;
  auto path=std::string(TILEMEGA_SOURCE_DIR)+"/test/fixtures/dnn/structure_search.json";
  auto bridge=ReadExportBridge(path);DnnPlanOptions options;
  options.batch=2;options.dwpw_fuse=options.deferred_layernorm=true;
  options.workspace_budget_bytes=98304;options.memory_l2_budget_bytes=65536;
  auto plan=BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  assert(plan.gemms.size()==5 && plan.deferred_layernorm_edges.size()==2);
  unsigned fused=0;for(auto const& stage:plan.stages)fused+=stage.kind==PlanTaskKind::kDwPwFused;
  assert(fused==2);
  plan.dm_reduction_mask=1;
  auto imported=TorchExportImporter{}.ImportSemantics(path,plan,context);
  SkeletonSearchOptions search;
  search.common.placement.target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  search.common.placement.target.res.num_sms=2;
  search.common.placement.dims={1,0,1};search.common.placement.dims.batch=2;
  search.seed={16,16,16,2,1};search.common.geometry_domain={search.seed};
  search.passes=1;search.top_m=1;search.search_only=true;
  std::set<std::string> probes;
  search.dm_variant_probe=[&](auto const& p,auto const&,auto const* g,auto) {
    // Host search checks use synthetic registers; a native probe and kernel
    // execution establish compiled resource and numerical evidence separately.
    auto text=DnnNonGemmProbeSource(p);probes.insert(text);
    auto shared=int(DnnNonGemmSharedBytes(p));
    if(g)shared=std::max(shared,DmServingBF16SmemBytes(g->tile_m,g->tile_n,g->tile_k,g->stages));
    return VariantResources{32,shared,128,false};
  };
  auto folder=std::filesystem::temp_directory_path()/"tilemega-dnn-structure-search";
  std::filesystem::create_directories(folder);search.artifact_prefix=(folder/"search").string();
  std::ostringstream evidence;
  auto result=SearchDnnStructures(imported,options,{"none","greedy","l2"},context,search,evidence);
  bool mixed_fusion=false,mixed_norm=false,c4=false,greedy=false,l2=false,la=false,no_la=false;
  std::set<std::string> keys;
  for(auto const& item:result.evaluated) {
    if(!item.error.empty())std::cerr<<item.choice.Key()<<": "<<item.error<<'\n';
    assert(item.error.empty() && std::isfinite(item.score) && !item.top.empty());
    assert(keys.insert(item.choice.Key()).second);
    la|=item.choice.reduction_mask==1;no_la|=item.choice.reduction_mask==0;
    mixed_fusion|=item.choice.fused_gemms.size()==1;
    mixed_norm|=item.choice.deferred_edges.size()==1;
    c4|=item.choice.small_channels==4;greedy|=item.choice.reuse=="greedy";l2|=item.choice.reuse=="l2";
    auto rebuilt=RebuildDnnStructure(imported,options,item.choice);
    assert(rebuilt.plan.deferred_layernorm_edges.size()==item.choice.deferred_edges.size());
  }
  assert(mixed_fusion && mixed_norm && c4 && greedy && l2 && la && no_la && probes.size()>1);
  if(argc==3 && std::string(argv[1])=="--emit") {
    search.search_only=false;search.artifact_prefix=std::string(argv[2])+".search";
    search.common.query_residency=[](auto,int){return 1;};
    auto solved=SolveDnnStructures(imported,options,{"none","greedy","l2"},context,search,nullptr,evidence);
    assert(solved.compiled.module && (*solved.compiled.module)->hasAttr("tilemega.dnn_structure"));
    auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*solved.compiled.module,1,1}});
    std::ofstream output(argv[2]);output<<source;assert(output);
    auto off=options;off.dwpw_fuse=off.deferred_layernorm=false;
    auto off_plan=BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,off);
    auto probe=std::string(argv[2])+".nongemm.cu";
    std::ofstream(probe)<<DnnNonGemmProbeSource(off_plan);
  }
  std::cout<<"DNN structure search states="<<result.evaluated.size()
      <<" independent fusion/LN edges, reuse, RGB padding PASS\n";
  std::filesystem::remove_all(folder);return 0;
}
}
