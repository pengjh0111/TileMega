// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeStructureSearch.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/DnnResourceProbe.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace tilemega::tests::moe_structure_search_test {
int TestMoeStructureSearch(int argc,char** argv) {
  using namespace frontend;using namespace solver;
  analysis::IslContext isl;mlir::MLIRContext context;
  auto path=std::string(TILEMEGA_SOURCE_DIR)+"/test/fixtures/moe/region_before.json";
  auto bridge=ReadExportBridge(path);
  for(auto& node:bridge.nodes)for(auto& d:node.shape) {
    if(d=="2048")d="128";else if(d=="768")d="64";
    else if(d=="1536")d="128";else if(d=="128")d="16";
  }
  MoeRegionOptions options;options.tokens=17;options.grouped=true;
  auto plan=BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
  auto imported=TorchExportImporter{}.ImportSemantics(path,plan,context);
  imported.bridge=bridge;
  SkeletonSearchOptions search;
  search.common.placement.target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  search.common.placement.target.res.num_sms=2;search.common.placement.dims={17,0,17};
  search.seed={16,32,32,2,1};search.common.geometry_domain={search.seed};
  search.passes=1;search.top_m=1;search.search_only=true;
  std::set<std::string> probes;
  search.dm_variant_probe=[&](auto const& model,auto const&,auto const* g,auto) {
    probes.insert(MoeRegionNonGemmProbeSource(model));
    return VariantResources{32,g?DmServingBF16SmemBytes(g->tile_m,g->tile_n,g->tile_k,g->stages):2560,128,false};
  };
  if(argc==2 && std::string(argv[1])=="--alignment") {
    search.seed={16,128,32,2,1};search.search_only=false;
    search.evaluation_cases={{{{16,32,32,2,1},{16,32,32,2,1},{16,32,32,2,1}},1,1}};
    search.common.query_residency=[](auto,int){return 1;};
    search.artifact_prefix="/tmp/tilemega-moe-multicolumn";
    std::ostringstream evidence;
    auto solved=SolveSkeletonImported(imported,context,search,nullptr,evidence);
    assert(solved.compiled.module);
    auto model=ModelDescription::FromCouplingGraph(*solved.compiled.module,{17,0,17},"multicolumn");
    assert(model.stages.back().width==32);
    std::cout<<"MoE multicolumn seed128 to selected32 counted materialization PASS\n";
    return 0;
  }
  auto directory=std::filesystem::temp_directory_path()/"tilemega-moe-binding-search";
  std::filesystem::create_directories(directory);search.artifact_prefix=(directory/"search").string();
  std::vector<MoeBindingChoice> choices{{false,1},{true,16},{true,32},{true,64},{true,128}};
  auto mma_choices=choices;for(auto choice:mma_choices){choice.gemv=true;choices.push_back(choice);}
  auto implementation_choices=choices;choices.clear();
  for(auto choice:implementation_choices)for(int mask:{0,2,4,6}) {
    choice.reduction_mask=mask;choices.push_back(choice);
  }
  std::ostringstream evidence;
  auto result=SearchMoeStructures(imported,choices,context,search,evidence);
  assert(result.evaluated.size()==40 && probes.size()>=5 && !result.budget_exhausted);
  for(auto const& entry:result.evaluated) {
    if(!entry.error.empty())std::cerr<<entry.choice.Key()<<": "<<entry.error<<'\n';
    assert(entry.error.empty() && std::isfinite(entry.score) && !entry.top.empty());
    auto rebuilt=RebuildMoeStructure(imported,entry.choice);
    assert(rebuilt.plan.moe_gemv==entry.choice.gemv);
    assert(rebuilt.plan.dm_reduction_mask==entry.choice.reduction_mask);
    unsigned experts=0;
    for(auto const& gemm:rebuilt.plan.gemms)if(gemm.access.b==codegen::DmBAccess::kExpertIndirect) {
      assert(gemm.access.block_rows==entry.choice.block_rows && gemm.access.binding_rows==17*8);++experts;
    }
    assert(experts==2);
  }
  if(argc==3 && std::string(argv[1])=="--emit") {
    search.search_only=false;search.artifact_prefix=std::string(argv[2])+".search";
    search.common.query_residency=[](auto,int){return 1;};
    auto solved=SolveMoeStructures(imported,choices,context,search,nullptr,evidence);
    assert(solved.compiled.module && (*solved.compiled.module)->hasAttr("tilemega.moe_structure"));
    auto proved=(*solved.compiled.module)->getAttrOfType<mlir::IntegerAttr>("tilemega.dm_reduction_proved_stages");
    assert(proved && proved.getInt()>=0);
    auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*solved.compiled.module,1,1}});
    assert(source.find("#define TILEMEGA_DM_MOE_LA_MASK ")!=std::string::npos);
    std::ofstream out(argv[2]);out<<source;assert(out);
  }
  std::cout<<"MoE binding search: slot and four BM families across MMA/GEMV, dispatch/combine LA coordinates, per-plan probes, geometry rescoring PASS\n";
  std::filesystem::remove_all(directory);return 0;
}
}
