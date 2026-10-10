// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <sstream>

namespace tilemega::tests::dnn_memory_codegen_test {
int TestDnnMemoryCodegen(int argc,char** argv) {
  using namespace frontend;using namespace codegen;
  analysis::IslContext isl;ModelPlan plan;
  plan.dm=plan.forward=true;plan.dtype="bf16";plan.serving_seq=7;
  plan.memory_reuse="greedy";
  llvm::json::Array nodes,inputs;
  auto node=[&](std::string name,char const* op,std::vector<std::string> shape) {
    llvm::json::Array dimensions;for(auto const& extent:shape)dimensions.push_back(extent);
    nodes.push_back(llvm::json::Object{{"index",int(nodes.size())},{"name",name},{"op",op},
        {"target",std::string(op)=="placeholder"?name:"aten.linear.default"},
        {"inputs",llvm::json::Array{}},{"shape",std::move(dimensions)},{"dtype","torch.bfloat16"}});
  };
  auto matrix=[&](std::string name,unsigned width,bool external) {
    PlanBuffer b;b.name=name;b.per_batch=7*width;b.layout.rank=2;
    b.layout.logical[0]=b.layout.physical[0]=14;b.layout.logical[1]=b.layout.physical[1]=width;
    b.layout.strides[0]=width;b.layout.strides[1]=1;
    if(external){b.role="external";b.external_name=name;}
    unsigned id=plan.buffers.size();plan.buffers.push_back(b);plan.node_buffer[name]=id;return id;
  };
  auto input=matrix("reuse_input",64,true);node("reuse_input","placeholder",{"B","7","64"});
  inputs.push_back(llvm::json::Object{{"name","reuse_input"},{"kind","USER_INPUT"},{"target","reuse_input"}});
  unsigned previous=input,k=64;
  std::vector<GemmGranularity> geometry;
  for(unsigned index=0;index<7;++index) {
    // Alternating live allocations also leave a large/small/large slot history.
    unsigned n=index%3==1?32:64;
    std::string name=index==6?"reuse_output":"reuse_value"+std::to_string(index);
    auto output=matrix(name,n,index==6);
    PlanBuffer weight;weight.name=weight.external_name="reuse_weight"+std::to_string(index);
    weight.constant=n*k;weight.role="external";weight.source=PlanBuffer::Source::kWeight;
    weight.pack_json=llvm::formatv("{0}",llvm::json::Value(llvm::json::Object{
        {"kind","alias"},{"source",weight.name}})).str();
    auto w=unsigned(plan.buffers.size());plan.buffers.push_back(weight);
    node(weight.name,"placeholder",{std::to_string(n),std::to_string(k)});
    inputs.push_back(llvm::json::Object{{"name",weight.name},{"kind","PARAMETER"},{"target",weight.name}});
    PlanGemm gemm;gemm.n=n;gemm.k=k;gemm.a=previous;gemm.b=w;gemm.c=gemm.d=output;
    gemm.access.rows_per_batch=7;gemm.access.write.layout=output;
    gemm.chain.count=1;gemm.chain.operations[0].kind=DmEpilogueKind::kActivation;
    gemm.chain.operations[0].activation=DmActivation::kRelu;
    PlanStage stage;stage.gemm=index;stage.operands.fill(kDmNoIndex);
    stage.representative=name;stage.representative_index=nodes.size();
    node(name,"call_function",{"B","7",std::to_string(n)});
    plan.gemms.push_back(gemm);plan.stages.push_back(stage);
    geometry.push_back({16,16,16,2,1});previous=output;k=n;
  }
  plan.outputs={{previous,{}}};
  llvm::json::Value bridge=llvm::json::Object{{"schema","tilemega.exported_program.v1"},
      {"guards",llvm::json::Array{}},{"range_constraints",llvm::json::Object{{"B","VR[1, 64]"}}},
      {"nodes",std::move(nodes)},{"signature",llvm::json::Object{{"inputs",std::move(inputs)},
      {"outputs",llvm::json::Array{llvm::json::Object{{"name","reuse_output"},{"kind","USER_OUTPUT"}}}}}}};
  if((argc!=3 && argc!=4) || std::string(argv[1])!="--emit" ||
      (argc==4 && std::string(argv[3])!="--search"))
    throw std::invalid_argument("dnn_memory_codegen requires --emit OUTPUT.cu");
  auto path=std::string(argv[2])+".bridge.json";
  {std::ofstream stream(path);stream<<llvm::formatv("{0:2}",bridge).str();}
  mlir::MLIRContext context;ImportOptions options;options.phase_batch=2;options.gemms=geometry;
  auto module=TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,options);
  auto stored=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  assert(stored.getAs<mlir::IntegerAttr>("dm_memory_arena_bytes").getInt()>0);
  solver::ModelDims dims{7,0,7};dims.batch=2;
  auto physical_model=solver::ModelDescription::FromCouplingGraph(*module,dims,"reuse-storage");
  std::uint64_t expected_bytes=physical_model.memory_arena_bytes;
  for(auto item:stored.getAs<mlir::ArrayAttr>("buffers")) {
    auto b=llvm::cast<mlir::DictionaryAttr>(item);
    if(b.get("dm_arena_offset"))continue;
    auto count=[&](char const* name){return b.getAs<mlir::IntegerAttr>(name).getInt();};
    auto name=b.getAs<mlir::StringAttr>("name").getValue().str();
    expected_bytes+=(count("constant")+7*count("per_seq")+2*count("per_batch"))*
        physical_model.buffer_element_bytes.at(name);
  }
  assert(physical_model.PhysicalFootprintBytes()==expected_bytes &&
         physical_model.LiveFootprintBytes()==double(expected_bytes));
  auto hazards=(*module)->getAttrOfType<mlir::IntegerAttr>("tilemega.memory_hazard_count").getInt();
  assert(hazards>0);
  auto source=CouplingGraphToCUDA{}.LowerVariants({{*module,7,7}});
  std::ofstream output(argv[2]);output<<source;assert(output);
  if(argc==4 && std::string(argv[3])=="--search") {
    solver::SkeletonSearchOptions search;
    search.common.placement.target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
        "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
    search.common.placement.target.res.num_sms=4;
    search.common.placement.dims={7,0,7};search.common.placement.dims.batch=2;
    search.seed={16,16,16,2,1};
    search.search_only=true;
    search.variant_probe=[](auto const&,auto const* g,auto) {
      return solver::VariantResources{32,g?solver::DmServingBF16SmemBytes(
          g->tile_m,g->tile_n,g->tile_k,g->stages):512,128,false};
    };
    auto imported=TorchExportImporter{}.ImportSemantics(path,plan,context);
    auto classes=solver::BuildOperatorClasses(imported);
    for(auto const& g:std::vector<solver::GemmConfig>{{16,16,16,2,1},
          {32,32,16,2,1},{16,32,16,2,2}})
      search.evaluation_cases.push_back({std::vector<solver::GemmConfig>(classes.size(),g),1,1});
    search.artifact_prefix=std::string(argv[2])+".search";
    std::ostringstream evidence;
    auto warm=solver::SolveSkeletonImported(imported,context,search,nullptr,evidence);
    assert(warm.evaluated.size()==search.evaluation_cases.size());
    for(std::size_t i=0;i<search.evaluation_cases.size();++i) {
      auto cold_options=search;cold_options.evaluation_cases={search.evaluation_cases[i]};
      cold_options.artifact_prefix+=".cold"+std::to_string(i);
      auto cold=solver::SolveSkeletonImported(imported,context,cold_options,nullptr,evidence);
      if(!warm.evaluated[i].error.empty() || !cold.evaluated[0].error.empty())
        std::cerr<<"reuse case "<<i<<" warm="<<warm.evaluated[i].error
                 <<" cold="<<cold.evaluated[0].error<<'\n';
      assert(warm.evaluated[i].error.empty() && cold.evaluated[0].error.empty());
      assert(std::isfinite(warm.evaluated[i].score));
      assert(std::abs(warm.evaluated[i].score-cold.evaluated[0].score)<1e-9);
    }
    auto base=solver::PrepareSymbolicProblem(*module,search.common.placement.target,
        search.common.placement.dims,4,1,1,nullptr,false);
    assert(base.model.storage_reuse);
    analysis::CouplingCache cache;bool rejected=false;
    try {solver::PrepareFlowStructure(base,base.geometry,4,1,cache);}
    catch(std::invalid_argument const& e) {
      rejected=std::string(e.what()).find("storage reuse")!=std::string::npos;
    }
    assert(rejected);
    auto split_options=options;
    for(auto& geometry:split_options.gemms)geometry={16,32,16,2,2};
    auto split=TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,split_options);
    auto runtime=ReadRuntimePlan(*split);
    assert(std::any_of(runtime.dependencies.begin(),runtime.dependencies.end(),
        [](auto const& edge){return edge.producer_main && edge.consumer_done;}));
    auto split_source=CouplingGraphToCUDA{}.LowerVariants({{*split,7,7}});
    std::ofstream(std::string(argv[2])+".split.cu")<<split_source;
    auto bad=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(split->clone()));
    bool corrupted=false;
    for(auto edge:bad->getOps<dialect::CouplingOp>()) {
      dialect::TileSpaceOp src,dst;
      for(auto task:bad->getOps<dialect::TileSpaceOp>()) {
        if(task.getSymName()==edge.getSrc())src=task;
        if(task.getSymName()==edge.getDst())dst=task;
      }
      auto name=src->getAttrOfType<mlir::StringAttr>("operator_name");
      if(src.getStage()<dst.getStage() && name && name.getValue().ends_with(".combine")) {
        edge->setAttr("dependency_producer_main",mlir::BoolAttr::get(&context,true));
        corrupted=true;break;
      }
    }
    assert(corrupted);rejected=false;
    try {(void)ReadRuntimePlan(*bad);}
    catch(std::invalid_argument const& e) {
      rejected=std::string(e.what()).find("endpoint disagrees")!=std::string::npos;
    }
    assert(rejected);
    std::cout<<"Reuse search: warm/cold geometry, split-K and RAW-only rejection PASS\n";
  }
  std::cout<<"Reusable storage generated CUDA: "<<hazards<<" exact anti-dependencies PASS\n";
  return 0;
}
}
