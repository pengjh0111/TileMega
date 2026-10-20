#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/AscendImplementation.h>
#include <tilemega/Solver/AscendPlan.h>
#include <tilemega/Solver/ImplementationContract.h>
#include <tilemega/Codegen/AscendCodegen.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

static void Write(std::filesystem::path const& path,std::string const& value) {
  std::ofstream out(path);out<<value<<'\n';if(!out)throw std::runtime_error("Cannot write "+path.string());
}
int main(int argc,char** argv) {
  try {
    if(argc!=4 && argc!=5)throw std::invalid_argument("usage: as1_compile_model prepared_directory output_directory groups [epochs]");
    namespace fs=std::filesystem;
    auto root=fs::path("/home/d50064638/junhuip/TileMega/runs/as1");
    auto input=fs::weakly_canonical(argv[1]),out=fs::weakly_canonical(argv[2]);
    auto below=[&](fs::path const& p){auto relative=p.lexically_relative(root);return !relative.empty() && *relative.begin()!=".." && p!=root;};
    if(!below(input)||!below(out)||fs::exists(out))throw std::invalid_argument("Use existing prepared inputs and a new controlled output directory");
    auto config=tilemega::json::ParseFile((input/"config.json").string());
    auto const& shape=config.At("shape_mnk").AsArray("shape_mnk");
    if(config.At("test_id").AsString("test_id")!="E-0" || shape.size()!=3)
      throw std::invalid_argument("Current codegen admits only the verified E-0 body");
    int m=shape[0].AsNumber("M"),groups=std::stoi(argv[3]);
    auto target=tilemega::TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
    auto workers=tilemega::ascend::WorkerMap::Build(*target.ascend_capability,tilemega::ascend::LaunchMode::kMix1_2,groups);
    tilemega::solver::CandidateGenerator candidates(target,tilemega::solver::ScalarType::kBF16);
    auto fixed=candidates.Enumerate();
    if(fixed.size()!=1)throw std::invalid_argument("Expected one fixed legal implementation");
    auto const& traits=fixed[0].traits();
    tilemega::analysis::IslContext isl;
    mlir::MLIRContext context;
    auto bridge=tilemega::frontend::ReadExportBridge((input/"export.json").string());
    tilemega::frontend::DnnPlanOptions dnn;dnn.batch=m;
    auto model_plan=tilemega::frontend::BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,dnn);
    tilemega::frontend::ImportOptions options;
    for(std::size_t i=0;i<model_plan.gemms.size();++i)
      options.gemms.push_back({traits.tile_m,traits.tile_n,traits.tile_k,traits.stages,1,0});
    tilemega::frontend::ImportSummary summary;
    auto start=std::chrono::steady_clock::now();
    auto module=tilemega::frontend::TorchExportImporter{}.ImportPlan((input/"export.json").string(),model_plan,context,&summary,options);
    if(!module || mlir::failed(mlir::verify(*module)))throw std::runtime_error("Importer produced invalid CG");
    fs::create_directories(out);
    std::string ctg;{llvm::raw_string_ostream os(ctg);module->print(os);}Write(out/"ctg.mlir",ctg);
    auto imported=std::chrono::steady_clock::now();
    tilemega::solver::ModelDims dims;dims.seq=model_plan.serving_seq;dims.total=dims.seq;dims.batch=m;
    auto problem=tilemega::solver::PrepareSymbolicProblem(*module,target,dims,groups,1,1,nullptr,false,128);
    auto graph=tilemega::solver::InstantiateModelTasks(problem.model,problem.geometry);
    unsigned checked_contracts=0;
    for(auto const& node:graph.nodes)if(node.kind==tilemega::analysis::OperatorKind::kMatmul) {
      if(node.output.axes.size()!=2 || node.operands.size()!=2)
        throw std::invalid_argument("Native GEMM requires two matrix operands");
      auto k=problem.model.gemms.at(0).k;
      auto n=problem.model.gemms.at(0).n;
      tilemega::solver::ImplementationContract contract;
      contract.name=traits.ascend_implementation->impl_id;contract.task=node.name;
      contract.backend="catlass_950_block_mmad";contract.arch_tag="ascend_3510";
      contract.ascend_implementation=traits.ascend_implementation;
      contract.tile_m=traits.tile_m;contract.tile_n=traits.tile_n;contract.tile_k=traits.tile_k;contract.stages=traits.stages;
      if(!node.element_access)throw std::invalid_argument("Native GEMM requires the exact task-element contract");
      auto const& access=*node.element_access;auto const& sem=access.semantic;
      auto known=problem.model.MetricBindings();
      std::vector<tilemega::analysis::CouplingRelation> derived;
      for(auto const& operand:sem.operands) {
        tilemega::analysis::CouplingRelation read;
        if(sem.element_reads.empty())read=tilemega::analysis::ProjectTaskRead(
            sem,node,access.partition,operand.tensor,operand.map,{},known);
        else for(auto const& physical:sem.element_reads)if(physical.tensor.name==operand.tensor.name)
          read=read.Union(tilemega::analysis::ProjectTaskRead(sem,node,access.partition,
              physical.tensor,physical.map,physical.nonnegative,known));
        derived.push_back(read.BindParams(known));
      }
      derived.push_back(tilemega::analysis::ProjectTaskWrite(sem,node,access.partition,
          sem.result,sem.result_map,{},known).BindParams(known));
      // The body contract is independently declared in logical coordinates;
      // K padding and NK→KN packing affect physical storage only.
      std::string domain="0<=tm<"+std::to_string((m+31)/32)+" and 0<=tn<"+std::to_string((n+63)/64);
      std::string rows="32*tm<=i<32*tm+32 and i<"+std::to_string(m);
      std::string cols="64*tn<=j<64*tn+64 and j<"+std::to_string(n);
      for(auto const& text:{"{ [tm,tn] -> [i,r] : "+domain+" and "+rows+" and 0<=r<"+std::to_string(k)+" }",
          "{ [tm,tn] -> [j,r] : "+domain+" and "+cols+" and 0<=r<"+std::to_string(k)+" }",
          "{ [tm,tn] -> [i,j] : "+domain+" and "+rows+" and "+cols+" }"})
        contract.exact_operands.push_back(tilemega::analysis::CouplingRelation::FromIslText(text));
      std::string reason;
      if(!tilemega::solver::VerifyTraits(contract,&reason) || !tilemega::solver::VerifyTarget(contract,target,&reason) ||
         !tilemega::solver::VerifyExactAccess(contract,derived,&reason))
        throw std::invalid_argument("Native logical operand contract: "+reason);
      ++checked_contracts;
    }
    if(checked_contracts!=problem.model.gemms.size())throw std::invalid_argument("Incomplete native implementation contract coverage");
    tilemega::analysis::CouplingCache cache;
    auto skeleton=tilemega::solver::BuildPlanSkeleton(problem,groups,1,1,false,cache,nullptr,nullptr,true);
    std::vector<tilemega::ascend::ImplementationRole> roles;
    std::vector<tilemega::ascend::TaskImplementation> implementations;
    for(auto const& stage:problem.projection.stages) {
      if(stage.combine || problem.model.stages.at(stage.logical_stage).kind!=tilemega::solver::StageKind::kGemm)
        throw std::invalid_argument("Unimplemented native TaskKind rejected before materialization");
      roles.push_back(traits.ascend_implementation->required_role);
      implementations.push_back(*traits.ascend_implementation);
    }
    skeleton.BindAscendWorkers(workers,roles);
    auto plan=tilemega::solver::MaterializeAscendPlan(problem,skeleton,*target.ascend_capability,implementations,
        argc==5?std::stoul(argv[4]):1);
    tilemega::solver::WritePlanSkeleton(*module,skeleton);
    if(mlir::failed(mlir::verify(*module)))throw std::runtime_error("Solved CG verification failed");
    std::string solved;{llvm::raw_string_ostream os(solved);module->print(os);}Write(out/"solved.mlir",solved);
    Write(out/"plan.json",plan.ToJson());
    Write(out/"task_domain.isl",problem.projection.tasks.ToString());
    Write(out/"dependencies.isl",problem.projection.dependencies.ToString());
    auto solved_time=std::chrono::steady_clock::now();
    Write(out/"kernel.cpp",tilemega::codegen::LowerAscend(*module,problem,plan));
    Write(out/"CMakeLists.txt",std::string("cmake_minimum_required(VERSION 3.22)\nfind_package(ASC REQUIRED)\n")+
      "project(TileMegaGeneratedAscend LANGUAGES ASC CXX)\nset(CMAKE_CXX_STANDARD 17)\n"+
      "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\nset(TILEMEGA_ROOT \""+TILEMEGA_SOURCE_DIR+"\")\n"+
      "set(CATLASS_ROOT \"/home/d50064638/junhuip/tilemega-as1/third_party/catlass\")\n"+
      "include(\"${TILEMEGA_ROOT}/cmake/AscendKernel.cmake\")\n"+
      "as1_asc_target(tilemega_as1_generated \"${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp\")\n");
    auto generated=std::chrono::steady_clock::now();
    auto seconds=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
    tilemega::json::Value receipt(tilemega::json::Object{{"status","generated_not_executed"},
      {"importer","TorchExportImporter::ImportPlan/BuildDnnModelPlan"},
      {"solver","PrepareSymbolicProblem/BuildPlanSkeleton/MaterializeAscendPlan"},
      {"import_seconds",seconds(start,imported)},{"solver_seconds",seconds(imported,solved_time)},
      {"codegen_seconds",seconds(solved_time,generated)},{"task_spaces",int(summary.task_spaces)},
      {"couplings",int(summary.couplings)},{"tasks",int(plan.tasks.size())},
      {"launch_groups",groups},{"cost_status","not_calibrated"}});
    receipt.Set("verified_logical_operand_contracts",int(checked_contracts));
    Write(out/"pipeline.json",receipt.Dump());std::cout<<receipt.Dump()<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
