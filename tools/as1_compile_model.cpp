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
#include <algorithm>
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
    if(argc<4 || argc>6)throw std::invalid_argument("usage: as1_compile_model prepared_directory output_directory groups [epochs] [cross_group|stage_barrier]");
    namespace fs=std::filesystem;
    auto root=fs::path("/home/d50064638/junhuip/TileMega/runs/as1");
    auto input=fs::weakly_canonical(argv[1]),out=fs::weakly_canonical(argv[2]);
    auto below=[&](fs::path const& p){auto relative=p.lexically_relative(root);return !relative.empty() && *relative.begin()!=".." && p!=root;};
    if(!below(input)||!below(out)||fs::exists(out))throw std::invalid_argument("Use existing prepared inputs and a new controlled output directory");
    auto config=tilemega::json::ParseFile((input/"config.json").string());
    auto test=config.At("test_id").AsString("test_id");
    if(test!="E-0" && test!="E-1" && test!="E-2")throw std::invalid_argument("Current codegen admits the fixed AS-1 workloads only");
    int m=test=="E-0"?config.At("shape_mnk").AsArray("shape_mnk").at(0).AsNumber("M"):
        config.At("batch").AsNumber("batch");
    int groups=std::stoi(argv[3]);bool cross=argc==6 && std::string(argv[5])=="cross_group";
    bool barrier=argc==6 && std::string(argv[5])=="stage_barrier";
    if((argc==6 && !cross && !barrier) || (cross && (groups!=2 || test!="E-1")) ||
       (barrier && (groups!=2 || test!="E-2")))
      throw std::invalid_argument("Invalid fixed placement/executor for this workload");
    auto target=tilemega::TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
    auto workers=tilemega::ascend::WorkerMap::Build(*target.ascend_capability,tilemega::ascend::LaunchMode::kMix1_2,groups);
    tilemega::solver::CandidateGenerator candidates(target,tilemega::solver::ScalarType::kBF16);
    auto fixed=candidates.Enumerate();
    if(fixed.size()!=1)throw std::invalid_argument("Expected one fixed legal implementation");
    auto const& traits=fixed[0].traits();
    tilemega::analysis::IslContext isl;
    mlir::MLIRContext context;
    auto bridge=tilemega::frontend::ReadExportBridge((input/"export.json").string());
    tilemega::frontend::DnnPlanOptions dnn;dnn.batch=m;dnn.materialize_pointwise=test!="E-0";
    if(test=="E-2") {dnn.layernorm_rows=1;dnn.attention_query_rows=1;}
    auto model_plan=tilemega::frontend::BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,dnn);
    tilemega::frontend::ImportOptions options;
    options.phase_batch=m;
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
      auto origin=std::find_if(problem.model.task_semantics.begin(),problem.model.task_semantics.end(),
          [&](auto const& x){return x.op.name==node.name;});
      if(origin==problem.model.task_semantics.end())throw std::invalid_argument("GEMM lacks its semantic stage");
      auto gi=problem.model.stages.at(origin->stage).gemm;
      auto k=problem.model.gemms.at(gi).k;
      auto n=problem.model.gemms.at(gi).n;
      auto matrix_rows=problem.model.gemm_access.at(gi).rows_per_batch*m;
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
      std::string domain="0<=tm<"+std::to_string((matrix_rows+31)/32)+" and 0<=tn<"+std::to_string((n+63)/64);
      std::string rows="32*tm<=i<32*tm+32 and i<"+std::to_string(matrix_rows);
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
    unsigned checked_aiv_contracts=0;
    for(auto const& node:graph.nodes)if(node.kind!=tilemega::analysis::OperatorKind::kMatmul) {
      auto origin=std::find_if(problem.model.task_semantics.begin(),problem.model.task_semantics.end(),
          [&](auto const& x){return x.op.name==node.name;});
      if(origin==problem.model.task_semantics.end() || !node.element_access)
        throw std::invalid_argument("AIV body lacks exact semantic access");
      auto kind=problem.model.stages.at(origin->stage).kind;
      auto const& access=*node.element_access;auto const& sem=access.semantic;
      std::vector<std::string> expected;
      auto const& stage=problem.model.stages.at(origin->stage);
      unsigned rows=(stage.rows_per_batch?stage.rows_per_batch:1)*m,width=stage.width;
      using Kind=tilemega::solver::StageKind;
      if(kind==Kind::kRMSNorm || kind==Kind::kElementwise || kind==Kind::kLayerNorm) {
        bool norm=kind!=Kind::kElementwise;
        if((norm && sem.arithmetic!=(kind==Kind::kRMSNorm?"rmsnorm":"layernorm")) ||
           (!norm && sem.arithmetic!="silu" && sem.arithmetic!="dm_gelu_erf") || (kind==Kind::kLayerNorm && stage.group!=1))
          throw std::invalid_argument("AIV mathematical operation or row ownership mismatch");
        auto matrix="{ [tm] -> [i,r] : 0<=tm<"+std::to_string(rows)+" and i=tm and 0<=r<"+std::to_string(width)+" }";
        expected.push_back(matrix);
        if(norm)expected.push_back("{ [tm] -> [r] : 0<=tm<"+std::to_string(rows)+" and 0<=r<"+std::to_string(width)+" }");
        if(kind==Kind::kLayerNorm)expected.push_back(expected.back());
        expected.push_back(matrix);
      }else if(kind==Kind::kAdd) {
        if(sem.arithmetic!="add")throw std::invalid_argument("Residual mathematical operation mismatch");
        auto matrix="{ [tm,tn] -> [i,j] : 0<=tm<"+std::to_string((rows+31)/32)+" and 0<=tn<"+
            std::to_string((width+63)/64)+" and 32*tm<=i<32*tm+32 and i<"+
            std::to_string(rows)+" and 64*tn<=j<64*tn+64 and j<"+std::to_string(width)+" }";
        expected={matrix,matrix,matrix};
      }else if(kind==Kind::kEncoderAttention) {
        if(test!="E-2" || m!=1 || stage.width!=16 || stage.extent!=4 || stage.group!=1 ||
            sem.arithmetic!="encoder_attention" || sem.operands.size()!=4)
          throw std::invalid_argument("Attention is outside the fixed S16/H256/mask contract");
        expected={
          "{ [q] -> [i,r] : 0<=q<64 and i=q%16 and 192*floor(q/16)<=r<192*floor(q/16)+64 }",
          "{ [q] -> [i,r] : 0<=q<64 and 0<=i<16 and 192*floor(q/16)+64<=r<192*floor(q/16)+128 }",
          "{ [q] -> [i,r] : 0<=q<64 and 0<=i<16 and 192*floor(q/16)+128<=r<192*floor(q/16)+192 }",
          "{ [q] -> [i,j] : 0<=q<64 and i=0 and 0<=j<16 }",
          "{ [q] -> [i,r] : 0<=q<64 and i=q%16 and 64*floor(q/16)<=r<64*floor(q/16)+64 }"};
      }else throw std::invalid_argument("Unknown AIV semantic body");
      std::vector<tilemega::analysis::CouplingRelation> derived;
      auto known=problem.model.MetricBindings();
      for(auto const& operand:sem.operands)derived.push_back(tilemega::analysis::ProjectTaskRead(
          sem,node,access.partition,operand.tensor,operand.map,{},known).BindParams(known));
      derived.push_back(tilemega::analysis::ProjectTaskWrite(sem,node,access.partition,
          sem.result,sem.result_map,{},known).BindParams(known));
      tilemega::solver::ImplementationContract contract;contract.name=sem.arithmetic;
      for(auto const& relation:expected)contract.exact_operands.push_back(tilemega::analysis::CouplingRelation::FromIslText(relation));
      std::string reason;
      if(!tilemega::solver::VerifyExactAccess(contract,derived,&reason))throw std::invalid_argument(reason);
      ++checked_aiv_contracts;
    }
    tilemega::analysis::CouplingCache cache;
    auto skeleton=tilemega::solver::BuildPlanSkeleton(problem,groups,1,1,false,cache,nullptr,nullptr,true);
    std::vector<tilemega::ascend::ImplementationRole> roles;
    std::vector<tilemega::ascend::TaskImplementation> implementations;
    for(auto const& stage:problem.projection.stages) {
      if(stage.combine)throw std::invalid_argument("Unimplemented native combine rejected before materialization");
      auto kind=problem.model.stages.at(stage.logical_stage).kind;
      auto impl=kind==tilemega::solver::StageKind::kGemm?*traits.ascend_implementation:
          (test=="E-2"?tilemega::solver::As1EncoderImplementation(static_cast<tilemega::codegen::TaskKind>(kind),target):
            tilemega::solver::As1AivImplementation(static_cast<tilemega::codegen::TaskKind>(kind),target));
      roles.push_back(impl.required_role);implementations.push_back(std::move(impl));
    }
    skeleton.BindAscendWorkers(workers,roles);
    if(cross) {
      if(skeleton.spaces.size()!=5)throw std::invalid_argument("Cross-group MLP requires exactly five stages");
      for(unsigned s=0;s<skeleton.spaces.size();++s) {
        auto& domain=skeleton.spaces[s].legal_workers;
        int group=(s==2 || s==3)?1:0;
        domain.erase(std::remove_if(domain.begin(),domain.end(),[&](int w){return workers.At(w).group!=group;}),domain.end());
        if(domain.empty())throw std::invalid_argument("Fixed placement has no legal worker");
      }
    }
    auto plan=tilemega::solver::MaterializeAscendPlan(problem,skeleton,*target.ascend_capability,implementations,
        argc>=5?std::stoul(argv[4]):1);
    if(barrier)plan.SelectStageBarrier();
    tilemega::solver::WritePlanSkeleton(*module,skeleton);
    if(mlir::failed(mlir::verify(*module)))throw std::runtime_error("Solved CG verification failed");
    std::string solved;{llvm::raw_string_ostream os(solved);module->print(os);}Write(out/"solved.mlir",solved);
    Write(out/"plan.json",plan.ToJson());
    tilemega::json::Array stage_buffers;
    for(unsigned i=0;i<model_plan.stages.size();++i) {
      auto const& s=model_plan.stages[i];unsigned buffer;
      using K=tilemega::frontend::PlanTaskKind;
      if(s.kind==K::kGemm)buffer=model_plan.gemms.at(s.gemm).d;
      else if(s.kind==K::kEncoderAttention || s.kind==K::kElementwise)buffer=s.operands[1];
      else if(s.kind==K::kLayerNorm)buffer=s.operands[3];
      else if(s.kind==K::kRMSNorm || s.kind==K::kAdd)buffer=s.operands[2];
      else throw std::invalid_argument("Unsupported stage buffer mapping");
      stage_buffers.emplace_back(tilemega::json::Object{{"stage",int(i)},{"output_buffer",int(buffer)},
          {"name",model_plan.buffers.at(buffer).name}});
    }
    Write(out/"stage_buffers.json",tilemega::json::Value(std::move(stage_buffers)).Dump());
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
    receipt.Set("verified_aiv_operand_contracts",int(checked_aiv_contracts));
    Write(out/"pipeline.json",receipt.Dump());std::cout<<receipt.Dump()<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
