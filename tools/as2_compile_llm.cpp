#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Codegen/AscendLlmCodegen.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>

static void Write(std::filesystem::path const& path,std::string const& text) {
  std::ofstream out(path);out<<text<<'\n';if(!out)throw std::runtime_error("Cannot write "+path.string());
}
int main(int argc,char** argv) {
  try {
    bool emit=(argc==6 || argc==7) && std::string(argv[1])=="emit-diagnostic";
    bool prepare=emit || (argc==6 && std::string(argv[1])=="prepare");
    if(!prepare && (argc!=5 || std::string(argv[1])!="inspect"))
      throw std::invalid_argument("usage: as2_compile_llm inspect export new_output batch; or prepare|emit-diagnostic export new_output batch groups [diagnostic_kv_block]");
    namespace fs=std::filesystem;
    auto root=fs::path("/home/d50064638/junhuip/TileMega/runs/as2");
    auto input=fs::weakly_canonical(argv[2]),output=fs::weakly_canonical(argv[3]);
    auto controlled=[&](auto const& p){auto r=p.lexically_relative(root);return p!=root && !r.empty() && *r.begin()!="..";};
    if(!controlled(input)||!controlled(output)||fs::exists(output))
      throw std::invalid_argument("Use existing AS-2 exports and a new AS-2 output directory");
    int batch=std::stoi(argv[4]);if(batch!=1 && batch!=4 && batch!=16)throw std::invalid_argument("Unsupported batch");
    auto receipt=tilemega::json::ParseFile((input/"as2_export.json").string());
    auto phase=receipt.At("phase").AsString("phase");
    if(phase!="prefill" && phase!="decode")throw std::invalid_argument("Unknown phase");
    tilemega::frontend::ServingOptions serving;
    serving.phase=phase=="prefill"?decltype(serving.phase)::kPrefill:decltype(serving.phase)::kDecode;
    serving.seq=phase=="prefill"?64:1;serving.capacity=1088;
    serving.deferred_norm=false;serving.kv_block=128;serving.argmax_tile_n=64;
    if(emit)serving.kv_block=1088;
    if(emit && argc==7) {
      serving.kv_block=std::stoi(argv[6]);
      if(serving.kv_block!=128 && serving.kv_block!=1088)throw std::invalid_argument("Bounded native KV candidates are 128 and 1088");
    }
    auto start=std::chrono::steady_clock::now();
    tilemega::analysis::IslContext isl;
    auto bridge=tilemega::frontend::ReadExportBridge((input/"export.json").string());
    auto plan=tilemega::frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,serving);
    if(emit && phase=="prefill")for(auto& stage:plan.stages)
      if(stage.kind==tilemega::frontend::PlanTaskKind::kFusedAttention)
        stage.attention_query_rows=16*stage.group;
    tilemega::frontend::ImportOptions options;options.phase_batch=batch;
    options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=options.combiner_tile_per_block=true;
    options.gemms.assign(plan.gemms.size(),{32,64,64,2,1});
    mlir::MLIRContext context;tilemega::frontend::ImportSummary summary;
    tilemega::frontend::TorchExportImporter importer;
    auto semantic=importer.ImportSemantics((input/"export.json").string(),plan,context);
    if(!semantic.lifted.degraded.empty())throw std::invalid_argument("Serving semantics contain degraded operators");
    for(auto const& op:semantic.lifted.ops)
      if(op.role==tilemega::frontend::OpRole::kGeneric)
        throw std::invalid_argument("Generic semantics cannot certify an imported LLM");
    auto module=importer.InstantiateForGranularity(semantic,context,options,nullptr,&summary);
    if(!module || mlir::failed(mlir::verify(*module)))
      throw std::runtime_error("LLM import verification failed");
    tilemega::solver::ModelDims dims;dims.batch=batch;dims.seq=serving.seq;
    dims.past=phase=="prefill"?0:emit?1087:64;dims.total=dims.seq+dims.past;
    auto model=tilemega::solver::ModelDescription::FromCouplingGraph(*module,dims,"as2-llm-inspection");
    fs::create_directories(output);
    std::string ir;{llvm::raw_string_ostream stream(ir);module->print(stream);}Write(output/"ctg.mlir",ir);
    tilemega::json::Array stages;
    tilemega::json::Array raw_unclassified;
    for(auto const& name:bridge.unsupported)raw_unclassified.emplace_back(name);
    for(unsigned i=0;i<model.stages.size();++i)
      stages.emplace_back(tilemega::json::Object{{"stage",int(i)},{"kind",int(model.stages[i].kind)}});
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    tilemega::json::Value result(tilemega::json::Object{
      {"status","verified_import_only"},{"model_id",receipt.At("model_id").AsString("model_id")},
      {"config_sha256",receipt.At("config_sha256").AsString("config_sha256")},
      {"phase",phase},{"batch",batch},{"seq",dims.seq},{"past",dims.past},{"capacity",serving.capacity},
      {"stages",std::move(stages)},{"gemms",int(model.gemms.size())},
      {"task_spaces",int(summary.task_spaces)},{"couplings",int(summary.couplings)},
      {"buffers",int(plan.buffers.size())},{"import_seconds",seconds},
      {"semantic_generic_stages",0},{"raw_unclassified_aten",std::move(raw_unclassified)},
      {"solver_executed",false},{"device_executed",false}});
    if(prepare) {
      const int groups=std::stoi(argv[5]);
      auto target=tilemega::TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
      auto workers=tilemega::ascend::WorkerMap::Build(*target.ascend_capability,
          tilemega::ascend::LaunchMode::kMix1_2,groups);
      auto problem=tilemega::solver::PrepareSymbolicProblem(*module,target,dims,groups,1,1,nullptr,false,128);
      tilemega::analysis::CouplingCache cache;
      auto skeleton=tilemega::solver::BuildPlanSkeleton(problem,groups,1,1,false,cache,nullptr,nullptr,true);
      tilemega::json::Array spaces,allocations;
      for(unsigned s=0;s<problem.counts.size();++s) {
        auto const& projected=problem.projection.stages.at(s);
        auto const& stage=problem.model.stages.at(projected.logical_stage);
        tilemega::json::Value item(tilemega::json::Object{{"stage",int(s)},
          {"logical_stage",projected.logical_stage},{"kind",int(stage.kind)},
          {"count",problem.counts[s]},{"offset",problem.offsets[s]},
          {"width",stage.width},{"extent",stage.extent},{"group",stage.group},
          {"combine",projected.combine},{"service_ns",tilemega::json::Value{}},
          {"implementation_status","requires_native_body_binding_and_empirical_price"}});
        if(stage.IsCollective()) {
          auto const& g=problem.model.gemms.at(stage.gemm);
          item.Set("gemm_n",g.n);item.Set("gemm_k",g.k);
        }
        spaces.push_back(std::move(item));
      }
      auto add=[](uint64_t a,uint64_t b) {
        if(a>std::numeric_limits<uint64_t>::max()-b)throw std::overflow_error("Buffer byte sum overflow");
        return a+b;
      };
      auto multiply=[](uint64_t a,uint64_t b) {
        if(b && a>std::numeric_limits<uint64_t>::max()/b)throw std::overflow_error("Buffer byte product overflow");
        return a*b;
      };
      uint64_t arena=0;
      if(problem.model.physical_buffers.size()!=plan.buffers.size())
        throw std::invalid_argument("Serving allocation metadata does not cover the model buffer table");
      for(auto const& [name,b]:problem.model.physical_buffers) {
        auto elements=add(add(b.constant,multiply(b.per_batch,dims.batch)),
          add(multiply(b.per_seq,dims.seq),add(multiply(b.per_past,dims.past),multiply(b.per_total,dims.total))));
        auto bytes=multiply(elements,b.element_bytes);
        auto allocated=multiply(add(add(bytes,512),511)/512,512);
        allocations.emplace_back(tilemega::json::Object{{"name",name},{"offset_bytes",double(arena)},
          {"logical_bytes",double(bytes)},{"guarded_allocation_bytes",double(allocated)}});
        arena=add(arena,allocated);
      }
      Write(output/"symbolic_dependencies.isl",problem.execution_dependencies.ToString());
      tilemega::json::Value preparation(tilemega::json::Object{
        {"status","verified_symbolic_preparation_only"},{"groups",groups},{"launch_blocks",workers.launch.num_blocks},
        {"task_count",problem.offsets.back()},{"projected_stages",int(problem.counts.size())},
        {"structural_skeleton_edges",int(skeleton.edges.size())},{"spaces",std::move(spaces)},
        {"guarded_arena_bytes",double(arena)},{"allocations",std::move(allocations)},
        {"byte_offsets_bits",64},{"costs_calibrated",false},{"priced_solver_executed",false},
        {"native_plan_materialized",false},{"device_executed",false}});
      Write(output/"preparation.json",preparation.Dump());
      result.Set("symbolic_preparation_executed",true);
      result.Set("structural_skeleton_executed",true);
      result.Set("task_count",problem.offsets.back());
      if(emit) {
        std::vector<tilemega::ascend::ImplementationRole> roles;
        for(auto const& s:plan.stages) {
          using R=tilemega::ascend::ImplementationRole;
          roles.push_back(s.kind==tilemega::frontend::PlanTaskKind::kGemm?
            (plan.gemms.at(s.gemm).epilogue==tilemega::frontend::PlanGemm::Epilogue::kStore?R::kAicOnly:R::kMix1_2):R::kAivOnly);
        }
        skeleton.BindAscendWorkers(workers,roles,true);
        std::vector<tilemega::solver::AscendServicePrice> unknown(problem.offsets.back());
        auto tasks=tilemega::solver::AscendTasksFromSymbolic(problem,skeleton,unknown);
        for(auto& t:tasks)t.mix_aiv_lanes=1;
        auto scheduled=tilemega::solver::DiagnosticAscendSchedule(tasks,workers);
        auto native=tilemega::codegen::MaterializeAscendLlm(*module,problem,workers,tasks,scheduled,true);
        native.Write(output/"plan");
        auto diagnostic_schedule=tilemega::json::Parse(scheduled.ToJson());
        diagnostic_schedule.Set("selection","diagnostic_unpriced_bootstrap");
        diagnostic_schedule.Set("makespan_ns",tilemega::json::Value{});
        diagnostic_schedule.Set("shared_gm_floor_ns",tilemega::json::Value{});
        Write(output/"schedule.json",diagnostic_schedule.Dump());
        Write(output/"persistent_llm.cpp",tilemega::codegen::LowerAscendLlm(native));
        result.Set("native_plan_materialized",true);
        result.Set("selection","diagnostic_unpriced_bootstrap_not_optimized");
        result.Set("priced_solver_executed",false);
        result.Set("arena_bytes",double(native.metadata.arena_bytes));
        result.Set("device_execution_verified",false);
      }
    }
    Write(output/"import.json",result.Dump());std::cout<<result.Dump()<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
