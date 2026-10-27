#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Codegen/AscendLlmCodegen.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SHA256.h>
#include <llvm/ADT/StringExtras.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstdlib>

static std::string ScheduleReport(tilemega::solver::AscendSchedule const& schedule) {
  if(auto raw=std::getenv("TILEMEGA_AS2_VERBOSE_SCHEDULE");raw && std::string(raw)=="1") {
    if(schedule.owners.size()>65536)throw std::invalid_argument("Verbose schedules require a bounded diagnostic instance");
    return schedule.ToJson();
  }
  llvm::SHA256 hash;
  auto bytes=[&](auto const& values) {
    uint64_t count=values.size();hash.update(llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t const*>(&count),sizeof(count)));
    hash.update(llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t const*>(values.data()),values.size()*sizeof(values[0])));
  };
  bytes(schedule.owners);bytes(schedule.ordinals);bytes(schedule.offsets);bytes(schedule.queue);
  bytes(schedule.start_ns);bytes(schedule.finish_ns);
  for(auto const& participants:schedule.participants)bytes(participants);
  for(auto const& waits:schedule.waits) {
    std::vector<uint32_t> pairs;for(auto wait:waits){pairs.push_back(wait.producer_worker);pairs.push_back(wait.required_prefix);}bytes(pairs);
  }
  tilemega::json::Array queues;
  for(unsigned w=0;w+1<schedule.offsets.size();++w) {
    unsigned first=schedule.offsets[w],end=schedule.offsets[w+1];tilemega::json::Array sample;
    for(unsigned i=first;i<end && i-first<4;++i)sample.emplace_back(double(schedule.queue.at(i)));
    queues.emplace_back(tilemega::json::Object{{"worker",int(w)},{"queue_count",double(end-first)},{"first_tasks",std::move(sample)}});
  }
  auto identity=hash.final();
  return tilemega::json::Value(tilemega::json::Object{
    {"schema","tilemega.ascend.schedule_summary.v1"},{"tasks",double(schedule.owners.size())},
    {"queue_refs",double(schedule.queue.size())},{"worker_queues",std::move(queues)},
    {"schedule_sha256",llvm::toHex(llvm::ArrayRef<uint8_t>(identity),true)},
    {"identity_encoding","little-endian host u64 array lengths, u32 owner/ordinal/offset/queue/participants/wait-pairs, f64 start/finish"},
    {"semantic_edges",double(schedule.semantic_edges)},{"fifo_edges",double(schedule.fifo_edges)},
    {"lowered_waits",double(schedule.lowered_waits)},{"epoch_stride",double(schedule.epoch_stride)},
    {"makespan_ns",schedule.makespan_ns},{"shared_gm_floor_ns",schedule.gm_floor_ns},
    {"wait_lowering",schedule.fifo_wait_elision?"experimental_fifo_observed_frontier":"per_task_frontier"},
    {"full_selected_tables","plan/*.bin; no task or dependency removed from execution"}}).Dump();
}

static void Write(std::filesystem::path const& path,std::string const& text) {
  std::ofstream out(path);out<<text<<'\n';if(!out)throw std::runtime_error("Cannot write "+path.string());
}
template<class T> static std::vector<T> ReadTable(std::filesystem::path const& path) {
  auto bytes=std::filesystem::file_size(path);
  if(bytes>(64ULL<<20) || bytes%sizeof(T))throw std::invalid_argument("Invalid bounded cached table");
  std::vector<T> data(bytes/sizeof(T));std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(data.data()),bytes);if(!f)throw std::runtime_error("Cannot read cached table");return data;
}
int main(int argc,char** argv) {
  try {
    if(argc==4 && std::string(argv[1])=="reemit") {
      namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
      auto plan=fs::weakly_canonical(argv[2]),out=fs::weakly_canonical(argv[3]);
      for(auto const& p:{plan,out})if(p==root || p.lexically_relative(root).empty() || *p.lexically_relative(root).begin()=="..")
        throw std::invalid_argument("Cached codegen must remain in AS2 runs");
      if(fs::exists(out))throw std::invalid_argument("Use a new codegen output directory");
      auto origin=tilemega::json::ParseFile((plan.parent_path()/"import.json").string());
      if(!origin.At("native_plan_materialized").AsBool("materialized"))throw std::invalid_argument("Missing materialized-plan provenance");
      auto metadata=ReadTable<tilemega::ascend::llm::Metadata>(plan/"metadata.bin");
      tilemega::codegen::AscendLlmPlan native;
      if(metadata.size()!=1)throw std::invalid_argument("Wrong cached metadata extent");
      native.metadata=metadata[0];native.stages=ReadTable<tilemega::ascend::llm::Stage>(plan/"stages.bin");
      if(native.metadata.stages!=native.stages.size() || native.metadata.tasks!=origin.At("task_count").AsNumber("task_count"))
        throw std::invalid_argument("Cached plan/provenance mismatch");
      auto source=tilemega::codegen::LowerAscendLlm(native);fs::create_directories(out);
      Write(out/"persistent_llm.cpp",source);
      Write(out/"codegen.json",tilemega::json::Value(tilemega::json::Object{
        {"source_plan",plan.string()},{"source_import",origin},{"tables_rewritten",false},
        {"solver_executed_this_call",false},{"scope","codegen from existing materialized CTG/Solver plan"}}).Dump());
      return 0;
    }
    const std::string command=argc>1?argv[1]:"";
    const bool fifo_elision=command=="emit-priced-fifo" || command=="emit-priced-decomposed-fifo" || command=="emit-priced-body-fifo";
    bool priced_decomposition=argc==11 && (command=="emit-priced-decomposed" || command=="emit-priced-decomposed-fifo");
    bool priced_body=argc==11 && (command=="emit-priced-body" || command=="emit-priced-body-fifo");
    bool body=priced_body || (argc==10 && std::string(argv[1])=="emit-body");
    bool priced=priced_decomposition || priced_body || (argc==10 && (command=="emit-priced" || command=="emit-priced-fifo"));
    bool decomposed=priced_decomposition || (argc==10 && std::string(argv[1])=="emit-decomposed");
    bool emit=priced || decomposed || body || ((argc>=6 && argc<=9) && std::string(argv[1])=="emit-diagnostic");
    bool prepare=emit || (argc==6 && std::string(argv[1])=="prepare");
    if(!prepare && (argc!=5 || std::string(argv[1])!="inspect"))
      throw std::invalid_argument("usage: as2_compile_llm inspect export new_output batch; or prepare|emit-diagnostic export new_output batch groups [diagnostic_kv_block] [gemm_tile_n:64|128] [attention:bf16|fp32|fp32-scores|math-sdpa|cube-fp32|cube-fp32-newton|cube-fp32-newton-full|cube-fp32-newton-fold|cube-fp32-fold-softmax]; or emit-priced|emit-decomposed export new_output batch groups kv_block tile_n attention prices_or_decomposition.json; or emit-body|emit-priced-body export new_output batch groups kv_block tile_n attention [prices.json] binding.json; priced modes accept an explicit -fifo suffix for experimental wait elision");
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
    if(emit && argc>=7) {
      serving.kv_block=std::stoi(argv[6]);
      if(serving.kv_block!=128 && serving.kv_block!=1088)throw std::invalid_argument("Bounded native KV candidates are 128 and 1088");
    }
    if(emit && argc>=8) {
      serving.argmax_tile_n=std::stoi(argv[7]);
      if(serving.argmax_tile_n!=64 && serving.argmax_tile_n!=128)
        throw std::invalid_argument("Native GEMM domain is 32x64x64 or 16x128x128");
    }
    unsigned attention_precision=0;
    if(emit && argc>=9) {
      std::string attention=argv[8];
      if(attention!="bf16" && attention!="fp32" && attention!="fp32-scores" && attention!="math-sdpa" && attention!="cube-fp32" && attention!="cube-fp32-newton" && attention!="cube-fp32-newton-full" && attention!="cube-fp32-newton-fold" && attention!="cube-fp32-fold-softmax")throw std::invalid_argument("Unknown attention implementation precision");
      attention_precision=attention=="fp32"?1:attention=="fp32-scores"?2:attention=="math-sdpa"?3:attention=="cube-fp32"?4:attention=="cube-fp32-newton"?5:attention=="cube-fp32-newton-full"?6:attention=="cube-fp32-newton-fold"?7:attention=="cube-fp32-fold-softmax"?8:0;
    }
    tilemega::json::Value decomposition;
    tilemega::json::Value binding;
    if(body) {
      auto path=fs::weakly_canonical(argv[priced_body?10:9]);
      auto root=fs::path(TILEMEGA_SOURCE_DIR)/"configs/ascend";
      auto relative=path.lexically_relative(root);
      if(relative.empty() || *relative.begin()=="..")throw std::invalid_argument("Use a checked-in Body binding");
      binding=tilemega::json::ParseFile(path.string());
      if(binding.At("schema").AsString("schema")!="tilemega.as2.body_binding.v1" ||
          binding.At("impl_id").AsString("impl_id")!="as2_mix_gemm_local_tail7_residual_16x128x128" ||
          binding.At("model_id").AsString("model")!=receipt.At("model_id").AsString("model") ||
          binding.At("phase").AsString("phase")!=phase || phase!="prefill" || batch!=16 ||
          binding.At("batch").AsNumber("batch")!=batch || serving.argmax_tile_n!=128)
        throw std::invalid_argument("Unverified local Body binding");
      auto model_id=receipt.At("model_id").AsString("model");
      auto k=binding.At("k").AsNumber("k"),count=binding.At("expected_gemms").AsNumber("expected_gemms");
      if(binding.At("n").AsNumber("n")!=2048 ||
          !((model_id=="llama3_2_1b" && k==8192 && count==16) ||
            (model_id=="qwen3_1_7b" && k==6144 && count==28)))
        throw std::invalid_argument("Unverified local Body model/shape/count");
    }
    if(decomposed) {
      auto path=fs::weakly_canonical(argv[priced_decomposition?10:9]);
      auto config_root=fs::path(TILEMEGA_SOURCE_DIR)/"configs/ascend";
      auto relative=path.lexically_relative(config_root);
      if(relative.empty() || *relative.begin()=="..")throw std::invalid_argument("Use a checked-in decomposition candidate");
      decomposition=tilemega::json::ParseFile(path.string());
      if(decomposition.At("schema").AsString("schema")!="tilemega.as2.observed_decomposition.v1" ||
          decomposition.At("model_id").AsString("model")!=receipt.At("model_id").AsString("model") ||
          decomposition.At("phase").AsString("phase")!=phase ||
          decomposition.At("batch").AsNumber("batch")!=batch || serving.argmax_tile_n!=128)
        throw std::invalid_argument("Observed decomposition model/phase/geometry mismatch");
      if(batch!=1 && !(batch==16 && phase=="decode" &&
          (receipt.At("model_id").AsString("model")=="llama3_2_1b" || receipt.At("model_id").AsString("model")=="qwen3_1_7b" || receipt.At("model_id").AsString("model")=="llama3_2_3b" || receipt.At("model_id").AsString("model")=="llama3_1_8b")) &&
          !(batch==16 && phase=="prefill" && receipt.At("model_id").AsString("model")=="llama3_1_8b"))
        throw std::invalid_argument("Unverified batch decomposition");
      if(auto separate=decomposition.Find("separate_qkv"))serving.separate_qkv=separate->AsBool("separate_qkv");
      if(serving.separate_qkv && receipt.At("model_id").AsString("model")!="llama3_1_8b")
        throw std::invalid_argument("Separate projection candidate is bounded to observed Llama8B");
    }
    auto start=std::chrono::steady_clock::now();
    tilemega::analysis::IslContext isl;
    auto bridge=tilemega::frontend::ReadExportBridge((input/"export.json").string());
    auto plan=tilemega::frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,serving);
    if(body) {
      unsigned matches=0;
      for(auto const& g:plan.gemms)if(g.n==binding.At("n").AsNumber("n") &&
          g.k==binding.At("k").AsNumber("k") && g.epilogue==tilemega::frontend::PlanGemm::Epilogue::kResidual)++matches;
      if(matches!=binding.At("expected_gemms").AsNumber("expected_gemms"))
        throw std::invalid_argument("Local Body binding does not cover the imported model");
    }
    if(emit && phase=="prefill")for(auto& stage:plan.stages)
      if(stage.kind==tilemega::frontend::PlanTaskKind::kFusedAttention)
        stage.attention_query_rows=16*stage.group;
    tilemega::frontend::ImportOptions options;options.phase_batch=batch;
    options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=options.combiner_tile_per_block=true;
    if(serving.argmax_tile_n==128)options.gemms.assign(plan.gemms.size(),{16,128,128,2,1});
    else options.gemms.assign(plan.gemms.size(),{32,64,64,2,1});
    unsigned decomposed_gemms=0;
    if(decomposed) {
      for(auto const& family:decomposition.At("families").AsArray("families")) {
        unsigned matches=0;
        auto family_k=family.At("k").AsNumber("k");
        auto family_n=family.At("n").AsNumber("n"),split=family.At("split_k").AsNumber("split_k");
        auto epilogue=family.At("epilogue").AsString("epilogue");
        bool residual=((family_n==2048 && split==3 && (family_k==6144 || family_k==8192)) ||
             (family_n==3072 && split==2 && family_k==8192)) && epilogue=="residual";
        bool store=serving.separate_qkv && family_n==1024 && family_k==4096 && split==7 && epilogue=="store";
        if(!residual && !store)
          throw std::invalid_argument("Unverified decomposition family");
        for(unsigned i=0;i<plan.gemms.size();++i) {
          auto const& g=plan.gemms[i];
          auto required=store?tilemega::frontend::PlanGemm::Epilogue::kStore:tilemega::frontend::PlanGemm::Epilogue::kResidual;
          if(g.n==family_n && g.k==family_k && g.epilogue==required) {
            if(options.gemms[i].split_k!=1)throw std::invalid_argument("Duplicate decomposition binding");
            options.gemms[i].split_k=unsigned(split);++matches;++decomposed_gemms;
          }
        }
        if(matches!=family.At("expected_gemms").AsNumber("expected_gemms"))
          throw std::invalid_argument("Decomposition did not match the complete model family");
      }
      if(!decomposed_gemms && !serving.separate_qkv)throw std::invalid_argument("Empty decomposition cannot prove optimization participation");
    }
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
    if(decomposed) {
      result.Set("decomposition_candidate",decomposition);
      result.Set("decomposed_logical_gemms",int(decomposed_gemms));
    }
    if(body)result.Set("body_binding_candidate",binding);
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
        for(auto const& projected:problem.projection.stages) {
          using R=tilemega::ascend::ImplementationRole;
          auto const& s=plan.stages.at(projected.logical_stage);
          roles.push_back(projected.combine?R::kAivOnly:s.kind==tilemega::frontend::PlanTaskKind::kGemm?
            (options.gemms.at(s.gemm).split_k>1 || plan.gemms.at(s.gemm).epilogue==tilemega::frontend::PlanGemm::Epilogue::kStore?R::kAicOnly:R::kMix1_2):
            s.kind==tilemega::frontend::PlanTaskKind::kFusedAttention && attention_precision>=4?R::kMix1_2:R::kAivOnly);
        }
        skeleton.BindAscendWorkers(workers,roles,true);
        std::vector<tilemega::solver::AscendServicePrice> unknown(problem.offsets.back());
        auto tasks=tilemega::solver::AscendTasksFromSymbolic(problem,skeleton,unknown);
        for(auto& t:tasks)t.mix_aiv_lanes=1;
        tilemega::json::Value prices;
        std::string selected_algorithm="diagnostic_unpriced_bootstrap";
        tilemega::solver::AscendSchedulingCosts scheduling_costs;
        if(priced) {
          auto path=fs::weakly_canonical(argv[9]);
          if(!controlled(path))throw std::invalid_argument("Native prices must be an AS-2 measured artifact");
          prices=tilemega::json::ParseFile(path.string());
          if(prices.At("schema").AsString("price schema")!="tilemega.as2.native_task_prices.v1" ||
              prices.At("status").AsString("price status")!="sourced_empirical_context" ||
              prices.At("model_id").AsString("model")!=receipt.At("model_id").AsString("model") ||
              prices.At("config_sha256").AsString("config")!=receipt.At("config_sha256").AsString("config") ||
              prices.At("batch").AsNumber("batch")!=batch || prices.At("seq").AsNumber("seq")!=dims.seq ||
              prices.At("groups").AsNumber("groups")!=groups || prices.At("phase").AsString("phase")!=phase)
            throw std::invalid_argument("Native empirical prices do not describe this model/phase/topology");
          if(prices.At("runtime_abi").AsNumber("runtime ABI")!=tilemega::ascend::llm::kVersion)
            throw std::invalid_argument("Native service prices belong to another runtime ABI");
          auto const& rows=prices.At("stages").AsArray("stage prices");
          if(rows.size()!=problem.counts.size())throw std::invalid_argument("Incomplete native price domain");
          for(unsigned s=0;s<rows.size();++s) {
            auto const& row=rows[s];auto const& bytes=row.At("logical_gm_bytes").AsArray("GM bytes");
            double service=row.At("service_ns").AsNumber("service_ns");
            auto tile_prices=row.Find("tile_service_ns");
            if(tile_prices && tile_prices->AsArray("tile services").size()!=bytes.size())
              throw std::invalid_argument("Incomplete coordinate-specific task prices");
            if(row.At("stage").AsNumber("stage")!=s || row.At("tasks").AsNumber("tasks")!=problem.counts[s] ||
                bytes.size()!=std::size_t(problem.counts[s]) || !std::isfinite(service) || service<=0)
              throw std::invalid_argument("Missing or nonpositive native task price");
            for(unsigned tile=0;tile<bytes.size();++tile) {
              double n=bytes[tile].AsNumber("logical bytes");
              double selected_service=tile_prices?tile_prices->AsArray("tile services")[tile].AsNumber("tile service"):service;
              if(!std::isfinite(selected_service) || selected_service<=0)
                throw std::invalid_argument("Nonpositive coordinate-specific task price");
              if(!std::isfinite(n) || n<0 || n>=double(UINT64_MAX) || std::floor(n)!=n)
                throw std::invalid_argument("Unrepresentable native GM demand");
              tasks.at(problem.offsets[s]+tile).price={selected_service,uint64_t(n),path.string()};
            }
          }
          scheduling_costs={prices.At("gm_bytes_per_ns").AsNumber("shared GM rate"),
            prices.At("cross_worker_ready_ns").AsNumber("ready latency"),path.string()};
        }
        auto scheduled=priced?tilemega::solver::ScheduleAscendTasks(tasks,workers,scheduling_costs):
          tilemega::solver::DiagnosticAscendSchedule(tasks,workers);
        if(priced) {
          auto baseline=tilemega::solver::PriceAscendSchedule(tasks,workers,scheduling_costs,
            tilemega::solver::DiagnosticAscendSchedule(tasks,workers));
          scheduled=tilemega::solver::PriceAscendSchedule(tasks,workers,scheduling_costs,std::move(scheduled));
          bool choose_eft=scheduled.makespan_ns<baseline.makespan_ns;
          selected_algorithm=choose_eft?"measured_native_rank_eft_shared_gm":"measured_native_structural_shared_gm";
          Write(output/"eft_candidate.json",ScheduleReport(scheduled));
          Write(output/"structural_candidate.json",ScheduleReport(baseline));
          Write(output/"search.json",tilemega::json::Value(tilemega::json::Object{
            {"schema","tilemega.as2.native_plan_search.v1"},{"compared_legal_plans",2},
            {"fixed_body_domain","one measured geometry/attention implementation"},
            {"cost_provenance",prices},{"predicted_eft_ns",scheduled.makespan_ns},
            {"predicted_structural_ns",baseline.makespan_ns},
            {"selected",choose_eft?"heterogeneous_rank_eft":"structural_topological_least_queue"},
            {"decision","minimum same-model shared-GM-calendar makespan"},
            {"queue_or_placement_differs",scheduled.queue!=baseline.queue || scheduled.owners!=baseline.owners},
            {"post_selection_wait_lowering",fifo_elision?"experimental_fifo_observed_frontier":"per_task_frontier"},
            {"device_execution_verified",false},{"end_to_end_performance_admitted",false}}).Dump());
          if(!choose_eft)scheduled=std::move(baseline);
        }
        if(fifo_elision) {
          Write(output/"uncompressed_schedule.json",ScheduleReport(scheduled));
          scheduled=tilemega::solver::ElideAscendFifoWaits(tasks,workers,std::move(scheduled));
        }
        auto native=tilemega::codegen::MaterializeAscendLlm(*module,problem,workers,tasks,scheduled,!priced,attention_precision,body);
        if(priced) {
          auto const& rows=prices.At("stages").AsArray("stage prices");
          for(unsigned s=0;s<rows.size();++s) {
            auto const& r=rows[s];auto const& n=native.stages[s];
            if(r.At("kind").AsNumber("kind")!=n.kind || r.At("variant").AsNumber("variant")!=n.variant ||
                r.At("rows").AsNumber("rows")!=n.rows || r.At("columns").AsNumber("columns")!=n.columns ||
                r.At("k").AsNumber("k")!=n.k)throw std::invalid_argument("Priced implementation/shape differs from actual CTG");
          }
        }
        native.Write(output/"plan");
        if(priced) {
          double low=prices.At("past_min").AsNumber("minimum past"),high=prices.At("past_max").AsNumber("maximum past");
          if(low!=0 || !std::isfinite(high) || high<0 || high>1086 || std::floor(high)!=high || (dims.seq==64 && high))
            throw std::invalid_argument("Invalid measured context interval");
          uint32_t bounds[4]={1,uint32_t(low),uint32_t(high),unsigned(groups)};
          std::ofstream bounds_file(output/"plan/pricing_bounds.bin",std::ios::binary);
          bounds_file.write(reinterpret_cast<char const*>(bounds),sizeof(bounds));
          if(!bounds_file)throw std::runtime_error("Cannot save empirical price bounds");
          Write(output/"plan/prices.json",prices.Dump());
        }
        auto diagnostic_schedule=tilemega::json::Parse(ScheduleReport(scheduled));
        diagnostic_schedule.Set("selection",selected_algorithm);
        if(!priced) {
          diagnostic_schedule.Set("makespan_ns",tilemega::json::Value{});
          diagnostic_schedule.Set("shared_gm_floor_ns",tilemega::json::Value{});
        }
        Write(output/"schedule.json",diagnostic_schedule.Dump());
        Write(output/"persistent_llm.cpp",tilemega::codegen::LowerAscendLlm(native));
        result.Set("native_plan_materialized",true);
        result.Set("status","native_plan_materialized_not_device_verified");
        result.Set("solver_executed",true);
        result.Set("solver_scope",priced?"two_plan_empirical_native_rank_EFT_search":"structural_native_schedule_only");
        result.Set("attention_implementation",attention_precision==8?"mix_fp32_cube_newton_norm_fold_softmax1088":attention_precision==7?"mix_fp32_cube_newton_qk_half_fold_fullnorm":attention_precision==6?"mix_fp32_cube_newton_qk_and_fullnorm":attention_precision==5?"mix_fp32_cube_newton_qknorm":attention_precision==1?"fp32_chunk32_merge":attention_precision==2?"fp32_scores_bf16_probability":attention_precision==3?"math_sdpa_fp32_split_scale":attention_precision>=4?"mix_fp32_cube_qk_vector_softmax_pv":"bf16_scores_probabilities");
        result.Set("selection",priced?selected_algorithm:"diagnostic_unpriced_bootstrap_not_optimized");
        result.Set("priced_solver_executed",priced);
        result.Set("wait_lowering",fifo_elision?"experimental_fifo_observed_frontier":"per_task_frontier");
        result.Set("wait_elision_device_verified",false);
        if(priced)result.Set("price_provenance",prices);
        result.Set("arena_bytes",double(native.metadata.arena_bytes));
        result.Set("device_execution_verified",false);
      }
    }
    Write(output/"import.json",result.Dump());std::cout<<result.Dump()<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
