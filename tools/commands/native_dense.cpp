// SPDX-License-Identifier: BSD-3-Clause
#include "NativeDense.h"
#include "NativeCompileOutcome.h"
#include "Toolchain.h"
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
#include <tilemega/Solver/Sm90DenseRegisterContract.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/raw_ostream.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <iterator>
#include <sys/wait.h>

namespace tilemega::commands {
namespace {
std::string Quote(std::string const& s) {
  std::string result="'";
  for(char c:s)result+=c=='\''?"'\\''":std::string(1,c);
  return result+"'";
}
std::string Hash(std::string const& path) {
  auto command="sha256sum "+Quote(path);auto* pipe=popen(command.c_str(),"r");
  if(!pipe)throw std::runtime_error("native input hash command failed");
  char text[65]{};auto* got=fgets(text,sizeof(text),pipe);int code=pclose(pipe);
  if(!got || code || std::string(text).size()!=64)throw std::runtime_error("native input hash failed");
  return text;
}
// The tensor parallel degree a per-rank export was written for: `tp` in the
// export directory's manifest.json (absent there, or no manifest: 1). A plan
// built at another degree would size every shard wrongly, so it is refused.
int ManifestTp(std::string const& input) {
  auto manifest=std::filesystem::path(input).parent_path()/"manifest.json";
  int recorded=1;
  if(std::filesystem::exists(manifest))
    if(auto const* value=json::ParseFile(manifest.string()).Find("tp"))
      recorded=int(value->AsNumber("manifest tp"));
  return recorded;
}
void CheckManifestTp(std::string const& input,int tp) {
  int recorded=ManifestTp(input);
  if(recorded!=tp)
    throw std::invalid_argument("native dense tp "+std::to_string(tp)+
        " differs from the export manifest's tp "+std::to_string(recorded));
}
}
static int CompileNativeDenseArtifact(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target_path,int batch,
    std::string const& dump_cg,int split_compile,std::string const& origin,
    json::Value const& selection={}) {
  namespace fs=std::filesystem;
  if(split_compile<1 || split_compile>8)
    throw std::invalid_argument("native split compilation must use 1 through 8 workers");
  auto requested=fs::absolute(output);
  bool shared=requested.extension()==".so";
  if(!shared && requested.extension()!=".cu")
    throw std::invalid_argument("native dense output must be .so or .cu");
  for(auto const& path:{requested.string(),requested.string()+".build_limits.json",
      requested.string()+".compile-result.json",requested.string()+".ptxas.log",
      requested.string()+".identity.json",requested.string()+".source.json",requested.string()+".build_command.txt"})
    if(fs::exists(path))throw std::invalid_argument("retain native artifact evidence; choose a new output path");
  auto input_hash=Hash(input),coordinates_hash=Hash(coordinates),target_hash=Hash(target_path);
  auto c=solver::Sm90DenseCoordinates::FromJson(json::ParseFile(coordinates));
  CheckManifestTp(input,c.tp);
  auto target=TargetSpec::FromJson(target_path);
  auto registers=solver::NativeDenseRegisterContract(target,c.residency);
  mlir::MLIRContext context;
  auto imported=solver::ImportSm90Dense(input,c,context);
  auto module=solver::MaterializeSm90Dense(imported,c,batch,target,context,origin);
  auto source=codegen::CouplingGraphToCUDA{}.Lower(*module);
  fs::create_directories(requested.parent_path());
  auto cu=shared?fs::path(requested.string()+".cu"):requested;
  std::ofstream(cu)<<source;
  auto cg=fs::path(dump_cg.empty()?requested.string()+".tmexec.mlir":dump_cg);
  std::error_code error;llvm::raw_fd_ostream ir(cg.string(),error);
  if(error)throw std::runtime_error("cannot write native CTG");
  module->print(ir);ir<<'\n';ir.close();
  json::Array gemms;
  for(unsigned i=0;i<imported.plan.gemms.size();++i) {
    auto const& g=imported.plan.gemms[i];auto family=solver::Sm90DenseFamily(imported.plan,g);
    auto const& shape=c.gemms[family];
    gemms.emplace_back(json::Object{{"index",int(i)},{"n",int(g.n)},{"k",int(g.k)},
        {"family",solver::kSm90DenseFamilies[family]},{"impl","native_mma_sync"},
        {"tile_m",16},{"tile_n",shape.tile_n},{"tile_k",shape.TileK()},
        {"stages",shape.slots},{"split_k",1},{"warps_n",shape.warps_n},{"warps_k",shape.warps_k},
        {"register_stages",shape.register_stages}});
  }
  json::Object manifest={{"identity_schema",1},{"model",fs::path(input).stem().string()},
      {"phase","decode"},{"runtime_batch",batch},{"batch_lo",batch},{"batch_hi",batch},
      {"seq",1},{"capacity",1088},{"prompt_len",64},
      {"past_lo",64},{"past_hi",1086},{"pg","native"},{"mode","L1"},{"loop",1},
      {"residency",c.residency},{"grid",target.res.num_sms*c.residency},{"kappa",1},
      {"register_contract",json::Object{{"threads",registers.threads},
          {"requested_residency",registers.residency},{"callee_limit",registers.callee_limit},
          {"allocation_granularity",registers.granularity},{"subpartitions",registers.subpartitions},
          {"runtime_occupancy_required",true}}},
      {"runtime_target",fs::absolute(target_path).string()},{"target_spec",json::Parse(target.ToJson())},
      {"native_dense",c.ToJson()},{"native_origin",origin},{"native_selection",selection},{"gemms",gemms},
      {"export_sha256",input_hash},{"coordinates_sha256",coordinates_hash},
      {"ctg_sha256",Hash(cg.string())},{"ctg",cg.string()}};
  std::ofstream(requested.string()+".plan.json")<<json::Value(manifest).Dump(2)<<'\n';
  if(!shared)return 0;
  std::string root=TILEMEGA_SOURCE_DIR;
  std::string python_path=root+"/python";
  if(auto* inherited=std::getenv("PYTHONPATH"))python_path+=":"+std::string(inherited);
  auto identity=[&](bool capture) {
    auto command="PYTHONPATH="+Quote(python_path)+" python3 -m tilemega.build.identity --root "+Quote(root)+
        " --snapshot "+Quote(requested.string()+".source.json");
    command+=capture?" --capture --compiler "+Quote(fs::read_symlink("/proc/self/exe").string()):
        " --so "+Quote(requested.string());
    if(std::system(command.c_str()))throw std::runtime_error("native serving artifact identity failed");
  };
  identity(true);
  std::vector<std::string> command_args={NvccPath(),"-rdc=true","-std=c++17","-O3",
      "-gencode","arch=compute_90a,code=sm_90a","--expt-relaxed-constexpr","-shared",
      "-Xcompiler","-fPIC","-cudart","shared","-Xptxas","-v","--split-compile",
      std::to_string(split_compile),"-I"+root+"/include","-I"+root+"/third_party/cutlass/include",
      cu.string(),"-o",requested.string()};
  if(registers.Constrained()) {
    command_args.push_back("-Xptxas");
    command_args.push_back("--device-function-maxrregcount="+std::to_string(registers.callee_limit));
  }
  std::string command;
  for(auto const& argument:command_args) {if(!command.empty())command+=' ';command+=Quote(argument);}
  auto command_path=requested.string()+".build_command.txt";
  std::ofstream(command_path)<<command<<'\n';
  auto snapshot_path=requested.string()+".source.json";
  auto snapshot=json::ParseFile(snapshot_path);
  std::map<std::string,std::string> frozen_inputs={{fs::absolute(input).string(),input_hash},
      {fs::absolute(coordinates).string(),coordinates_hash},{fs::absolute(target_path).string(),target_hash}};
  for(auto const& path:{cu.string(),cg.string(),requested.string()+".plan.json",snapshot_path,command_path,NvccPath()})
    frozen_inputs[fs::absolute(path).string()]=Hash(path);
  auto snapshot_root=fs::path(snapshot.At("root").AsString("snapshot root"));
  for(auto const& item:snapshot.At("source_files").AsObject("snapshot files")) {
    fs::path path(item.first);if(!path.is_absolute())path=snapshot_root/path;
    auto hash=item.second.AsString("snapshot source hash");
    auto inserted=frozen_inputs.emplace(path.string(),hash);
    if(!inserted.second && inserted.first->second!=hash)
      throw std::runtime_error("conflicting native compile source binding");
  }
  frozen_inputs[snapshot.At("compiler").AsString("snapshot compiler")]=
      snapshot.At("compiler_sha256").AsString("snapshot compiler hash");
  auto unchanged=[&] {for(auto const& item:frozen_inputs)if(Hash(item.first)!=item.second)return false;return true;};
  if(!unchanged())throw std::runtime_error("native compile inputs changed before invocation");
  auto log=requested.string()+".ptxas.log";
  // Candidate searches may run concurrently; bound and serialize their nvcc
  // process groups while retaining the unwrapped command for reproducibility.
  auto bounded="python3 "+Quote(root+"/scripts/run_sm90r1_build.py")+
      " --receipt "+Quote(requested.string()+".build_limits.json")+
      " --max-gib 48 --timeout 5400 --lock /tmp/tilemega_sm90r2_native_build.lock -- "+command;
  int status=std::system((bounded+" >"+Quote(log)+" 2>&1").c_str());
  auto limits_path=requested.string()+".build_limits.json";
  auto limits=json::ParseFile(limits_path);
  std::ifstream log_stream(log);if(!log_stream)throw std::runtime_error("missing native compile log");
  std::string log_text((std::istreambuf_iterator<char>(log_stream)),{});
  bool stable=unchanged();
  bool normal_failure=status!=-1 && WIFEXITED(status) && WEXITSTATUS(status)==1;
  auto diagnostics=stable?NativeResourceDiagnostics(limits,command_args,registers.callee_limit,
      normal_failure,log_text):std::vector<NativeRegisterCallLimit>{};
  bool succeeded=status==0 && limits.At("status").AsString("bounded status")=="completed" &&
      limits.At("exit_code").AsNumber("bounded exit")==0 &&
      limits.At("classification").AsString("bounded classification")=="compilation; not a numerical test" &&
      NativeBoundedCommandMatches(limits,command_args) && stable;
  json::Object inputs;
  for(auto const& item:frozen_inputs)inputs.emplace_back(item.first,item.second);
  json::Array errors;
  for(auto const& error:diagnostics)errors.emplace_back(json::Object{{"entry",error.entry},
      {"callee",error.callee},{"entry_limit",error.entry_limit},{"callee_registers",error.callee_registers}});
  auto outcome_path=requested.string()+".compile-result.json";
  json::Value outcome=json::Object{{"schema",1},{"classifier","native_ptxas_register_call_limit_v1"},
      {"status",succeeded?"compiled":diagnostics.empty()?"compiler_error":"resource_rejected"},
      {"classification","compilation; not a numerical test"},{"system_status",status},
      {"bounded_status",limits.At("status")},{"bounded_exit_code",limits.Find("exit_code")?*limits.Find("exit_code"):json::Value{}},
      {"bounded_receipt",limits_path},{"bounded_receipt_sha256",Hash(limits_path)},
      {"log",log},{"log_sha256",Hash(log)},{"raw_command_sha256",Hash(command_path)},
      {"generated_source_sha256",Hash(cu.string())},{"source_snapshot_sha256",Hash(snapshot_path)},
      {"source_digest",snapshot.At("source_digest")},
      {"coordinates_sha256",coordinates_hash},{"target_sha256",target_hash},{"export_sha256",input_hash},
      {"register_contract",json::Value(manifest).At("register_contract")},
      {"input_hashes",inputs},{"inputs_unchanged",stable},{"diagnostics",errors},
      {"output_present",fs::exists(requested)},{"artifact_identity_generated_before_classification",false}};
  {std::ofstream output(outcome_path);output<<outcome.Dump(2)<<'\n';
    if(!output)throw std::runtime_error("cannot retain native compile result");}
  if(!succeeded) {
    if(!diagnostics.empty())throw NativeResourceRejected(outcome_path);
    throw std::runtime_error("native nvcc compilation/evidence failed; inspect "+outcome_path);
  }
  identity(false);
  json::Value execution=json::Object{{"decode_mode","L1"},{"decode_loop",1},
      {"prefill_mode","L1"},{"source","native static program"}};
  std::ofstream(requested.string()+".serving.json")<<execution.Dump(2)<<'\n';
  std::cerr<<"NATIVE_DENSE origin="<<origin<<" grid="<<target.res.num_sms*c.residency
      <<" stages="<<imported.plan.stages.size()<<" ctg="<<cg.string()<<'\n';
  return 0;
}

int CompileNativeDenseFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target_path,int batch,
    std::string const& dump_cg,int split_compile) {
  return CompileNativeDenseArtifact(input,output,coordinates,target_path,batch,dump_cg,
      split_compile,"fixed");
}

bool IsNativeDenseInput(std::string const& input) {
  auto bridge=frontend::ReadExportBridge(input);
  // Native plans (consumer RMS) are the only ones with native MoE layers.
  frontend::ServingOptions options;options.dense_attention_chunks=1;options.native_rms=true;
  // A per-rank export only plans at its own degree (its lm_head holds V/T rows).
  options.tp=ManifestTp(input);
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  return plan.serving && !plan.dm && !plan.forward && plan.serving_seq==1;
}

int SolveNativeDense(NativeDenseSolveOptions const& o) {
  namespace fs=std::filesystem;
  if(o.measure_top<1 || o.measure_top>6 ||
      (o.measured && o.measure_command.empty()) || (!o.measured && !o.measure_command.empty()))
    throw std::invalid_argument("native solve requires measured top 1..6 and an explicit measure command");
  auto requested=fs::absolute(o.output);
  if(fs::exists(requested))throw std::invalid_argument("retain native solve evidence; choose a new output path");
  auto directory=fs::path(requested.string()+".search");
  if(fs::exists(directory))throw std::invalid_argument("retain prior native search evidence");
  fs::create_directories(directory);
  auto target=TargetSpec::FromJson(o.target);
  auto target_sha=Hash(o.target),input_sha=Hash(o.input);
  auto constants=solver::Sm90NativeConstants::FromTarget(json::ParseFile(o.target),TILEMEGA_SOURCE_DIR);
  // Constants are re-measured at the body milestones of prompt §5/§6; between
  // them a changed header is reported, never a reason to refuse the solve.
  json::Array stale;
  for(auto const& header:constants.StaleHeaders(TILEMEGA_SOURCE_DIR))stale.emplace_back(header);
  solver::Sm90DenseCostModel cost(target,constants);
  auto bridge=frontend::ReadExportBridge(o.input);
  // Native plans (consumer RMS) are the only ones with native MoE layers.
  CheckManifestTp(o.input,o.tp);
  frontend::ServingOptions options;options.dense_attention_chunks=1;options.native_rms=true;
  options.tp=o.tp;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  auto work=solver::Sm90DenseWorkloadFromPlan(plan,o.batch);
  solver::Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
  space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
  space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
  auto target_json=json::ParseFile(o.target);
  auto const* decisions=target_json.Find("sm90_native_decisions");
  if(!decisions)throw std::invalid_argument("target lacks sm90_native_decisions (prompt §11 decision tables)");
  space.weight_stream=decisions->At("weight_stream").AsBool("weight_stream");
  space.layer_loop=decisions->At("layer_loop").AsBool("layer_loop");
  if(o.search!="pipeline" && o.search!="additive")
    throw std::invalid_argument("--search takes pipeline or additive");
  solver::Sm90DenseSearchOptions search;search.top=o.measure_top;
  auto result=solver::SearchSm90Dense(space,work,cost,search);
  json::Value pipeline_record;
  // P3: Algorithm 1 replaces the additive ranking for every selection this
  // round. `--search additive` is kept only to reproduce the R3-R5 choices
  // (continuation 1 section 1.3), so it never decides a final coordinate here.
  if(o.search=="pipeline") {
    solver::Sm90NativePipelineOptions pipeline;
    pipeline.finalists=o.measure_top;
    pipeline.operator_sum=o.operator_sum;
    pipeline.retile_only=o.retile_only;
    pipeline.realize_aggregate=o.realize_aggregate;
    pipeline.budget_ms=o.search_budget_ms;
    std::vector<solver::Sm90DenseCoordinates> seeds;
    if(!o.seed_coordinates.empty())
      seeds.push_back(solver::Sm90DenseCoordinates::FromJson(json::ParseFile(o.seed_coordinates)));
    if(!o.pin_theta.empty()) {
      // AB-1: each family's geometry is fixed by a local criterion and the
      // rest of the decisions are still Algorithm 1's.
      auto pinned=solver::Sm90DenseCoordinates::FromJson(json::ParseFile(o.pin_theta));
      pipeline.pinned_theta=pinned;
      seeds.push_back(pinned);
    }
    mlir::MLIRContext context;
    auto imported=solver::ImportSm90Dense(o.input,result.ranked.front().coordinates,context);
    solver::Sm90NativeCtgCache cache;
    auto evaluate=[&](solver::Sm90DenseCoordinates const& c,bool isolated) {
      return solver::Sm90NativeCtgScore(imported,work,c,cost,context,cache,isolated);
    };
    auto pipelined=solver::SolveSm90NativePipeline(space,work,cost,pipeline,evaluate,seeds);
    auto rows=[&](std::vector<solver::Sm90NativePipelineCandidate> const& input) {
      json::Array output;
      for(auto const& candidate:input) {
        json::Value row=json::Object{
          {"coordinates",candidate.coordinates.ToJson()},{"structure",candidate.structural_class},
          {"pool_score_ns",candidate.pool_score_ns},{"replay_ns",candidate.replay_ns},
          {"evaluation_ms",candidate.evaluation_ms},{"origin",candidate.origin}};
        if(work.moe) {
          auto const& c=candidate.coordinates;
          row.Set("task_domain_approximation",json::Object{
              {"kind",solver::kSm90NativeMoeDomainApproximation},
              {"gate_up",solver::Sm90NativeMoeDomain(work.experts,work.top_k,work.batch,
                  work.shapes[6][0]/c.gemms[6].tile_n).ToJson()},
              {"down",solver::Sm90NativeMoeDomain(work.experts,work.top_k,work.batch,
                  work.shapes[7][0]/c.gemms[7].tile_n).ToJson()},
              {"route_price","older-body RouteLa proxy for per-CTA local routing"}});
        }
        output.push_back(std::move(row));
      }
      return json::Value(output);
    };
    json::Array trajectories;
    for(std::size_t seed=0;seed<pipelined.descent.size();++seed) {
      json::Array values;
      for(std::size_t move=0;move<pipelined.descent[seed].size();++move)
        values.emplace_back(json::Object{{"score_ns",pipelined.descent[seed][move]},
            {"coordinates",pipelined.descent_coordinates[seed][move].ToJson()}});
      trajectories.emplace_back(values);
    }
    pipeline_record=json::Object{{"objective",o.operator_sum?"operator-sum":"completion"},
        {"evaluated",rows(pipelined.evaluated)},{"shortlist",rows(pipelined.shortlist)},
        {"finalists",rows(pipelined.finalists)},{"descent",trajectories},
        {"memo_hits",double(pipelined.memo_hits)},{"coupling_hits",double(cache.coupling.hits)},
        {"coupling_misses",double(cache.coupling.misses)},
        {"release_hits",double(cache.release_hits)},{"release_misses",double(cache.release_misses)},
        {"space_hits",double(cache.space_hits)},{"space_misses",double(cache.space_misses)},
        {"edge_hits",double(cache.edge_hits)},{"edge_misses",double(cache.edge_misses)}};
    if(!o.seed_coordinates.empty())pipeline_record.Set("frozen_seed_sha256",Hash(o.seed_coordinates));
    {std::ofstream timing(directory/"ctg-timing.json");cache.timing.Write(timing,"native",o.input,o.batch);}
    result.ranked.clear();
    for(auto const& candidate:pipelined.finalists) {
      solver::Sm90DenseCandidate entry;
      entry.coordinates=candidate.coordinates;
      entry.prediction=cost.Predict(work,candidate.coordinates);
      entry.prediction.step_ns=candidate.replay_ns;
      entry.origin=candidate.structural_class;
      result.ranked.push_back(std::move(entry));
    }
    result.evaluated=pipelined.evaluations;
    result.elapsed_ms=pipelined.seconds*1000;
  }
  json::Array ranked,candidates;
  std::vector<fs::path> configs;
  for(unsigned i=0;i<result.ranked.size();++i) {
    auto path=directory/("candidate-"+std::to_string(i)+".tmexec.json");
    std::ofstream(path)<<result.ranked[i].coordinates.ToJson().Dump(2)<<'\n';configs.push_back(path);
    ranked.emplace_back(json::Object{{"rank",int(i)},{"origin",result.ranked[i].origin},
        {"coordinates",result.ranked[i].coordinates.ToJson()},
        {"prediction",result.ranked[i].prediction.ToJson()}});
  }
  json::Value measurement;int winner=0;
  auto save=[&] {
    json::Value receipt=json::Object{{"schema",o.search=="pipeline"?3:2},{"selection",o.measured?"measure":"predicted"},
        {"target_sha256",target_sha},{"export_sha256",input_sha},{"solve_command",o.solve_command},
        {"constants",constants.section},{"constants_from_older_body",stale},
        {"search_ms",result.elapsed_ms},{"evaluated",double(result.evaluated)},
        {"ranked",ranked},{"candidates",candidates},{"measurement",measurement},{"winner",winner},
        {"pipeline",pipeline_record}};
    std::ofstream(directory/"receipt.json")<<receipt.Dump(2)<<'\n';return receipt;
  };
  save();
  bool compiled_finalists=o.measured || o.search=="pipeline";
  if(compiled_finalists) {
    for(unsigned i=0;i<configs.size();++i) {
      auto binary=directory/("candidate-"+std::to_string(i)+".so");
      NativeCandidateCounts counts;
      auto rejected=TryMeasuredNativeCompile(counts,[&] {
        CompileNativeDenseArtifact(o.input,binary.string(),configs[i].string(),o.target,o.batch,
            "",o.split_compile,"solver",json::Object{{"selection","candidate"},{"rank",int(i)}});
      });
      json::Value row=json::Object{{"rank",int(i)},{"coordinates",configs[i].string()},
          {"predicted_ns",result.ranked[i].prediction.step_ns}};
      if(rejected) {row.Set("status","resource_rejected");row.Set("compile_result",*rejected);}
      else {row.Set("status","compiled");row.Set("binary",binary.string());row.Set("binary_sha256",Hash(binary.string()));}
      candidates.push_back(row);save();
    }
    if(o.search=="pipeline") {
      // A requested launch bound is not a measurement of the linked kernel.
      // Query the cubin before either predicted or measured final selection.
      for(int pass=0;pass<3;++pass) {
        auto list=directory/("occupancy-input-"+std::to_string(pass)+".json");
        std::ofstream(list)<<json::Value(candidates).Dump(2)<<'\n';
        auto probe=directory/("occupancy-"+std::to_string(pass));
        auto command="python3 "+Quote(std::string(TILEMEGA_SOURCE_DIR)+
            "/scripts/sm90r6_probe_native.py")+" --candidates "+Quote(list.string())+
            " --out "+Quote(probe.string());
        if(std::system(command.c_str()))
          throw std::runtime_error("native finalist occupancy query failed; evidence: "+probe.string());
        auto evidence=json::ParseFile((probe/"result.json").string());
        pipeline_record.Set("occupancy_pass_"+std::to_string(pass),evidence);
        bool changed=false;
        for(auto const& resource:evidence.At("candidates").AsArray("occupancy candidates")) {
          int i=int(resource.At("rank").AsNumber("occupancy rank"));
          if(i<0 || i>=int(candidates.size()) ||
             resource.At("binary_sha256").AsString("binary hash")!=
                 candidates[i].At("binary_sha256").AsString("candidate binary hash"))
            throw std::runtime_error("occupancy result does not bind a compiled finalist");
          int actual=int(resource.At("effective_residency").AsNumber("measured residency"));
          candidates[i].Set("occupancy",resource);
          if(actual<1) {candidates[i].Set("status","occupancy_rejected");continue;}
          auto c=result.ranked[i].coordinates;
          if(actual==c.residency)continue;
          if(pass==2)throw std::runtime_error("native occupancy correction did not converge");
          mlir::MLIRContext context;
          auto imported=solver::ImportSm90Dense(o.input,c,context);
          solver::Sm90NativeCtgCache cache;
          auto corrected=solver::RepriceSm90NativeResidency(space,c,actual,o.operator_sum,
              [&](solver::Sm90DenseCoordinates const& coordinate,bool isolated) {
                return solver::Sm90NativeCtgScore(imported,work,coordinate,cost,context,cache,isolated);
              });
          if(!corrected) {
            candidates[i].Set("status","occupancy_coordinate_rejected");continue;
          }
          c=corrected->coordinates;double score=corrected->replay_ns;
          candidates[i].Set("occupancy_pool_score_ns",corrected->pool_score_ns);
          candidates[i].Set("occupancy_rescore_ms",corrected->evaluation_ms);
          auto base=directory/("candidate-"+std::to_string(i)+"-occupancy-"+std::to_string(pass));
          configs[i]=base.string()+".tmexec.json";
          std::ofstream(configs[i])<<c.ToJson().Dump(2)<<'\n';
          result.ranked[i].coordinates=c;result.ranked[i].prediction=cost.Predict(work,c);
          result.ranked[i].prediction.step_ns=score;
          result.ranked[i].origin=corrected->structural_class;
          ranked[i].Set("origin",corrected->structural_class);
          ranked[i].Set("coordinates",c.ToJson());ranked[i].Set("prediction",result.ranked[i].prediction.ToJson());
          auto binary=base.string()+".so";
          NativeCandidateCounts counts;
          auto rejected=TryMeasuredNativeCompile(counts,[&] {
            CompileNativeDenseArtifact(o.input,binary,configs[i].string(),o.target,o.batch,
                "",o.split_compile,"solver",json::Object{{"selection","occupancy-corrected"},{"rank",i}});
          });
          candidates[i].Set("coordinates",configs[i].string());candidates[i].Set("predicted_ns",score);
          if(rejected) {
            candidates[i].Set("status","resource_rejected");candidates[i].Set("compile_result",*rejected);
          }else {
            candidates[i].Set("status","compiled");candidates[i].Set("binary",binary);
            candidates[i].Set("binary_sha256",Hash(binary));changed=true;
          }
          save();
        }
        save();
        if(!changed)break;
      }
    }
    if(!o.measured) {
      winner=-1;
      for(unsigned i=0;i<candidates.size();++i)
        if(candidates[i].At("status").AsString("status")=="compiled" &&
           (winner<0 || result.ranked[i].prediction.step_ns<result.ranked[winner].prediction.step_ns))
          winner=i;
      if(winner<0)throw std::runtime_error("no resource-verified native finalist remains");
    }
  }
  if(o.measured) {
    auto list=directory/"candidates.json";
    std::ofstream(list)<<json::Value(candidates).Dump(2)<<'\n';
    // One P-SEL session measures every compiled candidate in interleaved order.
    auto out=directory/"measure";
    auto command=o.measure_command+" --candidates "+Quote(list.string())+" --out "+Quote(out.string());
    int status=std::system(command.c_str());
    if(status)throw std::runtime_error("native P-SEL measurement failed: "+command);
    measurement=json::ParseFile((out/"result.json").string());
    winner=int(measurement.At("winner_rank").AsNumber("P-SEL winner"));
    if(winner<0 || winner>=int(candidates.size()) ||
       candidates[winner].At("status").AsString("candidate status")!="compiled" ||
       measurement.At("winner_binary_sha256").AsString("winner binary")!=
           Hash(candidates[winner].At("binary").AsString("binary")))
      throw std::runtime_error("P-SEL result does not name a compiled candidate");
  }
  if(Hash(o.target)!=target_sha || Hash(o.input)!=input_sha)
    throw std::runtime_error("native search inputs changed during selection");
  auto selected=fs::path(requested.string()+".tmexec.json");
  fs::copy_file(configs[winner],selected);
  auto receipt=save();
  CompileNativeDenseArtifact(o.input,o.output,selected.string(),o.target,o.batch,o.dump_cg,
      o.split_compile,"solver",receipt);
  if(compiled_finalists && requested.extension()==".so") {
    auto reference=fs::path(candidates[winner].At("binary").AsString("winner binary"));
    auto command="python3 "+Quote(std::string(TILEMEGA_SOURCE_DIR)+
        "/scripts/check_sm90r2_rebuild.py")+" --reference "+Quote(reference.string())+
        " --candidate "+Quote(requested.string())+" --out "+Quote(requested.string()+".rebuild");
    if(std::system(command.c_str()))
      throw std::runtime_error("selected native artifact changed during final rebuild");
  }else if(compiled_finalists && Hash(requested.string())!=
      Hash(candidates[winner].At("binary").AsString("winner binary")+".cu")) {
    throw std::runtime_error("selected native source changed after resource verification");
  }
  return 0;
}
}  // namespace tilemega::commands
