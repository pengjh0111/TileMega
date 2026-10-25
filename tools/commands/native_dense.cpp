// SPDX-License-Identifier: BSD-3-Clause
#include "NativeDense.h"
#include "NativeCompileOutcome.h"
#include "Toolchain.h"
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <tilemega/Solver/Sm90NativeConstants.h>
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
  frontend::ServingOptions options;options.dense_attention_chunks=1;
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
  frontend::ServingOptions options;options.dense_attention_chunks=1;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  auto work=solver::Sm90DenseWorkloadFromPlan(plan,o.batch);
  solver::Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
  auto target_json=json::ParseFile(o.target);
  auto const* decisions=target_json.Find("sm90_native_decisions");
  if(!decisions)throw std::invalid_argument("target lacks sm90_native_decisions (prompt §11 decision tables)");
  space.weight_stream=decisions->At("weight_stream").AsBool("weight_stream");
  space.layer_loop=decisions->At("layer_loop").AsBool("layer_loop");
  solver::Sm90DenseSearchOptions search;search.top=o.measure_top;
  auto result=solver::SearchSm90Dense(space,work,cost,search);
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
    json::Value receipt=json::Object{{"schema",2},{"selection",o.measured?"measure":"predicted"},
        {"target_sha256",target_sha},{"export_sha256",input_sha},{"solve_command",o.solve_command},
        {"constants",constants.section},{"constants_from_older_body",stale},
        {"search_ms",result.elapsed_ms},{"evaluated",double(result.evaluated)},
        {"ranked",ranked},{"candidates",candidates},{"measurement",measurement},{"winner",winner}};
    std::ofstream(directory/"receipt.json")<<receipt.Dump(2)<<'\n';return receipt;
  };
  save();
  if(o.measured) {
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
  if(o.measured) {
    auto reference=directory/("candidate-"+std::to_string(winner)+".so");
    auto command="python3 "+Quote(std::string(TILEMEGA_SOURCE_DIR)+
        "/scripts/check_sm90r2_rebuild.py")+" --reference "+Quote(reference.string())+
        " --candidate "+Quote(requested.string())+" --out "+Quote(requested.string()+".rebuild");
    if(std::system(command.c_str()))
      throw std::runtime_error("selected native artifact changed during final rebuild");
  }
  return 0;
}
}  // namespace tilemega::commands
