// SPDX-License-Identifier: BSD-3-Clause
#include "NativeForward.h"
#include "NativeCompileOutcome.h"
#include "Toolchain.h"
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Solver/Sm90ForwardPlan.h>
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/raw_ostream.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
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
}  // namespace

static int CompileNativeForwardArtifact(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target_path,int batch,bool fuse,
    std::string const& dump_cg,int split_compile,std::string const& origin,
    json::Value const& selection={}) {
  namespace fs=std::filesystem;
  if(split_compile<1 || split_compile>8)
    throw std::invalid_argument("native split compilation must use 1 through 8 workers");
  auto requested=fs::absolute(output);
  bool shared=requested.extension()==".so";
  if(!shared && requested.extension()!=".cu")
    throw std::invalid_argument("native forward output must be .so or .cu");
  for(auto const& path:{requested.string(),requested.string()+".build_limits.json",
      requested.string()+".compile-result.json",requested.string()+".ptxas.log",
      requested.string()+".source.json",requested.string()+".build_command.txt"})
    if(fs::exists(path))throw std::invalid_argument("retain native artifact evidence; choose a new output path");
  auto input_hash=Hash(input),coordinates_hash=Hash(coordinates),target_hash=Hash(target_path);
  auto c=solver::Sm90ForwardCoordinates::FromJson(json::ParseFile(coordinates));
  auto target=TargetSpec::FromJson(target_path);
  mlir::MLIRContext context;
  auto plan=solver::BuildSm90ForwardPlan(input,batch,fuse);
  auto module=solver::MaterializeSm90Forward(plan,c,batch,target,context,origin);
  auto source=codegen::CouplingGraphToCUDA{}.Lower(*module);
  fs::create_directories(requested.parent_path());
  auto cu=shared?fs::path(requested.string()+".cu"):requested;
  std::ofstream(cu)<<source;
  auto cg=fs::path(dump_cg.empty()?requested.string()+".tmexec.mlir":dump_cg);
  std::error_code error;llvm::raw_fd_ostream ir(cg.string(),error);
  if(error)throw std::runtime_error("cannot write the native forward plan module");
  module->print(ir);ir<<'\n';ir.close();
  json::Array stage_rows;
  auto shapes=solver::Sm90ForwardShapesFromPlan(plan);
  for(std::size_t i=0;i<plan.stages.size();++i)
    stage_rows.emplace_back(json::Object{{"index",int(i)},
        {"body",solver::kSm90ForwardBodies[unsigned(shapes[i].body)]},
        {"representative",plan.stages[i].representative},
        {"coordinate",c.stages[i].ToJson()}});
  // Activation layouts, so a stage-level comparison can read each buffer back
  // and map it onto the exported graph's own tensors.
  json::Array buffer_rows;
  for(unsigned i=0;i<plan.buffers.size();++i) {
    auto const& b=plan.buffers[i];
    auto const& l=b.layout;
    json::Array logical,physical;
    for(unsigned axis=0;axis<4;++axis) {
      logical.emplace_back(double(l.logical[axis]));
      physical.emplace_back(double(l.physical[axis]));
    }
    buffer_rows.emplace_back(json::Object{{"index",int(i)},{"name",b.name},{"role",b.role},
        {"dtype",b.dtype},{"rank",int(l.rank)},
        {"layout",l.kind==codegen::DmLayout::kNHWC?"nhwc":"row_major"},
        {"logical",logical},{"physical",physical},
        {"halo_top",int(l.halo_top)},{"halo_left",int(l.halo_left)},
        {"constant",double(b.constant)},{"per_batch",double(b.per_batch)}});
  }
  // Which exported node each buffer holds: a fused group writes one buffer,
  // so a stage-level comparison must know every node that aliases it.
  json::Object node_buffers;
  for(auto const& [name,buffer]:plan.node_buffer)
    node_buffers.emplace_back(name,double(buffer));
  json::Object manifest={{"identity_schema",1},{"buffers",buffer_rows},
      {"node_buffers",node_buffers},{"model",fs::path(input).stem().string()},
      {"phase","forward"},{"runtime_batch",batch},{"batch_lo",batch},{"batch_hi",batch},
      {"seq",plan.serving_seq},{"dwpw_fuse",fuse},{"mode","L1"},{"loop",1},
      {"residency",c.residency},{"grid",target.res.num_sms*c.residency},
      {"runtime_target",fs::absolute(target_path).string()},
      {"target_spec",json::Parse(target.ToJson())},
      {"native_forward",c.ToJson()},{"native_origin",origin},{"native_selection",selection},
      {"stages",stage_rows},{"export_sha256",input_hash},{"coordinates_sha256",coordinates_hash},
      {"plan_sha256",Hash(cg.string())},{"plan_module",cg.string()}};
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
    if(std::system(command.c_str()))throw std::runtime_error("native forward artifact identity failed");
  };
  identity(true);
  std::vector<std::string> command_args={NvccPath(),"-rdc=true","-std=c++17","-O3",
      "-gencode","arch=compute_90a,code=sm_90a","--expt-relaxed-constexpr","-shared",
      "-Xcompiler","-fPIC","-cudart","shared","-Xptxas","-v","--split-compile",
      std::to_string(split_compile),"-I"+root+"/include","-I"+root+"/third_party/cutlass/include",
      cu.string(),"-o",requested.string()};
  std::string command;
  for(auto const& argument:command_args) {if(!command.empty())command+=' ';command+=Quote(argument);}
  auto command_path=requested.string()+".build_command.txt";
  std::ofstream(command_path)<<command<<'\n';
  auto snapshot_path=requested.string()+".source.json";
  auto snapshot=json::ParseFile(snapshot_path);
  std::map<std::string,std::string> frozen_inputs={{fs::absolute(input).string(),input_hash},
      {fs::absolute(coordinates).string(),coordinates_hash},
      {fs::absolute(target_path).string(),target_hash}};
  for(auto const& path:{cu.string(),cg.string(),requested.string()+".plan.json",snapshot_path,
      command_path,NvccPath()})
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
  auto unchanged=[&] {
    for(auto const& item:frozen_inputs)if(Hash(item.first)!=item.second)return false;
    return true;
  };
  if(!unchanged())throw std::runtime_error("native compile inputs changed before invocation");
  auto log=requested.string()+".ptxas.log";
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
  auto diagnostics=stable?NativeResourceDiagnostics(limits,command_args,255,normal_failure,
      log_text,kNativeForwardEntryPrefix):std::vector<NativeRegisterCallLimit>{};
  bool succeeded=status==0 && limits.At("status").AsString("bounded status")=="completed" &&
      limits.At("exit_code").AsNumber("bounded exit")==0 &&
      limits.At("classification").AsString("bounded classification")=="compilation; not a numerical test" &&
      NativeBoundedCommandMatches(limits,command_args) && stable;
  json::Object inputs;
  for(auto const& item:frozen_inputs)inputs.emplace_back(item.first,item.second);
  json::Array errors;
  for(auto const& error:diagnostics)errors.emplace_back(json::Object{{"entry",error.entry},
      {"callee",error.callee},{"entry_limit",error.entry_limit},
      {"callee_registers",error.callee_registers}});
  auto outcome_path=requested.string()+".compile-result.json";
  json::Value outcome=json::Object{{"schema",1},{"classifier","native_ptxas_register_call_limit_v1"},
      {"status",succeeded?"compiled":diagnostics.empty()?"compiler_error":"resource_rejected"},
      {"classification","compilation; not a numerical test"},{"system_status",status},
      {"bounded_status",limits.At("status")},
      {"bounded_exit_code",limits.Find("exit_code")?*limits.Find("exit_code"):json::Value{}},
      {"bounded_receipt",limits_path},{"bounded_receipt_sha256",Hash(limits_path)},
      {"log",log},{"log_sha256",Hash(log)},{"raw_command_sha256",Hash(command_path)},
      {"generated_source_sha256",Hash(cu.string())},
      {"source_snapshot_sha256",Hash(snapshot_path)},
      {"source_digest",snapshot.At("source_digest")},
      {"coordinates_sha256",coordinates_hash},{"target_sha256",target_hash},
      {"export_sha256",input_hash},{"input_hashes",inputs},{"inputs_unchanged",stable},
      {"diagnostics",errors},{"output_present",fs::exists(requested)},
      {"artifact_identity_generated_before_classification",false}};
  {std::ofstream retained(outcome_path);retained<<outcome.Dump(2)<<'\n';
    if(!retained)throw std::runtime_error("cannot retain native compile result");}
  if(!succeeded) {
    if(!diagnostics.empty())throw NativeResourceRejected(outcome_path);
    throw std::runtime_error("native nvcc compilation/evidence failed; inspect "+outcome_path);
  }
  identity(false);
  std::ofstream(requested.string()+".serving.json")
      <<json::Value(json::Object{{"forward_mode","L1"},{"forward_loop",1},
          {"source","native static forward program"}}).Dump(2)<<'\n';
  std::cerr<<"NATIVE_FORWARD origin="<<origin<<" grid="<<target.res.num_sms*c.residency
      <<" stages="<<plan.stages.size()<<" plan="<<cg.string()<<'\n';
  return 0;
}

int CompileNativeForwardFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target,int batch,bool fuse,
    std::string const& dump_cg,int split_compile) {
  return CompileNativeForwardArtifact(input,output,coordinates,target,batch,fuse,dump_cg,
      split_compile,"fixed");
}

int SolveNativeForward(NativeForwardSolveOptions const& o) {
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
  json::Array stale;
  for(auto const& header:constants.StaleHeaders(TILEMEGA_SOURCE_DIR))stale.emplace_back(header);
  auto plan=solver::BuildSm90ForwardPlan(o.input,o.batch,o.fuse);
  solver::Sm90ForwardCostModel cost(target,constants);
  solver::Sm90ForwardSearchSpace space{solver::Sm90ForwardShapesFromPlan(plan),target};
  auto result=solver::SearchSm90Forward(space,cost,o.measure_top);
  json::Array ranked,candidates;
  std::vector<fs::path> configs;
  for(unsigned i=0;i<result.ranked.size();++i) {
    auto path=directory/("candidate-"+std::to_string(i)+".tmexec.json");
    std::ofstream(path)<<result.ranked[i].coordinates.ToJson().Dump(2)<<'\n';
    configs.push_back(path);
    ranked.emplace_back(json::Object{{"rank",int(i)},{"origin",result.ranked[i].origin},
        {"coordinates",result.ranked[i].coordinates.ToJson()},
        {"prediction",result.ranked[i].prediction.ToJson()}});
  }
  json::Value measurement;int winner=0;
  auto save=[&] {
    json::Value receipt=json::Object{{"schema",1},{"selection",o.measured?"measure":"predicted"},
        {"target_sha256",target_sha},{"export_sha256",input_sha},{"solve_command",o.solve_command},
        {"batch",o.batch},{"dwpw_fuse",o.fuse},{"constants",constants.section},
        {"constants_from_older_body",stale},{"search_ms",result.elapsed_ms},
        {"evaluated",double(result.evaluated)},{"ranked",ranked},{"candidates",candidates},
        {"measurement",measurement},{"winner",winner}};
    std::ofstream(directory/"receipt.json")<<receipt.Dump(2)<<'\n';
    return receipt;
  };
  save();
  if(o.measured) {
    for(unsigned i=0;i<configs.size();++i) {
      auto binary=directory/("candidate-"+std::to_string(i)+".so");
      NativeCandidateCounts counts;
      auto rejected=TryMeasuredNativeCompile(counts,[&] {
        CompileNativeForwardArtifact(o.input,binary.string(),configs[i].string(),o.target,o.batch,
            o.fuse,"",o.split_compile,"solver",
            json::Object{{"selection","candidate"},{"rank",int(i)}});
      });
      json::Value row=json::Object{{"rank",int(i)},{"coordinates",configs[i].string()},
          {"predicted_ns",result.ranked[i].prediction.total_ns}};
      if(rejected) {row.Set("status","resource_rejected");row.Set("compile_result",*rejected);}
      else {
        row.Set("status","compiled");row.Set("binary",binary.string());
        row.Set("binary_sha256",Hash(binary.string()));
      }
      candidates.push_back(row);save();
    }
    auto list=directory/"candidates.json";
    std::ofstream(list)<<json::Value(candidates).Dump(2)<<'\n';
    // One P-DNN session measures every compiled candidate in A B C D D C B A.
    auto out=directory/"measure";
    auto command=o.measure_command+" --candidates "+Quote(list.string())+" --out "+Quote(out.string());
    if(std::system(command.c_str()))
      throw std::runtime_error("native P-DNN measurement failed: "+command);
    measurement=json::ParseFile((out/"result.json").string());
    winner=int(measurement.At("winner_rank").AsNumber("P-DNN winner"));
    if(winner<0 || winner>=int(candidates.size()) ||
       candidates[winner].At("status").AsString("candidate status")!="compiled" ||
       measurement.At("winner_binary_sha256").AsString("winner binary")!=
           Hash(candidates[winner].At("binary").AsString("binary")))
      throw std::runtime_error("P-DNN result does not name a compiled candidate");
  }
  if(Hash(o.target)!=target_sha || Hash(o.input)!=input_sha)
    throw std::runtime_error("native search inputs changed during selection");
  auto selected=fs::path(requested.string()+".tmexec.json");
  fs::copy_file(configs[winner],selected);
  auto receipt=save();
  CompileNativeForwardArtifact(o.input,o.output,selected.string(),o.target,o.batch,o.fuse,
      o.dump_cg,o.split_compile,"solver",receipt);
  return 0;
}

}  // namespace tilemega::commands
