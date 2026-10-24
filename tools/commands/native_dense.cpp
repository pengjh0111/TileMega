// SPDX-License-Identifier: BSD-3-Clause
#include "NativeDense.h"
#include "Toolchain.h"
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearch.h>
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
  if(fs::exists(requested))throw std::invalid_argument("retain native artifact evidence; choose a new output path");
  auto c=solver::Sm90DenseCoordinates::FromJson(json::ParseFile(coordinates));
  auto target=TargetSpec::FromJson(target_path);
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
      {"runtime_target",fs::absolute(target_path).string()},{"target_spec",json::Parse(target.ToJson())},
      {"native_dense",c.ToJson()},{"native_origin",origin},{"native_selection",selection},{"gemms",gemms},
      {"export_sha256",Hash(input)},{"coordinates_sha256",Hash(coordinates)},
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
  auto command=Quote(NvccPath())+" -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a"
      " --expt-relaxed-constexpr -shared -Xcompiler -fPIC -cudart shared -Xptxas -v"
      " --split-compile "+std::to_string(split_compile)+" -I"+Quote(root+"/include")+
      " -I"+Quote(root+"/third_party/cutlass/include")+" "+Quote(cu.string())+" -o "+Quote(requested.string());
  std::ofstream(requested.string()+".build_command.txt")<<command<<'\n';
  auto log=requested.string()+".ptxas.log";
  // Candidate searches may run concurrently; bound and serialize their nvcc
  // process groups while retaining the unwrapped command for reproducibility.
  auto bounded="python3 "+Quote(root+"/scripts/run_sm90r1_build.py")+
      " --receipt "+Quote(requested.string()+".build_limits.json")+
      " --max-gib 48 --timeout 5400 --lock /tmp/tilemega_sm90r2_native_build.lock -- "+command;
  if(std::system((bounded+" >"+Quote(log)+" 2>&1").c_str()))
    throw std::runtime_error("native nvcc compilation failed; inspect "+log);
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
  if(o.measure_top<1 || o.measure_top>6 || o.passes<1 || o.budget_ms<1 ||
      (o.measured && o.measure_command.empty()) || (!o.measured && !o.measure_command.empty()))
    throw std::invalid_argument("native solve requires measured top 1..6 and an explicit measure command");
  auto requested=fs::absolute(o.output);
  if(fs::exists(requested))throw std::invalid_argument("retain native solve evidence; choose a new output path");
  auto directory=fs::path(requested.string()+".search");
  if(fs::exists(directory))throw std::invalid_argument("retain prior native search evidence");
  fs::create_directories(directory);
  auto target=TargetSpec::FromJson(o.target);
  auto target_sha=Hash(o.target),input_sha=Hash(o.input);
  auto calibration=json::ParseFile(o.target).At("sm90_dense_decode");
  if(calibration.At("native_gemm_header_sha256").AsString("native implementation hash")!=
      Hash(std::string(TILEMEGA_SOURCE_DIR)+"/include/tilemega/Backend/Sm90DenseGemm.h"))
    throw std::invalid_argument("native prices belong to a different GEMM implementation");
  for(auto const& header:calibration.At("native_headers").AsObject("native device inputs"))
    if(Hash(std::string(TILEMEGA_SOURCE_DIR)+"/"+header.first)!=
        header.second.AsString("calibrated device header"))
      throw std::invalid_argument("native prices belong to a different stage program");
  auto fit_status=calibration.At("fit_status").AsString("native fit status");
  bool validated=fit_status=="fitted";
  // E-3 is checkpoint/final acceptance, not a prerequisite for collecting
  // development selections that diagnose and improve the same price model.
  if(o.measured && !validated && fit_status!="initial_overlap_fitted")
    throw std::invalid_argument("measured native solve requires an initial overlap fit or validated calibration");
  solver::Sm90DenseCostModel cost(target,calibration);
  auto bridge=frontend::ReadExportBridge(o.input);
  frontend::ServingOptions options;options.dense_attention_chunks=1;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  solver::Sm90DenseWorkload work;work.shapes=solver::Sm90DenseShapes(plan);work.batch=o.batch;
  for(auto const& stage:plan.stages)if(stage.kind==frontend::PlanTaskKind::kFusedAttention) {
    ++work.layers;work.head_dim=stage.width;work.queries=stage.group;
    work.qk_norm=stage.operands[5]!=0xffffffffu;
  }
  solver::Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target};
  solver::Sm90DenseSearchOptions search;search.passes=o.passes;search.budget_ms=o.budget_ms;
  solver::Sm90DenseSearchResult result;
  try {result=solver::SearchSm90Dense(space,work,cost,search);}
  catch(solver::Sm90DenseCalibrationPending const& pending) {
    auto report=pending.report;
    report.Set("target_sha256",target_sha);report.Set("export_sha256",input_sha);
    report.Set("solve_command",o.solve_command);
    auto path=directory/"calibration-pending.json";
    std::ofstream output(path);
    if(!output)throw std::runtime_error("cannot write native calibration_pending receipt");
    output<<report.Dump(2)<<'\n';output.close();
    if(!output)throw std::runtime_error("cannot finish native calibration_pending receipt");
    std::cerr<<"NATIVE_DENSE calibration_pending "<<path.string()<<'\n';
    return 2;
  }
  json::Array ranked,measurements,evaluated;
  std::vector<fs::path> configs;
  for(unsigned i=0;i<result.ranked.size();++i) {
    auto path=directory/("candidate-"+std::to_string(i)+".tmexec.json");
    std::ofstream(path)<<result.ranked[i].coordinates.ToJson().Dump(2)<<'\n';configs.push_back(path);
    ranked.emplace_back(json::Object{{"rank",int(i)},{"origin",result.ranked[i].origin},
        {"coordinates",result.ranked[i].coordinates.ToJson()},
        {"prediction",result.ranked[i].prediction.ToJson()}});
  }
  if(!o.evaluate_configs.empty()) {
    auto values=json::ParseFile(o.evaluate_configs);
    for(auto const& item:values.AsArray("native evaluation configurations")) {
      auto c=solver::Sm90DenseCoordinates::FromJson(item);
      if(!space.Legal(c))throw std::invalid_argument("native evaluation coordinate is illegal");
      evaluated.emplace_back(json::Object{{"coordinates",c.ToJson()},
          {"prediction",cost.Request(work,c).ToJson()}});
    }
  }
  auto save=[&](int winner) {
    json::Value receipt=json::Object{{"schema",1},{"selection",o.measured?"measure":"predicted"},
        {"e3_validated",validated},{"development_only",!validated},
        {"target_sha256",target_sha},{"export_sha256",input_sha},{"solve_command",o.solve_command},
        {"calibration",calibration},{"search_ms",result.elapsed_ms},
        {"evaluated",double(result.evaluated)},{"converged",result.converged},
        {"budget_exhausted",result.budget_exhausted},{"ranked",ranked},
        {"evaluated_configs",evaluated},{"measurements",measurements},{"winner",winner}};
    std::ofstream(directory/"receipt.json")<<receipt.Dump(2)<<'\n';return receipt;
  };
  int winner=o.measured?-1:0;double best=std::numeric_limits<double>::infinity();save(winner);
  ValidateNativeSelectionSearch(o.measured,result.converged,result.budget_exhausted);
  int measured_unique=0;
  std::map<std::string,int> program_keys;
  if(o.measured)for(int i=0;i<int(result.ranked.size()) && measured_unique<o.measure_top;++i) {
    auto binary=directory/("candidate-"+std::to_string(i)+".so");
    CompileNativeDenseArtifact(o.input,binary.string(),configs[i].string(),o.target,o.batch,
        "",o.split_compile,"solver",json::Object{{"selection","candidate"},{"rank",i}});
    auto key_path=directory/("candidate-"+std::to_string(i)+".program-key.json");
    auto key_command="python3 "+Quote(std::string(TILEMEGA_SOURCE_DIR)+
        "/scripts/native_sm90r2_program_key.py")+" --binary "+Quote(binary.string())+
        " --out "+Quote(key_path.string());
    if(std::system(key_command.c_str()))throw std::runtime_error("native program-key audit failed");
    auto key=json::ParseFile(key_path).At("program_key").AsString("native program key");
    auto inserted=program_keys.emplace(key,i);
    if(!inserted.second) {
      measurements.emplace_back(json::Object{{"rank",i},{"status",0},{"measured",false},
          {"reason","compiler-equivalent register lookahead"},
          {"duplicate_of",inserted.first->second},{"program_key",key}});
      save(winner);continue;
    }
    ++measured_unique;
    auto out=directory/("measure-"+std::to_string(i));
    auto command=o.measure_command+" --so "+Quote(binary.string())+
        " --native-config "+Quote(configs[i].string())+" --out "+Quote(out.string());
    auto begin=std::chrono::steady_clock::now();int status=std::system(command.c_str());
    json::Value row=json::Object{{"rank",i},{"command",command},{"status",status},
        {"measured",true},{"program_key",key},
        {"elapsed_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}};
    if(!status) {
      auto measured=json::ParseFile(out/"result.json");
      if(measured.At("accepted").AsBool("accepted native measurement") &&
          measured.At("binary_sha256").AsString("measured binary") ==Hash(binary.string()) &&
          measured.At("c1_pass").AsBool("C-1") && measured.At("c2_pass").AsBool("C-2")) {
        double ns=measured.At("tpot_ns").AsNumber("native measured TPOT");
        if(!std::isfinite(ns) || ns<=0)throw std::runtime_error("invalid native measured TPOT");
        row.Set("result",measured);
        if(ns<best) {best=ns;winner=i;}
      }else throw std::runtime_error("native callback lacks accepted correctness and artifact evidence");
    }
    measurements.push_back(row);save(winner);
  }
  if(winner<0)throw std::runtime_error("no native candidate passed resource/correctness/measurement checks");
  if(Hash(o.target)!=target_sha || Hash(o.input)!=input_sha)
    throw std::runtime_error("native search inputs changed during selection");
  auto selected=fs::path(requested.string()+".tmexec.json");
  fs::copy_file(configs[winner],selected);
  auto receipt=save(winner);
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
