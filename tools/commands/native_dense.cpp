// SPDX-License-Identifier: BSD-3-Clause
#include "NativeDense.h"
#include "Toolchain.h"
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/raw_ostream.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

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
int CompileNativeDenseFixed(std::string const& input,std::string const& output,
    std::string const& coordinates,std::string const& target_path,int batch,
    std::string const& dump_cg,int split_compile) {
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
  auto module=solver::MaterializeSm90Dense(imported,c,batch,target,context,"fixed");
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
        {"stages",shape.slots},{"split_k",1},{"warps_n",shape.warps_n},{"warps_k",shape.warps_k}});
  }
  json::Object manifest={{"identity_schema",1},{"model",fs::path(input).stem().string()},
      {"phase","decode"},{"runtime_batch",batch},{"capacity",1088},{"prompt_len",64},
      {"past_lo",64},{"past_hi",1086},{"pg","native"},{"mode","L1"},{"loop",1},
      {"residency",c.residency},{"grid",target.res.num_sms*c.residency},{"kappa",1},
      {"runtime_target",fs::absolute(target_path).string()},{"target_spec",json::Parse(target.ToJson())},
      {"native_dense",c.ToJson()},{"native_origin","fixed"},{"gemms",gemms},
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
  if(std::system((command+" >"+Quote(log)+" 2>&1").c_str()))
    throw std::runtime_error("native nvcc compilation failed; inspect "+log);
  identity(false);
  json::Value execution=json::Object{{"decode_mode","L1"},{"decode_loop",1},
      {"prefill_mode","L1"},{"source","native static program"}};
  std::ofstream(requested.string()+".serving.json")<<execution.Dump(2)<<'\n';
  std::cerr<<"NATIVE_DENSE origin=fixed grid="<<target.res.num_sms*c.residency
      <<" stages="<<imported.plan.stages.size()<<" ctg="<<cg.string()<<'\n';
  return 0;
}
}  // namespace tilemega::commands
