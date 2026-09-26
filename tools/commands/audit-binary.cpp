// SPDX-License-Identifier: BSD-3-Clause
#include "Toolchain.h"
#include <tilemega/Support/Json.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace tilemega::commands::audit_binary {
namespace {
std::string Quote(std::string const& value) {
  std::string out="'";
  for(char c:value) out+=c=='\'' ? "'\\''" : std::string(1,c);
  return out+"'";
}
std::string Capture(std::string const& command) {
  FILE* pipe=popen(command.c_str(),"r");
  if(!pipe)throw std::runtime_error("cannot start: "+command);
  std::string result; char buffer[4096];
  while(fgets(buffer,sizeof(buffer),pipe))result+=buffer;
  if(pclose(pipe)!=0)throw std::runtime_error("command failed: "+command);
  return result;
}
std::string Cuobjdump() {
  if(auto* path=std::getenv("CUOBJDUMP"))return path;
  return (std::filesystem::path(NvccPath()).parent_path()/"cuobjdump").string();
}
json::Value Scan(std::string const& binary,std::string const& dump) {
  std::regex kernel("tilemega_l[12]_kernel");
  std::regex fp64(R"(\b(DADD|DMUL|DFMA|DSETP|F2F\.F64|F2F\.F32\.F64|I2F\.F64|F64)\b)");
  std::regex async(R"(\b(UTMALDG|UBLKCP|SYNCS[^ ;]*|ACQBULK|PREEXIT|LDGSTS)\b)");
  std::map<std::string,int> counts,instructions;
  std::map<std::string,json::Array> matches;
  std::string function,line; std::istringstream input(dump); int total=0;
  while(std::getline(input,line)) {
    auto at=line.find("Function : ");
    if(at!=std::string::npos) {
      function=line.substr(at+11);
      if(std::regex_search(function,kernel))counts.emplace(function,0);
      continue;
    }
    if(!std::regex_search(function,kernel))continue;
    auto start=line.find_first_not_of(" \t");
    if(start==std::string::npos || line.compare(start,2,"/*")!=0)continue;
    auto op=line.substr(0,line.find(';'));
    if(std::regex_search(op,fp64)) {++counts[function];++total;matches[function].push_back(line);}
    std::smatch match;if(std::regex_search(op,match,async))++instructions[match.str()];
  }
  if(counts.empty())throw std::runtime_error("no serving megakernel found in "+binary);
  json::Value by(json::Object{}), evidence(json::Object{}), ops(json::Object{});
  for(auto const& [name,count]:counts){by.Set(name,count);evidence.Set(name,matches[name]);}
  for(auto const& [name,count]:instructions)ops.Set(name,count);
  return json::Object{{"binary",binary},{"fp64_total",total},{"fp64_by_kernel",by},
                      {"instructions",evidence},{"async_instructions",ops}};
}
void Write(std::string const& path,json::Value const& report) {
  if(!path.empty()) {
    auto parent=std::filesystem::path(path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot write "+path);
    out<<report.Dump()<<'\n';
  }
  std::cout<<report.Dump()<<'\n';
}
}
int RunSass(int argc,char** argv) {
  std::string output,tool=Cuobjdump();std::vector<std::string> binaries;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(arg=="--out" && i+1<argc)output=argv[++i];
    else if(arg=="--cuobjdump" && i+1<argc)tool=argv[++i];
    else if(arg.rfind("--",0)==0)throw std::runtime_error("unknown SASS option: "+arg);
    else binaries.push_back(arg);
  }
  if(binaries.empty())throw std::runtime_error("usage: tilemega audit sass BINARY... [--out FILE]");
  json::Array records;int total=0;
  for(auto const& binary:binaries) {
    auto item=Scan(binary,Capture(Quote(tool)+" -sass "+Quote(binary)));
    total+=int(item.At("fp64_total").AsNumber("fp64_total"));records.push_back(item);
  }
  Write(output,json::Object{{"binaries",records},{"fp64_total",total}});
  return total?1:0;
}
int RunArch(int argc,char** argv) {
  std::string source,output;std::vector<std::string> arches;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(i+1==argc)throw std::runtime_error("missing arch audit argument");
    if(arg=="--cu")source=argv[++i];else if(arg=="--out")output=argv[++i];
    else if(arg=="--arch")arches.push_back(argv[++i]);
    else throw std::runtime_error("unknown arch audit option: "+arg);
  }
  if(source.empty()||output.empty()||arches.empty())throw std::runtime_error(
      "usage: tilemega audit arch --cu PLAN.cu --arch sm_80 [--arch sm_120] --out DIR");
  std::filesystem::create_directories(output);json::Array records;int failed=0;
  auto supported=Capture(Quote(NvccPath())+" --list-gpu-code");
  std::string root=TILEMEGA_SOURCE_DIR;
  for(auto const& arch:arches) {
    if(!std::regex_match(arch,std::regex("sm_[0-9]+a?")))throw std::runtime_error("invalid arch: "+arch);
    json::Value record(json::Object{{"arch",arch}});
    if(supported.find(arch+"\n")==std::string::npos) {record.Set("status","UNCHECKED: nvcc unsupported");records.push_back(record);continue;}
    auto stem=(std::filesystem::path(output)/arch).string();
    std::string command=Quote(NvccPath())+" -std=c++17 -O3 -c -Xptxas=-v --expt-relaxed-constexpr -arch="+
        Quote(arch)+" -I"+Quote(root+"/include")+" -I"+Quote(root+"/third_party/cutlass/include")+
        " -I"+Quote(root+"/third_party/cutlass/tools/util/include")+
        " -I"+Quote(root+"/third_party/cutlass/test")+" "+Quote(source)+" -o "+Quote(stem+".o");
    std::ofstream(stem+".command.txt")<<command<<'\n';
    int code=std::system((command+" >"+Quote(stem+".ptxas.txt")+" 2>&1").c_str());
    record.Set("status",code?"FAIL":"PASS");record.Set("command",command);
    if(code)++failed;
    else {
      auto sass=Capture(Quote(Cuobjdump())+" -sass "+Quote(stem+".o"));
      std::ofstream(stem+".sass.txt")<<sass;
      record.Set("sass",Scan(stem+".o",sass));
    }
    records.push_back(record);
  }
  Write((std::filesystem::path(output)/"report.json").string(),json::Object{{"architectures",records}});
  return failed?1:0;
}
} // namespace tilemega::commands::audit_binary
