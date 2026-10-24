// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace tilemega::commands {
struct NativeRegisterCallLimit {
  std::string entry,callee;
  int entry_limit=0,callee_registers=0;
};
inline bool NativeSymbol(std::string const& value) {
  return !value.empty() && std::all_of(value.begin(),value.end(),[](unsigned char c) {
    return std::isalnum(c) || c=='_' || c=='$' || c=='.';
  });
}
inline std::optional<NativeRegisterCallLimit> NativeRegisterDiagnostic(std::string const& line) {
  std::string const prefix="ptxas error   : Entry function '";
  std::string const first="' with max regcount of ",second=" calls function '",third="' with regcount of ";
  if(line.rfind(prefix,0)!=0)return {};
  auto a=line.find(first,prefix.size());if(a==std::string::npos)return {};
  auto b=line.find(second,a+first.size());if(b==std::string::npos)return {};
  auto c=line.find(third,b+second.size());if(c==std::string::npos)return {};
  auto number=[](std::string const& s) {
    if(s.empty() || s.size()>3 || !std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isdigit(c);}))return 0;
    int value=std::stoi(s);return value>=1 && value<=256?value:0;
  };
  NativeRegisterCallLimit result{line.substr(prefix.size(),a-prefix.size()),
      line.substr(b+second.size(),c-b-second.size()),
      number(line.substr(a+first.size(),b-a-first.size())),number(line.substr(c+third.size()))};
  if(result.entry.rfind("_ZN8tilemega7codegen12native_dense24sm90_dense_native_kernel",0)!=0 ||
      !NativeSymbol(result.entry) || !NativeSymbol(result.callee) || !result.entry_limit ||
      result.callee_registers<=result.entry_limit)return {};
  return result;
}
inline bool NativeBoundedCommandMatches(json::Value const& bounded,
    std::vector<std::string> const& expected_command) {
  auto const& argv=bounded.At("command").AsArray("bounded command");
  if(argv.size()!=expected_command.size() || expected_command.empty() ||
      expected_command.front().find("nvcc")==std::string::npos)return false;
  for(std::size_t i=0;i<argv.size();++i)
    if(argv[i].AsString("bounded argument")!=expected_command[i])return false;
  return true;
}
// Unknown output stays a compiler error. This classifier never changes the
// bounded runner's original build_failed/exit_code receipt into success.
inline std::vector<NativeRegisterCallLimit> NativeResourceDiagnostics(
    json::Value const& bounded,std::vector<std::string> const& expected_command,
    int expected_entry_limit,bool wrapper_failed_normally,std::string const& log) {
  if(!wrapper_failed_normally || expected_entry_limit<1 || expected_entry_limit>256 ||
      bounded.At("status").AsString("bounded status")!="build_failed" ||
      bounded.At("classification").AsString("bounded classification")!="compilation; not a numerical test" ||
      bounded.At("exit_code").AsNumber("bounded exit")!=255)return {};
  if(!NativeBoundedCommandMatches(bounded,expected_command))return {};
  auto started=bounded.At("started").AsNumber("bounded start"),ended=bounded.At("ended").AsNumber("bounded end");
  double pid=bounded.At("pid").AsNumber("bounded pid");
  if(!std::isfinite(started) || !std::isfinite(ended) || !(ended>=started) ||
      !std::isfinite(pid) || pid<=0 || pid!=std::floor(pid))return {};
  std::istringstream input(log);std::string line,entry;std::vector<NativeRegisterCallLimit> result;
  bool unused_warning=false;
  while(std::getline(input,line)) {
    if(!line.empty() && line.back()=='\r')line.pop_back();
    auto diagnostic=NativeRegisterDiagnostic(line);
    if(diagnostic) {
      if(diagnostic->entry_limit!=expected_entry_limit || (!entry.empty() && entry!=diagnostic->entry))return {};
      if(std::any_of(result.begin(),result.end(),[&](auto const& r){return r.callee==diagnostic->callee;}))return {};
      entry=diagnostic->entry;result.push_back(*diagnostic);continue;
    }
    if(line.empty())continue;
    std::string lower=line;
    std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return std::tolower(c);});
    for(auto word:{"error","fatal","out of memory","cannot allocate memory","killed","segmentation"})
      if(lower.find(word)!=std::string::npos)return {};
    if(line.rfind("ptxas info    :",0)==0)continue;
    // The generated coordinate identity has this known harmless CUDA warning.
    if(line.find(": warning #177-D: variable \"tilemega::codegen::native_dense::kNativeCoordinateIdentity\" was declared but never referenced")!=std::string::npos) {
      unused_warning=true;continue;
    }
    if(unused_warning && (line.rfind("  constexpr char kNativeCoordinateIdentity[]=",0)==0 ||
        line.find_first_not_of(" ^\t")==std::string::npos ||
        line=="Remark: The warnings can be suppressed with \"-diag-suppress <warning-number>\""))continue;
    // Short PTXAS property lines; do not use recursive regex on huge mangles.
    std::istringstream properties(line);std::string f,s,l,b1,b2,b3,stack,frame,spill1,spill2,stores,loads,extra;
    if(properties>>f>>b1>>stack>>frame>>s>>b2>>spill1>>stores>>l>>b3>>spill2>>loads) {
      auto digits=[](std::string const& n){return !n.empty() && std::all_of(n.begin(),n.end(),[](unsigned char c){return std::isdigit(c);});};
      if(digits(f) && digits(s) && digits(l) && b1=="bytes" && b2=="bytes" && b3=="bytes" &&
          stack=="stack" && frame=="frame," && spill1=="spill" && spill2=="spill" &&
          stores=="stores," && loads=="loads" && !(properties>>extra))continue;
    }
    return {};
  }
  return result;
}
class NativeResourceRejected:public std::runtime_error {
 public:
  explicit NativeResourceRejected(std::string path):std::runtime_error("native resource-rejected compile; inspect "+path),receipt(std::move(path)) {
    if(receipt.empty())throw std::invalid_argument("resource rejection needs a retained receipt");
  }
  std::string receipt;
};
struct NativeCandidateCounts {
  int compile_attempts=0,compiled_candidates=0,resource_rejections=0;
  int equivalent_duplicates=0,measured_unique=0,accepted_measurements=0;
};
template<class Compile>
std::optional<std::string> TryMeasuredNativeCompile(NativeCandidateCounts& counts,Compile compile) {
  ++counts.compile_attempts;
  try {compile();++counts.compiled_candidates;return {};}
  catch(NativeResourceRejected const& error) {++counts.resource_rejections;return error.receipt;}
}
}  // namespace tilemega::commands
