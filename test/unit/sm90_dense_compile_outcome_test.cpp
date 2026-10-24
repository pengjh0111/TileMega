// SPDX-License-Identifier: BSD-3-Clause
#include "NativeCompileOutcome.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <limits>
using namespace tilemega;
using namespace tilemega::commands;
int main(int argc,char** argv) {
  std::vector<std::string> command{"/usr/local/cuda/bin/nvcc","-rdc=true","unit.cu","-o","candidate.so"};
  json::Array args;for(auto const& v:command)args.emplace_back(v);
  json::Value limits=json::Object{{"status","build_failed"},{"exit_code",255},
      {"classification","compilation; not a numerical test"},{"command",args},
      {"started",1},{"ended",2},{"pid",3}};
  std::string entry="_ZN8tilemega7codegen12native_dense24sm90_dense_native_kernelIExampleEv";
  std::string callee="_ZN8tilemega7backend13Sm90DenseGemm7ConsumeEv";
  auto diagnostic=[&](std::string const& target,int limit=168,int registers=254) {
    return "ptxas error   : Entry function '"+target+"' with max regcount of "+std::to_string(limit)+
        " calls function '"+callee+"' with regcount of "+std::to_string(registers)+"\n";
  };
  auto log=diagnostic(entry)+"ptxas info    : Used 120 registers, used 2 barriers\n"+
      "    8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\n";
  auto classify=[&](json::Value const& r,std::string const& text,bool normal=true) {
    return NativeResourceDiagnostics(r,command,168,normal,text);
  };
  auto result=classify(limits,log);assert(result.size()==1 && result[0].entry==entry &&
      result[0].callee==callee && result[0].entry_limit==168 && result[0].callee_registers==254);
  for(auto status:{"completed","build_timeout","cancelled","build_spawn_failed","running"}) {
    auto r=limits;r.Set("status",status);assert(classify(r,log).empty());
  }
  for(int code:{0,1,-9,-15,137}) {auto r=limits;r.Set("exit_code",code);assert(classify(r,log).empty());}
  assert(classify(limits,log,false).empty());
  auto changed=command;changed[1]="-rdc=false";
  assert(NativeResourceDiagnostics(limits,changed,168,true,log).empty());
  assert(NativeResourceDiagnostics(limits,command,169,true,log).empty());
  auto r=limits;r.Set("classification","numerical test");assert(classify(r,log).empty());
  r=limits;r.Set("ended",0);assert(classify(r,log).empty());
  r=limits;r.Set("pid",0);assert(classify(r,log).empty());
  r=limits;r.Set("pid",std::numeric_limits<double>::quiet_NaN());assert(classify(r,log).empty());
  for(auto suffix:{"ptxas fatal   : Internal error\n","Killed\n","out of memory\n",
      "nvlink error   : unresolved symbol\n","unit.cu: error: invalid template\n",
      "ptxas info    : internal compiler error\n","unexpected driver output\n"})
    assert(classify(limits,log+suffix).empty());
  assert(classify(limits,log+diagnostic(entry)).empty());
  assert(classify(limits,diagnostic("other_kernel")).empty());
  assert(classify(limits,diagnostic(entry,168,168)).empty());
  assert(classify(limits,diagnostic(entry,0,254)).empty());
  assert(classify(limits,diagnostic(entry,168,257)).empty());
  assert(classify(limits,log.substr(0,40)).empty());
  assert(!NativeRegisterDiagnostic("ptxas error   : Entry function '"+entry+
      "' with max regcount of 999999999999999999999 calls function '"+callee+"' with regcount of 254"));
  auto long_entry=entry+std::string(30000,'A');
  assert(classify(limits,diagnostic(long_entry)).front().entry==long_entry);

  NativeCandidateCounts counts;counts.measured_unique=2;int keys=0,timings=0;
  auto rejected=TryMeasuredNativeCompile(counts,[]{throw NativeResourceRejected("candidate.compile-result.json");});
  if(!rejected) {++keys;++timings;}
  assert(rejected && *rejected=="candidate.compile-result.json" && keys==0 && timings==0 &&
      counts.compile_attempts==1 && counts.resource_rejections==1 && counts.compiled_candidates==0 && counts.measured_unique==2);
  auto built=TryMeasuredNativeCompile(counts,[]{});assert(!built && counts.compiled_candidates==1 && counts.measured_unique==2);
  bool propagated=false;
  try {TryMeasuredNativeCompile(counts,[]{throw std::runtime_error("compiler crashed");});}
  catch(std::runtime_error const&) {propagated=true;}
  assert(propagated && counts.compile_attempts==3 && counts.compiled_candidates==1 && counts.resource_rejections==1);
  if(argc==3) {
    auto actual=json::ParseFile(argv[1]);std::ifstream input(argv[2]);assert(input);
    std::string raw((std::istreambuf_iterator<char>(input)),{});std::vector<std::string> actual_args;
    for(auto const& a:actual.At("command").AsArray("fixture command"))actual_args.push_back(a.AsString("arg"));
    auto witness=NativeResourceDiagnostics(actual,actual_args,168,true,raw);
    assert(witness.size()==2 && witness[0].entry_limit==168 && witness[1].entry_limit==168 &&
        witness[0].callee_registers==254 && witness[1].callee_registers==254);
  }else assert(argc==1);
}
