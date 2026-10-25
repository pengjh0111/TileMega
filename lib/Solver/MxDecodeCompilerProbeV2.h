// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <zlib.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace tilemega::solver::compiler_probe_v2 {
using json::Value;
[[noreturn]] inline void Fail(std::string const& reason) {
  throw std::invalid_argument("mx decode v2 compiler probe: " + reason);
}
inline std::string Str(Value const& value, char const* name) { return value.At(name).AsString(name); }
inline int Int(Value const& value, char const* name) {
  double number=value.At(name).AsNumber(name);
  if (!std::isfinite(number) || number<0 || number>std::numeric_limits<int>::max() || std::floor(number)!=number)
    Fail(std::string("invalid attribute ")+name);
  return int(number);
}
inline std::string Digest(std::string const& bytes) {
  llvm::SHA256 hash;hash.update(bytes);return llvm::toHex(hash.final(),true);
}
inline std::string File(Value const& descriptor) {
  auto input=llvm::MemoryBuffer::getFile(Str(descriptor,"path"));
  if (!input) Fail("missing proof file "+Str(descriptor,"path"));
  std::string bytes=input.get()->getBuffer().str();
  if (Digest(bytes)!=Str(descriptor,"sha256")) Fail("proof file hash mismatch "+Str(descriptor,"path"));
  return bytes;
}
inline std::string Field(std::string const& block,std::size_t begin,std::size_t count) {
  auto end=block.find('\0',begin);if(end==std::string::npos || end>begin+count)end=begin+count;
  return block.substr(begin,end-begin);
}
inline std::size_t Octal(std::string value) {
  auto first=value.find_first_not_of(" \0",0,2);if(first==std::string::npos)return 0;
  auto end=value.find_first_of(" \0",first,2);auto token=value.substr(first,end-first);
  if(token.empty() || token.find_first_not_of("01234567")!=std::string::npos)Fail("malformed tar size/checksum");
  try {return std::stoull(token,nullptr,8);} catch(std::exception const&) {Fail("oversized tar member");}
}
inline std::string PaxPath(std::string const& body) {
  std::string path;
  for(std::size_t offset=0;offset<body.size();) {
    auto space=body.find(' ',offset);if(space==std::string::npos)Fail("malformed PAX record");
    std::string digits=body.substr(offset,space-offset);
    if(digits.empty() || digits.find_first_not_of("0123456789")!=std::string::npos)Fail("malformed PAX length");
    std::size_t length=0;try {length=std::stoull(digits);}catch(std::exception const&){Fail("oversized PAX record");}
    if(length<=space-offset+1 || length>body.size()-offset || body[offset+length-1]!='\n')Fail("truncated PAX record");
    std::string item=body.substr(space+1,offset+length-space-2);
    if(item.rfind("path=",0)==0)path=item.substr(5);
    offset+=length;
  }
  return path;
}
inline std::map<std::string,std::string> Archive(std::string const& path) {
  gzFile stream=gzopen(path.c_str(),"rb");if(!stream)Fail("cannot open source archive");
  std::string data;char buffer[32768];int amount=0;
  while((amount=gzread(stream,buffer,sizeof(buffer)))>0)data.append(buffer,amount);
  bool valid=amount==0 && gzeof(stream);int closed=gzclose(stream);
  if(!valid || closed!=Z_OK)Fail("corrupt source archive gzip stream");
  std::map<std::string,std::string> files;std::string pending,global;
  for(std::size_t offset=0;offset<data.size();) {
    if(data.size()-offset<512)Fail("truncated tar header");
    auto block=data.substr(offset,512);offset+=512;
    if(block.find_first_not_of('\0')==std::string::npos)break;
    std::size_t sum=0;for(std::size_t i=0;i<512;++i)sum+=(i>=148 && i<156)?32:static_cast<unsigned char>(block[i]);
    if(sum!=Octal(block.substr(148,8)))Fail("tar header checksum mismatch");
    auto size=Octal(block.substr(124,12));if(size>data.size()-offset)Fail("truncated tar body");
    std::string body=data.substr(offset,size);
    auto padded=(size+511)/512*512;if(padded>data.size()-offset)Fail("truncated tar padding");offset+=padded;
    char type=block[156];
    if(type=='x') {pending=PaxPath(body);continue;}
    if(type=='g') {global=PaxPath(body);continue;}
    if(type=='L') {pending=Field(body,0,body.size());continue;}
    auto name=Field(block,0,100);auto prefix=Field(block,345,155);if(!prefix.empty())name=prefix+"/"+name;
    if(!global.empty())name=global;
    if(!pending.empty())name=pending;
    pending.clear();
    if(type=='0' || type=='\0')if(!files.emplace(name,std::move(body)).second)Fail("duplicate archived source member");
  }
  return files;
}
inline bool Equal(Value const& a,Value const& b) {
  if(a.kind()!=b.kind())return false;
  if(a.kind()==Value::Kind::kObject) {
    auto const& aa=a.AsObject("object");auto const& bb=b.AsObject("object");
    if(aa.size()!=bb.size())return false;
    for(auto const& [name,value]:aa) {auto* other=b.Find(name);if(!other || !Equal(value,*other))return false;}
    return true;
  }
  if(a.kind()==Value::Kind::kArray) {
    auto const& aa=a.AsArray("array");auto const& bb=b.AsArray("array");if(aa.size()!=bb.size())return false;
    for(std::size_t i=0;i<aa.size();++i)if(!Equal(aa[i],bb[i]))return false;
    return true;
  }
  return a.Dump(0)==b.Dump(0);
}
inline void NormalCommand(std::string const& kind,Value const& key,
                          std::vector<std::string> const& command,Value const& binary) {
  if(std::count(command.begin(),command.end(),"-o")!=1)Fail("normal bench command requires exactly one output");
  auto output=std::find(command.begin(),command.end(),"-o");
  if(++output==command.end() || std::filesystem::canonical(*output)!=std::filesystem::canonical(Str(binary,"path")))
    Fail("normal bench command output differs from binary");
  std::vector<std::string> normal{command[0],"-std=c++17","-O3","-shared","-Xcompiler","-fPIC,-resource-usage",
    "-fno-strict-aliasing","-DTILEMEGA_SYNC_V3=1","-DTILEMEGA_WATCHDOG=0","-DMX5_TEST_THREADS="+std::to_string(Int(key,"threads"))};
  auto define=[&](char const* macro,char const* dimension){normal.emplace_back(std::string("-D")+macro+"="+std::to_string(Int(key,dimension)));};
  if(kind=="attention") {define("MX5_TEST_DIM","dim");define("MX5_TEST_Q","q");define("MX5_BENCH_PREFETCH_KIND","prefetch_kind");}
  else {define("MX5_TEST_TN","tn");define("MX5_TEST_DEPTH","depth");define("MX5_BENCH_PW","prefetch");}
  auto source="experiments/mx5/"+(kind=="gemm"?std::string("stage_bench.cu"):kind+"_stage_bench.cu");
  normal.insert(normal.end(),{"-Iinclude",source,"-o",*output});
  auto observed=command;auto mir=std::find(observed.begin(),observed.end(),"-print-after=stack-frame-layout");
  if(mir!=observed.end()) {
    if(mir==observed.begin() || *(mir-1)!="-mllvm")Fail("invalid MIR observation argument");
    observed.erase(mir-1,mir+1);
  }
  if(observed!=normal)Fail("diagnostic/forced-register flags differ from normal bench build policy");
}
inline void Schema3LegacyReference(Value const& reference) {
  if(reference.Find("compiler_probe_only"))return;
  for(char const* field:{"probe","build_metadata","metadata","command","library","source_archive","compiler"})
    if(reference.Find(field))Fail("schema3 unmarked R_next contains build evidence; modern marker required");
  if(reference.At("key").Find("library"))Fail("schema3 unmarked R_next contains library metadata");
}
inline void Validate(Value const& reference,std::string const& kind,int threads,bool primary=false,bool normal_policy=false) {
  if(!reference.At(primary?"primary_resource_rejection":"compiler_probe_only").AsBool("probe marker"))Fail(primary?"primary_resource_rejection must be true":"compiler_probe_only must be true");
  auto const& probe=reference.At("probe");
  if(!probe.At("query_only").AsBool("query_only") || Int(probe,"iterations")!=0 || Int(probe,"query_status")!=0)
    Fail("requires query_only, zero iterations and successful attribute status");
  auto const& key=reference.At("key");auto const& provenance=reference.At("provenance");auto const& resources=reference.At("resources");
  if(!primary && Int(key,kind=="attention"?"prefetch_kind":"prefetch")!=0)Fail("reference must use exact Pw0/PF0");
  int mtreg=Int(resources,"mtreg");if(mtreg<1)Fail("MTreg must be positive");
  // Diagnostic non-fit must not short-circuit validation of the remaining attributes.
  int local=Int(resources,"local_bytes"),shared=Int(resources,"static_smem_bytes"),resident=Int(resources,"resident_ctas");
  bool fits=mtreg<=(threads==576?227:255) && local==0 && shared==0 && resident==1;
  if(reference.At("fits").AsBool("fits")!=fits)Fail("diagnostic fits disagrees with observed attributes");
  auto const& binary=probe.At("binary");File(binary);
  if(Str(binary,"sha256")!=Str(provenance,"binary_sha256"))Fail("binary provenance mismatch");
  auto metadata=json::Parse(File(probe.At("build_metadata")));auto const& archive=probe.At("source_archive");File(archive);
  for(char const* field:{"source_sha256","binary_sha256","base_head"})
    if(Str(metadata,field)!=Str(provenance,field))Fail(std::string("build/provenance mismatch ")+field);
  if(Str(metadata,"body")!=kind || !metadata.At("bench").AsBool("bench") || Int(metadata,"returncode")!=0)
    Fail("reference requires a successful matching bench build");
  std::map<std::string,int> expected{{"MX5_TEST_THREADS",threads}};
  if(Int(metadata,"threads")!=threads)Fail("build threads mismatch");
  auto coordinate=[&](char const* meta,char const* dimension,char const* macro) {
    int actual=dimension?Int(key,dimension):0;if(Int(metadata,meta)!=actual)Fail(std::string("build coordinate mismatch ")+meta);
    expected[macro]=actual;
  };
  if(kind=="attention") {coordinate("dim","dim","MX5_TEST_DIM");coordinate("query","q","MX5_TEST_Q");coordinate("prefetch_kind",primary?"prefetch_kind":nullptr,"MX5_BENCH_PREFETCH_KIND");}
  else {coordinate("tn","tn","MX5_TEST_TN");coordinate("depth","depth","MX5_TEST_DEPTH");coordinate("pw",primary?"prefetch":nullptr,"MX5_BENCH_PW");}
  std::map<std::string,int> definitions;std::vector<std::string> command;
  auto const& arguments=metadata.At("command").AsArray("command");
  for(auto const& value:arguments)command.push_back(value.AsString("command argument"));
  for(std::size_t i=0;i<command.size();++i) {
    std::string argument=command[i];
    if(argument=="-D") {if(++i==command.size())Fail("missing macro argument");argument=command[i];}
    else if(argument.rfind("-D",0)==0)argument=argument.substr(2);else continue;
    auto equal=argument.find('=');auto macro=argument.substr(0,equal);if(!expected.count(macro))continue;
    auto raw=equal==std::string::npos?std::string():argument.substr(equal+1);
    if(raw.empty() || raw.find_first_not_of("0123456789")!=std::string::npos || definitions.count(macro))Fail("invalid or duplicate build macro");
    try {definitions[macro]=std::stoi(raw);}catch(std::exception const&){Fail("oversized build macro");}
  }
  if(definitions!=expected)Fail("build macros differ from exact reference coordinates");
  auto files=Archive(Str(archive,"path"));std::string source="experiments/mx5/"+(kind=="gemm"?std::string("stage_bench.cu"):kind+"_stage_bench.cu");
  if(!files.count(source) || std::find(command.begin(),command.end(),source)==command.end())Fail("archived bench/source command mismatch");
  llvm::SHA256 hash;
  for(auto const& [name,body]:files) {hash.update(name);hash.update(llvm::StringRef("\0",1));hash.update(body);hash.update(llvm::StringRef("\0",1));}
  if(llvm::toHex(hash.final(),true)!=Str(provenance,"source_sha256"))Fail("archived source digest mismatch");
  if(primary || normal_policy)NormalCommand(kind,key,command,binary);
  if(primary) {
    if(fits)Fail("primary rejection actually fits executable resource gates");
    auto const& compiler=probe.At("compiler");File(compiler);
    if(Str(reference,"compiler_binding")!="retained_executable_at_qualification"
       || reference.At("original_compiler_identity_recorded").AsBool("original_compiler_identity_recorded")
       || !reference.At("original_build_compiler_sha256").IsNull())Fail("primary must distinguish qualification-time compiler identity");
    if(command.empty() || !std::filesystem::path(command[0]).is_absolute()
       || std::filesystem::canonical(command[0])!=std::filesystem::canonical(Str(compiler,"path"))
       || ::access(Str(compiler,"path").c_str(),X_OK)!=0)Fail("primary retained executable/command binding mismatch");
    if(Str(reference,"build_policy")!="mx5_build_bodies_default_v1")Fail("primary rejection requires normal stage-bench build policy");
    auto const& descriptor=probe.At("raw_query");auto text=File(descriptor);
    if(descriptor.Find("line")) {
      int line=Int(descriptor,"line");if(line<1)Fail("invalid primary raw query line");
      std::istringstream stream(text);std::string selected;
      for(int i=0;i<line;++i)if(!std::getline(stream,selected))Fail("primary raw query line outside file");
      text=selected;
    }
    auto raw=json::Parse(text);
    if(std::filesystem::canonical(Str(raw,"library"))!=std::filesystem::canonical(Str(binary,"path")))Fail("primary raw query library differs from compiled binary");
    for(char const* field:{"kind","key","resources","provenance","fits"})
      if(!Equal(raw.At(field),reference.At(field)))Fail(std::string("primary raw query differs: ")+field);
    if(!raw.At("query_only").AsBool("query_only") || Int(raw,"iterations")!=0 || Int(raw,"query_status")!=0)
      Fail("primary raw query must be successful zero-execution attributes");
    for(char const* field:{"median_ns","p90_ns","rounds","spread","processes"})if(raw.Find(field))Fail("primary raw query contains execution costs");
  }
}
}  // namespace tilemega::solver::compiler_probe_v2
