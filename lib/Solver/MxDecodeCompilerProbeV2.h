// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
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
inline void Validate(Value const& reference,std::string const& kind,int threads) {
  if(!reference.At("compiler_probe_only").AsBool("compiler_probe_only"))Fail("compiler_probe_only must be true");
  auto const& probe=reference.At("probe");
  if(!probe.At("query_only").AsBool("query_only") || Int(probe,"iterations")!=0 || Int(probe,"query_status")!=0)
    Fail("requires query_only, zero iterations and successful attribute status");
  auto const& key=reference.At("key");auto const& provenance=reference.At("provenance");auto const& resources=reference.At("resources");
  if(Int(key,kind=="attention"?"prefetch_kind":"prefetch")!=0)Fail("reference must use exact Pw0/PF0");
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
  if(kind=="attention") {coordinate("dim","dim","MX5_TEST_DIM");coordinate("query","q","MX5_TEST_Q");coordinate("prefetch_kind",nullptr,"MX5_BENCH_PREFETCH_KIND");}
  else {coordinate("tn","tn","MX5_TEST_TN");coordinate("depth","depth","MX5_TEST_DEPTH");coordinate("pw",nullptr,"MX5_BENCH_PW");}
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
}
}  // namespace tilemega::solver::compiler_probe_v2
