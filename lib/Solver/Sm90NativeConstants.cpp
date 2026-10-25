// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <cstdio>

namespace tilemega::solver {
namespace {
std::string FileHash(std::string const& path) {
  auto command="sha256sum '"+path+"'";auto* pipe=popen(command.c_str(),"r");
  if(!pipe)throw std::runtime_error("cannot hash "+path);
  char text[65]{};auto* got=fgets(text,sizeof(text),pipe);pclose(pipe);
  return got?std::string(text):std::string();
}
double Positive(json::Value const& v,char const* what) {
  double x=v.AsNumber(what);
  if(!(x>0))throw std::invalid_argument(std::string("nonpositive native constant ")+what);
  return x;
}
}  // namespace

Sm90NativeConstants Sm90NativeConstants::FromTarget(json::Value const& target,std::string const& root) {
  auto const* found=target.Find("sm90_native_constants");
  if(!found)throw std::invalid_argument("target lacks sm90_native_constants; run the SM90R3 constants collector");
  Sm90NativeConstants c;c.section=*found;auto const& s=*found;
  if(s.At("schema").AsNumber("constants schema")!=1)
    throw std::invalid_argument("unknown sm90_native_constants schema");
  for(auto const& item:s.At("c_bar_ns").AsObject("c_bar"))
    c.barrier_[std::stoi(item.first)]=Positive(item.second,"c_bar");
  c.l2_latency=Positive(s.At("l2_latency_ns"),"l2 latency");
  c.dram_latency=Positive(s.At("dram_latency_ns"),"dram latency");
  c.dram_gbps=Positive(s.At("dram_gbps"),"dram bandwidth");
  c.l2_gbps=Positive(s.At("l2_gbps"),"l2 bandwidth");
  c.sm_gbps=Positive(s.At("sm_gbps"),"sm bandwidth");
  c.cp_async_gbps=Positive(s.At("activation_cp_async_gbps"),"cp.async activation bandwidth");
  c.epilogue_=s.At("epilogue_ns");c.embedding_=s.At("embedding_ns");c.argmax_=s.At("argmax_ns");
  c.attention_=s.At("attention_ns");
  return c;
}
double Sm90NativeConstants::Barrier(int grid) const {
  auto found=barrier_.find(grid);
  if(found==barrier_.end())throw Sm90NativeMissing("no measured barrier for grid "+std::to_string(grid));
  return found->second;
}
double Sm90NativeConstants::Epilogue(std::string const& family,int tile_n,int batch,
                                     int warps_k) const {
  auto const* f=epilogue_.Find(family);auto const* w=f?f->Find("wk"+std::to_string(warps_k)):nullptr;
  auto const* tn=w?w->Find(std::to_string(tile_n)):nullptr;
  auto const* b=tn?tn->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured epilogue "+family+" TN"+std::to_string(tile_n)+
      " B"+std::to_string(batch)+" WK"+std::to_string(warps_k));
  return Positive(*b,"epilogue");
}
double Sm90NativeConstants::Embedding(int grid,int width,int batch) const {
  auto const* g=embedding_.Find("g"+std::to_string(grid));
  auto const* w=g?g->Find("w"+std::to_string(width)):nullptr;
  auto const* b=w?w->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured embedding stage");
  return Positive(*b,"embedding");
}
double Sm90NativeConstants::Argmax(int grid,int parts,int batch) const {
  auto const* g=argmax_.Find("g"+std::to_string(grid));
  auto const* p=g?g->Find("p"+std::to_string(parts)):nullptr;
  auto const* b=p?p->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured argmax stage for "+std::to_string(parts)+" parts");
  return Positive(*b,"argmax");
}
json::Value const& Sm90NativeConstants::AttentionShape(int dim,int queries,bool norm) const {
  auto key="d"+std::to_string(dim)+":q"+std::to_string(queries)+":norm"+std::to_string(int(norm));
  auto const* found=attention_.Find(key);
  if(!found)throw Sm90NativeMissing("no measured attention constants for "+key);
  return *found;
}
double Sm90NativeConstants::AttentionBlock(int dim,int queries,bool norm) const {
  return Positive(AttentionShape(dim,queries,norm).At("block"),"attention block");
}
double Sm90NativeConstants::AttentionPrepare(int dim,int queries,bool norm) const {
  return Positive(AttentionShape(dim,queries,norm).At("prepare"),"attention preparation");
}
double Sm90NativeConstants::AttentionMerge(int dim,int queries,bool norm,int chunks) const {
  auto const* found=AttentionShape(dim,queries,norm).At("merge").Find(std::to_string(chunks));
  if(!found)throw Sm90NativeMissing("no measured attention merge for "+std::to_string(chunks)+" chunks");
  return Positive(*found,"attention merge");
}
std::vector<std::string> Sm90NativeConstants::StaleHeaders(std::string const& root) const {
  std::vector<std::string> result;
  for(auto const& item:section.At("header_sha256").AsObject("constant headers"))
    if(FileHash(root+"/"+item.first)!=item.second.AsString("header hash"))result.push_back(item.first);
  return result;
}
}  // namespace tilemega::solver
