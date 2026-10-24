// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace tilemega::solver {
namespace {
std::string FileHash(std::string const& path) {
  auto command="sha256sum '"+path+"'";auto* pipe=popen(command.c_str(),"r");
  if(!pipe)throw std::runtime_error("cannot hash "+path);
  char text[65]{};auto* got=fgets(text,sizeof(text),pipe);pclose(pipe);
  return got?std::string(text):std::string();
}
std::vector<std::string> Split(std::string const& line,char separator) {
  std::vector<std::string> result;std::string item;std::istringstream stream(line);
  while(std::getline(stream,item,separator))result.push_back(item);
  return result;
}
// Both the R2 calibration CSV (one column per key field) and the T5 rows
// (name=attention_stage, key=field=value:...) define the same lookup.
void ReadAttention(std::string const& path,
    std::map<std::tuple<int,int,int,int,int,int,int>,Sm90AttentionEntry>& out) {
  std::ifstream stream(path);
  if(!stream)throw std::invalid_argument("missing native attention table "+path);
  std::string line;std::getline(stream,line);
  auto header=Split(line,',');
  auto column=[&](char const* name) {
    for(unsigned i=0;i<header.size();++i)if(header[i]==name)return int(i);
    return -1;
  };
  bool compact=column("key")>=0;
  while(std::getline(stream,line)) {
    auto cells=Split(line,',');
    std::map<std::string,int> f;double ns=0;
    if(compact) {
      if(cells.at(column("name"))!="attention_stage")continue;
      for(auto const& part:Split(cells.at(column("key")),':')) {
        auto kv=Split(part,'=');f[kv.at(0)]=std::stoi(kv.at(1));
      }
      ns=std::stod(cells.at(column("median")));
    }else {
      for(auto name:{"batch","head_dim","queries","qk_norm","chunks","block_keys","slots",
          "new_row_work","grid","past"})f[name]=std::stoi(cells.at(column(name)));
      ns=std::stod(cells.at(column("median_ns")));
    }
    if(f.at("past")!=575)continue;
    // The low-batch body copies a whole chunk; block keys do not apply to it.
    int keys=f.at("batch")>=8?f.at("block_keys"):0;
    auto key=std::make_tuple(f.at("batch"),f.at("head_dim"),f.at("queries"),f.at("qk_norm"),
        f.at("chunks"),f.at("grid"),keys);
    auto found=out.find(key);
    if(found==out.end() || ns<found->second.ns)
      out[key]={ns,f.at("slots"),f.at("new_row_work")};
  }
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
  for(auto const& path:s.At("attention_tables").AsArray("attention tables"))
    ReadAttention(root+"/"+path.AsString("attention table"),c.attention_);
  return c;
}
double Sm90NativeConstants::Barrier(int grid) const {
  auto found=barrier_.find(grid);
  if(found==barrier_.end())throw Sm90NativeMissing("no measured barrier for grid "+std::to_string(grid));
  return found->second;
}
double Sm90NativeConstants::Epilogue(std::string const& family,int tile_n,int batch,int k,
                                     bool vector_sums) const {
  auto const* f=epilogue_.Find(family);auto const* v=f?f->Find(vector_sums?"vs1":"vs0"):nullptr;
  auto const* kk=v?v->Find("k"+std::to_string(k)):nullptr;
  auto const* tn=kk?kk->Find(std::to_string(tile_n)):nullptr;
  auto const* b=tn?tn->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured epilogue "+family+" TN"+std::to_string(tile_n)+
      " B"+std::to_string(batch)+" K"+std::to_string(k)+" vs"+std::to_string(vector_sums));
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
std::optional<Sm90AttentionEntry> Sm90NativeConstants::Attention(int batch,int dim,int queries,
    bool norm,int chunks,int grid,int block_keys) const {
  auto found=attention_.find({batch,dim,queries,int(norm),chunks,grid,batch>=8?block_keys:0});
  if(found==attention_.end())return std::nullopt;
  return found->second;
}
std::vector<std::string> Sm90NativeConstants::StaleHeaders(std::string const& root) const {
  std::vector<std::string> result;
  for(auto const& item:section.At("header_sha256").AsObject("constant headers"))
    if(FileHash(root+"/"+item.first)!=item.second.AsString("header hash"))result.push_back(item.first);
  return result;
}
}  // namespace tilemega::solver
