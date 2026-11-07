// SPDX-License-Identifier: BSD-3-Clause
#include "NativeSweep.h"
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/SHA256.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>

namespace tilemega::commands::native_sweep {
namespace {
std::string Hash(std::string const& text) {
  auto hash=llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<uint8_t const*>(text.data()),text.size()));
  return llvm::toHex(hash,true);
}
void Write(std::filesystem::path const& path,std::string const& text) {
  std::ofstream stream(path);stream<<text;
  if(!stream)throw std::runtime_error("cannot write "+path.string());
}
}
int RunNativeSweep(int argc,char** argv) {
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2) {
    if(i+1==argc || !args.emplace(argv[i],argv[i+1]).second)
      throw std::invalid_argument("expected unique --option value pairs");
  }
  for(auto const& item:args)
    if(item.first!="--export" && item.first!="--target" && item.first!="--batch" &&
       item.first!="--reference" && item.first!="--family" && item.first!="--top" &&
       item.first!="--out" && item.first!="--mode" && item.first!="--selection")throw std::invalid_argument("unknown option: "+item.first);
  auto required=[&](char const* key) {
    auto it=args.find(key);if(it==args.end())throw std::invalid_argument(std::string("missing ")+key);
    return it->second;
  };
  auto input=required("--export"),target_path=required("--target"),reference_path=required("--reference");
  int batch=std::stoi(required("--batch")),top=args.count("--top")?std::stoi(args.at("--top")):4;
  auto reference=json::ParseFile(reference_path);
  if(auto* nested=reference.Find("coordinates"))reference=json::Value(*nested);
  auto w=Sm90DenseCoordinates::FromJson(reference);
  auto target=TargetSpec::FromJson(target_path);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(target_path),TILEMEGA_SOURCE_DIR);
  Sm90DenseCostModel cost(target,constants);
  auto bridge=frontend::ReadExportBridge(input);
  frontend::ServingOptions options;options.native_rms=true;options.dense_attention_chunks=1;options.tp=w.tp;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  auto work=Sm90DenseWorkloadFromPlan(plan,batch);
  Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
  space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
  space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
  if(!space.Legal(w))throw std::invalid_argument("illegal reference: "+Reason(space,w));
  auto mode=args.count("--mode")?args.at("--mode"):"sweep";
  auto out=std::filesystem::absolute(required("--out"));
  if(std::filesystem::exists(out))throw std::invalid_argument("retain sweep evidence; choose a new output directory");
  std::vector<std::pair<std::string,Variant>> entries;json::Array choices;
  if(mode=="sweep") {
    auto family=required("--family");
    auto families=family=="all"?(w.moe?std::vector<std::string>{"qkv","o","lm_head","attention","experts"}:
        std::vector<std::string>{"qkv","o","gate_up","down","lm_head","attention"}):std::vector<std::string>{family};
    for(auto const& f:families)for(auto const& v:Sweep(space,work,cost,w,f,top))entries.emplace_back(f,v);
  }else if(mode=="check") {
    Variant v;v.coordinates=w;v.legal=v.reference=v.selected=true;v.key="W";
    v.step_ns=cost.Predict(work,w).step_ns;entries.emplace_back("all",v);
  }else if(mode=="compose") {
    auto selection=json::ParseFile(required("--selection"));
    std::vector<std::string> families;std::vector<std::vector<Sm90DenseCoordinates>> lists;
    for(auto const& item:selection.AsObject("selection")) {
      families.push_back(item.first);lists.emplace_back();
      for(auto const& path:item.second.AsArray("ranked variants"))
        lists.back().push_back(Sm90DenseCoordinates::FromJson(json::ParseFile(path.AsString("coordinate path"))));
      if(lists.back().empty())throw std::invalid_argument("empty family ranking");
    }
    using Node=std::pair<int,std::vector<int>>;
    std::priority_queue<Node,std::vector<Node>,std::greater<Node>> pending;
    std::set<std::vector<int>> seen;
    pending.push({0,std::vector<int>(families.size())});bool found=false;
    while(!pending.empty()) {
      auto node=pending.top();pending.pop();if(!seen.insert(node.second).second)continue;
      auto c=w;
      for(unsigned i=0;i<families.size();++i)c=Replace(c,lists[i][node.second[i]],families[i]);
      if(space.Legal(c)) {
        Variant v;v.coordinates=c;v.key="composed";v.legal=v.selected=true;
        v.step_ns=cost.Predict(work,c).step_ns;entries.emplace_back("all",v);
        for(unsigned i=0;i<families.size();++i)choices.emplace_back(json::Object{
            {"family",families[i]},{"rank",node.second[i]},
            {"key",Key(lists[i][node.second[i]],families[i])}});
        found=true;break;
      }
      for(unsigned i=0;i<lists.size();++i)if(unsigned(node.second[i]+1)<lists[i].size()) {
        auto next=node.second;++next[i];pending.push({node.first+1,std::move(next)});
      }
    }
    if(!found)throw std::invalid_argument("no legal composition in measured rankings");
  }else if(mode=="T" || mode=="N") {
    Variant v;v.coordinates=Control(space,work,cost,w,mode,choices);
    v.key=mode;v.legal=v.selected=true;v.step_ns=cost.Predict(work,v.coordinates).step_ns;
    entries.emplace_back("all",v);
  }else throw std::invalid_argument("mode must be sweep, check, compose, T or N");
  std::filesystem::create_directories(out);
  json::Array manifest;int index=0;
  for(auto const& entry:entries) {
    auto const& v=entry.second;
    auto name=v.reference?"W":mode=="T" || mode=="N"?mode:"v"+std::to_string(index++);
    std::string text=(v.reference?reference:v.coordinates.ToJson()).Dump(2)+"\n";
    std::string path;
    if(v.selected) {
      auto directory=mode=="sweep" && args.at("--family")=="all"?out/entry.first:out;
      std::filesystem::create_directories(directory);path=(directory/(name+".json")).string();Write(path,text);
    }
    manifest.emplace_back(json::Object{{"family",entry.first},{"variant",name},{"decomposition_key",v.key},
        {"coordinates_sha256",Hash(text)},{"coordinates",v.coordinates.ToJson()},
        {"path",path},{"reference",v.reference},{"selected",v.selected},{"legal",v.legal},{"reason",v.reason},
        {"predicted_stage_ns",v.stage_ns},{"predicted_step_ns",v.legal?json::Value(v.step_ns):json::Value()},
        {"modeled_bytes",v.bytes}});
  }
  Write(out/"manifest.json",json::Value(json::Object{{"schema",1},{"mode",mode},{"top",top},
      {"reference",std::filesystem::absolute(reference_path).string()},{"choices",choices},{"variants",manifest}}).Dump(2)+"\n");
  std::cout<<(out/"manifest.json").string()<<'\n';return 0;
}
}  // namespace tilemega::commands::native_sweep
