// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
using namespace tilemega;
using namespace tilemega::solver;

int main(int argc,char** argv) {try {
  if(argc!=6)throw std::invalid_argument("usage: ae-local target root D.json export.json output-dir");
  std::filesystem::path out=argv[5];
  if(std::filesystem::exists(out))throw std::invalid_argument("retain the previous local selection");
  auto target_json=json::ParseFile(argv[1]);auto target=TargetSpec::FromJson(argv[1]);
  auto raw=json::ParseFile(argv[3]);auto d=Sm90DenseCoordinates::FromJson(raw);
  if(d.moe || d.cluster!=1 || d.residency!=1 || d.weight_stream || !d.group_wait)
    throw std::invalid_argument("A-E requires the prescribed dense TP1 D coordinate");
  auto bridge=frontend::ReadExportBridge(argv[4]);
  frontend::ServingOptions options;
  options.interleave_u=d.swiglu_u;options.native_rms=true;options.native_shared_scratch=true;
  options.argmax_tile_n=d.gemms[4].tile_n;options.dense_attention_chunks=d.attention.chunks;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  auto work=Sm90DenseWorkloadFromPlan(plan,1);
  Sm90DenseSearchSpace space{work.shapes,1,work.head_dim,work.queries,target,work.hkv};
  auto constants=Sm90NativeConstants::FromTarget(target_json,argv[2]);
  Sm90DenseCostModel cost(target,constants);
  auto common_legal=[&](auto c) {
    c.group_wait=false;bool basic=space.Legal(c);
    c.group_wait=true;return basic && space.Legal(c);
  };
  if(!common_legal(d))throw std::invalid_argument("D is outside the common legal domain");
  std::filesystem::create_directories(out);
  std::ofstream scores(out/"candidate_scores.csv");
  scores<<std::setprecision(17)<<"family,value,is_D,legal_both_policies,cost_available,predicted_ns,selected\n";
  auto local=raw;json::Array records;
  for(unsigned family=0;family<6;++family) {
    std::string name=family<5?kSm90DenseFamilies[family]:"attention";
    int baseline=family<5?d.gemms[family].tile_n:d.attention.chunks;
    std::set<int> values{baseline};
    if(family<5)for(auto const& g:space.FamilyCandidates(family,d.residency))values.insert(g.tile_n);
    else for(auto const& a:space.AttentionCandidates(d.residency))values.insert(a.chunks);
    json::Array candidates;
    double best=std::numeric_limits<double>::infinity();int selected=-1;
    for(int value:values) {
      auto candidate=d;
      if(family<5) {
        auto& g=candidate.gemms[family];int tk=g.TileK();
        g.tile_n=value;g.slot_bytes=2*value*tk;
      }else candidate.attention.chunks=value;
      bool legal=common_legal(candidate);
      json::Object row{{"value",value},{"is_D",value==baseline},{"legal_both_policies",legal}};
      if(legal) {
        try {
          json::Value terms;
          double score=family<5?cost.Gemm(work,family,candidate.gemms[family],d.residency,&terms,false):
              cost.Attention(work,candidate.attention,d.residency,&terms);
          if(!std::isfinite(score) || score<=0)throw std::invalid_argument("invalid local cost");
          row.emplace_back("cost_available",true);row.emplace_back("predicted_ns",score);
          row.emplace_back("terms",terms);
          if(score<best || (score==best && value==baseline)) {best=score;selected=value;}
        }catch(Sm90NativeMissing const& error) {
          row.emplace_back("cost_available",false);row.emplace_back("reason",error.what());
        }
      }else row.emplace_back("reason","illegal with all other D parameters fixed");
      candidates.emplace_back(row);
    }
    if(selected<0)throw std::invalid_argument("no common priced candidate for "+name);
    for(auto const& row:candidates) {
      auto available=row.Find("cost_available");
      scores<<name<<','<<row.At("value").AsNumber("value")<<','<<row.At("is_D").AsBool("is_D")
            <<','<<row.At("legal_both_policies").AsBool("legal")<<',';
      if(available)scores<<available->AsBool("available");
      scores<<',';
      if(auto score=row.Find("predicted_ns"))scores<<score->AsNumber("score");
      scores<<','<<(row.At("value").AsNumber("value")==selected)<<'\n';
    }
    if(family<5) {
      auto gemms=local.At("gemms");auto g=gemms.At(name);
      g.Set("tile_n",selected);g.Set("slot_bytes",2*selected*d.gemms[family].TileK());
      gemms.Set(name,g);local.Set("gemms",gemms);
    }else {auto a=local.At("attention");a.Set("chunks",selected);local.Set("attention",a);}
    records.emplace_back(json::Object{{"family",name},{"D_value",baseline},{"selected",selected},
        {"predicted_ns",best},{"candidates",candidates}});
  }
  if(!common_legal(Sm90DenseCoordinates::FromJson(local)))
    throw std::invalid_argument("assembled L is not legal for both policies");
  std::ofstream(out/"L.json")<<local.Dump(2)<<'\n';
  json::Value receipt=json::Object{{"evidence","verified CPU cost-model enumeration"},
      {"objective","independent calibrated operator-local Gemm(pre=false) and complete Attention"},
      {"tie_break","exact score ties prefer D; remaining ties prefer the smaller value"},
      {"source_root",argv[2]},{"target",argv[1]},{"D",argv[3]},{"export",argv[4]},
      {"choices",records},{"legal_both_policies",true},{"gpu_candidate_sweeps",0},
      {"structure_or_derive_calls",0},{"stale_constant_headers",json::Array{}}};
  json::Array stale;for(auto const& path:constants.StaleHeaders(argv[2]))stale.emplace_back(path);
  receipt.Set("stale_constant_headers",stale);
  std::ofstream(out/"selection.json")<<receipt.Dump(2)<<'\n';
  std::cout<<receipt.Dump(2)<<'\n';return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}}
