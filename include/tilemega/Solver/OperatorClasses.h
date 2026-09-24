// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/CostModel.h>
#include <set>

namespace tilemega::solver {
struct OperatorClass {
  std::string signature;
  std::vector<std::size_t> gemms;
  std::vector<std::string> operators;
};
inline std::vector<OperatorClass> BuildOperatorClasses(frontend::ImportedSemantics const& imported) {
  std::vector<OperatorClass> classes;std::map<std::string,std::size_t> by_signature;
  for(std::size_t i=0;i<imported.lifted.ops.size();++i) {
    auto const& info=imported.lifted.ops[i];auto const& op=imported.lifted.sem.ops[i];
    if(op.kind!=analysis::OperatorKind::kMatmul)continue;
    int gemm=imported.plan.stages.at(info.stage).gemm;
    if(gemm<0)throw std::invalid_argument("matmul lacks GEMM invocation");
    auto signature=analysis::SemanticSignature(op);
    auto [it,added]=by_signature.emplace(signature,classes.size());
    if(added)classes.push_back({signature,{},{}});
    classes[it->second].gemms.push_back(std::size_t(gemm));
    classes[it->second].operators.push_back(op.name);
  }
  return classes;
}
inline auto GeometryKey(GemmConfig const& g) {return std::make_tuple(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k);}
inline frontend::ImportOptions ClassGranularity(frontend::ImportedSemantics const& imported,
    std::vector<OperatorClass> const& classes,std::vector<GemmConfig> const& config) {
  if(classes.size()!=config.size())throw std::invalid_argument("one tile required per semantic class");
  frontend::ImportOptions result;result.gemms.resize(imported.plan.gemms.size());
  std::vector<bool> covered(result.gemms.size());
  std::set<std::tuple<int,int,int,int>> shapes;
  for(std::size_t c=0;c<classes.size();++c) {
    auto const& g=config[c];shapes.emplace(g.tile_m,g.tile_n,g.tile_k,g.stages);
    for(auto index:classes[c].gemms) {
      if(covered.at(index))throw std::invalid_argument("GEMM belongs to two semantic classes");
      covered[index]=true;result.gemms[index]={g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k};
    }
  }
  if(shapes.size()>16)throw std::invalid_argument("per-class plan exceeds 16 compiled GEMM variants");
  if(std::find(covered.begin(),covered.end(),false)!=covered.end())throw std::invalid_argument("unclassified GEMM");
  result.rope_tile_per_block=result.kv_tile_per_block=result.activation_tile_per_block=result.combiner_tile_per_block=true;
  return result;
}
inline std::vector<GemmConfig> ClassCandidates(OperatorClass const& cls,
    frontend::ImportedSemantics const& imported,TargetSpec const& target,ScalarType dtype) {
  CandidateGenerator generator(target,dtype,{256,64,5});std::vector<GemmConfig> result;
  for(auto const& candidate:generator.Enumerate()) {
    auto const& t=candidate.traits();
    bool legal=true;
    for(auto id:cls.gemms) {
      auto const& op=imported.plan.gemms.at(id);
      legal &= op.k%t.alignment.a==0 && op.k%t.alignment.b==0;
    }
    if(!legal)continue;
    for(int split:{1,2,4,8,16,32})result.push_back({t.tile_m,t.tile_n,t.tile_k,t.stages,split});
  }
  return result;
}
}
