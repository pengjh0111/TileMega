// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Solver/ServingImplementation.h>
#include <tilemega/Support/Json.h>
#include <array>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {
// Decode seed geometry by GEMM family. A decode step streams every weight
// once, so the task count of a GEMM bounds how many SMs read it: on H800 a
// fixed TN128/split-1 seed left o/down/qkv on 16-24 of 132 SMs (verified
// stage trace) and doubled TPOT.
struct SeedGeometryRule {
  std::optional<int> tile_n,tile_k,stages,impl,split,min_tasks,max_split,task_multiplier,min_tile_n;
};
struct SeedGeometryPolicy {
  SeedGeometryRule base;
  std::map<std::string,SeedGeometryRule> families;
};

inline int SeedGemmImpl(std::string const& name) {
  if(name=="mma16")return 0;
  if(name=="gemv")return 1;
  if(name=="wgmma")return 2;
  throw std::invalid_argument("unknown seed geometry implementation: "+name);
}

inline SeedGeometryRule ParseSeedGeometryRule(json::Value const& value) {
  SeedGeometryRule rule;
  for(auto const& [key,item]:value.AsObject("seed geometry rule")) {
    auto integer=[&]{return int(item.AsNumber(key.c_str()));};
    if(key=="tile_n")rule.tile_n=integer();
    else if(key=="tile_k")rule.tile_k=integer();
    else if(key=="stages")rule.stages=integer();
    else if(key=="impl")rule.impl=SeedGemmImpl(item.AsString("impl"));
    else if(key=="split")rule.split=integer();
    else if(key=="min_tasks")rule.min_tasks=integer();
    else if(key=="max_split")rule.max_split=integer();
    else if(key=="task_multiplier")rule.task_multiplier=integer();
    else if(key=="min_tile_n")rule.min_tile_n=integer();
    else throw std::invalid_argument("unknown seed geometry field: "+key);
  }
  return rule;
}

inline SeedGeometryPolicy ParseSeedGeometryPolicy(json::Value const& value) {
  SeedGeometryPolicy policy;
  if(auto const* base=value.Find("base"))policy.base=ParseSeedGeometryRule(*base);
  if(auto const* families=value.Find("families"))
    for(auto const& [name,rule]:families->AsObject("seed geometry families"))
      policy.families[name]=ParseSeedGeometryRule(rule);
  for(auto const& [key,item]:value.AsObject("seed geometry policy"))
    if(key!="base" && key!="families" && key!="comment")
      throw std::invalid_argument("unknown seed geometry policy field: "+key);
  return policy;
}

// Families: ServingGemmFamily names, plus router (feeds MoE top-k) and expert
// (expert-indirect weights). A router is identified by its consumer, not shape.
inline std::string SeedGeometryFamily(frontend::ModelPlan const& plan,std::size_t index) {
  auto const& g=plan.gemms[index];
  if(g.access.b==codegen::DmBAccess::kExpertIndirect)return "expert";
  for(auto const& stage:plan.stages)
    if(stage.kind==frontend::PlanTaskKind::kMoETopK) {
      // Top-k consumes epilogue side outputs, so it need not read g.d.
      if(stage.moe.router_gemm==index)return "router";
      for(auto operand:stage.operands)if(operand==g.d && g.d)return "router";
    }
  static char const* const names[]={"qkv","o","gate_up","down","lm_head"};
  return names[ServingGemmFamily(g)];
}

// Smallest power-of-two split whose task count reaches min_tasks, limited to
// whole TK chunks per split.
inline int SeedSplitK(int n_tiles,int k,int tile_k,int min_tasks,int max_split) {
  int split=1;
  while(split<max_split && n_tiles*split<min_tasks && k%(tile_k*split*2)==0)split*=2;
  return split;
}

inline void ApplySeedGeometryPolicy(frontend::ModelPlan const& plan,
    std::vector<frontend::GemmGranularity>& geometry,SeedGeometryPolicy const& policy,
    int num_sms,bool hopper_native,std::array<int,5> const& requested={-1,-1,-1,-1,-1}) {
  if(geometry.size()!=plan.gemms.size())throw std::invalid_argument("seed geometry count differs");
  for(std::size_t i=0;i<plan.gemms.size();++i) {
    auto const& g=plan.gemms[i];auto& tile=geometry[i];
    auto family=SeedGeometryFamily(plan,i);
    SeedGeometryRule rule=policy.base;
    if(auto it=policy.families.find(family);it!=policy.families.end()) {
      auto const& o=it->second;
      for(auto [field,value]:{std::pair{&rule.tile_n,o.tile_n},{&rule.tile_k,o.tile_k},
          {&rule.stages,o.stages},{&rule.impl,o.impl},{&rule.split,o.split},
          {&rule.min_tasks,o.min_tasks},{&rule.max_split,o.max_split},
          {&rule.task_multiplier,o.task_multiplier},{&rule.min_tile_n,o.min_tile_n}})
        if(value)*field=value;
    }
    tile.tile_n=rule.tile_n.value_or(tile.tile_n);tile.tile_k=rule.tile_k.value_or(tile.tile_k);
    tile.stages=rule.stages.value_or(tile.stages);tile.impl=rule.impl.value_or(tile.impl);
    // --gemm-impl-* is applied after the seed; legality must see that choice.
    int impl=requested.at(ServingGemmFamily(g))>=0?requested.at(ServingGemmFamily(g)):tile.impl;
    int multiplier=rule.task_multiplier.value_or(1),min_tasks=rule.min_tasks.value_or(num_sms);
    // Narrower tiles add tasks without a split-K combine stage.
    while(rule.min_tile_n && tile.tile_n/2>=*rule.min_tile_n &&
          int((g.n+tile.tile_n-1)/tile.tile_n)*multiplier<min_tasks)tile.tile_n/=2;
    int tiles=int((g.n+tile.tile_n-1)/tile.tile_n)*multiplier;
    bool legal=impl==0?ServingBF16ShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages):
        impl==1?ServingGemvShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages):
        hopper_native && ServingWgmmaShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages);
    if(!legal || g.k%tile.tile_k)
      throw std::invalid_argument("seed geometry for "+family+" is not a legal decode shape");
    tile.split_k=rule.split?*rule.split:SeedSplitK(tiles,int(g.k),tile.tile_k,
        min_tasks,rule.max_split.value_or(16));
    // Argmax partials are reduced per head tile; a K split has no combiner.
    if(g.epilogue==frontend::PlanGemm::Epilogue::kArgmaxPartial && tile.split_k!=1)
      throw std::invalid_argument("seed geometry cannot split the argmax head");
    if(tile.split_k<1 || tile.split_k>int((g.k+tile.tile_k-1)/tile.tile_k))
      throw std::invalid_argument("seed geometry split exceeds K tiles for "+family);
  }
}
} // namespace tilemega::solver
