// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/DmGemmCandidates.h>
#include <tilemega/Support/Json.h>
#include <cmath>
#include <map>

namespace tilemega::solver {
inline json::Value CanonicalDmRecipe(json::Value const& value) {
  if(value.kind()==json::Value::Kind::kObject) {
    std::map<std::string,json::Value> fields;
    for(auto const& [name,item]:value.AsObject("weight recipe"))
      if(!fields.emplace(name,CanonicalDmRecipe(item)).second)
        throw std::invalid_argument("duplicate weight recipe field");
    json::Object result;
    for(auto& field:fields)result.push_back(std::move(field));
    return result;
  }
  if(value.kind()==json::Value::Kind::kArray) {
    json::Array result;
    for(auto const& item:value.AsArray("weight recipe"))result.push_back(CanonicalDmRecipe(item));
    return result;
  }
  return value;
}

inline std::map<std::size_t,DmWeightLayoutConstraint> SharedDmWeightLayouts(
    frontend::ModelPlan const& plan,json::Value const& manifest) {
  if(!plan.dm || plan.dtype!="bf16")
    throw std::invalid_argument("shared packed weights require a BF16 DM plan");
  std::map<std::string,DmWeightLayoutConstraint> recipes;
  for(auto const& buffer:manifest.At("buffers").AsArray("shared weight buffers")) {
    auto recipe=buffer.Find("recipe");
    if(!recipe || recipe->At("kind").AsString("recipe kind")!="tile_pages")continue;
    if(buffer.At("dtype").AsString("shared weight dtype")!="bf16")
      throw std::invalid_argument("shared packed weight is not BF16");
    auto tile=[&](char const* key) {
      auto value=recipe->At(key).AsNumber(key);
      if(!std::isfinite(value) || value<16 || value>256 || std::floor(value)!=value)
        throw std::invalid_argument("invalid shared packed weight tile");
      return int(value);
    };
    DmWeightLayoutConstraint layout{tile("tile_n"),tile("tile_k")};
    auto key=CanonicalDmRecipe(recipe->At("source")).Dump();
    auto [found,inserted]=recipes.emplace(key,layout);
    if(!inserted && (found->second.tile_n!=layout.tile_n || found->second.tile_k!=layout.tile_k))
      throw std::invalid_argument("shared weight recipe has multiple packed layouts");
  }
  std::map<std::size_t,DmWeightLayoutConstraint> result;
  for(std::size_t i=0;i<plan.gemms.size();++i) {
    auto const& weight=plan.buffers.at(plan.gemms[i].b);
    if(weight.pack_json.empty())throw std::invalid_argument("shared GEMM weight lacks a recipe");
    auto recipe=json::Parse(weight.pack_json);
    if(recipe.At("kind").AsString("recipe kind")=="tile_pages")recipe=json::Value(recipe.At("source"));
    auto found=recipes.find(CanonicalDmRecipe(recipe).Dump());
    if(found==recipes.end()) {
      if(plan.gemms[i].access.b==codegen::DmBAccess::kExpertIndirect)
        throw std::invalid_argument("shared deployment has no matching packed recipe for "+weight.name);
      // Phase-specific dense projections can have different norm folding.
      // They retain distinct allocations; never bind them to another recipe.
      continue;
    }
    result.emplace(i,found->second);
  }
  if(result.empty())throw std::invalid_argument("shared deployment has no matching GEMM weights");
  return result;
}
} // namespace tilemega::solver
