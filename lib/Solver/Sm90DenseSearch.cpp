// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
std::string Sm90DenseScheduleKey(Sm90DenseCoordinates const& c,Sm90DensePrediction const& prediction) {
  auto structural=c.ToJson();
  auto strip=[](json::Value const& body) {
    json::Object result;
    for(auto const& field:body.AsObject("native body"))
      if(field.first!="workspace_offset" && field.first!="register_stages" &&
         field.first!="preissue_late" && field.first!="prefetch_l2_bytes")result.push_back(field);
    return json::Value(result);
  };
  auto gemms=structural.At("gemms").AsObject("GEMM families");
  for(auto& family:gemms)family.second=strip(family.second);
  structural.Set("gemms",gemms);structural.Set("attention",strip(structural.At("attention")));
  json::Array prices;
  for(auto const& stage:prediction.stages)prices.emplace_back(json::Object{
      {"kind",stage.kind},{"instances",stage.instances},{"stage_ns",stage.ns}});
  return json::Value(json::Array{structural,prediction.step_ns,prices}).Dump(0);
}
Sm90DenseSearchResult SearchSm90Dense(Sm90DenseSearchSpace const& space,
    Sm90DenseWorkload const& work,Sm90DenseCostModel const& cost,Sm90DenseSearchOptions const& o) {
  if(o.passes<1 || o.beam_width<1 || o.retained<8 || o.budget_ms<1)
    throw std::invalid_argument("invalid native search budget");
  using Clock=std::chrono::steady_clock;auto start=Clock::now();
  Sm90DenseSearchResult result;std::set<std::string> seen;
  std::vector<Sm90DenseCandidate> pool;
  auto elapsed=[&] {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
  auto less=[](auto const& a,auto const& b) {
    if(a.prediction.step_ns!=b.prediction.step_ns)return a.prediction.step_ns<b.prediction.step_ns;
    return a.coordinates.ToJson().Dump(0)<b.coordinates.ToJson().Dump(0);
  };
  using Frontier=std::vector<std::pair<Sm90DenseCoordinates,std::string>>;
  auto require_frontier=[&](Frontier const& frontier,char const* name,int pass) {
    std::set<std::string> checked;
    std::map<std::string,std::pair<json::Value,json::Array>> missing;
    for(auto const& entry:frontier) {
      auto coordinates=entry.first.ToJson();auto key=coordinates.Dump(0);
      if(seen.count(key) || !checked.insert(key).second)continue;
      try {cost.RequireAttentionCalibration(work,entry.first);}
      catch(Sm90DenseMissingMeasurement const& error) {
        if(error.metric.rfind("attention_",0)!=0)throw;
        auto measurement=error.ToJson();auto measurement_key=measurement.Dump(0);
        auto found=missing.emplace(measurement_key,
            std::make_pair(measurement,json::Array{})).first;
        found->second.second.emplace_back(json::Object{
            {"coordinates",coordinates},{"origin",entry.second}});
      }
    }
    if(missing.empty())return;
    json::Array measurements;
    for(auto& entry:missing) {
      entry.second.first.Set("candidates",entry.second.second);
      measurements.push_back(std::move(entry.second.first));
    }
    // A missing member invalidates this entire search, including already
    // priced seeds. Never return a partial ranking as a completed selection.
    throw Sm90DenseCalibrationPending(json::Object{{"schema",1},{"status","calibration_pending"},
        {"frontier",name},{"pass",pass},{"frontier_candidates",double(checked.size())},
        {"evaluated",double(result.evaluated)},{"elapsed_ms",elapsed()},
        {"converged",false},{"selection_performed",false},{"missing",measurements}});
  };
  auto evaluate=[&](Sm90DenseCoordinates const& c,std::string const& origin) {
    auto key=c.ToJson().Dump(0);
    if(seen.count(key))return;
    auto prediction=cost.Request(work,c);
    if(!(prediction.step_ns>0) || !std::isfinite(prediction.step_ns))
      throw std::runtime_error("native pricing produced an invalid prediction");
    auto schedule=Sm90DenseScheduleKey(c,prediction);
    pool.push_back({c,std::move(prediction),origin,std::move(schedule)});
    seen.insert(std::move(key));++result.evaluated;
  };
  auto seeds=space.Seeds();
  if(seeds.empty())throw std::invalid_argument("native search has no legal seed");
  Frontier seed_frontier;
  for(unsigned i=0;i<seeds.size();++i)
    seed_frontier.emplace_back(seeds[i],std::string(i/3?"resident2_":"resident1_")+
        (i%3==0?"exact":i%3==1?"preissue":"preissue_l2"));
  // Keeping one zeroed arena needs a joint change to every operand layout;
  // single-coordinate neighbours cannot reach it on mixed-TK programs.
  for(auto const& seed:space.ActivationReuseSeeds())seed_frontier.emplace_back(seed,"activation_reuse");
  require_frontier(seed_frontier,"seeds",-1);
  for(auto const& entry:seed_frontier)evaluate(entry.first,entry.second);
  std::sort(pool.begin(),pool.end(),less);
  std::set<std::string> expanded;
  auto leaders=[&] {
    std::set<std::string> unique;std::vector<std::string> keys;
    for(auto const& candidate:pool)if(unique.insert(candidate.schedule_key).second) {
      keys.push_back(candidate.schedule_key);if(int(keys.size())==o.beam_width)break;
    }
    return keys;
  };
  for(int pass=0;pass<o.passes;++pass) {
    std::vector<Sm90DenseCandidate> beam;
    // A protected native seed is expanded once even if its first prediction
    // loses; its neighbours can change both geometry and resource coupling.
    if(!pass)beam=pool;
    else {
      std::set<std::string> unique;
      for(auto const& p:pool) {
        if(!expanded.count(p.schedule_key) && unique.insert(p.schedule_key).second)beam.push_back(p);
        if(int(beam.size())==o.beam_width)break;
      }
    }
    if(beam.empty()) {result.converged=true;break;}
    auto previous=leaders();
    Frontier frontier;
    for(auto const& center:beam) {
      expanded.insert(center.schedule_key);
      for(auto const& neighbour:space.Neighbours(center.coordinates))
        frontier.emplace_back(neighbour,"search");
    }
    require_frontier(frontier,"beam",pass);
    for(auto const& entry:frontier) {
      if(elapsed()>=o.budget_ms) {result.budget_exhausted=true;break;}
      evaluate(entry.first,entry.second);
    }
    std::sort(pool.begin(),pool.end(),less);
    if(result.budget_exhausted)break;
    if(previous==leaders()) {result.converged=true;break;}
  }
  result.elapsed_ms=elapsed();
  // Equal-price translations must not crowd every different division of work
  // out of the measured shortlist. Protected seeds remain explicit controls.
  std::set<std::string> retained;
  for(auto const& p:pool) {
    bool fresh=retained.insert(p.schedule_key).second;
    if(p.origin!="search" || (fresh && int(retained.size())<=o.retained))
      result.ranked.push_back(p);
  }
  return result;
}
}  // namespace tilemega::solver
