// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
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
  auto evaluate=[&](Sm90DenseCoordinates const& c,std::string const& origin) {
    if(!seen.insert(c.ToJson().Dump(0)).second)return;
    auto prediction=cost.Request(work,c);
    if(!(prediction.step_ns>0) || !std::isfinite(prediction.step_ns))
      throw std::runtime_error("native pricing produced an invalid prediction");
    pool.push_back({c,std::move(prediction),origin});++result.evaluated;
  };
  auto seeds=space.Seeds();
  if(seeds.empty())throw std::invalid_argument("native search has no legal seed");
  for(unsigned i=0;i<seeds.size();++i)
    evaluate(seeds[i],std::string(i/3?"resident2_":"resident1_")+
        (i%3==0?"exact":i%3==1?"preissue":"preissue_l2"));
  std::sort(pool.begin(),pool.end(),less);
  std::set<std::string> expanded;
  for(int pass=0;pass<o.passes;++pass) {
    std::vector<Sm90DenseCandidate> beam;
    // A protected native seed is expanded once even if its first prediction
    // loses; its neighbours can change both geometry and resource coupling.
    if(!pass)beam=pool;
    else for(auto const& p:pool) {
      if(!expanded.count(p.coordinates.ToJson().Dump(0)))beam.push_back(p);
      if(int(beam.size())==o.beam_width)break;
    }
    if(beam.empty()) {result.converged=true;break;}
    std::vector<std::string> previous;
    for(int i=0;i<std::min(int(pool.size()),o.beam_width);++i)
      previous.push_back(pool[i].coordinates.ToJson().Dump(0));
    for(auto const& center:beam) {
      auto key=center.coordinates.ToJson().Dump(0);expanded.insert(key);
      for(auto const& neighbour:space.Neighbours(center.coordinates)) {
        if(elapsed()>=o.budget_ms) {result.budget_exhausted=true;break;}
        evaluate(neighbour,"search");
      }
      if(result.budget_exhausted)break;
    }
    std::sort(pool.begin(),pool.end(),less);
    if(result.budget_exhausted)break;
    bool unchanged=int(previous.size())==std::min(int(pool.size()),o.beam_width);
    for(unsigned i=0;i<previous.size() && unchanged;++i)
      unchanged=previous[i]==pool[i].coordinates.ToJson().Dump(0);
    if(unchanged) {result.converged=true;break;}
  }
  result.elapsed_ms=elapsed();
  for(auto const& p:pool)if(int(result.ranked.size())<o.retained || p.origin!="search")
    result.ranked.push_back(p);
  return result;
}
}  // namespace tilemega::solver
