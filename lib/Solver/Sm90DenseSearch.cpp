// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

namespace tilemega::solver {
namespace {
struct Option {
  double ns;
  Sm90DenseGemm gemm;
  Sm90DenseAttention attention;
};
}

// Stage times are additive and families independent for a fixed residency and
// vector-sum layout, so the per-family optimum is exact; the top-K list adds
// single-family alternatives around it (prompt §4 L0-2 item 3).
Sm90DenseSearchResult SearchSm90Dense(Sm90DenseSearchSpace const& space,
    Sm90DenseWorkload const& work,Sm90DenseCostModel const& cost,Sm90DenseSearchOptions const& o) {
  if(o.top<1 || o.alternatives<1)throw std::invalid_argument("invalid native search options");
  auto start=std::chrono::steady_clock::now();
  Sm90DenseSearchResult result;
  std::vector<Sm90DenseCandidate> pool;
  std::set<std::string> seen;
  for(int residency:{1,2}) {
    // Index 5 of `families` stays the attention list; GEMM families are
    // stored at their family number (MoE: 0, 1, 4, 5, 6, 7).
    std::array<std::vector<Option>,8> families;
    std::vector<Option> attention;
    auto active=Sm90DenseActiveFamilies(space.moe);
    for(unsigned f:active) {
      for(auto const& g:space.FamilyCandidates(f,residency)) {
        families[f].push_back({cost.Gemm(work,f,g,residency),g,{}});
        ++result.evaluated;
      }
    }
    for(auto const& a:space.AttentionCandidates(residency))
      attention.push_back({cost.Attention(work,a,residency),{},a});
    // Equal predictions are one model choice: keep the configuration with
    // the least shared memory (then the deeper tile) and offer only distinct
    // predicted costs as alternatives.
    auto rank=[&](std::vector<Option>& list,int family) {
      std::stable_sort(list.begin(),list.end(),[&](Option const& a,Option const& b) {
        if(a.ns!=b.ns)return a.ns<b.ns;
        if(family<0)return false;
        bool residual=Sm90DenseCoordinates::ResidualFamily(unsigned(family));
        if(a.gemm.SharedBytes(work.batch,residual)!=b.gemm.SharedBytes(work.batch,residual))
          return a.gemm.SharedBytes(work.batch,residual)<b.gemm.SharedBytes(work.batch,residual);
        return a.gemm.TileK()>b.gemm.TileK();
      });
      list.erase(std::unique(list.begin(),list.end(),[](Option const& a,Option const& b) {
        return std::abs(a.ns-b.ns)<=1e-9*b.ns;
      }),list.end());
      return !list.empty();
    };
    bool complete=rank(attention,-1);
    for(unsigned f:active)complete&=rank(families[f],int(f));
    if(!complete)continue;
    // The embedding placement (D9 ②) is a structural dimension: both values
    // of every family combination enter the predicted ranking. `family` is a
    // GEMM family number or kAttention.
    constexpr unsigned kAttention=~0u;
    auto build=[&](unsigned family,unsigned alternative,std::string const& origin) {
      Sm90DenseCoordinates c;c.residency=residency;c.weight_stream=space.weight_stream;
      c.layer_loop=space.layer_loop;c.moe=space.moe;
      for(unsigned f:active)c.gemms[f]=families[f][f==family?alternative:0].gemm;
      c.attention=attention[family==kAttention?alternative:0].attention;
      Sm90DenseSearchSpace::Structure(c);
      if(!space.Legal(c))return;
      for(bool embed_la:{false,true}) {
        c.embed_la=embed_la;
        if(!seen.insert(c.ToJson().Dump(0)).second)continue;
        pool.push_back({c,cost.Predict(work,c),origin+(embed_la?"_emb_la":"_emb_stage")});
      }
    };
    std::string prefix="r"+std::to_string(residency);
    build(kAttention-1,0,prefix+"_optimum");
    for(unsigned f:active)
      for(unsigned alt=1;alt<=unsigned(o.alternatives) && alt<families[f].size();++alt)
        build(f,alt,prefix+"_"+kSm90DenseFamilies[f]+"_alt"+std::to_string(alt));
    for(unsigned alt=1;alt<=unsigned(o.alternatives) && alt<attention.size();++alt)
      build(kAttention,alt,prefix+"_attention_alt"+std::to_string(alt));
  }
  std::stable_sort(pool.begin(),pool.end(),[](auto const& a,auto const& b) {
    return a.prediction.step_ns<b.prediction.step_ns;
  });
  if(pool.empty())throw std::invalid_argument("native search found no legal coordinate");
  if(int(pool.size())>o.top)pool.resize(o.top);
  result.ranked=std::move(pool);
  result.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  return result;
}
}  // namespace tilemega::solver
