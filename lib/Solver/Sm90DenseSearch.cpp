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
    std::array<std::vector<Option>,6> families;
    for(unsigned f=0;f<5;++f) {
      for(auto const& g:space.FamilyCandidates(f,residency)) {
        families[f].push_back({cost.Gemm(work,f,g,residency),g,{}});
        ++result.evaluated;
      }
    }
    for(auto const& a:space.AttentionCandidates(residency))
      families[5].push_back({cost.Attention(work,a,residency),{},a});
    bool empty=false;
    for(unsigned i=0;i<families.size();++i) {
      auto& f=families[i];
      // Equal predictions are one model choice: keep the configuration with
      // the least shared memory (then the deeper tile) and offer only
      // distinct predicted costs as alternatives.
      std::stable_sort(f.begin(),f.end(),[&](Option const& a,Option const& b) {
        if(a.ns!=b.ns)return a.ns<b.ns;
        if(i==5)return false;
        bool residual=Sm90DenseCoordinates::ResidualFamily(i);
        if(a.gemm.SharedBytes(work.batch,residual)!=b.gemm.SharedBytes(work.batch,residual))
          return a.gemm.SharedBytes(work.batch,residual)<b.gemm.SharedBytes(work.batch,residual);
        return a.gemm.TileK()>b.gemm.TileK();
      });
      f.erase(std::unique(f.begin(),f.end(),[](Option const& a,Option const& b) {
        return std::abs(a.ns-b.ns)<=1e-9*b.ns;
      }),f.end());
      empty|=f.empty();
    }
    if(empty)continue;
    auto build=[&](unsigned family,unsigned rank,std::string const& origin) {
      Sm90DenseCoordinates c;c.residency=residency;c.weight_stream=space.weight_stream;
      for(unsigned f=0;f<5;++f)c.gemms[f]=families[f][f==family?rank:0].gemm;
      c.attention=families[5][family==5?rank:0].attention;
      Sm90DenseSearchSpace::Structure(c);
      if(!space.Legal(c))return;
      auto key=c.ToJson().Dump(0);
      if(!seen.insert(key).second)return;
      pool.push_back({c,cost.Predict(work,c),origin});
    };
    std::string prefix="r"+std::to_string(residency);
    build(6,0,prefix+"_optimum");
    static constexpr char const* kNames[]={"qkv","o","gate_up","down","lm_head","attention"};
    for(unsigned f=0;f<6;++f)
      for(unsigned rank=1;rank<=unsigned(o.alternatives) && rank<families[f].size();++rank)
        build(f,rank,prefix+"_"+kNames[f]+"_alt"+std::to_string(rank));
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
