// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace tilemega::solver {

std::vector<Sm90ForwardStage> Sm90ForwardSearchSpace::Candidates(std::size_t index,
                                                                 int residency) const {
  auto const& shape=shapes[index];
  std::vector<Sm90ForwardStage> result;
  auto push=[&](Sm90ForwardStage stage) {
    stage.body=shape.body;
    try {stage.Validate();}catch(std::invalid_argument const&) {return;}
    result.push_back(stage);
  };
  switch(shape.body) {
    case Sm90ForwardBody::kGemm:
    case Sm90ForwardBody::kConv: {
      for(int tn:{32,64,128,256}) {
        if(tn>shape.n && tn!=32)continue;  // N is padded up to 32
        for(int split:{1,2,4,8,16}) {
          if(split>1 && shape.whole_row)continue;
          // K is the padded contraction: a slice is a whole number of k tiles.
          if(split>1 && (shape.k%(std::int64_t(split)*64) || shape.k/split<128))continue;
          // The ring depth follows the slot bytes, as D5 does for decode.
          Sm90ForwardStage stage;stage.tile_n=tn;stage.split_k=split;
          stage.slots=shape.k/64>=4?4:2;
          push(stage);
        }
      }
      break;
    }
    case Sm90ForwardBody::kDepthwise: {
      for(auto tile:{std::pair{8,8},std::pair{8,16},std::pair{16,16}}) {
        Sm90ForwardStage stage;stage.tile_w=tile.first;stage.tile_h=tile.second;
        stage.channel_block=64;stage.slots=2;push(stage);
      }
      break;
    }
    case Sm90ForwardBody::kDwPw:
    case Sm90ForwardBody::kInvertedResidual: {
      for(bool fused:{true,false}) {
        if(fused && !shape.fusable)continue;
        for(auto tile:{std::pair{8,8},std::pair{8,16},std::pair{16,16}})
          for(int tn:{64,128,256}) {
            if(tn>shape.out_channels && tn!=64)continue;
            Sm90ForwardStage stage;stage.fused=fused;stage.tile_w=tile.first;
            stage.tile_h=tile.second;stage.tile_n=tn;stage.channel_block=64;stage.slots=2;
            push(stage);
          }
      }
      break;
    }
    default: {
      Sm90ForwardStage stage;stage.tile_n=64;stage.split_k=1;stage.slots=2;push(stage);
      break;
    }
  }
  // Prefer task counts inside [G/2, 2G]; keep the closest when none qualify.
  int grid=Grid(residency);
  auto inside=[&](Sm90ForwardStage const& stage) {
    long long tasks=Sm90ForwardCostModel::Tasks(shape,stage);
    return tasks*2>=grid && tasks<=2LL*grid;
  };
  std::vector<Sm90ForwardStage> kept;
  for(auto const& stage:result)if(inside(stage))kept.push_back(stage);
  if(!kept.empty())return kept;
  if(result.empty())return result;
  auto distance=[&](Sm90ForwardStage const& stage) {
    long long tasks=Sm90ForwardCostModel::Tasks(shape,stage);
    double low=grid/2.0,high=2.0*grid;
    return tasks<low?low-tasks:tasks>high?tasks-high:0.0;
  };
  std::stable_sort(result.begin(),result.end(),[&](auto const& a,auto const& b) {
    return distance(a)<distance(b);
  });
  result.resize(1);
  return result;
}

// Stage times are additive and the stages independent at a fixed residency,
// so the per-stage optimum is exact; the top-K list adds the single-stage
// alternatives around it, exactly as the decode solver does (prompt §4 L0-2).
Sm90ForwardSearchResult SearchSm90Forward(Sm90ForwardSearchSpace const& space,
    Sm90ForwardCostModel const& cost,int top) {
  if(top<1)throw std::invalid_argument("native forward search needs a positive top count");
  auto start=std::chrono::steady_clock::now();
  Sm90ForwardSearchResult result;
  std::vector<Sm90ForwardCandidate> pool;
  for(int residency:{1,2}) {
    std::vector<std::vector<Sm90ForwardStage>> options(space.shapes.size());
    std::vector<std::vector<double>> costs(space.shapes.size());
    bool complete=true;
    for(std::size_t i=0;i<space.shapes.size();++i) {
      options[i]=space.Candidates(i,residency);
      if(options[i].empty()) {complete=false;break;}
      for(auto const& stage:options[i]) {
        costs[i].push_back(cost.Stage(space.shapes[i],stage,residency));
        ++result.evaluated;
      }
      // Rank each stage's own candidates; equal predictions keep the first.
      std::vector<std::size_t> order(options[i].size());
      for(std::size_t j=0;j<order.size();++j)order[j]=j;
      std::stable_sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b) {
        return costs[i][a]<costs[i][b];
      });
      std::vector<Sm90ForwardStage> sorted;std::vector<double> sorted_costs;
      for(auto j:order) {sorted.push_back(options[i][j]);sorted_costs.push_back(costs[i][j]);}
      options[i]=std::move(sorted);costs[i]=std::move(sorted_costs);
    }
    if(!complete)continue;
    auto build=[&](std::size_t stage,std::size_t alternative,std::string const& origin) {
      Sm90ForwardCoordinates c;c.residency=residency;
      for(std::size_t i=0;i<options.size();++i)
        c.stages.push_back(options[i][i==stage && alternative<options[i].size()?alternative:0]);
      pool.push_back({c,cost.Predict(space.shapes,c),origin});
    };
    std::string prefix="r"+std::to_string(residency);
    build(space.shapes.size(),0,prefix+"_optimum");
    for(std::size_t i=0;i<options.size();++i)
      if(options[i].size()>1)build(i,1,prefix+"_stage"+std::to_string(i)+"_alt1");
  }
  if(pool.empty())throw std::invalid_argument("native forward search found no legal coordinate");
  std::stable_sort(pool.begin(),pool.end(),[](auto const& a,auto const& b) {
    return a.prediction.total_ns<b.prediction.total_ns;
  });
  if(int(pool.size())>top)pool.resize(top);
  result.ranked=std::move(pool);
  result.elapsed_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-start).count();
  return result;
}

}  // namespace tilemega::solver
