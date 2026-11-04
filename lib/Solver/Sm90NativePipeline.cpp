// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace tilemega::solver {
namespace {
std::string CoordinateKey(Sm90DenseCoordinates const& c) {
  return c.ToJson().Dump();
}

double Replay(Sm90NativeFlowResult const& flow, Sm90DenseCoordinates const& c) {
  std::vector<int> offsets(flow.problem.spaces.size());
  if(c.expert_fuse)for(std::size_t s=1;s<offsets.size();++s)
    if(flow.nodes[s].price.name=="gemm7" && flow.nodes[s-1].price.name=="gemm6")
      offsets[s]=flow.problem.spaces[s-1].count;
  return RefineAndReplaySm90Native(flow.problem,
      Sm90NativeCandidateTracks(flow.problem,offsets)).replay.makespan_ns;
}
}  // namespace

std::vector<std::vector<int>> Sm90NativeCandidateTracks(
    FlowProblem const& p,std::vector<int> const& offsets) {
  if(p.workers<=0 || (!offsets.empty() && offsets.size()!=p.spaces.size()))
    throw std::invalid_argument("native track contract has invalid dimensions");
  std::vector<std::vector<int>> tracks(p.spaces.size());
  for(std::size_t s=0;s<p.spaces.size();++s) {
    int offset=offsets.empty()?0:offsets[s];
    if(offset<0)throw std::invalid_argument("native track offset is negative");
    for(int t=0;t<p.spaces[s].count;++t)tracks[s].push_back((offset+t)%p.workers);
  }
  return tracks;
}

Sm90NativeReplayResult RefineAndReplaySm90Native(FlowProblem const& p,
    std::vector<std::vector<int>> const& tracks) {
  if(p.workers<=0 || tracks.size()!=p.spaces.size())
    throw std::invalid_argument("native replay lacks the singleton track contract");
  Sm90NativeReplayResult out;
  out.placement.worker_of_task=tracks;out.placement.order.resize(p.workers);
  out.estimated_start_ns.resize(p.spaces.size());
  out.estimated_finish_ns.resize(p.spaces.size());
  std::vector<std::vector<int>> pending(p.spaces.size());
  std::vector<std::vector<double>> ready(p.spaces.size()),duration(p.spaces.size());
  std::vector<double> available(p.workers),rank(p.spaces.size());
  std::vector<int> prefix(p.spaces.size(),-1);
  std::size_t tasks=0;
  for(std::size_t s=0;s<p.spaces.size();++s) {
    auto const& space=p.spaces[s];
    if(tracks[s].size()!=std::size_t(space.count))
      throw std::invalid_argument("native replay track count differs from its CTG");
    pending[s].resize(space.count);ready[s].resize(space.count);
    out.estimated_start_ns[s].resize(space.count,-1);
    out.estimated_finish_ns[s].resize(space.count,-1);
    for(int t=0;t<space.count;++t) {
      if(tracks[s][t]<0 || tracks[s][t]>=p.workers)
        throw std::invalid_argument("native replay track is outside the grid");
      auto const& price=space.pieces.at(space.piece_of_task.at(t)).parts;
      double rate=std::min(price.dram_rate_cap,p.dram_gbps/p.workers);
      double memory=price.dram_bytes?price.dram_bytes/rate:0;
      double d=price.fixed_ns+std::max(price.compute_ns,memory);
      if(price.pipeline_steps>0)d+=std::min(memory,price.pipeline_compute_ns<0?
          price.compute_ns:price.pipeline_compute_ns)/price.pipeline_steps;
      duration[s].push_back(d);rank[s]=std::max(rank[s],d);
    }
    rank[s]*=(space.count+p.workers-1)/p.workers;tasks+=space.count;
  }
  struct Release {int producer,consumer;double hop;std::size_t cursor=0;
    std::vector<std::pair<int,int>> endpoints;};
  std::vector<Release> releases;
  std::vector<std::vector<int>> outgoing(p.spaces.size());
  for(auto const& edge:p.edges) {
    if(edge.producer<0 || edge.consumer<=edge.producer ||
       edge.consumer>=int(p.spaces.size()) || !edge.sorted)
      throw std::invalid_argument("native replay requires forward CTG boundaries");
    Release release{edge.producer,edge.consumer,edge.hop_ns<0?p.hop_ns:edge.hop_ns};
    for(auto [last,task]:*edge.sorted) {
      if(edge.all_producer)last=p.spaces[edge.producer].count-1;
      if(last<0)continue;
      if(last>=p.spaces[edge.producer].count || task<0 || task>=p.spaces[edge.consumer].count)
        throw std::invalid_argument("native replay release is outside its task space");
      release.endpoints.emplace_back(last,task);++pending[edge.consumer][task];
    }
    std::sort(release.endpoints.begin(),release.endpoints.end());
    outgoing[edge.producer].push_back(releases.size());releases.push_back(std::move(release));
  }
  for(int s=int(p.spaces.size())-1;s>=0;--s) {
    double tail=0;
    for(int e:outgoing[s])tail=std::max(tail,releases[e].hop+rank[releases[e].consumer]);
    rank[s]+=tail;
  }
  using Key=std::tuple<double,double,int,int>;
  std::priority_queue<Key,std::vector<Key>,std::greater<Key>> frontier;
  auto push=[&](int s,int t) {
    frontier.emplace(std::max(ready[s][t],available[tracks[s][t]]),-rank[s],s,t);
  };
  for(std::size_t s=0;s<p.spaces.size();++s)
    for(int t=0;t<p.spaces[s].count;++t)if(!pending[s][t])push(s,t);
  std::vector<double> prefix_end(p.spaces.size());
  std::size_t planned=0;
  while(!frontier.empty()) {
    auto [estimate,remaining,s,t]=frontier.top();frontier.pop();
    int worker=tracks[s][t];
    double start=std::max(ready[s][t],available[worker]);
    if(start>estimate) {push(s,t);continue;}
    double finish=start+duration[s][t];
    out.estimated_start_ns[s][t]=start;out.estimated_finish_ns[s][t]=finish;
    available[worker]=finish;out.placement.order[worker].emplace_back(s,t);++planned;
    // The oracle's scalar endpoint denotes a completed prefix. Never
    // release from a later planned task while an earlier index is unplanned.
    while(prefix[s]+1<p.spaces[s].count && out.estimated_finish_ns[s][prefix[s]+1]>=0) {
      ++prefix[s];prefix_end[s]=std::max(prefix_end[s],out.estimated_finish_ns[s][prefix[s]]);
      for(int e:outgoing[s]) {
        auto& release=releases[e];
        while(release.cursor<release.endpoints.size() &&
              release.endpoints[release.cursor].first<=prefix[s]) {
          int consumer=release.consumer,task=release.endpoints[release.cursor++].second;
          ready[consumer][task]=std::max(ready[consumer][task],prefix_end[s]+release.hop);
          if(--pending[consumer][task]==0)push(consumer,task);
        }
      }
    }
  }
  if(planned!=tasks)throw std::invalid_argument("native replay has an unresolved CTG frontier");
  FlowOptions options;options.placement=out.placement;
  out.replay=EvaluateFlow(p,options);
  return out;
}

std::string Sm90NativeStructuralClass(Sm90DenseCoordinates const& c) {
  // Program level, attention structure and the handoff choices: everything
  // except each family's (TN, TK, wn, wk, multicast, slots).
  std::string key;
  key += "res" + std::to_string(c.residency);
  key += "/cl" + std::to_string(c.cluster);
  key += c.group_wait ? "/gw1" : "/gw0";
  key += c.embed_la ? "/la1" : "/la0";
  key += c.expert_fuse ? "/ef1" : "/ef0";
  key += "/ch" + std::to_string(c.attention.chunks);
  key += c.attention.dsmem ? "/ds1" : "/ds0";
  return key;
}

Sm90NativePipelineResult SolveSm90NativePipeline(
    Sm90DenseSearchSpace const& space, Sm90DenseWorkload const& work,
    Sm90DenseCostModel const& cost, Sm90NativePipelineOptions const& options,
    std::function<Sm90NativeFlowResult(Sm90DenseCoordinates const&,bool)> const& evaluate,
    std::vector<Sm90DenseCoordinates> const& seeds) {
  if(!evaluate)throw std::invalid_argument("native search requires a CTG evaluator");
  using Clock = std::chrono::steady_clock;
  auto started = Clock::now();
  auto spent_ms = [&] {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  };

  Sm90NativePipelineResult result;
  std::map<std::string, double> memo;  // Algorithm 1 line 11's E
  auto derive=[&](Sm90DenseCoordinates& c) {
    Sm90DenseSearchSpace::Structure(c);
    if(c.expert_fuse) {
      for(int slots=8;slots>=2;--slots) {
        c.gemms[6].slots=c.gemms[7].slots=slots;
        if(space.Legal(c))return true;
      }
      return false;
    }
    return space.Legal(c);
  };

  auto score = [&](Sm90DenseCoordinates const& c) -> double {
    auto key = CoordinateKey(c);
    auto found = memo.find(key);
    if (found != memo.end()) {
      ++result.memo_hits;
      return found->second;
    }
    auto before=Clock::now();
    double value=evaluate(c,options.operator_sum).step_ns;
    memo.emplace(key, value);
    ++result.evaluations;
    Sm90NativePipelineCandidate candidate;
    candidate.coordinates = c;
    candidate.pool_score_ns = value;
    candidate.evaluation_ms=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
    candidate.structural_class = Sm90NativeStructuralClass(c);
    result.evaluated.push_back(std::move(candidate));
    return value;
  };

  // Algorithm 1 line 4: the seeds. (1) the caller's current final coordinates,
  // (2) the additive model's top-1 and (3) its per-family optima, which the
  // existing search already produces.
  std::vector<Sm90DenseCoordinates> starts;
  auto project=[&](Sm90DenseCoordinates seed) {
    if(options.pinned_theta)seed.gemms=options.pinned_theta->gemms;
    if(options.retile_only) {seed.attention.chunks=1;seed.expert_fuse=false;}
    if(options.realize_aggregate)seed.group_wait=false;
    if(derive(seed))starts.push_back(seed);
  };
  for (auto const& seed : seeds)project(seed);
  Sm90DenseSearchOptions additive;
  additive.top = 1;
  auto seeded = SearchSm90Dense(space, work, cost, additive);
  for (auto const& candidate : seeded.ranked)
    project(candidate.coordinates);
  for(auto const& candidate:seeded.per_structure_optima)project(candidate.coordinates);
  if (starts.empty())
    throw std::invalid_argument("native pipeline search found no legal seed");

  // Lines 5-13: coordinate descent. A move is accepted only when it improves
  // the whole-program score, and derived coordinates are re-derived after each
  // move exactly as the additive search does.
  for (auto const& start : starts) {
    if (spent_ms() > options.budget_ms) break;
    auto current = start;
    double best = score(current);
    result.descent.emplace_back();
    result.descent.back().push_back(best);
    for (bool changed = true; changed && spent_ms() <= options.budget_ms;) {
      changed = false;
      // The per-family geometry.
      for (unsigned family : current.Active()) {
        if(options.pinned_theta)continue;
        for (auto const& geometry : space.FamilyCandidates(family, current.residency)) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          trial.gemms[family] = geometry;
          if (!derive(trial)) continue;
          double value = score(trial);
          if (value < best) {
            best = value; current = trial; changed = true;
            result.descent.back().push_back(value);
          }
        }
      }
      // The attention structure, unless AB-4a forbids changing it.
      if (!options.retile_only) {
        for (auto const& attention : space.AttentionCandidates(current.residency)) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          trial.attention = attention;
          if (!derive(trial)) continue;
          double value = score(trial);
          if (value < best) {
            best = value; current = trial; changed = true;
            result.descent.back().push_back(value);
          }
        }
      }
      // The program-level coordinates, which are the structural class.
      for(int dimension=0;dimension<5;++dimension) {
        for(int setting:{0,1}) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          if(dimension==0)trial.residency=setting+1;
          if(dimension==1)trial.cluster=setting+1;
          if(dimension==2)trial.group_wait=options.realize_aggregate?false:bool(setting);
          if(dimension==3)trial.embed_la=bool(setting);
          if(dimension==4)trial.expert_fuse=options.retile_only?false:bool(setting);
          if (!derive(trial)) continue;
          double value = score(trial);
          if (value < best) {
            best = value; current = trial; changed = true;
            result.descent.back().push_back(value);
          }
        }
      }
    }
  }

  // Line 14: the shortlist. The best `shortlist` by score, then the best of
  // each structural class not yet represented, up to `structural_extra`.
  auto ranked = result.evaluated;
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](auto const& a, auto const& b) {return a.pool_score_ns < b.pool_score_ns;});
  std::set<std::string> classes;
  for (auto const& candidate : ranked) {
    if (int(result.shortlist.size()) >= options.shortlist) break;
    result.shortlist.push_back(candidate);
    classes.insert(candidate.structural_class);
  }
  int extra = 0;
  for (auto const& candidate : ranked) {
    if (extra >= options.structural_extra) break;
    if (classes.insert(candidate.structural_class).second) {
      result.shortlist.push_back(candidate);
      ++extra;
    }
  }

  // Lines 15-19: materialize, replay, and keep at most `per_class` finalists
  // from any one structural class so the measured set stays varied.
  for (auto& candidate : result.shortlist)
    candidate.replay_ns = options.operator_sum?candidate.pool_score_ns:
        Replay(evaluate(candidate.coordinates,false), candidate.coordinates);
  auto by_replay = result.shortlist;
  std::stable_sort(by_replay.begin(), by_replay.end(),
                   [](auto const& a, auto const& b) {return a.replay_ns < b.replay_ns;});
  std::map<std::string, int> taken;
  for (auto const& candidate : by_replay) {
    if (int(result.finalists.size()) >= options.finalists) break;
    if (taken[candidate.structural_class] >= options.per_class) continue;
    ++taken[candidate.structural_class];
    result.finalists.push_back(candidate);
  }
  // Short of the quota, fill by replay time regardless of class.
  for (auto const& candidate : by_replay) {
    if (int(result.finalists.size()) >= options.finalists) break;
    bool present = std::any_of(result.finalists.begin(), result.finalists.end(),
        [&](auto const& f) {return CoordinateKey(f.coordinates) ==
                                   CoordinateKey(candidate.coordinates);});
    if (!present) result.finalists.push_back(candidate);
  }
  result.seconds = spent_ms() / 1000.0;
  return result;
}
}  // namespace tilemega::solver
