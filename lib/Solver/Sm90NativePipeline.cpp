// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
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

bool RequiredDenseService(Sm90DenseWorkload const& work,Sm90DenseCoordinates const& c,unsigned family) {
  auto const& gemm=c.gemms[family];
  return (c.live_epilogue_rows && work.batch<16 && (family==2 || gemm.warps_k>1)) ||
      (c.residual_copy_issue && Sm90DenseCoordinates::ResidualFamily(family)) ||
      (family==2 && (c.copy_issue_group ||
          c.swiglu_u!=Sm90DenseSearchSpace::DefaultGateInterleave(gemm.tile_n)));
}

bool RequiredFusedService(Sm90DenseCoordinates const& c) {
  return c.copy_issue_group || c.expert_gu_route_bases;
}

bool HasCompactAttentionService(Sm90DenseWorkload const& work,Sm90DenseCostModel const& cost,int grid) {
  int norm=work.qk_norm?(work.qk_norm_row?2:1):0;
  return cost.Constants().AttentionFresh(work.head_dim,work.queries,norm) &&
      cost.Constants().CompactAttentionBarrier(grid,work.batch,work.head_dim,work.queries,norm);
}

double Replay(Sm90NativeFlowResult const& flow, Sm90DenseCoordinates const& c) {
  std::vector<int> offsets(flow.problem.spaces.size());
  if(c.expert_fuse)for(std::size_t s=1;s<offsets.size();++s)
    if(flow.nodes[s].price.name=="gemm7" && flow.nodes[s-1].price.name=="gemm6")
      offsets[s]=flow.problem.spaces[s-1].count;
  auto tracks=Sm90NativeCandidateTracks(flow.problem,offsets);
  return Sm90NativeRequestScore(flow,[&](FlowProblem const& problem,int) {
    return Sm90NativeScoreWithFusedService(flow,RefineAndReplaySm90Native(problem,tracks).replay);
  });
}
}  // namespace

json::Value Sm80NativeCandidateCoverage(Sm90DenseSearchSpace const& space,
    Sm90DenseWorkload const& work,Sm90DenseCostModel const& cost,Sm90DenseCoordinates const& c) {
  json::Value row=json::Object{{"coordinates",c.ToJson()}};
  if(c.moe!=space.moe || c.tp!=space.tp) {
    row.Set("status","abi_incompatible");
    row.Set("reason","coordinate and workload disagree on the model kind or tensor parallel degree");
    return row;
  }
  if(!space.Legal(c)) {
    row.Set("status","resource_illegal");
    return row;
  }

  json::Array missing,proxies,services;
  int grid=space.Grid(c.residency);
  auto fused=c.moe && cost.FusedService(work,c).has_value();
  for(unsigned family:c.Active()) {
    auto const& g=c.gemms[family];
    std::string name=kSm90DenseFamilies[family];
    auto mma=Sm90NativeConstants::AmpereMmaKey(grid,g.tile_n,g.TileK(),g.warps_n,g.register_stages);
    if(!cost.HasCalibratedMma(g,c.residency))
      missing.emplace_back(json::Object{{"kind","mma"},{"family",name},{"key",mma},
          {"warps_n",g.warps_n},{"warps_k",g.warps_k}});
    auto shape=cost.DenseServiceShape(work,family,g,c.residency,&c);
    if(!shape)continue;
    json::Value service=json::Object{{"kind","dense_service"},{"family",name},{"key",shape->Key()}};
    if(cost.DenseService(work,family,g,c.residency,&c))services.emplace_back(service);
    else if(!c.moe && RequiredDenseService(work,c,family))
      missing.emplace_back(service);
    else proxies.emplace_back(service);
  }
  if(c.moe) {
    auto shape=cost.FusedServiceShape(work,c);
    json::Array key;
    if(shape)for(int value:shape->Key())key.emplace_back(value);
    json::Value service=json::Object{{"kind","fused_service"},{"family","gu_dn"},{"key",key}};
    if(fused)services.emplace_back(service);
    else if(RequiredFusedService(c))missing.emplace_back(service);
    else proxies.emplace_back(service);
  }
  if(c.attention.compact_workspace) {
    int norm=work.qk_norm?(work.qk_norm_row?2:1):0;
    json::Value service=json::Object{{"kind","attention_service"},{"family","compact_attention"},
        {"key",json::Object{{"grid",grid},{"batch",work.batch},{"head_dim",work.head_dim},
            {"queries",work.queries},{"norm",norm}}}};
    if(HasCompactAttentionService(work,cost,grid))services.emplace_back(std::move(service));
    else missing.emplace_back(std::move(service));
  }
  row.Set("status",missing.empty()?"legal_priced":"legal_missing_calibration");
  row.Set("missing",std::move(missing));
  row.Set("modeled_services",std::move(proxies));
  row.Set("measured_services",std::move(services));
  return row;
}

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

std::string Sm90NativeStructuralClass(Sm90DenseCoordinates const& c,bool ampere) {
  // Program level, attention structure and the handoff choices: everything
  // except each family's (TN, TK, wn, wk, multicast, slots).
  std::string key;
  key += "res" + std::to_string(c.residency);
  key += "/cl" + std::to_string(c.cluster);
  key += c.group_wait ? "/gw1" : "/gw0";
  key += c.embed_la ? "/la1" : "/la0";
  key += c.argmax_stage ? "/argstage" : "/argla";
  if(c.moe)key+=std::string("/route-")+c.RouteName();
  if(c.moe)key+=c.combine_stage?"/combinestage":"/combinela";
  key += c.expert_fuse ? "/ef1" : "/ef0";
  if(ampere && c.expert_fuse)
    key += "/slots"+std::to_string(c.CommonExpertSlots())+
           "/register"+std::to_string(c.CommonExpertRegisterStages());
  if(ampere && !c.moe)key += "/u"+std::to_string(c.swiglu_u);
  if(c.live_epilogue_rows)key += "/live_rows";
  if(c.copy_issue_group)key += "/copy"+std::to_string(c.copy_issue_group);
  if(c.residual_copy_issue)key += "/residual_copy";
  if(c.expert_page_segments)key += "/page_segments";
  if(c.expert_gu_route_bases)key += "/gu_route_bases";
  if(c.reuse_payload)key += "/payload_reuse";
  key += "/ch" + std::to_string(c.attention.chunks);
  if(c.attention.compact_workspace)key += "/compact";
  key += c.attention.dsmem ? "/ds1" : "/ds0";
  key += c.attention.merge_stage ? "/attstage" : "/attla";
  return key;
}

std::optional<Sm90NativePipelineCandidate> RepriceSm90NativeResidency(
    Sm90DenseSearchSpace const& space,Sm90DenseCoordinates const& original,int actual,
    bool isolated,std::function<Sm90NativeFlowResult(Sm90DenseCoordinates const&,bool)> const& evaluate) {
  if(actual<1)return std::nullopt;
  if(actual>2 || !evaluate)throw std::invalid_argument("invalid native occupancy correction");
  Sm90NativePipelineCandidate result;result.coordinates=original;result.coordinates.residency=actual;
  // Keep the compiled geometry and ring depths. Changing them would turn a
  // residency correction into an unmeasured new search move.
  if(!space.Legal(result.coordinates))return std::nullopt;
  auto start=std::chrono::steady_clock::now();
  Sm90NativeFlowResult flow;
  try {flow=evaluate(result.coordinates,isolated);}
  catch(Sm90NativeMissing const&) {return std::nullopt;}
  result.pool_score_ns=flow.step_ns;
  result.replay_ns=isolated?flow.step_ns:Replay(flow,result.coordinates);
  result.evaluation_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-start).count();
  result.structural_class=Sm90NativeStructuralClass(result.coordinates,space.target.arch_tag=="sm_80");result.origin="compiled occupancy";
  return result;
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
  bool ampere=space.target.arch_tag=="sm_80";
  int norm=work.qk_norm?(work.qk_norm_row?2:1):0;
  std::map<std::string, double> memo;  // Algorithm 1 line 11's E
  std::string origin;
  std::map<std::tuple<unsigned,int,bool>,std::vector<Sm90DenseGemm>> family_domain;
  auto geometries=[&](unsigned family,Sm90DenseCoordinates const& c)->std::vector<Sm90DenseGemm> const& {
    auto key=std::make_tuple(family,c.residency,c.reuse_payload);
    auto found=family_domain.find(key);
    if(found==family_domain.end())found=family_domain.emplace(key,
        space.FamilyCandidates(family,c.residency,c.reuse_payload)).first;
    return found->second;
  };
  auto constrain=[&](Sm90DenseCoordinates& c) {
    if(options.retile_only) {c.attention.chunks=1;c.attention.dsmem=false;c.expert_fuse=false;}
    if(options.realize_aggregate) {
      c.group_wait=false;c.embed_la=false;c.argmax_stage=true;
      c.attention.dsmem=false;c.attention.merge_stage=true;c.expert_fuse=false;
      if(c.moe) {c.route=Sm90DenseCoordinates::Route::kStage;c.combine_stage=true;}
    }
    // An unfuse move also chooses the minimal global publication needed by
    // its two stages. A local route cannot survive outside the fused stage.
    if(c.moe && !c.expert_fuse && !c.GlobalRoute())c.route=Sm90DenseCoordinates::Route::kLastArriver;
  };
  auto derive=[&](Sm90DenseCoordinates& c) {
    constrain(c);
    Sm90DenseSearchSpace::Structure(c,ampere);
    // The pinned dense gate tile also fixes the gate/up weight permutation.
    if(options.pinned_theta && !c.moe && ampere) {
      c.swiglu_u=options.pinned_theta->swiglu_u;
      c.live_epilogue_rows=options.pinned_theta->live_epilogue_rows;
    }
    for(unsigned family:c.Active()) {
      auto& geometry=c.gemms[family];
      auto const& domain=geometries(family,c);
      auto matches=[&](auto const& g) {
        return g.tile_n==geometry.tile_n && g.TileK()==geometry.TileK() &&
               g.warps_n==geometry.warps_n && g.warps_k==geometry.warps_k &&
               (!ampere || g.register_stages==geometry.register_stages) &&
               (!ampere || !c.expert_fuse || family<6 || g.slots==geometry.slots);
      };
      auto same=std::find_if(domain.begin(),domain.end(),matches);
      if(same==domain.end())return false;
      // SM80 calibrates several depths per geometry. Keep the chosen depth
      // until a smaller arena makes it infeasible.
      if(ampere)
        for(auto candidate=std::next(same);candidate!=domain.end();++candidate)
          if(candidate->slots<=geometry.slots && candidate->slots>same->slots &&
             matches(*candidate))same=candidate;
      geometry.slots=same->slots;
    }
    auto attention=space.AttentionCandidates(c.residency,&cost.Constants(),norm,c.reuse_payload);
    auto partition=[&](auto const& a) {
      return a.chunks==c.attention.chunks && a.block_keys==c.attention.block_keys;
    };
    auto same=std::find_if(attention.begin(),attention.end(),partition);
    if(same==attention.end())return false;
    c.attention.compact_workspace=same->compact_workspace;
    if(std::none_of(same,attention.end(),[&](auto const& a) {
         return partition(a) && a.slots==c.attention.slots;
       }))c.attention.slots=same->slots;
    if(c.expert_fuse && !ampere) {
      for(int slots=8;slots>=2;--slots) {
        c.SetCommonExpertSlots(slots);
        if(space.Legal(c))return true;
      }
      return false;
    }
    return space.Legal(c);
  };

  auto priced=[&](Sm90DenseCoordinates const& c) {
    if(!cost.HasCalibratedMma(c))return false;
    if(!ampere)return true;
    if(c.attention.compact_workspace && !HasCompactAttentionService(work,cost,space.Grid(c.residency)))
      return false;
    if(c.moe)return !RequiredFusedService(c) || cost.FusedService(work,c).has_value();
    for(unsigned f:{0u,1u,2u,3u}) {
      auto const& g=c.gemms[f];
      if(RequiredDenseService(work,c,f) && !cost.DenseService(work,f,g,c.residency,&c))return false;
    }
    return true;
  };

  auto body_neighbours=[&](Sm90DenseCoordinates const& c,bool audit=false) {
    std::vector<std::pair<Sm90DenseCoordinates,std::string>> result;
    if(!ampere || c.weight_stream || options.pinned_theta)return result;
    if(c.moe) {
      auto calibrated=[&](auto const& trial,std::string const& label) {
        if(audit && options.audit_candidate)options.audit_candidate(trial,"body_neighbour",false);
        if(cost.FusedService(work,trial))result.emplace_back(trial,label);
      };
      for(int group:{0,4,8})if(group!=c.copy_issue_group) {
        auto trial=c;trial.copy_issue_group=group;
        calibrated(trial,"expert weight copy issue");
      }
      auto page=c;page.expert_page_segments=!c.expert_page_segments;
      calibrated(page,"expert weight pages");
      auto routes=c;routes.expert_gu_route_bases=!c.expert_gu_route_bases;
      calibrated(routes,"expert Gu activation routes");
      auto rows=c;rows.live_epilogue_rows=!c.live_epilogue_rows;
      calibrated(rows,"expert accumulator rows");
      return result;
    }
    if(work.batch<16) {
      auto trial=c;trial.live_epilogue_rows=!c.live_epilogue_rows;
      result.emplace_back(trial,"accumulator rows");
    }
    if(c.copy_issue_group) {
      auto trial=c;trial.residual_copy_issue=!c.residual_copy_issue;
      if(audit && options.audit_candidate)options.audit_candidate(trial,"body_neighbour",false);
      if(priced(trial))result.emplace_back(trial,"residual weight copy issue");
    }
    for(int group:{0,4,8})if(group!=c.copy_issue_group) {
      auto trial=c;trial.copy_issue_group=group;
      if(!group)trial.residual_copy_issue=false;
      result.emplace_back(trial,"weight copy issue");
    }
    for(int u:{4,8,16})if(u!=c.swiglu_u && c.gemms[2].tile_n%(2*u)==0) {
      auto trial=c;trial.swiglu_u=u;
      result.emplace_back(trial,"gate interleave");
    }
    return result;
  };

  auto measured_body_seeds=[&](Sm90DenseCoordinates const& base) {
    std::vector<Sm90DenseCoordinates> candidates;
    if(!ampere || base.weight_stream || options.pinned_theta)return candidates;
    if(base.moe) {
      if(base.residency!=1 || base.reuse_payload)return candidates;
      for(auto const& shape:cost.Constants().AmpereFusedShapes()) {
        if(shape.batch!=work.batch || shape.experts!=work.experts ||
           shape.gate_up_n!=work.shapes[6][0] || shape.down_n!=work.shapes[7][0] ||
           shape.hidden!=work.shapes[6][1])continue;
        auto trial=base;
        trial.expert_fuse=true;
        trial.route=Sm90DenseCoordinates::Route::kLocal;
        trial.combine_stage=false;
        trial.expert_page_segments=shape.segments!=0;
        trial.expert_gu_route_bases=shape.gu_route_bases!=0;
        trial.live_epilogue_rows=shape.live_rows!=0;
        trial.copy_issue_group=shape.copy_group;
        auto retile=[&](unsigned family,int tn,int tk,int wn) {
          auto& g=trial.gemms[family];
          g.tile_n=tn;g.slot_bytes=2*tn*tk;g.slots=shape.slots;
          g.warps_n=wn;g.warps_k=4/wn;g.register_stages=shape.register_stages;
        };
        retile(6,shape.gate_up_tn,shape.gate_up_tk,shape.gate_up_warps_n);
        retile(7,shape.down_tn,shape.down_tk,shape.down_warps_n);
        if(cost.FusedService(work,trial))candidates.push_back(trial);
      }
      return candidates;
    }
    auto const& measured=cost.Constants().AmpereDenseShapes();
    auto matches=[&](Sm80DenseServiceShape const& s,unsigned family) {
      return s.grid==space.Grid(base.residency) && s.batch==work.batch &&
          s.n==work.shapes[family][0] && s.k==work.shapes[family][1];
    };
    auto retile=[](Sm90DenseGemm& g,Sm80DenseServiceShape const& s) {
      g.tile_n=s.tn;g.slot_bytes=2*s.tn*s.tk;g.slots=s.slots;
      g.warps_n=s.warps_n;g.warps_k=s.warps_k;g.register_stages=s.register_stages;
    };
    for(auto const& gate:measured) {
      if(gate.epilogue!="swiglu" || !matches(gate,2) ||
         (!gate.live_rows && !gate.copy_issue_group &&
           gate.u==Sm90DenseSearchSpace::DefaultGateInterleave(gate.tn)))continue;
      auto trial=base;
      retile(trial.gemms[2],gate);
      trial.swiglu_u=gate.u;trial.copy_issue_group=gate.copy_issue_group;
      trial.live_epilogue_rows=gate.live_rows;
      if(gate.live_rows)for(unsigned f:{0u,1u,3u}) {
        auto& g=trial.gemms[f];
        if(g.warps_k==1 || cost.DenseService(work,f,g,trial.residency,&trial))continue;
        auto epilogue=f==0?"store":"residual";
        auto replacement=std::find_if(measured.begin(),measured.end(),[&](auto const& s) {
          return s.epilogue==epilogue && matches(s,f) && s.live_rows;
        });
        if(replacement!=measured.end())retile(g,*replacement);
      }
      candidates.push_back(trial);
    }
    auto gate_candidates=candidates;
    for(auto trial:gate_candidates) {
      if(!trial.copy_issue_group)continue;
      trial.residual_copy_issue=true;
      for(unsigned f:{1u,3u}) {
        auto& g=trial.gemms[f];
        if(cost.DenseService(work,f,g,trial.residency,&trial))continue;
        auto replacement=std::find_if(measured.begin(),measured.end(),[&](auto const& s) {
          return s.epilogue=="residual" && matches(s,f) && s.copy_issue_group==trial.copy_issue_group &&
              s.live_rows==(trial.live_epilogue_rows && work.batch<16 && s.warps_k>1);
        });
        if(replacement!=measured.end())retile(g,*replacement);
      }
      if(priced(trial))candidates.push_back(trial);
    }
    return candidates;
  };

  auto ring_neighbours=[&](Sm90DenseCoordinates const& base) {
    std::vector<Sm90DenseCoordinates> result;
    if(!ampere || base.weight_stream || options.pinned_theta)return result;
    if(base.moe) {
      if(!base.expert_fuse)return result;
      for(int depth=2;depth<=8;++depth) {
        if(depth==base.CommonExpertSlots())continue;
        auto trial=base;trial.SetCommonExpertSlots(depth);
        if(options.audit_candidate)options.audit_candidate(trial,"ring_neighbour",false);
        if(cost.FusedService(work,trial))result.push_back(trial);
      }
      return result;
    }
    for(unsigned f:{0u,1u,2u,3u})
      for(int depth=2;depth<=8;++depth) {
        if(depth==base.gemms[f].slots)continue;
        auto trial=base;trial.gemms[f].slots=depth;
        if(options.audit_candidate)options.audit_candidate(trial,"ring_neighbour",false);
        if(cost.DenseService(work,f,trial.gemms[f],trial.residency,&trial))result.push_back(trial);
      }
    return result;
  };

  auto score = [&](Sm90DenseCoordinates const& c) -> double {
    auto key = CoordinateKey(c);
    auto found = memo.find(key);
    if (found != memo.end()) {
      ++result.memo_hits;
      return found->second;
    }
    auto before=Clock::now();
    double value;
    try {value=evaluate(c,options.operator_sum).step_ns;}
    catch(Sm90NativeMissing const&) {
      if(!ampere)throw;
      return memo.emplace(key,std::numeric_limits<double>::infinity()).first->second;
    }
    memo.emplace(key, value);
    ++result.evaluations;
    Sm90NativePipelineCandidate candidate;
    candidate.coordinates = c;
    candidate.pool_score_ns = value;
    candidate.evaluation_ms=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
    candidate.discovered_ms=spent_ms();
    candidate.structural_class = Sm90NativeStructuralClass(c,ampere);
    candidate.origin=origin;
    result.evaluated.push_back(std::move(candidate));
    return value;
  };

  // Algorithm 1 line 4: the seeds. (1) the caller's current final coordinates,
  // (2) the additive model's top-1 and (3) its per-family optima, which the
  // existing search already produces.
  std::vector<Sm90DenseCoordinates> starts;
  std::set<std::string> start_keys;
  std::set<std::string> protected_anchors;
  if(options.require_anchors && (!ampere || seeds.empty() || options.pinned_theta ||
      options.retile_only || options.realize_aggregate || options.operator_sum))
    throw std::invalid_argument("required anchors need the SM80 end-to-end pipeline");
  auto project=[&](Sm90DenseCoordinates seed,bool required=false) {
    if(options.pinned_theta) {
      seed.gemms=options.pinned_theta->gemms;
      seed.reuse_payload=options.pinned_theta->reuse_payload;
    }
    constrain(seed);
    bool projected=options.pinned_theta || options.retile_only || options.realize_aggregate;
    if(!(projected?derive(seed):space.Legal(seed))) {
      if(required)throw std::invalid_argument("required SM80 anchor violates resource or planning constraints: "+
          CoordinateKey(seed));
      return;
    }
    if(!priced(seed)) {
      if(required) {
        for(unsigned family:seed.Active())cost.MmaK16Ns(seed.gemms[family],seed.residency);
        throw Sm90NativeMissing("required SM80 anchor is missing its exact body price: "+CoordinateKey(seed));
      }
      return;
    }
    auto key=CoordinateKey(seed);
    if(required)protected_anchors.insert(key);
    if(start_keys.insert(key).second)starts.push_back(seed);
  };
  for(std::size_t i=0;i<seeds.size();++i) {
    if(options.audit_candidate)
      options.audit_candidate(seeds[i],i?"submitted_trial":"incumbent",options.require_anchors);
    project(seeds[i],options.require_anchors);
  }
  if(int(protected_anchors.size())>options.finalists)
    throw std::invalid_argument("more required anchors than the finalist budget");
  Sm90DenseSearchOptions additive;
  additive.top = 1;
  additive.allow_empty=ampere && !starts.empty();
  auto seeded=SearchSm90Dense(space,work,cost,additive);
  for(auto const& candidate:seeded.ranked)project(candidate.coordinates);
  for(auto const& candidate:seeded.per_structure_optima)project(candidate.coordinates);
  // Score a finite implementation frontier before the large tile domain.
  // Keep one seed per implementation, residency and arena for this pass.
  auto base_seeds=starts;
  std::set<std::tuple<bool,int,int,int,bool,bool,bool,bool>> seeded_bodies;
  auto project_body=[&](Sm90DenseCoordinates const& trial) {
    if(options.audit_candidate)options.audit_candidate(trial,"body_neighbour",false);
    auto body=std::make_tuple(trial.live_epilogue_rows,trial.copy_issue_group,
        trial.swiglu_u,trial.residency,trial.reuse_payload,trial.expert_page_segments,
        trial.expert_gu_route_bases,trial.residual_copy_issue);
    if(!seeded_bodies.count(body) && space.Legal(trial) && priced(trial)) {
      seeded_bodies.insert(body);
      project(trial);
    }
  };
  for(auto const& base:base_seeds) {
    for(auto const& [trial,label]:body_neighbours(base,true))project_body(trial);
    for(auto const& trial:measured_body_seeds(base))project_body(trial);
  }
  // Slot depth may leave the body class unchanged. Visit calibrated depths
  // on the leading seed before the geometry descent spends its budget.
  auto ring_peers=base_seeds.empty()?std::vector<Sm90DenseCoordinates>{}:
      ring_neighbours(base_seeds.front());
  for(auto const& trial:ring_peers) {
    if(options.audit_candidate)options.audit_candidate(trial,"ring_neighbour",false);
    project(trial);
  }
  if (starts.empty())
    throw std::invalid_argument("native pipeline search found no legal seed");
  for(std::size_t i=0;i<starts.size();++i) {
    origin="seed:"+std::to_string(i);
    double value=score(starts[i]);
    if(!std::isfinite(value) && protected_anchors.count(CoordinateKey(starts[i])))
      throw Sm90NativeMissing("full flow is missing a price for required anchor "+CoordinateKey(starts[i]));
  }
  if(result.evaluated.empty())
    throw std::invalid_argument("native pipeline search found no priced seed");

  // Lines 5-13: coordinate descent. A move is accepted only when it improves
  // the whole-program score, and derived coordinates are re-derived after each
  // move exactly as the additive search does.
  for (std::size_t seed=0;seed<starts.size();++seed) {
    if (spent_ms() > options.budget_ms) break;
    auto current = starts[seed];
    double best = score(current);
    result.descent.emplace_back();
    result.descent.back().push_back(best);
    result.descent_coordinates.push_back({current});
    auto consider=[&](Sm90DenseCoordinates trial,std::string coordinate) {
      if(spent_ms()>options.budget_ms || !derive(trial) || !priced(trial))return false;
      origin="seed:"+std::to_string(seed)+"/"+coordinate;
      double value=score(trial);
      if(value>=best)return false;
      best=value;current=trial;
      result.descent.back().push_back(value);
      result.descent_coordinates.back().push_back(current);
      return true;
    };
    for (bool changed = true; changed && spent_ms() <= options.budget_ms;) {
      changed = false;
      for(auto const& [trial,implementation]:body_neighbours(current))
        changed|=consider(trial,implementation);
      // The per-family geometry.
      for (unsigned family : current.Active()) {
        if(options.pinned_theta)continue;
        for (auto const& geometry : geometries(family,current)) {
          for(bool multicast:{false,true}) {
            if(multicast && (current.cluster<2 || family>4 || geometry.slots%2 ||
                double(work.batch)/geometry.tile_n<=cost.Constants().MulticastKappa()))continue;
            auto trial=current;trial.gemms[family]=geometry;
            trial.gemms[family].multicast=multicast;
            if(ampere && trial.expert_fuse && family>=6) {
              trial.SetCommonExpertSlots(geometry.slots);
              trial.SetCommonExpertRegisterStages(geometry.register_stages);
            }
            auto label="family:"+std::to_string(family);
            changed|=consider(trial,label);
            if(ampere && !trial.moe && family==2) {
              for(int u:{4,8,16}) {
                if(u==trial.swiglu_u || geometry.tile_n%(2*u))continue;
                auto interleave=trial;interleave.swiglu_u=u;
                if(cost.DenseService(work,2,geometry,interleave.residency,&interleave))
                  changed|=consider(interleave,label+"/gate interleave");
              }
            }
          }
        }
      }
      // The attention structure, unless AB-4a forbids changing it.
      if (!options.retile_only) {
        for (auto const& attention : space.AttentionCandidates(current.residency,&cost.Constants(),norm,current.reuse_payload)) {
          for(int realization:{0,1,2}) {
            bool dsmem=realization==1,independent=realization==2;
            if(options.realize_aggregate && !independent)continue;
            if(independent && attention.chunks==1 && !options.realize_aggregate)continue;
            if(dsmem && (current.cluster<2 || attention.chunks<2 ||
                         current.cluster%attention.chunks))continue;
            auto trial=current;trial.attention=attention;trial.attention.dsmem=dsmem;
            trial.attention.merge_stage=independent;
            if(dsmem)try {cost.Constants().AttentionMergeDsmem(work.head_dim,work.queries,
                work.qk_norm?(work.qk_norm_row?2:1):0,attention.chunks);}
              catch(Sm90NativeMissing const&) {continue;}
            changed|=consider(trial,"attention");
          }
        }
      }
      if(ampere && space.search_payload_reuse && !current.weight_stream && !options.pinned_theta) {
        auto trial=current;trial.reuse_payload=!trial.reuse_payload;
        changed|=consider(trial,"shared payload");
      }
      // The program-level coordinates, which are the structural class.
      for(int dimension=0;dimension<8;++dimension) {
        if(dimension==5 && current.tp>1)continue;
        if(dimension>=6 && !current.moe)continue;
        for(int setting=0;setting<(dimension==7?3:2);++setting) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          if(dimension==0)trial.residency=setting+1;
          if(dimension==1)trial.cluster=setting+1;
          if(dimension==2)trial.group_wait=options.realize_aggregate?false:bool(setting);
          if(dimension==3)trial.embed_la=bool(setting);
          if(dimension==4) {
            trial.expert_fuse=!options.retile_only && bool(setting);
            if(ampere && trial.expert_fuse && !current.expert_fuse) {
              trial.SetCommonExpertSlots(std::min(trial.gemms[6].slots,trial.gemms[7].slots));
              trial.SetCommonExpertRegisterStages(
                  std::min(trial.gemms[6].register_stages,trial.gemms[7].register_stages));
            }
          }
          if(dimension==5)trial.argmax_stage=bool(setting);
          if(dimension==6)trial.combine_stage=bool(setting);
          if(dimension==7) {
            trial.route=static_cast<Sm90DenseCoordinates::Route>(setting);
            if(!trial.expert_fuse && !trial.GlobalRoute())continue;
          }
          if(trial.cluster>1 && (space.target.arch_tag=="sm_80" ||
              trial.cluster>cost.Constants().ClusterStar(trial.residency)))continue;
          changed|=consider(trial,"structure:"+std::to_string(dimension));
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
  // A calibrated SM80 incumbent can have the same structural class as
  // several newly tried slot depths. Keep an exact body comparison when
  // measured prices exist for a legal neighbour at the same geometry.
  if(!options.require_anchors && ampere && options.finalists>0 && !seeds.empty() && !options.pinned_theta && !options.operator_sum &&
      !options.retile_only && !options.realize_aggregate) {
    auto key=CoordinateKey(seeds.front());
    if(memo.count(key) && std::isfinite(memo.at(key))) {
      protected_anchors.insert(key);
      if(options.finalists>1) {
        auto const& incumbent=seeds.front();
        std::set<std::string> neighbours;
        for(auto const& [trial,label]:body_neighbours(incumbent)) {
          bool measured=true;
          if(!trial.moe)for(unsigned f:{0u,1u,2u,3u}) {
            auto const& g=trial.gemms[f];
            bool rows=trial.live_epilogue_rows!=incumbent.live_epilogue_rows && (f==2 || g.warps_k>1);
            bool gate=f==2 && (trial.swiglu_u!=incumbent.swiglu_u || trial.copy_issue_group!=incumbent.copy_issue_group);
            int copy=trial.residual_copy_issue?trial.copy_issue_group:0;
            int original=incumbent.residual_copy_issue?incumbent.copy_issue_group:0;
            bool residual=Sm90DenseCoordinates::ResidualFamily(f) && copy!=original;
            if(rows || gate || residual)measured &= cost.DenseService(work,f,g,trial.residency,&trial).has_value();
          }
          if(measured)neighbours.insert(CoordinateKey(trial));
        }
        auto body=std::find_if(ranked.begin(),ranked.end(),[&](auto const& candidate) {
          return neighbours.count(CoordinateKey(candidate.coordinates));
        });
        std::set<std::string> ring_keys;
        for(auto const& trial:ring_peers)ring_keys.insert(CoordinateKey(trial));
        auto ring=std::find_if(ranked.begin(),ranked.end(),[&](auto const& candidate) {
          return ring_keys.count(CoordinateKey(candidate.coordinates));
        });
        if(options.finalists==2) {
          // A two-arm budget leaves one comparison. Use the same PoolScore
          // ordering for the calibrated body and ring alternatives.
          auto best=std::min(body,ring);
          if(best!=ranked.end())protected_anchors.insert(CoordinateKey(best->coordinates));
        }else {
          if(body!=ranked.end())protected_anchors.insert(CoordinateKey(body->coordinates));
          if(ring!=ranked.end())protected_anchors.insert(CoordinateKey(ring->coordinates));
        }
      }
    }
  }
  for(auto const& key:protected_anchors) {
    auto same=[&](auto const& x){return CoordinateKey(x.coordinates)==key;};
    if(std::none_of(result.shortlist.begin(),result.shortlist.end(),same))
      result.shortlist.push_back(*std::find_if(ranked.begin(),ranked.end(),same));
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
  for(auto const& c:by_replay)if(protected_anchors.count(CoordinateKey(c.coordinates))) {
    ++taken[c.structural_class];
    result.finalists.push_back(c);
  }
  for (auto const& candidate : by_replay) {
    if (int(result.finalists.size()) >= options.finalists) break;
    if(protected_anchors.count(CoordinateKey(candidate.coordinates)))continue;
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
  std::stable_sort(result.finalists.begin(),result.finalists.end(),
                   [](auto const& a,auto const& b){return a.replay_ns<b.replay_ns;});
  result.seconds = spent_ms() / 1000.0;
  return result;
}
}  // namespace tilemega::solver
