// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <algorithm>
#include <map>
#include <tuple>

namespace tilemega::commands::native_sweep {
using namespace solver;
struct Variant {
  std::string key, reason;
  Sm90DenseCoordinates coordinates;
  double stage_ns=0, step_ns=0, bytes=0;
  bool legal=false, reference=false, selected=false;
};

inline std::string Key(Sm90DenseCoordinates const& c,std::string const& family) {
  if(family=="attention")return std::to_string(c.attention.chunks)+":"+std::to_string(c.attention.block_keys);
  if(family=="experts")return std::to_string(c.gemms[6].tile_n)+":"+std::to_string(c.gemms[7].tile_n);
  for(unsigned f:c.Active())if(family==kSm90DenseFamilies[f])return std::to_string(c.gemms[f].tile_n);
  throw std::invalid_argument("unknown sweep family: "+family);
}

inline std::string Reason(Sm90DenseSearchSpace const& space,Sm90DenseCoordinates const& c) {
  if(space.Legal(c))return "";
  if(c.group_wait && ((space.queries+2)*space.head_dim)%c.gemms[0].tile_n)return "D8_qkv_group";
  if(c.group_wait && space.qk_norm_row)return "whole_row_norm_group_wait";
  try {c.Validate(space.shapes,space.batch);}
  catch(std::invalid_argument const& e) {return std::string("coordinate_validation: ")+e.what();}
  try {c.ValidateSchedule(space.shapes,space.batch,space.head_dim,space.queries,1088,
      Sm90DenseSharedBudget(space.target,c.residency));}
  catch(std::invalid_argument const& e) {return std::string("schedule_validation: ")+e.what();}
  return "other_legal_rule";
}

inline double Bytes(Sm90DenseWorkload const& work,Sm90DenseCoordinates const& c,unsigned f) {
  auto const& g=c.gemms[f];double n=work.shapes[f][0],k=work.shapes[f][1];
  double active=f>=6?work.experts*(1-std::pow(1-double(work.top_k)/work.experts,work.batch)):1;
  double rows=f==6?work.batch*double(work.top_k)/active:f==7?(work.batch<=8?8:16):work.batch;
  double w=2*n*k*active,a=(n/g.tile_n)*active*rows*k*2;
  return w+a/(g.multicast?2:1);
}

inline std::vector<Variant> Sweep(Sm90DenseSearchSpace const& space,Sm90DenseWorkload const& work,
    Sm90DenseCostModel const& cost,Sm90DenseCoordinates const& w,std::string const& family,int top,bool traffic=false) {
  if(top<1 || !space.Legal(w))throw std::invalid_argument("sweep needs a legal W and positive top");
  if(family=="router" || (work.moe && (family=="expert_gate_up" || family=="expert_down")))
    throw std::invalid_argument("sweep the fused experts together; router has one decomposition");
  auto reference_key=Key(w,family);
  std::vector<Sm90DenseCoordinates> raw;
  unsigned f=8;
  if(family=="attention") {
    for(auto a:space.AttentionCandidates(w.residency)) {
      auto c=w;c.attention=a;raw.push_back(c);
      if(w.cluster>1 && a.chunks>1 && w.cluster%a.chunks==0) {
        c.attention.dsmem=true;
        try {cost.Attention(work,c.attention,w.residency);raw.push_back(c);}
        catch(Sm90NativeMissing const&) {}
      }
    }
  }else if(family=="experts") {
    if(!w.moe || !w.expert_fuse)throw std::invalid_argument("experts needs a fused MoE W");
    for(auto gu:space.FamilyCandidates(6,w.residency))for(auto dn:space.FamilyCandidates(7,w.residency)) {
      auto c=w;c.gemms[6]=gu;c.gemms[7]=dn;
      Sm90DenseSearchSpace::Structure(c);
      for(int slots=8;slots>=2;--slots) {
        c.gemms[6].slots=c.gemms[7].slots=slots;
        if(space.Legal(c))break;
      }
      raw.push_back(c);
    }
  }else {
    for(unsigned g:w.Active())if(family==kSm90DenseFamilies[g])f=g;
    if(f==8)throw std::invalid_argument("inactive family");
    for(auto g:space.FamilyCandidates(f,w.residency)) {
      auto c=w;c.gemms[f]=g;raw.push_back(c);
      if(w.cluster>1 && f<=4 && double(work.batch)/g.tile_n>cost.Constants().MulticastKappa() && g.slots%2==0) {
        c.gemms[f].multicast=true;raw.push_back(c);
      }
    }
  }
  auto evaluate=[&](Sm90DenseCoordinates c,bool reference) {
    // W's structural coordinates are preserved; Structure only derives U and new_row_work.
    if(!reference)Sm90DenseSearchSpace::Structure(c);
    Variant v;v.coordinates=c;v.key=Key(c,family);v.reference=reference;
    v.reason=Reason(space,c);v.legal=v.reason.empty();
    if(family=="attention") {
      v.stage_ns=cost.Attention(work,c.attention,c.residency);
      v.bytes=4.0*work.batch*work.hkv*575*work.head_dim;
    }else if(family=="experts") {
      v.stage_ns=cost.Gemm(work,6,c.gemms[6],c.residency)+cost.Gemm(work,7,c.gemms[7],c.residency)
          -cost.Constants().Barrier(space.Grid(c.residency))-cost.Constants().dram_latency;
      v.bytes=Bytes(work,c,6)+Bytes(work,c,7);
    }else {
      v.stage_ns=cost.Gemm(work,f,c.gemms[f],c.residency);v.bytes=Bytes(work,c,f);
    }
    if(v.legal)v.step_ns=cost.Predict(work,c).step_ns;
    return v;
  };
  // Stable ties follow SearchSm90Dense::rank: ns, shared bytes, then deeper tile.
  auto less=[&](Variant const& a,Variant const& b) {
    if(traffic && a.bytes!=b.bytes)return a.bytes<b.bytes;
    if(a.stage_ns!=b.stage_ns)return a.stage_ns<b.stage_ns;
    if(family=="attention")return false;
    auto const& x=a.coordinates;auto const& y=b.coordinates;
    if(family=="experts") {
      auto sx=x.FusedExpertBytes(work.batch,work.experts),sy=y.FusedExpertBytes(work.batch,work.experts);
      if(sx!=sy)return sx<sy;
      return std::make_pair(x.gemms[6].TileK(),x.gemms[7].TileK())>
          std::make_pair(y.gemms[6].TileK(),y.gemms[7].TileK());
    }
    auto const& gx=x.gemms[f];auto const& gy=y.gemms[f];
    auto sx=gx.SharedBytes(work.batch,Sm90DenseCoordinates::ResidualFamily(f));
    auto sy=gy.SharedBytes(work.batch,Sm90DenseCoordinates::ResidualFamily(f));
    return sx!=sy?sx<sy:gx.TileK()>gy.TileK();
  };
  std::map<std::string,Variant> groups;
  std::vector<std::string> group_order;
  for(auto const& c:raw) {
    auto v=evaluate(c,false);
    if(!traffic && v.key==reference_key)continue;
    auto it=groups.find(v.key);
    if(it==groups.end())group_order.push_back(v.key);
    if(it==groups.end() || less(v,it->second))groups.insert_or_assign(v.key,std::move(v));
  }
  std::vector<Variant> result;
  if(!traffic) {result.push_back(evaluate(w,true));result[0].selected=true;}
  std::vector<Variant> ranked;
  for(auto const& key:group_order)ranked.push_back(groups.at(key));
  std::stable_sort(ranked.begin(),ranked.end(),less);
  int count=int(result.size());
  for(auto& v:ranked)if(v.legal && count<top) {v.selected=true;++count;}
  result.insert(result.end(),ranked.begin(),ranked.end());return result;
}
inline Sm90DenseCoordinates Replace(Sm90DenseCoordinates c,Sm90DenseCoordinates const& v,
    std::string const& family) {
  if(family=="attention")c.attention=v.attention;
  else if(family=="experts") {c.gemms[6]=v.gemms[6];c.gemms[7]=v.gemms[7];}
  else for(unsigned f:c.Active())if(family==kSm90DenseFamilies[f])c.gemms[f]=v.gemms[f];
  Sm90DenseSearchSpace::Structure(c);return c;
}

inline Sm90DenseCoordinates Control(Sm90DenseSearchSpace const& space,Sm90DenseWorkload const& work,
    Sm90DenseCostModel const& cost,Sm90DenseCoordinates const& w,std::string const& mode,
    json::Array& choices) {
  auto c=w;
  if(mode=="N") {
    if(w.moe)throw std::invalid_argument("N is dense only");
    std::vector<Sm90DenseAttention> attention;
    for(auto a:space.AttentionCandidates(w.residency))if(a.chunks==1)attention.push_back(a);
    std::stable_sort(attention.begin(),attention.end(),[&](auto const& a,auto const& b) {
      return cost.Attention(work,a,w.residency)<cost.Attention(work,b,w.residency);
    });
    for(auto const& a:attention) {
      c=w;c.attention=a;c.embed_la=false;Sm90DenseSearchSpace::Structure(c);
      if(space.Legal(c))return c;
    }
    throw std::invalid_argument("no legal N control");
  }
  auto families=w.moe?std::vector<std::string>{"qkv","o","lm_head","attention","experts"}:
      std::vector<std::string>{"qkv","o","gate_up","down","lm_head","attention"};
  for(auto const& family:families) {
    auto candidates=Sweep(space,work,cost,w,family,1,true);
    candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[](auto const& v) {return !v.legal;}),
        candidates.end());
    auto enough=[&](Variant const& v) {
      if(family=="attention")return work.batch*work.hkv*v.coordinates.attention.chunks>=space.Grid(w.residency);
      if(family=="experts")return work.shapes[6][0]/v.coordinates.gemms[6].tile_n*space.ExpectedActive()>=space.Grid(w.residency)
          && work.shapes[7][0]/v.coordinates.gemms[7].tile_n*space.ExpectedActive()>=space.Grid(w.residency);
      for(unsigned f:w.Active())if(family==kSm90DenseFamilies[f])
        return work.shapes[f][0]/v.coordinates.gemms[f].tile_n>=space.Grid(w.residency);
      return false;
    };
    bool has_enough=std::any_of(candidates.begin(),candidates.end(),enough);
    if(family!="attention" || has_enough)
      candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](auto const& v) {return !enough(v);}),candidates.end());
    if(family=="attention")std::stable_sort(candidates.begin(),candidates.end(),[&](auto const& a,auto const& b) {
      auto x=a.coordinates.attention.chunks,y=b.coordinates.attention.chunks;
      return x!=y?(has_enough?x<y:x>y):a.stage_ns<b.stage_ns;
    });
    bool found=false;int skipped=0;
    for(auto const& v:candidates) {
      auto next=Replace(c,v.coordinates,family);
      if(!space.Legal(next)) {++skipped;continue;}
      c=next;choices.emplace_back(json::Object{{"family",family},{"key",v.key},{"conflicts_skipped",skipped}});
      found=true;break;
    }
    if(!found)throw std::invalid_argument("no legal T candidate for "+family);
  }
  return c;
}
}  // namespace tilemega::commands::native_sweep
