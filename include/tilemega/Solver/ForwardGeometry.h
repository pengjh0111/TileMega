// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/ServingSeedGeometry.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Backend/Sm90ForwardResources.h>
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <cmath>
#include <limits>

namespace tilemega::solver {
// The complete indexed table prevents a model revision from silently inheriting
// a partial tuning table for different stages.
inline void ApplyForwardGeometry(frontend::ModelPlan const& plan,
    std::vector<frontend::GemmGranularity>& geometry,json::Value const& value) {
  auto const& rows=value.At("gemms").AsArray("forward geometry GEMMs");
  if(rows.size()!=plan.gemms.size() || geometry.size()!=rows.size())
    throw std::invalid_argument("forward geometry must cover every GEMM");
  for(auto const& [key,item]:value.AsObject("forward geometry"))
    if(key!="gemms" && key!="comment")
      throw std::invalid_argument("unknown forward geometry field: "+key);
  auto integer=[](json::Value const& v,char const* key) {
    double n=v.At(key).AsNumber(key);
    if(!std::isfinite(n) || n<0 || n>std::numeric_limits<int>::max() || std::floor(n)!=n)
      throw std::invalid_argument(std::string("invalid forward geometry integer: ")+key);
    return int(n);
  };
  auto proposed=geometry;std::vector<bool> seen(rows.size(),false);
  for(auto const& row:rows) {
    int index=integer(row,"index");
    if(index>=int(rows.size()) || seen[index])
      throw std::invalid_argument("duplicate or out-of-range forward GEMM index");
    seen[index]=true;
    for(auto const& [key,item]:row.AsObject("forward GEMM geometry"))
      if(key!="index" && key!="tile_m" && key!="tile_n" && key!="tile_k" &&
         key!="stages" && key!="split_k" && key!="impl")
        throw std::invalid_argument("unknown forward GEMM geometry field: "+key);
    frontend::GemmGranularity tile{integer(row,"tile_m"),integer(row,"tile_n"),
        integer(row,"tile_k"),integer(row,"stages"),integer(row,"split_k"),
        SeedGemmImpl(row.At("impl").AsString("impl"))};
    bool legal=tile.impl==0?(tile.stages<=8 &&
        DmServingBF16ShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages)):
        tile.impl==2 && ServingWgmmaShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages);
    auto const& g=plan.gemms[index];
    if(!legal || tile.split_k<1 || (tile.split_k>1 &&
        (std::uint64_t(tile.tile_k)*tile.split_k>g.k || g.k%(std::uint64_t(tile.tile_k)*tile.split_k))))
      throw std::invalid_argument("illegal forward tile or split at GEMM "+std::to_string(index));
    if(g.access.b==codegen::DmBAccess::kExpertIndirect)
      throw std::invalid_argument("forward geometry table does not admit expert-indirect tasks");
    proposed[index]=tile;
  }
  geometry.swap(proposed);
}
// kSharedBytes includes the 1KB header. Arena bytes name one payload half.
inline int Sm90ForwardArenaBytes(TargetSpec const& target,int residency) {
  int budget=Sm90DenseSharedBudget(target,residency);
  int arena=(budget-1024)/2/1024*1024;
  if(arena<1024)throw std::invalid_argument("target leaves no forward arena");
  return arena;
}
struct Sm90ForwardResourceUsage {
  int shared_bytes=0,slot_bytes=0,store_columns=0,arena_bytes=0,budget_bytes=0;
  long long task_count=0;
  bool exclusive=false,feasible=false;
  std::string reason;
};
inline Sm90ForwardResourceUsage QuerySm90ForwardResources(Sm90ForwardShape const& shape,
    Sm90ForwardStage const& c,TargetSpec const& target,int residency) {
  using namespace tilemega::backend;
  c.Validate();
  auto const& g=shape.dnn1_geometry;
  auto integer=[&](char const* key) {
    double value=g.At(key).AsNumber(key);
    if(!std::isfinite(value) || value<0 || value>std::numeric_limits<int>::max() || std::floor(value)!=value)
      throw std::invalid_argument(std::string("invalid forward resource geometry: ")+key);
    return int(value);
  };
  auto positive=[&](char const* key) {
    int value=integer(key);
    if(!value)throw std::invalid_argument(std::string("missing positive forward resource geometry: ")+key);
    return value;
  };
  auto ceildiv=[](long long x,long long y) {return (x+y-1)/y;};
  Sm90ForwardResourceUsage r;
  r.budget_bytes=Sm90DenseSharedBudget(target,residency);
  r.arena_bytes=Sm90ForwardArenaBytes(target,residency);
  bool needs_stats=g.At("needs_stats").AsBool("needs_stats");
  switch(c.body) {
    case Sm90ForwardBody::kGemm:
    case Sm90ForwardBody::kConv: {
      r.slot_bytes=Sm90ForwardGemmSlotBytes(c.tile_m,c.tile_n,64);
      r.shared_bytes=Sm90ForwardGemmLayout(c.tile_m,c.tile_n,64,c.slots,
          Sm90ForwardChainScratchBytes(c.tile_m,needs_stats)).end;
      r.task_count=ceildiv(shape.m,c.tile_m)*ceildiv(shape.n,c.tile_n)*c.split_k;
      r.store_columns=std::min(c.tile_n/(g.At("gated").AsBool("gated")?2:1),positive("store_columns"));
      break;
    }
    case Sm90ForwardBody::kDepthwise: {
      if(c.slots>4)throw std::invalid_argument("depthwise supports at most four ring slots");
      int pack=positive("depthwise_pack"),rows=c.tile_w*c.tile_h;
      r.slot_bytes=Sm90ForwardDepthwiseSlotBytes(c.tile_w,c.tile_h,c.channel_block,shape.stride,pack);
      r.shared_bytes=Sm90ForwardDepthwiseLayout(c.tile_w,c.tile_h,c.channel_block,shape.stride,
          c.slots,pack,Sm90ForwardChainScratchBytes(rows,needs_stats)).end;
      r.task_count=1LL*positive("batch")*ceildiv(positive("output_width"),c.tile_w)*
          ceildiv(positive("output_height"),c.tile_h)*ceildiv(shape.channels,c.channel_block);
      r.store_columns=std::min(c.channel_block/(g.At("gated").AsBool("gated")?2:1),positive("store_columns"));
      break;
    }
    case Sm90ForwardBody::kDwPw:
    case Sm90ForwardBody::kInvertedResidual: {
      if(c.tile_w*c.tile_h>64)throw std::invalid_argument("fused forward uses a 64-row MMA tile");
      auto f=Sm90ForwardFusedLayout(c.tile_w,c.tile_h,c.channel_block,c.tile_n,shape.stride,
          c.slots,positive("depthwise_pack"),Sm90ForwardChainScratchBytes(64,needs_stats));
      r.slot_bytes=f.slot_bytes;r.shared_bytes=f.layout.end;
      r.task_count=1LL*positive("batch")*ceildiv(positive("output_width"),c.tile_w)*
          ceildiv(positive("output_height"),c.tile_h)*ceildiv(shape.out_channels,c.tile_n);
      r.store_columns=std::min(c.tile_n/(g.At("gated").AsBool("gated")?2:1),positive("store_columns"));
      break;
    }
    case Sm90ForwardBody::kEncoderAttention: {
      if(c.slots>2)throw std::invalid_argument("encoder attention supports at most two slots");
      int seq=positive("seq");
      if(positive("head_dim")!=64 || seq%64 || seq%c.tile_m)
        throw std::invalid_argument("attention resource shape requires D64 and whole query blocks");
      r.slot_bytes=Sm90ForwardAttentionSlotBytes(seq,c.tile_m);
      r.shared_bytes=Sm90ForwardAttentionLayout(seq,c.tile_m,c.slots).end;
      r.task_count=1LL*positive("batch")*positive("heads")*(seq/c.tile_m);
      r.store_columns=64;
      break;
    }
    case Sm90ForwardBody::kGemv:
      if(shape.k<=0 || shape.k%64)throw std::invalid_argument("GEMV stages whole K-major panels");
      r.slot_bytes=2*c.tile_n*64;
      r.shared_bytes=Sm90ForwardGemvLayout(int(shape.k),c.tile_n).end;
      r.task_count=shape.m*ceildiv(shape.n,c.tile_n);
      r.store_columns=std::min<long long>(c.tile_n,shape.n);
      break;
    case Sm90ForwardBody::kLayout:
      r.shared_bytes=1024;
      r.task_count=1LL*positive("batch")*ceildiv(1LL*positive("width")*positive("height"),128);
      break;
    case Sm90ForwardBody::kPool:
      r.shared_bytes=1024;
      r.task_count=1LL*positive("batch")*positive("output_width")*positive("output_height");
      break;
    case Sm90ForwardBody::kGlobalPool:
      r.shared_bytes=Sm90ForwardGlobalPoolLayout().end;
      r.task_count=1LL*positive("batch")*ceildiv(positive("channels"),128);
      break;
    case Sm90ForwardBody::kEmbedding:
    case Sm90ForwardBody::kLayerNorm:
      r.shared_bytes=1024;r.task_count=ceildiv(positive("rows"),4);
      break;
  }
  // Matches generated Placement: parity arena, or offset zero for an exclusive body.
  r.exclusive=r.shared_bytes>1024+r.arena_bytes;
  r.feasible=r.shared_bytes>=1024 && r.shared_bytes<=r.budget_bytes && r.task_count>0;
  r.reason=r.feasible?"":"body shared bytes exceed the residency budget or has no tasks";
  return r;
}
} // namespace tilemega::solver
