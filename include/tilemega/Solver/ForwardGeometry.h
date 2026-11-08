// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/ServingSeedGeometry.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Backend/Sm90ForwardResources.h>
#include <tilemega/Backend/Sm90F3Ring.cuh>
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
// One chain class identifies both register feasibility and measured cost.
enum class Sm90F3ChainClass : unsigned {
  kPlain, kBiasAct, kBiasResAct, kDeferredLn, kDeferredLnGelu,
  kResidualLnRowStats, kSwapAbBiasAct
};
inline constexpr char const* kSm90F3ChainClasses[]={
    "plain","bias_act","bias_res_act","deferred_ln","deferred_ln_gelu",
    "residual_ln_rowstats","swapab_bias_act"};

// DT-SPILL: the fully inline production sweep on nvcc 12.8.93 rejects
// 256x128 with residual or normalization chains (180/12/504/316 B stores).
// Receipt: runs/dnn3/spill_inline_predicate_v2/spill.json. Keep the prior
// full-model 128x256 normalization exclusions until that artifact passes.
// Resident-probe loop spills are diagnosed separately and do not ban tiles.
inline bool Sm90F3TileFeasible(int tile_m,int tile_n,Sm90F3ChainClass chain) {
  int wg_cols=tile_m==64?tile_n/2:tile_n;
  int sub_tiles=tile_m==256?2:1;
  int accumulator=sub_tiles*64*wg_cols/128;
  if(accumulator>128)return false;
  if(accumulator<128)return true;
  using C=Sm90F3ChainClass;
  if(tile_m==128 && tile_n==256)
    return chain==C::kPlain || chain==C::kBiasAct;
  if(tile_m==256 && tile_n==128)
    return chain==C::kPlain || chain==C::kBiasAct;
  return true;
}
// The class of a chain, from what its steps need. Codegen and the solver call
// this on the same plan chain, so neither can name a different class.
inline Sm90F3ChainClass Sm90F3ClassOf(bool residual,bool deferred_norm,
    bool residual_norm,bool row_stats,bool activation,bool swap_ab) {
  using C=Sm90F3ChainClass;
  if(swap_ab)return C::kSwapAbBiasAct;
  if(residual_norm || (row_stats && residual))return C::kResidualLnRowStats;
  if(deferred_norm)return activation?C::kDeferredLnGelu:C::kDeferredLn;
  if(residual)return C::kBiasResAct;
  return activation?C::kBiasAct:C::kPlain;
}
// F3 fixes the GAP producer's partial-sum granularity at 64 rows (DNN3 prompt
// §4 F-7): one slot per (image, 64-row block). A dense M tile may straddle two
// images, so a block's leading overlap gets its own slot. The plan and the
// body both call this, so neither can compute a different count.
inline constexpr int kSm90ForwardF3PoolRows=64;
inline int Sm90ForwardF3PoolParts(long long rows_per_image) {
  if(rows_per_image<=0)throw std::invalid_argument("a pooled image has pixels");
  return int((rows_per_image+kSm90ForwardF3PoolRows-1)/kSm90ForwardF3PoolRows+1);
}
// Each producer task owns one partial per image/channel. A dense M tile may
// straddle images, so reserve the maximum count including its leading overlap.
inline int Sm90ForwardPoolParts(Sm90ForwardShape const& producer,
    Sm90ForwardStage const& coordinate,
    Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy) {
  if(producer.body!=coordinate.body)
    throw std::invalid_argument("pool producer body disagrees with coordinates");
  coordinate.Validate(abi);
  auto positive=[&](char const* key) -> long long {
    double value=producer.dnn1_geometry.At(key).AsNumber(key);
    if(!std::isfinite(value) || value<=0 || value>std::numeric_limits<int>::max() || std::floor(value)!=value)
      throw std::invalid_argument(std::string("invalid pool producer geometry: ")+key);
    return static_cast<long long>(value);
  };
  long long parts=0;
  if(coordinate.body==Sm90ForwardBody::kGemm || coordinate.body==Sm90ForwardBody::kConv) {
    auto rows=positive("rows"),batch=positive("batch");
    if(rows%batch)throw std::invalid_argument("pool producer rows do not form complete images");
    rows/=batch;
    parts=abi==Sm90ForwardAbi::kF3?Sm90ForwardF3PoolParts(rows):
        (rows+2LL*coordinate.tile_m-2)/coordinate.tile_m;
  }else if(coordinate.body==Sm90ForwardBody::kDepthwise || coordinate.body==Sm90ForwardBody::kDwPw ||
      coordinate.body==Sm90ForwardBody::kInvertedResidual) {
    auto width=positive("output_width"),height=positive("output_height");
    parts=((width+coordinate.tile_w-1)/coordinate.tile_w)*
        ((height+coordinate.tile_h-1)/coordinate.tile_h);
  }else throw std::invalid_argument("global pool has no supported tile producer");
  if(parts<=0 || parts>std::numeric_limits<int>::max())
    throw std::invalid_argument("pool partial count exceeds the native range");
  return int(parts);
}
struct Sm90ForwardResourceUsage {
  int shared_bytes=0,slot_bytes=0,store_columns=0,arena_bytes=0,budget_bytes=0;
  long long task_count=0;
  bool exclusive=false,feasible=false;
  std::string reason;
};
inline Sm90ForwardResourceUsage QuerySm90ForwardF3Resources(
    Sm90ForwardShape const& shape,Sm90ForwardStage const& c,
    TargetSpec const& target,int residency) {
  using namespace tilemega::backend;
  c.Validate(Sm90ForwardAbi::kF3);
  Sm90ForwardResourceUsage r;
  r.budget_bytes=Sm90F3Budget::kBudget;r.arena_bytes=Sm90F3Budget::kArena;
  if(residency!=1 || target.res.max_dynamic_smem_per_cta<r.budget_bytes) {
    r.reason="F3 requires one CTA and its 232448-byte shared budget";return r;
  }
  auto const& g=shape.dnn1_geometry;
  auto integer=[&](char const* key,bool zero=false) {
    double value=g.At(key).AsNumber(key);
    if(!std::isfinite(value) || value<(zero?0:1) ||
        value>std::numeric_limits<int>::max() || std::floor(value)!=value)
      throw std::invalid_argument(std::string("invalid F3 resource geometry: ")+key);
    return int(value);
  };
  auto ceildiv=[](long long x,long long y){return (x+y-1)/y;};
  bool matrix=c.body==Sm90ForwardBody::kGemm || c.body==Sm90ForwardBody::kConv;
  int payload=1024;
  if(matrix) {
    if(shape.m<=0 || shape.n<=0 || shape.k<=0 || shape.k%(1LL*c.tk*c.split_k)) {
      r.reason="F3 matrix extents or whole-slot split are invalid";return r;
    }
    bool swap=c.mode==Sm90ForwardMode::kSwapAB;
    if(c.store!=Sm90ForwardStore::kReg || c.mc) {
      r.reason="F3 store and multicast mechanisms have not passed their gates";return r;
    }
    if(!swap && c.tile_m==256 && shape.m<512) {
      r.reason="F3 TM256 requires at least 512 matrix rows";return r;
    }
    if(c.mode!=shape.mode || (swap && shape.m>16)) {
      r.reason="F3 operand mode differs from plan rows";return r;
    }
    int vectors=integer("f3_vectors",true),rows=shape.m<=8?8:16;
    bool residual=g.At("f3_residual").AsBool("F3 residual");
    bool stats=g.At("needs_stats").AsBool("F3 input statistics");
    if(g.At("has_row_stats").AsBool("F3 output statistics") && c.tile_n%64) {
      r.reason="F3 statistics producers require 64-column blocks";return r;
    }
    int stat_parts=g.Find("f3_stat_parts")?integer("f3_stat_parts"):1;
    int pool_images=0;
    if(g.At("has_channel_pool").AsBool("F3 GAP side output")) {
      int area=integer("pool_rows");
      pool_images=area>=64?2:(64+area-1)/area+1;
    }
    int columns=c.tile_m==64?c.tile_n/2:c.tile_n;
    auto layout=Sm90F3MakeGemmLayout(swap?rows:c.tile_m,swap?128:c.tile_n,c.slots,
        swap,rows,residual,vectors,stats,swap?0:2*columns,pool_images,c.tk,stat_parts);
    payload=layout.end;r.slot_bytes=layout.slot_bytes;
    r.store_columns=int(std::min<long long>(shape.n,swap?128:c.tile_n));
    r.task_count=(swap?1:ceildiv(shape.m,c.tile_m))*
        ceildiv(shape.n,swap?128:c.tile_n)*c.split_k;
    r.exclusive=layout.Spanning();
  }else {
    long long items=0,per_cta=256;
    switch(c.body) {
      case Sm90ForwardBody::kEncoderAttention:
        payload=2*(std::min(128,16*integer("batch"))+256)*128;
        r.task_count=1LL*integer("batch")*integer("heads")*
            (128/std::min(128,16*integer("batch")));break;
      case Sm90ForwardBody::kLayout:
        items=1LL*integer("batch")*ceildiv(1LL*integer("width")*integer("height"),8);break;
      case Sm90ForwardBody::kPool:
        items=1LL*integer("batch")*integer("output_width")*integer("output_height")*
            ceildiv(integer("channels"),8);break;
      case Sm90ForwardBody::kGlobalPool:
        items=1LL*integer("batch")*integer("channels");break;
      case Sm90ForwardBody::kLayerNorm:case Sm90ForwardBody::kEmbedding:
        items=integer("rows");per_cta=8;break;
      default:r.reason="unsupported F3 body";return r;
    }
    if(items)r.task_count=ceildiv(items,per_cta);
  }
  r.shared_bytes=Sm90F3Budget::kHeader+payload;
  r.feasible=Sm90F3Budget::Fits(payload) && r.task_count>0;
  r.reason=r.feasible?"F3 production layout fits":"F3 payload exceeds shared budget";
  return r;
}
inline Sm90ForwardResourceUsage QuerySm90ForwardResources(Sm90ForwardShape const& shape,
    Sm90ForwardStage const& c,TargetSpec const& target,int residency,
    Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy) {
  if(abi==Sm90ForwardAbi::kF3)return QuerySm90ForwardF3Resources(shape,c,target,residency);
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
      r.slot_bytes=Sm90ForwardGemmSlotBytes(c.tile_m,c.tile_n,64,c.body==Sm90ForwardBody::kConv);
      r.shared_bytes=Sm90ForwardGemmLayout(c.tile_m,c.tile_n,64,c.slots,
          Sm90ForwardChainScratchBytes(c.tile_m,needs_stats),c.body==Sm90ForwardBody::kConv).end;
      r.task_count=ceildiv(shape.m,c.tile_m)*ceildiv(shape.n,c.tile_n)*c.split_k;
      r.store_columns=std::min(c.tile_n/(g.At("gated").AsBool("gated")?2:1),positive("store_columns"));
      break;
    }
    case Sm90ForwardBody::kDepthwise: {
      if(c.slots>8)throw std::invalid_argument("depthwise supports at most eight ring slots");
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
      r.task_count=ceildiv(1LL*positive("batch")*positive("output_width")*
          positive("output_height")*shape.channels,128);
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
