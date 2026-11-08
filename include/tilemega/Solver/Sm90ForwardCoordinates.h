// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Support/Json.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {

// Native DNN forward coordinates (prompt §10 N-1 item 8). One record per
// stage of the forward program: the GEMM/conv stages carry TN and split-K,
// the depthwise and fused stages their spatial tile, and the fused stages
// the structural choice of fusing at all.
enum class Sm90ForwardBody : unsigned {
  kGemm, kConv, kDepthwise, kDwPw, kInvertedResidual, kEncoderAttention,
  kLayout, kPool, kGlobalPool, kGemv, kLayerNorm, kEmbedding
};

inline constexpr char const* kSm90ForwardBodies[]={
    "gemm","conv","depthwise","dwpw","inverted_residual","encoder_attention",
    "layout","pool","global_pool","gemv","layer_norm","embedding"};

// Which forward skeleton a plan and its program use (DNN3 prompt §9.1). The
// legacy 160-thread bodies keep every earlier coordinate meaning; F3 is the
// 384-thread skeleton, whose GEMM stages also carry an operand mode.
enum class Sm90ForwardAbi : unsigned {kLegacy, kF3};
inline constexpr char const* kSm90ForwardAbis[]={"legacy","f3"};

// How an F3 GEMM stage reads its operands. The plan decides it, never search:
// a gather stem has a physically 8-channel input, and a swap-AB stage has at
// most sixteen rows, so the weight becomes the WGMMA's A operand.
enum class Sm90ForwardMode : unsigned {kNormal, kGather, kSwapAB};
inline constexpr char const* kSm90ForwardModes[]={"normal","gather","swapab"};
enum class Sm90ForwardStore : unsigned {kReg,kTma};
inline constexpr char const* kSm90ForwardStores[]={"reg","tma"};

struct Sm90ForwardStage {
  Sm90ForwardBody body=Sm90ForwardBody::kGemm;
  // F3 only: the operand mode the plan assigned to this stage.
  Sm90ForwardMode mode=Sm90ForwardMode::kNormal;
  // GEMM/conv: N tile and the K split; both 1 for the small bodies.
  int tile_m=64,tile_n=64,split_k=1,slots=2;
  int tk=64,mc=0;
  Sm90ForwardStore store=Sm90ForwardStore::kReg;
  // Depthwise and fused bodies: the spatial tile and its channel block.
  int tile_w=8,tile_h=8,channel_block=64;
  // Fused bodies only: whether the plan's fusion is taken (structural).
  bool fused=false;

  void Validate(Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy) const {
    bool matrix=body==Sm90ForwardBody::kGemm || body==Sm90ForwardBody::kConv;
    bool f3=abi==Sm90ForwardAbi::kF3;
    if(f3 && (tk!=64 && tk!=128))
      throw std::invalid_argument("F3 slot K width must be 64 or 128");
    if(mc<0 || mc>1 || unsigned(store)>unsigned(Sm90ForwardStore::kTma))
      throw std::invalid_argument("invalid F3 store or multicast coordinate");
    if(!f3 && (tk!=64 || mc || store!=Sm90ForwardStore::kReg))
      throw std::invalid_argument("F3 coordinates require the F3 skeleton");
    if(f3 && !matrix && (tk!=64 || mc || store!=Sm90ForwardStore::kReg))
      throw std::invalid_argument("only matrix stages carry F3 pipeline coordinates");
    if(f3 && store==Sm90ForwardStore::kTma &&
        (mode==Sm90ForwardMode::kSwapAB || tile_m<128 || tile_n%64))
      throw std::invalid_argument("F3 TMA stores require whole 64-column normal tiles");
    if(f3 && mc && mode!=Sm90ForwardMode::kNormal)
      throw std::invalid_argument("F3 multicast requires normal operand mode");
    if(f3 && mode!=Sm90ForwardMode::kNormal && !matrix)
      throw std::invalid_argument("only a forward GEMM stage carries an operand mode");
    if(!f3 && mode!=Sm90ForwardMode::kNormal)
      throw std::invalid_argument("the legacy forward skeleton has one operand mode");
    if(f3 && mode==Sm90ForwardMode::kSwapAB) {
      // The weight is the WGMMA A operand, so TN names its 128 weight rows.
      if(tile_n!=128)throw std::invalid_argument("a swap-AB stage keeps TN 128");
      if(split_k<1 || split_k>32)throw std::invalid_argument("swap-AB split-K is 1..32");
      if(slots<2 || slots>8)throw std::invalid_argument("forward ring slots must be 2..8");
      return;
    }
    if(matrix && f3 && tile_m!=64 && tile_m!=128 && tile_m!=256)
      throw std::invalid_argument("F3 forward TM must be 64, 128 or 256");
    if(matrix && !f3 && tile_m!=32 && tile_m!=64 && tile_m!=128)
      throw std::invalid_argument("forward TM must be 32, 64 or 128");
    if(matrix && !f3 && tile_m*tile_n>16384)
      throw std::invalid_argument("forward WGMMA accumulator exceeds Sm90Gmma.h capacity");
    // F3 splits one CTA tile over two consumer warpgroups, so the bound is on
    // each warpgroup's own accumulator rather than on the whole tile.
    if(matrix && f3 && tile_m*tile_n>32768)
      throw std::invalid_argument("an F3 warpgroup accumulator exceeds 128 registers");
    if(body==Sm90ForwardBody::kEncoderAttention &&
        tile_m!=16 && tile_m!=32 && tile_m!=64 && tile_m!=128)
      throw std::invalid_argument("forward attention query rows must be 16, 32, 64 or 128");
    if(body==Sm90ForwardBody::kGemv && tile_n!=64 && tile_n!=128 && tile_n!=256)
      throw std::invalid_argument("forward GEMV block must be 64, 128 or 256");
    bool gemm=body==Sm90ForwardBody::kGemm || body==Sm90ForwardBody::kConv ||
        body==Sm90ForwardBody::kDwPw || body==Sm90ForwardBody::kInvertedResidual;
    if(gemm && tile_n!=32 && tile_n!=64 && tile_n!=128 && tile_n!=256 &&
        !(f3 && (tile_n==96 || tile_n==192)))
      throw std::invalid_argument("forward TN is outside the skeleton's tile set");
    // F3 drops the power-of-two rule: split-K only has to divide kIter.
    if(f3) {
      if(split_k<1 || split_k>16)
        throw std::invalid_argument("F3 normal-mode split-K is 1..16");
    }else if(split_k<1 || split_k>16 || (split_k&(split_k-1)))
      throw std::invalid_argument("forward split-K must be a power of two up to 16");
    if(slots<2 || slots>8)throw std::invalid_argument("forward ring slots must be 2..8");
    bool spatial=body==Sm90ForwardBody::kDepthwise || body==Sm90ForwardBody::kDwPw ||
        body==Sm90ForwardBody::kInvertedResidual;
    if(spatial && !((tile_w==8 && tile_h==8) || (tile_w==8 && tile_h==16) ||
                    (tile_w==16 && tile_h==16)))
      throw std::invalid_argument("forward spatial tile must be 8x8, 8x16 or 16x16");
    if(spatial && (channel_block<=0 || channel_block%64))throw std::invalid_argument("forward channel block is a multiple of 64");
    if(fused && body!=Sm90ForwardBody::kDwPw && body!=Sm90ForwardBody::kInvertedResidual)
      throw std::invalid_argument("only the DWPW and inverted-residual bodies fuse");
  }
  json::Value ToJson(Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy) const {
    json::Value result=json::Object{{"body",kSm90ForwardBodies[unsigned(body)]},
        {"mode",kSm90ForwardModes[unsigned(mode)]},{"tile_m",tile_m},{"tile_n",tile_n},
        {"split_k",split_k},{"slots",slots},{"tile_w",tile_w},{"tile_h",tile_h},
        {"channel_block",channel_block},{"fused",fused}};
    if(abi==Sm90ForwardAbi::kF3) {
      result.Set("tk",tk);result.Set("store",kSm90ForwardStores[unsigned(store)]);
      result.Set("mc",mc);
    }
    return result;
  }
};

struct Sm90ForwardCoordinates {
  std::vector<Sm90ForwardStage> stages;
  int residency=1;
  // Schema 1 and 2 predate the F3 skeleton and are always read as legacy.
  Sm90ForwardAbi abi=Sm90ForwardAbi::kLegacy;
  json::Value ToJson() const {
    json::Array rows;
    for(auto const& stage:stages)rows.emplace_back(stage.ToJson(abi));
    return json::Object{{"schema",abi==Sm90ForwardAbi::kF3?4:3},{"model","dnn"},{"abi",kSm90ForwardAbis[unsigned(abi)]},
        {"residency",residency},{"stages",std::move(rows)}};
  }
  static Sm90ForwardCoordinates FromJson(json::Value const& value) {
    auto schema=value.At("schema").AsNumber("schema");
    if(schema!=1 && schema!=2 && schema!=3 && schema!=4)
      throw std::invalid_argument("unknown sm90 forward coordinate schema");
    Sm90ForwardCoordinates result;
    result.residency=int(value.At("residency").AsNumber("residency"));
    if(schema>=3) {
      auto abi=value.At("abi").AsString("abi");
      bool found=false;
      for(unsigned i=0;i<sizeof(kSm90ForwardAbis)/sizeof(*kSm90ForwardAbis);++i)
        if(abi==kSm90ForwardAbis[i]) {result.abi=Sm90ForwardAbi(i);found=true;break;}
      if(!found)throw std::invalid_argument("unknown sm90 forward abi: "+abi);
    }
    for(auto const& row:value.At("stages").AsArray("stages")) {
      Sm90ForwardStage stage;
      auto body=row.At("body").AsString("body");
      bool found=false;
      for(unsigned i=0;i<sizeof(kSm90ForwardBodies)/sizeof(*kSm90ForwardBodies);++i)
        if(body==kSm90ForwardBodies[i]) {stage.body=Sm90ForwardBody(i);found=true;break;}
      if(!found)throw std::invalid_argument("unknown sm90 forward body: "+body);
      if(schema>=3) {
        auto mode=row.At("mode").AsString("mode");
        found=false;
        for(unsigned i=0;i<sizeof(kSm90ForwardModes)/sizeof(*kSm90ForwardModes);++i)
          if(mode==kSm90ForwardModes[i]) {stage.mode=Sm90ForwardMode(i);found=true;break;}
        if(!found)throw std::invalid_argument("unknown sm90 forward mode: "+mode);
      }
      stage.tile_m=schema==1?64:int(row.At("tile_m").AsNumber("tile_m"));
      stage.tile_n=int(row.At("tile_n").AsNumber("tile_n"));
      stage.split_k=int(row.At("split_k").AsNumber("split_k"));
      stage.slots=int(row.At("slots").AsNumber("slots"));
      if(schema==4) {
        if(result.abi!=Sm90ForwardAbi::kF3)
          throw std::invalid_argument("schema 4 coordinates require F3");
        auto integer=[&](char const* key) {
          double n=row.At(key).AsNumber(key);
          if(!std::isfinite(n) || n<0 || n>128 || std::floor(n)!=n)
            throw std::invalid_argument(std::string("invalid F3 coordinate: ")+key);
          return int(n);
        };
        stage.tk=integer("tk");stage.mc=integer("mc");
        auto store=row.At("store").AsString("store");
        if(store=="reg")stage.store=Sm90ForwardStore::kReg;
        else if(store=="tma")stage.store=Sm90ForwardStore::kTma;
        else throw std::invalid_argument("unknown F3 store: "+store);
      }
      stage.tile_w=int(row.At("tile_w").AsNumber("tile_w"));
      stage.tile_h=int(row.At("tile_h").AsNumber("tile_h"));
      stage.channel_block=int(row.At("channel_block").AsNumber("channel_block"));
      stage.fused=row.At("fused").AsBool("fused");
      stage.Validate(result.abi);
      result.stages.push_back(stage);
    }
    if(result.residency!=1 && result.residency!=2)
      throw std::invalid_argument("sm90 forward residency must be 1 or 2");
    if(result.abi==Sm90ForwardAbi::kF3 && result.residency!=1)
      throw std::invalid_argument("the F3 skeleton runs one 384-thread CTA per SM");
    return result;
  }
};

// One stage's shape, taken from the DNN plan: the GEMM extents, the conv
// geometry and the bytes each task moves. The solver never re-derives these.
struct Sm90ForwardShape {
  Sm90ForwardBody body=Sm90ForwardBody::kGemm;
  Sm90ForwardMode mode=Sm90ForwardMode::kNormal;
  long long m=0,n=0,k=0;        // GEMM extents after padding
  long long pixels=0,channels=0,out_channels=0;
  int kernel=1,stride=1;
  long long weight_bytes=0,activation_bytes=0,output_bytes=0;
  bool fusable=false;           // the plan offers a fused form
  // A deferred or reconstructed LayerNorm needs the consumer's whole A row,
  // so its stage cannot split the contraction (prompt §10 N-1 item 6).
  bool whole_row=false;
  int epilogue_ops=0;
  // DNN1 exact plan/body mapping metadata; populated before search, never inferred from totals.
  json::Value dnn1_geometry;
};

}  // namespace tilemega::solver
