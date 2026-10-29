// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Support/Json.h>
#include <tilemega/Backend/Sm90DenseLayout.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {

enum class Sm90DenseMma { kMmaSync, kWgmma };
inline constexpr int kSm90DenseGemmImpl=3;
// Dense decode runs families 0-4; MoE decode (prompt §8) runs qkv, o, the
// router (5), the grouped expert gate_up (6) and down (7), and lm_head.
inline constexpr std::array<char const*,8> kSm90DenseFamilies={
    "qkv","o","gate_up","down","lm_head","router","expert_gate_up","expert_down"};
using Sm90DenseShapeTable=std::array<std::array<int,2>,8>;
inline std::vector<unsigned> Sm90DenseActiveFamilies(bool moe) {
  return moe?std::vector<unsigned>{0,1,5,6,7,4}:std::vector<unsigned>{0,1,2,3,4};
}

struct Sm90DenseGemm {
  int tile_n=0,warps_n=1,warps_k=4,slot_bytes=0,slots=3;
  Sm90DenseMma mma=Sm90DenseMma::kMmaSync;
  int register_stages=1;
  // D10c: the CTA pair of a cluster shares each A slot by TMA multicast.
  bool multicast=false;
  int TileK() const { return int(slot_bytes/(2ll*tile_n)); }
  std::int64_t SharedBytes(int batch,bool residual) const {
    return backend::Sm90DenseLayout::Gemm(tile_n,TileK(),slots,warps_k,batch,residual).end;
  }
  // D5 full residency: one task's whole K is in flight at once.
  bool FullyResident(int k) const {return slots==k/TileK();}
  // D6 preissue: the first task's slots, weights only (prompt §6).
  int StreamBytes(int k) const {return std::min(slots,k/TileK())*slot_bytes;}
  void Validate(int n,int k) const {
    if(n<=0 || k<=0 || tile_n<8 || tile_n>n || tile_n%8 || n%tile_n || warps_n<1 ||
       (warps_n!=1 && warps_n!=2 && warps_n!=4) || warps_n*warps_k!=4 ||
       tile_n%(8*warps_n) || slot_bytes<=0 || slot_bytes%(2ll*tile_n) ||
       TileK()%64 || k%TileK() || slots<2 || slots>8 ||
       (register_stages!=1 && register_stages!=2 && register_stages!=4) ||
       register_stages>TileK()/(16*warps_k) || (multicast && slots%2))
      throw std::invalid_argument("illegal sm90 dense GEMM coordinate");
    if(mma==Sm90DenseMma::kWgmma && (tile_n%64 || warps_n!=4 || warps_k!=1))
      throw std::invalid_argument("sm90 dense WGMMA requires a complete N warpgroup");
  }
};

struct Sm90DenseAttention {
  int chunks=1,block_keys=64,slots=3;
  int new_row_work=0;
  // D10b: the S chunks merge through the chunk-0 CTA's DSMEM inbox.
  bool dsmem=false;
  std::int64_t SharedBytes(int dim,int queries) const {
    return backend::Sm90DenseLayout::Attention(dim,queries,chunks,block_keys,slots,dsmem).end;
  }
  // D5 full width: the storage exceeds one arena and takes the whole budget.
  bool FullWidth(int dim,int queries,int arena) const {return SharedBytes(dim,queries)-1024>arena;}
  // Every block of a task at `past` is in flight at once.
  bool FullyResident(int past) const {
    int keys=(past+1+chunks-1)/chunks;return slots>=(keys+block_keys-1)/block_keys;
  }
  int StreamBytes(int dim) const {return slots*4*block_keys*dim;}
  void Validate() const {
    if(chunks<1 || chunks>64 || (block_keys!=64 && block_keys!=128) || slots<2 || slots>4 ||
       new_row_work<0 || (dsmem && chunks<2))
      throw std::invalid_argument("illegal sm90 dense attention coordinate");
  }
};

inline constexpr int Sm90DenseArena(std::int64_t budget) {return int((budget-1024)/2/1024*1024);}

struct Sm90DenseCoordinates {
  std::array<Sm90DenseGemm,8> gemms;
  Sm90DenseAttention attention;
  bool moe=false;
  std::vector<unsigned> Active() const {return Sm90DenseActiveFamilies(moe);}
  int residency=1,swiglu_u=16;
  // D6 (prompt §6): every GEMM/attention boundary between two arena stages
  // streams the next stage's first slots; set by decision table DT-1.
  bool weight_stream=false;
  // D7: the layer pairs run in a runtime loop rather than unrolled; set by
  // decision table DT-3.
  bool layer_loop=false;
  // D9 ②: the head's last arriver writes the next step's embedding ("la")
  // or a separate embedding stage runs each step ("stage"); P-SEL decides.
  bool embed_la=false;
  // D8 (prompt §7): attention waits only for its KV group's qkv tiles
  // (group counters) instead of the grid barrier after qkv; P-SEL decides.
  bool group_wait=false;
  // D10 (DT-2): CTAs launch in clusters of `cluster` ranks (1 or 2; larger
  // clusters change the grid away from the measured 132/264).
  int cluster=1;
  // MR7 (MC7): the grouped gate_up and down stages run as one typed stage
  // (M56) whose down tasks wait on the per-slot counters; P-SEL decides.
  // MoE only, and one ring serves both task types, so the two expert
  // families share its depth.
  bool expert_fuse=false;
  static constexpr bool ResidualFamily(unsigned family) {return family==1 || family==3;}

  // MR7: the fused stage's storage; a slot holds either task type, so each
  // geometry takes the larger of the two (Sm90DenseLayout::MoeFused).
  std::int64_t FusedExpertBytes(int batch) const {
    return backend::Sm90DenseLayout::MoeFused(gemms[6].tile_n,gemms[6].TileK(),gemms[7].tile_n,
        gemms[7].TileK(),gemms[6].slots,gemms[6].warps_k,gemms[7].warps_k,batch).end;
  }

  static Sm90DenseCoordinates Exact(Sm90DenseShapeTable const& nk,
                                    int batch,std::int64_t shared_budget) {
    if(batch!=1 && batch!=2 && batch!=4 && batch!=8 && batch!=16)
      throw std::invalid_argument("sm90 dense batch must be 1,2,4,8,16");
    Sm90DenseCoordinates result;
    result.attention.chunks=16/batch;
    for(unsigned i:result.Active()) {
      auto& g=result.gemms[i];int n=nk[i][0],k=nk[i][1];
      if(n<=0 || k<=0 || n%128 || (i!=4 && (n/128)%8))
        throw std::invalid_argument("exact dense partition needs 128 integral N tiles");
      g.tile_n=i==4?128:n/128;
      g.warps_n=std::gcd(4,g.tile_n/8);g.warps_k=4/g.warps_n;
      bool fits=false;
      for(int tile_k:{256,128,64}) {
        if(k%tile_k)continue;
        g.slot_bytes=2*g.tile_n*tile_k;
        if(g.SharedBytes(batch,ResidualFamily(i))<=shared_budget) {fits=true;break;}
      }
      if(!fits)throw std::invalid_argument("dense exact seed exceeds shared-memory budget");
      if(i==2)while(g.tile_n%(2*result.swiglu_u))result.swiglu_u/=2;
    }
    result.Validate(nk,batch);
    return result;
  }

  void Validate(Sm90DenseShapeTable const& nk,int batch) const {
    if(batch!=1 && batch!=2 && batch!=4 && batch!=8 && batch!=16)
      throw std::invalid_argument("sm90 dense batch must be 1,2,4,8,16");
    if((residency!=1 && residency!=2) ||
       (swiglu_u!=4 && swiglu_u!=8 && swiglu_u!=16) || (cluster!=1 && cluster!=2))
      throw std::invalid_argument("illegal sm90 dense structural coordinate");
    // D10b chunks share one cluster; D10c multicast pairs exist only in
    // clusters and only for the static dense-shaped GEMM families.
    if((attention.dsmem && cluster%attention.chunks) ||
       std::any_of(gemms.begin(),gemms.end(),[&](Sm90DenseGemm const& g) {
         return g.multicast && cluster<2;}) || gemms[5].multicast || gemms[6].multicast || gemms[7].multicast)
      throw std::invalid_argument("illegal sm90 dense cluster coordinate");
    for(unsigned i:Active())gemms[i].Validate(nk[i][0],nk[i][1]);
    // Expert gate_up pages interleave gate/up rows in U = 16 blocks.
    if((!moe && gemms[2].tile_n%(2*swiglu_u)) || (moe && gemms[6].tile_n%32))
      throw std::invalid_argument("sm90 dense gate tile splits a gate/up pair");
    // MR7: the fused stage is MoE only and runs one ring for both task types.
    if(expert_fuse && (!moe || gemms[6].slots!=gemms[7].slots))
      throw std::invalid_argument("illegal sm90 dense expert fusion coordinate");
    attention.Validate();
  }

  void ValidateSchedule(Sm90DenseShapeTable const& nk,int batch,
      int dim,int queries,int capacity,std::int64_t budget) const {
    Validate(nk,batch);
    // Arena i % 2 holds stage i (D5); D6 payloads live in the next stage's
    // arena, so the program's static_asserts are the only overlap check.
    for(unsigned i:Active())
      if(gemms[i].SharedBytes(batch,ResidualFamily(i))-1024>Sm90DenseArena(budget))
        throw std::invalid_argument("native GEMM storage exceeds its arena");
    // Attention may take the whole budget when two slots exceed one arena.
    if(attention.SharedBytes(dim,queries)>budget)
      throw std::invalid_argument("native attention exceeds the shared-memory budget");
    // MR7: the fused stage replaces both expert stages, so its own ring and
    // scratch have to fit the arena the surviving stage occupies.
    if(expert_fuse && FusedExpertBytes(batch)-1024>Sm90DenseArena(budget))
      throw std::invalid_argument("native fused expert storage exceeds its arena");
  }

  json::Value ToJson() const {
    json::Object families;
    for(unsigned i:Active()) {
      auto const& g=gemms[i];
      families.emplace_back(kSm90DenseFamilies[i],json::Object{
          {"tile_n",g.tile_n},{"warps_n",g.warps_n},{"warps_k",g.warps_k},
          {"slot_bytes",g.slot_bytes},{"slots",g.slots},
          {"mma",g.mma==Sm90DenseMma::kMmaSync?"mma_sync":"wgmma"},
          {"register_stages",g.register_stages},{"multicast",g.multicast}});
    }
    return json::Object{{"schema",6},{"model",moe?"moe":"dense"},{"gemms",families},
        {"attention",json::Object{{"chunks",attention.chunks},{"block_keys",attention.block_keys},
            {"slots",attention.slots},{"new_row_work",attention.new_row_work},
            {"merge",attention.dsmem?"dsmem":"la"}}},
        {"residency",residency},{"swiglu_u",swiglu_u},{"weight_stream",weight_stream},
        {"layer_loop",layer_loop},{"emb",embed_la?"la":"stage"},{"group_wait",group_wait},
        {"cluster",cluster},{"expert_fuse",expert_fuse}};
  }

  static Sm90DenseCoordinates FromJson(json::Value const& value) {
    auto keys=[](json::Value const& object,std::initializer_list<char const*> names) {
      std::vector<std::string> seen;
      for(auto const& item:object.AsObject("sm90 dense coordinate")) {
        if(std::find(names.begin(),names.end(),item.first)==names.end() ||
           std::find(seen.begin(),seen.end(),item.first)!=seen.end())
          throw std::invalid_argument("unknown or duplicate sm90 dense coordinate: "+item.first);
        seen.push_back(item.first);
      }
    };
    auto integer=[](json::Value const& object,char const* key) {
      double number=object.At(key).AsNumber(key);
      if(!std::isfinite(number) || number!=std::floor(number) || number<0 || number>2147483647)
        throw std::invalid_argument(std::string("invalid sm90 dense integer: ")+key);
      return int(number);
    };
    // Schema 3 (prompt §6): arena offsets and D6 prefixes are derived by
    // rule; schema 4 (§8) names the model and with it the GEMM families;
    // schema 5 (§7) adds the L3 structural dimensions; schema 6 (MR7) adds
    // `expert_fuse`. Schema 5 still loads: REF-R3's 30 cells are written in
    // it and predate the fusion, which is exactly `expert_fuse` false.
    int schema=integer(value,"schema");
    if(schema!=5 && schema!=6)throw std::invalid_argument("unknown sm90 dense coordinate schema");
    keys(value,{"schema","model","gemms","attention","residency","swiglu_u","weight_stream","layer_loop",
        "emb","group_wait","cluster","expert_fuse"});
    Sm90DenseCoordinates result;
    auto model=value.At("model").AsString("model");
    if(model!="dense" && model!="moe")throw std::invalid_argument("unknown sm90 dense model");
    result.moe=model=="moe";
    std::vector<std::string> names;
    for(unsigned i:result.Active())names.push_back(kSm90DenseFamilies[i]);
    for(auto const& item:value.At("gemms").AsObject("sm90 dense coordinate"))
      if(std::find(names.begin(),names.end(),item.first)==names.end())
        throw std::invalid_argument("unknown or duplicate sm90 dense coordinate: "+item.first);
    if(value.At("gemms").AsObject("sm90 dense coordinate").size()!=names.size())
      throw std::invalid_argument("sm90 dense coordinates lack a GEMM family");
    for(unsigned i:result.Active()) {
      auto const& v=value.At("gemms").At(kSm90DenseFamilies[i]);auto& g=result.gemms[i];
      keys(v,{"tile_n","warps_n","warps_k","slot_bytes","slots","mma","register_stages","multicast"});
      g.tile_n=integer(v,"tile_n");g.warps_n=integer(v,"warps_n");g.warps_k=integer(v,"warps_k");
      g.slot_bytes=integer(v,"slot_bytes");g.slots=integer(v,"slots");
      g.register_stages=integer(v,"register_stages");g.multicast=v.At("multicast").AsBool("multicast");
      auto mma=v.At("mma").AsString("mma");
      if(mma!="mma_sync" && mma!="wgmma")throw std::invalid_argument("unknown sm90 dense MMA instruction");
      g.mma=mma=="mma_sync"?Sm90DenseMma::kMmaSync:Sm90DenseMma::kWgmma;
    }
    auto const& a=value.At("attention");
    keys(a,{"chunks","block_keys","slots","new_row_work","merge"});
    result.attention={integer(a,"chunks"),integer(a,"block_keys"),integer(a,"slots"),
        integer(a,"new_row_work")};
    auto merge=a.At("merge").AsString("merge");
    if(merge!="la" && merge!="dsmem")throw std::invalid_argument("unknown sm90 dense attention merge");
    result.attention.dsmem=merge=="dsmem";
    result.residency=integer(value,"residency");result.swiglu_u=integer(value,"swiglu_u");
    result.weight_stream=value.At("weight_stream").AsBool("weight_stream");
    result.layer_loop=value.At("layer_loop").AsBool("layer_loop");
    auto emb=value.At("emb").AsString("emb");
    if(emb!="la" && emb!="stage")throw std::invalid_argument("unknown sm90 dense embedding placement");
    result.embed_la=emb=="la";
    result.group_wait=value.At("group_wait").AsBool("group_wait");
    result.cluster=integer(value,"cluster");
    if(auto const* fuse=value.Find("expert_fuse")) {
      if(schema<6)throw std::invalid_argument("sm90 dense expert fusion needs coordinate schema 6");
      result.expert_fuse=fuse->AsBool("expert_fuse");
    }
    return result;
  }
};

}  // namespace tilemega::solver
