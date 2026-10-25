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

namespace tilemega::solver {

enum class Sm90DenseMma { kMmaSync, kWgmma };
inline constexpr int kSm90DenseGemmImpl=3;
inline constexpr std::array<char const*,5> kSm90DenseFamilies={
    "qkv","o","gate_up","down","lm_head"};

struct Sm90DenseGemm {
  int tile_n=0,warps_n=1,warps_k=4,slot_bytes=0,slots=3;
  Sm90DenseMma mma=Sm90DenseMma::kMmaSync;
  int register_stages=1;
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
       register_stages>TileK()/(16*warps_k))
      throw std::invalid_argument("illegal sm90 dense GEMM coordinate");
    if(mma==Sm90DenseMma::kWgmma && (tile_n%64 || warps_n!=4 || warps_k!=1))
      throw std::invalid_argument("sm90 dense WGMMA requires a complete N warpgroup");
  }
};

struct Sm90DenseAttention {
  int chunks=1,block_keys=64,slots=3;
  int new_row_work=0;
  std::int64_t SharedBytes(int dim,int queries) const {
    return backend::Sm90DenseLayout::Attention(dim,queries,chunks,block_keys,slots).end;
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
       new_row_work<0)
      throw std::invalid_argument("illegal sm90 dense attention coordinate");
  }
};

inline constexpr int Sm90DenseArena(std::int64_t budget) {return int((budget-1024)/2/1024*1024);}

struct Sm90DenseCoordinates {
  std::array<Sm90DenseGemm,5> gemms;
  Sm90DenseAttention attention;
  int residency=1,swiglu_u=16;
  // D6 (prompt §6): every GEMM/attention boundary between two arena stages
  // streams the next stage's first slots; set by decision table DT-1.
  bool weight_stream=false;
  // D7: the layer pairs run in a runtime loop rather than unrolled; set by
  // decision table DT-3.
  bool layer_loop=false;
  static constexpr bool ResidualFamily(unsigned family) {return family==1 || family==3;}

  static Sm90DenseCoordinates Exact(std::array<std::array<int,2>,5> const& nk,
                                    int batch,std::int64_t shared_budget) {
    if(batch!=1 && batch!=2 && batch!=4 && batch!=8 && batch!=16)
      throw std::invalid_argument("sm90 dense batch must be 1,2,4,8,16");
    Sm90DenseCoordinates result;
    result.attention.chunks=16/batch;
    for(unsigned i=0;i<result.gemms.size();++i) {
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

  void Validate(std::array<std::array<int,2>,5> const& nk,int batch) const {
    if(batch!=1 && batch!=2 && batch!=4 && batch!=8 && batch!=16)
      throw std::invalid_argument("sm90 dense batch must be 1,2,4,8,16");
    if((residency!=1 && residency!=2) ||
       (swiglu_u!=4 && swiglu_u!=8 && swiglu_u!=16))
      throw std::invalid_argument("illegal sm90 dense structural coordinate");
    for(unsigned i=0;i<gemms.size();++i)gemms[i].Validate(nk[i][0],nk[i][1]);
    if(gemms[2].tile_n%(2*swiglu_u))
      throw std::invalid_argument("sm90 dense gate tile splits a gate/up pair");
    attention.Validate();
  }

  void ValidateSchedule(std::array<std::array<int,2>,5> const& nk,int batch,
      int dim,int queries,int capacity,std::int64_t budget) const {
    Validate(nk,batch);
    // Arena i % 2 holds stage i (D5); D6 payloads live in the next stage's
    // arena, so the program's static_asserts are the only overlap check.
    for(unsigned i=0;i<5;++i)
      if(gemms[i].SharedBytes(batch,ResidualFamily(i))-1024>Sm90DenseArena(budget))
        throw std::invalid_argument("native GEMM storage exceeds its arena");
    // Attention may take the whole budget when two slots exceed one arena.
    if(attention.SharedBytes(dim,queries)>budget)
      throw std::invalid_argument("native attention exceeds the shared-memory budget");
  }

  json::Value ToJson() const {
    json::Object families;
    for(unsigned i=0;i<gemms.size();++i) {
      auto const& g=gemms[i];
      families.emplace_back(kSm90DenseFamilies[i],json::Object{
          {"tile_n",g.tile_n},{"warps_n",g.warps_n},{"warps_k",g.warps_k},
          {"slot_bytes",g.slot_bytes},{"slots",g.slots},
          {"mma",g.mma==Sm90DenseMma::kMmaSync?"mma_sync":"wgmma"},
          {"register_stages",g.register_stages}});
    }
    return json::Object{{"schema",3},{"gemms",families},
        {"attention",json::Object{{"chunks",attention.chunks},{"block_keys",attention.block_keys},
            {"slots",attention.slots},{"new_row_work",attention.new_row_work}}},
        {"residency",residency},{"swiglu_u",swiglu_u},{"weight_stream",weight_stream},
        {"layer_loop",layer_loop}};
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
    // Schema 3 (prompt §6): arena offsets and D6 prefixes are derived by rule.
    if(integer(value,"schema")!=3)throw std::invalid_argument("unknown sm90 dense coordinate schema");
    keys(value,{"schema","gemms","attention","residency","swiglu_u","weight_stream","layer_loop"});
    keys(value.At("gemms"),{"qkv","o","gate_up","down","lm_head"});
    Sm90DenseCoordinates result;
    for(unsigned i=0;i<result.gemms.size();++i) {
      auto const& v=value.At("gemms").At(kSm90DenseFamilies[i]);auto& g=result.gemms[i];
      keys(v,{"tile_n","warps_n","warps_k","slot_bytes","slots","mma","register_stages"});
      g.tile_n=integer(v,"tile_n");g.warps_n=integer(v,"warps_n");g.warps_k=integer(v,"warps_k");
      g.slot_bytes=integer(v,"slot_bytes");g.slots=integer(v,"slots");
      g.register_stages=integer(v,"register_stages");
      auto mma=v.At("mma").AsString("mma");
      if(mma!="mma_sync" && mma!="wgmma")throw std::invalid_argument("unknown sm90 dense MMA instruction");
      g.mma=mma=="mma_sync"?Sm90DenseMma::kMmaSync:Sm90DenseMma::kWgmma;
    }
    auto const& a=value.At("attention");
    keys(a,{"chunks","block_keys","slots","new_row_work"});
    result.attention={integer(a,"chunks"),integer(a,"block_keys"),integer(a,"slots"),
        integer(a,"new_row_work")};
    result.residency=integer(value,"residency");result.swiglu_u=integer(value,"swiglu_u");
    result.weight_stream=value.At("weight_stream").AsBool("weight_stream");
    result.layer_loop=value.At("layer_loop").AsBool("layer_loop");
    return result;
  }
};

}  // namespace tilemega::solver
