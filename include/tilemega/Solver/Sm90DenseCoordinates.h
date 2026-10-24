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
  int preissue_smem_bytes=0,prefetch_l2_bytes=0;
  int register_stages=1;
  int workspace_offset=0;
  bool preissue_late=false;
  int TileK() const { return int(slot_bytes/(2ll*tile_n)); }
  std::int64_t SharedBytes(int batch) const {
    std::int64_t operands=slots*(std::int64_t(slot_bytes)+
        2ll*backend::Sm90DenseLayout::ActivationRows(batch)*TileK());
    std::int64_t reduction=64ll*tile_n*warps_k+512;
    return workspace_offset+1024+operands+reduction;
  }
  void Validate(int n,int k) const {
    if(n<=0 || k<=0 || tile_n<8 || tile_n>n || tile_n%8 || n%tile_n || warps_n<1 ||
       (warps_n!=1 && warps_n!=2 && warps_n!=4) || warps_n*warps_k!=4 ||
       tile_n%(8*warps_n) || slot_bytes<=0 || slot_bytes%(2ll*tile_n) ||
       TileK()%64 || k%TileK() || slots<2 || slots>8 ||
       preissue_smem_bytes<0 || preissue_smem_bytes%16 ||
       preissue_smem_bytes>2ll*tile_n*k || prefetch_l2_bytes<0 || prefetch_l2_bytes%16 ||
       (register_stages!=1 && register_stages!=2 && register_stages!=4) ||
       register_stages>TileK()/(16*warps_k))
      throw std::invalid_argument("illegal sm90 dense GEMM coordinate");
    if(workspace_offset<0 || workspace_offset%1024 ||
       preissue_smem_bytes>std::min(slots,k/TileK())*std::int64_t(slot_bytes) ||
       (preissue_late && !preissue_smem_bytes && !prefetch_l2_bytes))
      throw std::invalid_argument("native GEMM preissue exceeds its first-task slots");
    if(mma==Sm90DenseMma::kWgmma && (tile_n%64 || warps_n!=4 || warps_k!=1))
      throw std::invalid_argument("sm90 dense WGMMA requires a complete N warpgroup");
  }
};

struct Sm90DenseAttention {
  int chunks=1,block_keys=64,slots=3;
  int preissue_smem_bytes=0,prefetch_l2_bytes=0;
  int new_row_work=0;
  int workspace_offset=0;
  bool preissue_late=false;
  std::int64_t SharedBytes(int batch,int dim,int queries,int capacity) const {
    std::int64_t operands=batch<=4?
        2ll*queries*dim+4ll*((std::int64_t(capacity)+new_row_work+chunks-1)/chunks)*dim:
        36ll*dim+4ll*slots*block_keys*dim;
    std::int64_t scratch=16ll*queries*dim+32ll*queries+
        4ll*std::max(4,chunks)*queries+4ll*queries+4;
    return workspace_offset+1024+((operands+scratch+15)/16)*16;
  }
  void Validate() const {
    if(chunks<1 || chunks>64 || (block_keys!=16 && block_keys!=32 &&
       block_keys!=64 && block_keys!=128) || slots<2 || slots>8 ||
       preissue_smem_bytes<0 || preissue_smem_bytes%16 ||
       prefetch_l2_bytes<0 || prefetch_l2_bytes%16 || new_row_work<0 ||
       workspace_offset<0 || workspace_offset%1024 ||
       (preissue_late && !preissue_smem_bytes && !prefetch_l2_bytes))
      throw std::invalid_argument("illegal sm90 dense attention coordinate");
  }
};

struct Sm90DenseCoordinates {
  std::array<Sm90DenseGemm,5> gemms;
  Sm90DenseAttention attention;
  int residency=1,swiglu_u=16;
  bool dn_vector_sums=false;

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
        if(g.SharedBytes(batch)<=shared_budget) {fits=true;break;}
      }
      if(!fits)throw std::invalid_argument("dense exact seed exceeds shared-memory budget");
      if((i==1 || i==3) && g.tile_n%32)result.dn_vector_sums=true;
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
    if(!dn_vector_sums && (gemms[1].tile_n%32 || gemms[3].tile_n%32))
      throw std::invalid_argument("sm90 dense residual tiles require vector sums");
    attention.Validate();
  }

  void ValidateSchedule(std::array<std::array<int,2>,5> const& nk,int batch,
      int dim,int queries,int capacity,std::int64_t budget) const {
    Validate(nk,batch);
    using Layout=backend::Sm90DenseLayout;
    struct Stage {Layout layout;backend::Sm90DensePayload payload;int offset;};
    std::array<Stage,6> stages;
    for(unsigned i=0;i<5;++i) {
      auto const& g=gemms[i];
      if(g.SharedBytes(batch)>budget)
        throw std::invalid_argument("native GEMM arena exceeds target budget");
      stages[i]={Layout::Gemm(g.tile_n,g.TileK(),g.slots,g.warps_k,batch),
          Layout::GemmPayload(g.tile_n,g.TileK(),g.slots,nk[i][1],batch,g.preissue_smem_bytes)
              .Shift(g.workspace_offset),g.workspace_offset};
    }
    int rows=batch<=4?Layout::AttentionKeys(attention.chunks,attention.new_row_work,capacity):
        attention.slots*attention.block_keys;
    if(attention.SharedBytes(batch,dim,queries,capacity)>budget ||
       attention.prefetch_l2_bytes%32 || attention.preissue_smem_bytes%(4*dim) ||
       attention.preissue_smem_bytes>4ll*dim*rows)
      throw std::invalid_argument("native KV preissue must fit whole historical rows");
    stages[5]={Layout::Attention(batch,dim,queries,attention.chunks,attention.block_keys,
        attention.slots,attention.new_row_work,capacity),
        Layout::AttentionPayload(batch,dim,queries,attention.chunks,attention.block_keys,
            attention.slots,attention.new_row_work,capacity,attention.preissue_smem_bytes)
            .Shift(attention.workspace_offset),attention.workspace_offset};
    for(auto const& s:stages)if(s.offset+s.layout.end>budget)
      throw std::invalid_argument("native arena exceeds target shared-memory budget");
    // Both the layer back-edge and final down->head edge must be safe.
    for(auto edge:std::array<std::array<int,2>,6>{{{0,5},{5,1},{1,2},{2,3},{3,0},{3,4}}}) {
      auto const& previous=stages[edge[0]];auto const& next=stages[edge[1]];
      if(next.payload.Overlaps(previous.layout.Scratch(previous.offset)))
        throw std::invalid_argument("native preissue overlaps live epilogue scratch");
    }
  }

  json::Value ToJson() const {
    json::Object families;
    for(unsigned i=0;i<gemms.size();++i) {
      auto const& g=gemms[i];
      families.emplace_back(kSm90DenseFamilies[i],json::Object{
          {"tile_n",g.tile_n},{"warps_n",g.warps_n},{"warps_k",g.warps_k},
          {"slot_bytes",g.slot_bytes},{"slots",g.slots},
          {"mma",g.mma==Sm90DenseMma::kMmaSync?"mma_sync":"wgmma"},
          {"preissue_smem_bytes",g.preissue_smem_bytes},{"prefetch_l2_bytes",g.prefetch_l2_bytes},
          {"register_stages",g.register_stages},{"workspace_offset",g.workspace_offset},
          {"preissue_late",g.preissue_late}});
    }
    return json::Object{{"schema",1},{"gemms",families},
        {"attention",json::Object{{"chunks",attention.chunks},{"block_keys",attention.block_keys},
            {"slots",attention.slots},{"preissue_smem_bytes",attention.preissue_smem_bytes},
            {"prefetch_l2_bytes",attention.prefetch_l2_bytes},{"new_row_work",attention.new_row_work},
            {"workspace_offset",attention.workspace_offset},{"preissue_late",attention.preissue_late}}},
        {"residency",residency},{"swiglu_u",swiglu_u},{"dn_vector_sums",dn_vector_sums}};
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
    if(integer(value,"schema")!=1)throw std::invalid_argument("unknown sm90 dense coordinate schema");
    keys(value,{"schema","gemms","attention","residency","swiglu_u","dn_vector_sums"});
    keys(value.At("gemms"),{"qkv","o","gate_up","down","lm_head"});
    Sm90DenseCoordinates result;
    for(unsigned i=0;i<result.gemms.size();++i) {
      auto const& v=value.At("gemms").At(kSm90DenseFamilies[i]);auto& g=result.gemms[i];
      keys(v,{"tile_n","warps_n","warps_k","slot_bytes","slots","mma",
          "preissue_smem_bytes","prefetch_l2_bytes","register_stages","workspace_offset","preissue_late"});
      g.tile_n=integer(v,"tile_n");g.warps_n=integer(v,"warps_n");g.warps_k=integer(v,"warps_k");
      g.slot_bytes=integer(v,"slot_bytes");g.slots=integer(v,"slots");
      g.preissue_smem_bytes=integer(v,"preissue_smem_bytes");
      g.prefetch_l2_bytes=integer(v,"prefetch_l2_bytes");
      g.register_stages=v.Find("register_stages")?integer(v,"register_stages"):1;
      g.workspace_offset=v.Find("workspace_offset")?integer(v,"workspace_offset"):0;
      g.preissue_late=v.Find("preissue_late")?v.At("preissue_late").AsBool("preissue_late"):false;
      auto mma=v.At("mma").AsString("mma");
      if(mma!="mma_sync" && mma!="wgmma")throw std::invalid_argument("unknown sm90 dense MMA instruction");
      g.mma=mma=="mma_sync"?Sm90DenseMma::kMmaSync:Sm90DenseMma::kWgmma;
    }
    auto const& a=value.At("attention");
    keys(a,{"chunks","block_keys","slots","preissue_smem_bytes","prefetch_l2_bytes","new_row_work","workspace_offset","preissue_late"});
    result.attention={integer(a,"chunks"),integer(a,"block_keys"),integer(a,"slots"),
        integer(a,"preissue_smem_bytes"),integer(a,"prefetch_l2_bytes"),integer(a,"new_row_work")};
    result.attention.workspace_offset=a.Find("workspace_offset")?integer(a,"workspace_offset"):0;
    result.attention.preissue_late=a.Find("preissue_late")?a.At("preissue_late").AsBool("preissue_late"):false;
    result.residency=integer(value,"residency");result.swiglu_u=integer(value,"swiglu_u");
    result.dn_vector_sums=value.At("dn_vector_sums").AsBool("dn_vector_sums");
    return result;
  }
};

}  // namespace tilemega::solver
