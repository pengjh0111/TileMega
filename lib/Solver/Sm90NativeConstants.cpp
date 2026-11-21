// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tilemega::solver {
namespace {
std::string FileHash(std::string const& path) {
  auto command="sha256sum '"+path+"'";auto* pipe=popen(command.c_str(),"r");
  if(!pipe)throw std::runtime_error("cannot hash "+path);
  char text[65]{};auto* got=fgets(text,sizeof(text),pipe);pclose(pipe);
  return got?std::string(text):std::string();
}
double Positive(json::Value const& v,char const* what) {
  double x=v.AsNumber(what);
  if(!(x>0))throw std::invalid_argument(std::string("nonpositive native constant ")+what);
  return x;
}
}  // namespace

std::array<int,19> Sm80FusedExpertShape::Key() const {
  return {batch,gate_up_n,down_n,hidden,gate_up_tn,gate_up_tk,down_tn,down_tk,slots,
      gate_up_warps_n,down_warps_n,experts,float_routes,page,register_stages,
      softmax_all,segments,live_rows,copy_group};
}

Sm90NativeConstants Sm90NativeConstants::FromTarget(json::Value const& target,std::string const& root) {
  auto const* tag=target.Find("arch_tag");
  bool ampere=tag && tag->AsString("target architecture")=="sm_80";
  auto key=ampere?"sm80_native_constants":"sm90_native_constants";
  auto const* found=target.Find(key);
  if(!found)throw std::invalid_argument(std::string("target lacks measured ")+key);
  Sm90NativeConstants c;c.section=*found;auto const& s=*found;
  if(s.At("schema").AsNumber("constants schema")!=1)
    throw std::invalid_argument("unknown sm90_native_constants schema");
  for(auto const& item:s.At("c_bar_ns").AsObject("c_bar"))
    c.barrier_[std::stoi(item.first)]=Positive(item.second,"c_bar");
  c.l2_latency=Positive(s.At("l2_latency_ns"),"l2 latency");
  c.dram_latency=Positive(s.At("dram_latency_ns"),"dram latency");
  c.dram_gbps=Positive(s.At("dram_gbps"),"dram bandwidth");
  c.l2_gbps=Positive(s.At("l2_gbps"),"l2 bandwidth");
  c.sm_gbps=Positive(s.At("sm_gbps"),"sm bandwidth");
  c.cp_async_gbps=Positive(s.At("activation_cp_async_gbps"),"cp.async activation bandwidth");
  c.epilogue_=s.At("epilogue_ns");c.embedding_=s.At("embedding_ns");c.argmax_la_=s.At("argmax_la_ns");
  c.route_la_=s.At("route_la_ns");c.round_=s.At("round_ns");
  if(!ampere) {c.clusters_=s.At("clusters");c.dsmem_=s.At("dsmem_merge_ns");}
  else if(auto const* mma=s.Find("mma_mainloop"))c.ampere_mma_=mma->At("median_ns");
  if(auto const* service=s.Find("dense_gate_service")) {
    auto schema=service->At("schema").AsNumber("dense gate service schema");
    if(!ampere || (schema!=1 && schema!=2))
      throw std::invalid_argument("invalid SM80 dense gate service profile");
    auto profiles=schema==1?json::Array{*service}:service->At("profiles").AsArray("dense gate service profiles");
    json::Value matched(json::Object{});
    int const sms=int(target.At("resources").At("num_sms").AsNumber("target SM count"));
    for(auto const& profile:profiles) {
      auto const& hashes=profile.At("body_headers");
      bool exact=true;
      for(auto name:{"Sm90DenseGemm.h","Sm80DenseService.cuh","Sm90DenseLayout.h",
                     "Sm90DenseRing.cuh","Sm90DenseEpilogue.h"}) {
        auto path=root+"/include/tilemega/Backend/"+name;
        if(FileHash(path)==hashes.At(name).AsString("body header hash"))continue;
        if(schema==1)
          throw Sm90NativeMissing(std::string("SM80 dense service profile no longer matches ")+name);
        exact=false;
        break;
      }
      if(!exact)continue;
      for(auto const& [key,row]:profile.At("measurements").AsObject("dense service measurements")) {
        // Occupancy headroom alone does not establish equivalent concurrent service.
        if(schema==2 && row.At("grid").AsNumber("service grid")!=sms &&
            !row.Find("production_identity_sha256"))continue;
        if(auto const* prior=matched.Find(key);prior && prior->Dump(0)!=row.Dump(0))
          throw std::invalid_argument("conflicting compatible dense gate services for "+key);
        matched.Set(key,row);
      }
    }
    if(!matched.AsObject("matched dense services").empty())c.ampere_gate_service_=std::move(matched);
  }
  if(auto const* profiles=s.Find("dense_gemm_service")) {
    if(!ampere || profiles->At("schema").AsNumber("dense gemm service schema")!=1)
      throw std::invalid_argument("invalid SM80 dense GEMM service registry");
    json::Value matched(json::Object{});
    for(auto const& profile:profiles->At("profiles").AsArray("dense GEMM profiles")) {
      auto const& hashes=profile.At("body_headers");
      bool compatible=true;
      for(auto name:{"Backend/Sm90DenseGemm.h","Backend/Sm80DenseService.cuh",
                     "Backend/Sm90DenseLayout.h","Backend/Sm90DenseRing.cuh",
                     "Backend/Sm90DenseEpilogue.h","Codegen/executor/Async.cuh"}) {
        if(FileHash(root+"/include/tilemega/"+name)==hashes.At(name).AsString("body hash"))continue;
        compatible=false;break;
      }
      if(!compatible)continue;
      for(auto const& [key,row]:profile.At("measurements").AsObject("dense GEMM measurements")) {
        if(auto const* prior=matched.Find(key);prior && prior->Dump(0)!=row.Dump(0))
          throw std::invalid_argument("conflicting compatible dense GEMM service for "+key);
        matched.Set(key,row);
      }
    }
    for(auto const& [key,row]:matched.AsObject("matched dense GEMM services")) {
      auto number=[&](char const* name){return int(row.At(name).AsNumber(name));};
      Sm80DenseServiceShape shape{row.At("epilogue").AsString("production epilogue"),
          number("grid"),number("n"),number("k"),number("batch"),number("tn"),number("tk"),
          number("slots"),number("warps_n"),number("warps_k"),number("register_stages"),
          number("u"),number("residual_tiles"),row.At("input_norm").AsBool("input norm"),
          row.At("live_rows").AsBool("live rows"),number("copy_issue_group")};
      if(shape.Key()!=key)throw std::invalid_argument("inconsistent SM80 production service shape");
      c.ampere_dense_shapes_.push_back(std::move(shape));
    }
    if(!c.ampere_dense_shapes_.empty())c.ampere_dense_service_=std::move(matched);
  }
  if(auto const* profiles=s.Find("fused_expert_service")) {
    if(!ampere || profiles->At("schema").AsNumber("fused expert service schema")!=1)
      throw std::invalid_argument("invalid SM80 fused expert service registry");
    int sms=int(target.At("resources").At("num_sms").AsNumber("target SM count"));
    for(auto const& profile:profiles->At("profiles").AsArray("fused expert profiles")) {
      auto const& hashes=profile.At("body_headers");
      bool compatible=true;
      for(auto name:{"Backend/Sm90DenseGemm.h","Backend/Sm90DenseLayout.h","Backend/Sm90DenseMoe.cuh",
                     "Backend/Sm80MoePageOffsets.h","Backend/Sm90DenseMark.cuh",
                     "Backend/Sm90DenseRing.cuh","Backend/Sm90DenseEpilogue.h","Codegen/executor/Async.cuh"})
        compatible &= FileHash(root+"/include/tilemega/"+name)==hashes.At(name).AsString("body hash");
      if(!compatible)continue;
      auto const& row=profile.At("measurement");
      if(row.At("grid").AsNumber("fused expert grid")!=sms ||
         row.At("plain_residency").AsNumber("fused expert residency")!=1)continue;
      auto const& shape=profile.At("shape");
      auto dimension=[&](char const* name) {
        double value=shape.At(name).AsNumber(name);
        if(!std::isfinite(value) || value<0 || value>1000000 || std::floor(value)!=value)
          throw std::invalid_argument(std::string("invalid fused shape dimension ")+name);
        return int(value);
      };
      Sm80FusedExpertShape geometry{dimension("batch"),dimension("gate_up_n"),dimension("down_n"),
          dimension("hidden"),dimension("gate_up_tn"),dimension("gate_up_tk"),dimension("down_tn"),
          dimension("down_tk"),dimension("slots"),dimension("gate_up_warps_n"),dimension("down_warps_n"),
          dimension("experts"),dimension("float_routes"),dimension("page"),dimension("register_stages"),
          dimension("softmax_all"),dimension("segments"),dimension("live_rows"),dimension("copy_group")};
      if(geometry.gate_up_warps_n*dimension("gate_up_warps_k")!=4 ||
         geometry.down_warps_n*dimension("down_warps_k")!=4)
        throw std::invalid_argument("fused expert shapes require four consumer warps");
      auto positive=[&](char const* name){return Positive(row.At(name),name);};
      // Dn need not have a mainloop after its prefetched prefix.
      double dn_issue=row.At("dn_issue_ns_per_tile").AsNumber("Dn steady issue");
      if(!std::isfinite(dn_issue) || dn_issue<0)
        throw std::invalid_argument("invalid fused expert Dn issue");
      Sm80FusedExpertService service{
          positive("gu_issue_ns_per_tile"),positive("gu_mma_ns_per_tile"),
          positive("gu_epilogue_ns_per_task"),dn_issue,positive("dn_mma_ns_per_tile"),
          positive("dn_epilogue_ns_per_task"),positive("dn_preissue_ns_per_task"),
          positive("dn_activation_ns_per_task")};
      if(!c.ampere_fused_service_.emplace(geometry.Key(),service).second)
        throw std::invalid_argument("conflicting compatible fused expert service");
      c.ampere_fused_shapes_.push_back(geometry);
    }
  }
  if(auto const* forward=s.Find("forward"))c.forward_=*forward;
  if(auto const* cta=s.Find("cta_dram_gbps"))c.cta_dram_gbps_=Positive(*cta,"per-CTA DRAM bandwidth");
  if(auto const* task=s.Find("task_expert_ns"))c.task_expert_=*task;
  if(auto const* tp=s.Find("tp"))c.tp_=*tp;
  c.attention_=s.At("attention_ns");
  if(auto const* fresh=s.Find("attention_fresh_service")) {
    if(!ampere || fresh->At("schema").AsNumber("attention fresh schema")!=1)
      throw std::invalid_argument("invalid SM80 attention fresh service");
    bool compatible=true;
    auto const& hashes=fresh->At("body_headers");
    for(auto name:{"Backend/Sm90DenseAttentionBody.cuh","Backend/Sm90DenseAttentionPartition.h",
                   "Codegen/executor/Async.cuh"})
      compatible &= FileHash(root+"/include/tilemega/"+name)==hashes.At(name).AsString("body hash");
    if(compatible)c.ampere_attention_fresh_=fresh->At("measurements");
  }
  return c;
}

std::optional<Sm80FusedExpertService> Sm90NativeConstants::AmpereFusedService(
    Sm80FusedExpertShape const& shape) const {
  auto found=ampere_fused_service_.find(shape.Key());
  if(found==ampere_fused_service_.end())return std::nullopt;
  return found->second;
}
double Sm90NativeConstants::Barrier(int grid) const {
  auto found=barrier_.find(grid);
  if(found==barrier_.end())throw Sm90NativeMissing("no measured barrier for grid "+std::to_string(grid));
  return found->second;
}
double Sm90NativeConstants::Epilogue(std::string const& family,int tile_n,int batch,
                                     int warps_k) const {
  auto const* f=epilogue_.Find(family);auto const* w=f?f->Find("wk"+std::to_string(warps_k)):nullptr;
  auto const* tn=w?w->Find(std::to_string(tile_n)):nullptr;
  auto const* b=tn?tn->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured epilogue "+family+" TN"+std::to_string(tile_n)+
      " B"+std::to_string(batch)+" WK"+std::to_string(warps_k));
  return Positive(*b,"epilogue");
}
double Sm90NativeConstants::Embedding(int grid,int width,int batch) const {
  auto const* g=embedding_.Find("g"+std::to_string(grid));
  auto const* w=g?g->Find("w"+std::to_string(width)):nullptr;
  auto const* b=w?w->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured embedding stage");
  return Positive(*b,"embedding");
}
double Sm90NativeConstants::ArgmaxLa(int grid,int batch) const {
  auto const* g=argmax_la_.Find("g"+std::to_string(grid));
  auto const* b=g?g->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured last-arriver argmax for grid "+std::to_string(grid));
  return Positive(*b,"argmax last arriver");
}
double Sm90NativeConstants::RouteLa(int grid,int batch,int experts,bool softmax_all) const {
  auto key="g"+std::to_string(grid);
  if(experts!=128 || softmax_all)key+=":e"+std::to_string(experts)+":all"+(softmax_all?"1":"0");
  auto const* g=route_la_.Find(key);
  auto const* b=g?g->Find(std::to_string(batch)):nullptr;
  if(!b)throw Sm90NativeMissing("no measured MoE route last arriver for "+key+" B"+std::to_string(batch));
  return Positive(*b,"route last arriver");
}
double Sm90NativeConstants::GroupWaitSaving(int grid) const {
  auto const* v3=round_.Find("v3");auto const* group=round_.Find("group16");
  auto const* a=v3?v3->Find(std::to_string(grid)):nullptr;
  auto const* b=group?group->Find(std::to_string(grid)):nullptr;
  if(!a || !b)throw Sm90NativeMissing("no measured group wait for grid "+std::to_string(grid));
  return Positive(*a,"V3 round")-Positive(*b,"group round");
}
double Sm90NativeConstants::CtaDramGbps() const {
  if(!(cta_dram_gbps_>0))throw Sm90NativeMissing("no measured per-CTA DRAM bandwidth (MR6 BW_cta_dram)");
  return cta_dram_gbps_;
}
double Sm90NativeConstants::TaskExpert(std::string const& family,int batch) const {
  auto const* table=task_expert_.Find(family);
  auto const* found=table?table->Find(std::to_string(batch)):nullptr;
  if(!found)throw Sm90NativeMissing("no measured expert task overhead for "+family);
  // A per-task cost at or below the subtraction's noise is recorded as zero.
  double value=found->AsNumber("expert task overhead");
  if(value<0)throw std::invalid_argument("negative expert task overhead");
  return value;
}
double Sm90NativeConstants::AllReduce(int batch,int tile_n,int tp) const {
  auto key="b="+std::to_string(batch)+":tn="+std::to_string(tile_n)+":t="+std::to_string(tp);
  auto const* table=tp_.Find("ar_ns");
  auto const* found=table?table->Find(key):nullptr;
  if(!found)throw Sm90NativeMissing("no measured tensor parallel all-reduce for "+key);
  return Positive(*found,"tensor parallel all-reduce");
}
double Sm90NativeConstants::HeadExchange(int batch,int tp) const {
  auto key="b="+std::to_string(batch)+":t="+std::to_string(tp);
  auto const* table=tp_.Find("head_x_ns");
  auto const* found=table?table->Find(key):nullptr;
  if(!found)throw Sm90NativeMissing("no measured tensor parallel head exchange for "+key);
  return Positive(*found,"tensor parallel head exchange");
}
double Sm90NativeConstants::WgmmaEfficiency(int tile_n) const {
  auto const* table=forward_.Find("wgmma_efficiency");
  auto const* found=table?table->Find(std::to_string(tile_n)):nullptr;
  if(!found)throw Sm90NativeMissing("no measured WGMMA efficiency for N"+std::to_string(tile_n)+
      "; run the forward constants probe");
  return Positive(*found,"wgmma efficiency");
}
double Sm90NativeConstants::SmFlops() const {
  auto const* found=forward_.Find("sm_flops_per_ns");
  if(!found)throw Sm90NativeMissing("no measured SM throughput; run the forward constants probe");
  return Positive(*found,"sm throughput");
}
std::string Sm90NativeConstants::AmpereMmaKey(int grid,int tn,int tk,int wn,int stages) {
  return std::to_string(grid)+":"+std::to_string(tn)+":"+std::to_string(tk)+":"+
      std::to_string(wn)+":"+std::to_string(stages);
}
std::optional<double> Sm90NativeConstants::AmpereMmaK16(int grid,int tn,int tk,int wn,int stages) const {
  if(ampere_mma_.IsNull())return std::nullopt;
  auto key=AmpereMmaKey(grid,tn,tk,wn,stages);
  auto const* value=ampere_mma_.Find(key);
  if(!value)return std::nullopt;
  return Positive(*value,"Ampere MMA K16 slice");
}
std::string Sm80DenseServiceShape::Key() const {
  std::string key=epilogue;
  for(int v:{grid,n,k,batch,tn,tk,slots,warps_n,warps_k,register_stages,u,residual_tiles,
             int(input_norm),int(live_rows),copy_issue_group})key+=":"+std::to_string(v);
  return key;
}
std::optional<Sm80DenseService> Sm90NativeConstants::AmpereDenseService(Sm80DenseServiceShape const& shape) const {
  auto const* row=ampere_dense_service_.Find(shape.Key());
  if(!row)return std::nullopt;
  auto epilogue=row->At("consumer_epilogue_ns_per_task").AsNumber("dense epilogue");
  if(epilogue<0)throw std::invalid_argument("negative SM80 dense epilogue measurement");
  return Sm80DenseService{
      Positive(row->At("producer_issue_ns_per_tile"),"dense producer issue"),
      Positive(row->At("consumer_mma_ns_per_tile"),"dense consumer MMA and norm"),
      epilogue>0?std::optional<double>(epilogue):std::nullopt};
}
std::optional<Sm80DenseService> Sm90NativeConstants::AmpereGateService(
    int grid,int n,int k,int batch,int tn,int tk,int slots,int wn,int stages,int u) const {
  if(ampere_gate_service_.IsNull())return std::nullopt;
  std::string key=std::to_string(grid);
  for(int v:{n,k,batch,tn,tk,slots,wn,stages,u})key+=":"+std::to_string(v);
  auto const* row=ampere_gate_service_.Find(key);
  if(!row)return std::nullopt;
  return Sm80DenseService{
      Positive(row->At("producer_issue_ns_per_tile"),"dense gate producer issue"),
      Positive(row->At("consumer_mma_ns_per_tile"),"dense gate consumer MMA"),
      Positive(row->At("consumer_epilogue_ns_per_task"),"dense gate epilogue")};
}

bool Sm90NativeConstants::HasDenseGateService() const {
  return !ampere_gate_service_.IsNull() ||
      std::any_of(ampere_dense_shapes_.begin(),ampere_dense_shapes_.end(),
          [](auto const& shape) {return shape.epilogue=="swiglu" && shape.n/shape.tn>2*shape.grid;});
}

int Sm90NativeConstants::ClusterStar(int residency) const {
  auto const* r=clusters_.Find("r"+std::to_string(residency));
  auto const* star=r?r->Find("c_star"):nullptr;
  if(!star)throw Sm90NativeMissing("no measured cluster occupancy for residency "+std::to_string(residency));
  return int(star->AsNumber("c_star"));
}
double Sm90NativeConstants::AttentionMergeDsmem(int dim,int queries,int norm,int chunks) const {
  auto key="d"+std::to_string(dim)+":q"+std::to_string(queries)+":norm"+std::to_string(int(norm));
  auto const* shape=dsmem_.Find(key);
  auto const* found=shape?shape->Find(std::to_string(chunks)):nullptr;
  if(!found)throw Sm90NativeMissing("no measured DSMEM merge for "+key+" S"+std::to_string(chunks));
  return Positive(*found,"DSMEM merge");
}
json::Value const& Sm90NativeConstants::AttentionShape(int dim,int queries,int norm) const {
  auto key="d"+std::to_string(dim)+":q"+std::to_string(queries)+":norm"+std::to_string(int(norm));
  auto const* found=attention_.Find(key);
  if(!found)throw Sm90NativeMissing("no measured attention constants for "+key);
  return *found;
}
double Sm90NativeConstants::AttentionBlock(int dim,int queries,int norm) const {
  return Positive(AttentionShape(dim,queries,norm).At("block"),"attention block");
}
std::optional<double> Sm90NativeConstants::AttentionFresh(int dim,int queries,int norm) const {
  auto key="d"+std::to_string(dim)+":q"+std::to_string(queries)+":norm"+std::to_string(norm);
  auto const* row=ampere_attention_fresh_.Find(key);
  if(!row)return std::nullopt;
  return Positive(row->At("fresh_ns"),"attention new-row service");
}
double Sm90NativeConstants::AttentionPrepare(int dim,int queries,int norm) const {
  return Positive(AttentionShape(dim,queries,norm).At("prepare"),"attention preparation");
}
bool Sm90NativeConstants::HasAttentionMerge(int dim,int queries,int norm,int chunks) const {
  return AttentionShape(dim,queries,norm).At("merge").Find(std::to_string(chunks))!=nullptr;
}
double Sm90NativeConstants::AttentionMerge(int dim,int queries,int norm,int chunks) const {
  auto const* found=AttentionShape(dim,queries,norm).At("merge").Find(std::to_string(chunks));
  if(!found)throw Sm90NativeMissing("no measured attention merge for "+std::to_string(chunks)+" chunks");
  return Positive(*found,"attention merge");
}
std::vector<std::string> Sm90NativeConstants::StaleHeaders(std::string const& root) const {
  std::vector<std::string> result;
  for(auto const& item:section.At("header_sha256").AsObject("constant headers"))
    if(FileHash(root+"/"+item.first)!=item.second.AsString("header hash"))result.push_back(item.first);
  return result;
}
}  // namespace tilemega::solver
