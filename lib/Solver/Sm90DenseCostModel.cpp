// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseAttentionPartition.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace tilemega::solver {
namespace {
constexpr int kFirstDecodePast=64,kLastDecodePast=1086;
}

json::Value Sm90DensePrediction::ToJson() const {
  json::Array rows;
  for(auto const& s:stages)rows.emplace_back(json::Object{{"kind",s.kind},{"instances",s.instances},
      {"stage_ns",s.ns},{"terms",s.terms}});
  return json::Object{{"step_ns",step_ns},{"stages",rows}};
}

Sm90DenseCostModel::Sm90DenseCostModel(TargetSpec target,Sm90NativeConstants constants)
    :target_(std::move(target)),constants_(std::move(constants)) {
  if((target_.arch_tag!="sm_90" && target_.arch_tag!="sm_80") || target_.res.num_sms<=0)
    throw std::invalid_argument("native dense cost model requires an SM80 or SM90 target");
}

bool Sm90DenseCostModel::HasCalibratedMma(Sm90DenseGemm const& g,int residency) const {
  return target_.arch_tag!="sm_80" ||
      constants_.AmpereMmaK16(Grid(residency),g.tile_n,g.TileK(),g.warps_n,g.register_stages).has_value();
}

bool Sm90DenseCostModel::HasCalibratedMma(Sm90DenseCoordinates const& c) const {
  for(unsigned f:c.Active())if(!HasCalibratedMma(c.gemms[f],c.residency))return false;
  return true;
}

double Sm90DenseCostModel::MmaK16Ns(Sm90DenseGemm const& g,int residency) const {
  if(target_.arch_tag!="sm_80")return 0;  // SM90's existing shared service term.
  auto measured=constants_.AmpereMmaK16(Grid(residency),g.tile_n,g.TileK(),g.warps_n,g.register_stages);
  if(!measured)throw Sm90NativeMissing("no measured SM80 MMA mainloop for "+
      Sm90NativeConstants::AmpereMmaKey(Grid(residency),g.tile_n,g.TileK(),g.warps_n,g.register_stages));
  return *measured;
}

std::optional<Sm80DenseService> Sm90DenseCostModel::DenseService(
    Sm90DenseWorkload const& work,unsigned family,Sm90DenseGemm const& g,int residency,
    Sm90DenseCoordinates const* coordinates) const {
  if(target_.arch_tag!="sm_80" || family>3 || (work.moe && family==2) || g.multicast ||
     (coordinates && coordinates->weight_stream))return std::nullopt;
  bool residual=Sm90DenseCoordinates::ResidualFamily(family);
  bool gate=family==2;
  int u=gate?(coordinates?coordinates->swiglu_u:Sm90DenseSearchSpace::DefaultGateInterleave(g.tile_n)):16;
  bool live=coordinates && coordinates->live_epilogue_rows && work.batch<16 && (gate || g.warps_k>1);
  int copy=coordinates && gate?coordinates->copy_issue_group:0;
  int grid=Grid(residency),n=work.shapes[family][0],k=work.shapes[family][1];
  Sm80DenseServiceShape shape{gate?"swiglu":residual?"residual":"store",grid,n,k,work.batch,
      g.tile_n,g.TileK(),g.slots,g.warps_n,g.warps_k,g.register_stages,u,
      g.ResidualTiles(residual,n,grid),!residual,live,copy};
  if(auto measured=constants_.AmpereDenseService(shape))return measured;
  if(gate && !live && !copy)
    return constants_.AmpereGateService(grid,n,k,work.batch,g.tile_n,g.TileK(),g.slots,
        g.warps_n,g.register_stages,u);
  return std::nullopt;
}

std::optional<Sm80FusedExpertService> Sm90DenseCostModel::FusedService(
    Sm90DenseWorkload const& work,Sm90DenseCoordinates const& c) const {
  if(target_.arch_tag!="sm_80" || !work.moe || work.top_k!=8 || !c.expert_fuse ||
     c.residency!=1 || c.reuse_payload || c.weight_stream || c.GlobalRoute() || c.combine_stage)
    return std::nullopt;
  auto const& gu=c.gemms[6];
  auto const& dn=c.gemms[7];
  return constants_.AmpereFusedService({work.batch,work.shapes[6][0],work.shapes[7][0],
      work.shapes[6][1],gu.tile_n,gu.TileK(),dn.tile_n,dn.TileK(),gu.slots,
      gu.warps_n,dn.warps_n,work.experts,work.float_routes,
      frontend::NativeMoeSharedExpertPageSide(work.experts),gu.register_stages,
      work.softmax_all,c.expert_page_segments,c.live_epilogue_rows,c.copy_issue_group,
      c.expert_gu_route_bases});
}

std::optional<Sm80DenseService> Sm90DenseCostModel::GemmService(
    Sm90DenseWorkload const& work,unsigned family,Sm90DenseGemm const& g,int residency,
    Sm90DenseCoordinates const* coordinates) const {
  if(family<6)return DenseService(work,family,g,residency,coordinates);
  if(!coordinates)return std::nullopt;
  auto fused=FusedService(work,*coordinates);
  if(!fused)return std::nullopt;
  if(family==6)return Sm80DenseService{fused->gu_issue,fused->gu_mma,fused->gu_epilogue};
  int tiles=work.shapes[family][1]/g.TileK();
  int ahead=std::min(tiles,g.slots);
  double producer=fused->dn_preissue+fused->dn_activation+(tiles-ahead)*fused->dn_issue;
  return Sm80DenseService{producer/tiles,fused->dn_mma,fused->dn_epilogue};
}

// MR6: a CTA with n = ceil(t / G) tasks streams its first n - 1 at the shared
// rate BW_dram / G; in the last wave only r = t - (n - 1) G CTAs are active,
// each at min(BW_cta_dram, BW_dram / r).
double Sm90DenseCostModel::Waves(double tasks,int grid,double task_bytes) const {
  auto const& c=constants_;
  double n=std::ceil(tasks/grid),last=tasks-(n-1)*grid;
  return (n-1)*task_bytes/(c.dram_gbps/grid)+task_bytes/std::min(c.CtaDramGbps(),c.dram_gbps/last);
}

double Sm90DenseCostModel::Gemm(Sm90DenseWorkload const& work,unsigned family,
    Sm90DenseGemm const& g,int residency,json::Value* terms,bool pre,Sm90DenseCoordinates const* coordinates) const {
  auto const& c=constants_;
  int grid=Grid(residency),n=work.shapes[family][0],k=work.shapes[family][1],b=work.batch;
  // MoE experts (prompt §8): n_active x N/TN tasks over n_active expert
  // stacks; gate_up gathers each expert's rows, down reads R_A-row boxes.
  // MR6: n_active = E (1 - (1 - k/E)^B) from the workload.
  double active=family>=6?work.experts*(1-std::pow(1-double(work.top_k)/work.experts,b)):1;
  double rows=family==6?b*double(work.top_k)/active:family==7?(b<=8?8:16):b;
  double tasks=n/g.tile_n*active,per_cta=std::ceil(tasks/grid);
  double w_tot=2.0*n*k*active,a_tot=tasks*rows*k*2.0;
  double w_cta=per_cta*g.tile_n*k*2.0,a_cta=per_cta*rows*k*2.0;
  // D10c: each CTA of a multicast pair issues half the A slots, and L2
  // serves every A slot once per pair.
  if(g.multicast) {a_tot/=2;a_cta/=2;}
  // D3: weights (bulk) and activations (TMA) share one CTA's async stream.
  double cta=(w_cta+a_cta)/c.sm_gbps;
  double ring=w_cta*c.dram_latency/(double(g.slots)*g.slot_bytes);
  double waves=Waves(tasks,grid,2.0*g.tile_n*k);
  double arithmetic=MmaK16Ns(g,residency)*per_cta*k/(16*g.warps_k);
  auto service=GemmService(work,family,g,residency,coordinates);
  double producer=0;
  if(service) {
    double tiles=per_cta*k/g.TileK();
    producer=service->producer_issue_ns*tiles;
    arithmetic=service->consumer_mma_ns*tiles;
  }
  double stream=std::max({w_tot/c.dram_gbps,waves,(w_tot+a_tot)/c.l2_gbps,cta,ring,arithmetic,producer});
  // A3: OLMoE's 64-expert stages key their own measurements; rows per slot
  // follow n_active(E), so Qwen3's E = 128 entries do not stand in for them.
  std::string geometry=family>=5 && work.experts==64?"_e64":"";
  double task_fixed=family>=6?
      per_cta*c.TaskExpert((family==6?"expert_gate_up":"expert_down")+geometry,b):0;
  static constexpr char const* kEpilogue[]={"qkv","residual","gate_up","residual","lm_head",
      "router","expert_gate_up","expert_down"};
  double epilogue=service && service->epilogue_ns?*service->epilogue_ns*per_cta:
      c.Epilogue(kEpilogue[family]+geometry,g.tile_n,b,g.warps_k);
  double first=pre?c.l2_latency:c.dram_latency;
  // Tensor parallel o/down sum their tile across ranks inside the epilogue.
  double all_reduce=work.tp>1 && (family==1 || family==3)?c.AllReduce(b,g.tile_n,work.tp):0;
  double total=c.Barrier(grid)+first+stream+epilogue+task_fixed+all_reduce;
  if(terms) {
    json::Object fields{{"barrier",c.Barrier(grid)},{"first_byte",first},{"pre",pre},
        {"dram",w_tot/c.dram_gbps},{"waves",waves},{"l2",(w_tot+a_tot)/c.l2_gbps},{"cta",cta},{"ring",ring},
        {"epilogue",epilogue},{"task_fixed",task_fixed},{"tasks",tasks},{"multicast",g.multicast}};
    if(target_.arch_tag=="sm_80")fields.emplace_back("mma",arithmetic);
    if(service)fields.emplace_back("producer_issue",producer);
    if(work.tp>1 && (family==1 || family==3))fields.emplace_back("all_reduce",all_reduce);
    *terms=std::move(fields);
  }
  return total;
}

Sm90DenseHistoryWork Sm90DenseCostModel::AttentionHistory(Sm90DenseAttention const& a,int past,int chunk) const {
  auto range=backend::Sm90DensePartitionRange(past,chunk,a.chunks,a.new_row_work);
  int keys=std::min(range.end,past)-range.begin;
  int slots=(keys+a.block_keys-1)/a.block_keys;
  return {double(keys),double(slots*(a.block_keys/64)),double(slots),range.Contains(past)};
}

std::vector<Sm80AttentionScenario> const& Sm90DenseCostModel::AttentionScenarios(
    Sm90DenseAttention const& a,bool exhaustive) const {
  if(exhaustive) {
    static auto const positions=[] {
      std::vector<Sm80AttentionScenario> rows;
      for(int past=kFirstDecodePast;past<=kLastDecodePast;++past)rows.push_back({past,1});
      return rows;
    }();
    return positions;
  }
  auto key=std::make_tuple(a.chunks,a.block_keys,a.new_row_work);
  static thread_local std::map<decltype(key),std::vector<Sm80AttentionScenario>> cache;
  auto found=cache.find(key);
  if(found!=cache.end())return found->second;
  a.Validate();
  auto signature=[&](int past) {
    std::vector<int> slots;
    for(int chunk=0;chunk<a.chunks;++chunk) {
      auto history=AttentionHistory(a,past,chunk);
      slots.push_back(2*int(history.ring_slots)+int(history.fresh));
    }
    return slots;
  };
  std::vector<Sm80AttentionScenario> samples;
  int begin=kFirstDecodePast;
  auto shape=signature(begin);
  auto emit=[&](int end) {
    int count=end-begin+1;
    if(count==1)samples.push_back({begin,1});
    else {
      // Endpoint quadrature within a constant per-chunk slot signature.
      // Keep both actual byte counts and both sides of every slot boundary.
      samples.push_back({begin,count/2});
      samples.push_back({end,count-count/2});
    }
  };
  for(int past=kFirstDecodePast+1;past<=kLastDecodePast;++past) {
    auto next=signature(past);
    if(next==shape)continue;
    emit(past-1);
    begin=past;shape=std::move(next);
  }
  emit(kLastDecodePast);
  return cache.emplace(key,std::move(samples)).first->second;
}

Sm90DenseHistoryWork Sm90DenseCostModel::AttentionHistory(Sm90DenseAttention const& a) const {
  if(target_.arch_tag!="sm_80") {
    double keys=std::ceil(576./a.chunks),blocks=std::ceil(keys/64);
    return {keys,blocks,blocks};
  }
  a.Validate();
  auto key=std::make_tuple(a.chunks,a.block_keys,a.new_row_work);
  static thread_local std::map<decltype(key),Sm90DenseHistoryWork> cache;
  auto found=cache.find(key);
  if(found!=cache.end())return found->second;
  std::int64_t slots=0,keys=0;
  for(int past=kFirstDecodePast;past<=kLastDecodePast;++past)for(int chunk=0;chunk<a.chunks;++chunk) {
    auto history=AttentionHistory(a,past,chunk);
    keys+=std::int64_t(history.keys);
    slots+=std::int64_t(history.ring_slots);
  }
  double count=(kLastDecodePast-kFirstDecodePast+1)*a.chunks;
  // The consumer issues every 64-key subdivision of a loaded slot, even
  // when that subdivision contains only masked keys.
  return cache.emplace(key,Sm90DenseHistoryWork{keys/count,slots/count*(a.block_keys/64),slots/count}).first->second;
}

double Sm90DenseCostModel::AmpereAttentionStream(Sm90DenseWorkload const& work,
    Sm90DenseAttention const& a,int grid,int norm) const {
  int groups=work.batch*work.hkv,d=work.head_dim,q=work.queries;
  auto const& c=constants_;
  int tasks=groups*a.chunks;
  double bytes_per_key=4.*d;
  double rate=std::min(c.CtaDramGbps(),a.slots*a.block_keys*bytes_per_key/c.dram_latency);
  double mma=c.AttentionBlock(d,q,norm),fresh=c.AttentionFresh(d,q,norm).value_or(0);
  auto key=std::make_tuple(a.chunks,a.block_keys,a.new_row_work,grid,groups,d,rate,c.dram_gbps,mma,fresh);
  static thread_local std::map<decltype(key),double> cache;
  auto existing=cache.find(key);
  if(existing!=cache.end())return existing->second;
  // CTAs in either wave stratum with the same chunk residue own the same work.
  int tail=tasks%grid,active=std::min(tasks,grid);
  std::vector<int> owners;
  for(int worker=0;worker<std::min(tail,a.chunks);++worker)owners.push_back(worker);
  for(int worker=tail;worker<std::min(active,tail+a.chunks);++worker)owners.push_back(worker);
  std::vector<Sm90DenseHistoryWork> chunks(a.chunks);
  double total=0;
  for(int past=kFirstDecodePast;past<=kLastDecodePast;++past) {
    for(int chunk=0;chunk<a.chunks;++chunk)chunks[chunk]=AttentionHistory(a,past,chunk);
    double longest=0;
    for(int worker:owners) {
      double service=0;
      for(int task=worker;task<tasks;task+=grid) {
        auto const& chunk=chunks[task%a.chunks];
        int active_wave=tail && task>=tasks-tail?tail:active;
        service+=chunk.keys*bytes_per_key/std::min(rate,c.dram_gbps/active_wave)+
            chunk.mma_blocks*mma+chunk.fresh*fresh;
      }
      longest=std::max(longest,service);
    }
    total+=std::max(longest,groups*past*bytes_per_key/c.dram_gbps);
  }
  return cache.emplace(key,total/(kLastDecodePast-kFirstDecodePast+1)).first->second;
}

// The SM80 range follows the production history partition and tile padding.
double Sm90DenseCostModel::Attention(Sm90DenseWorkload const& work,
    Sm90DenseAttention const& a,int residency,json::Value* terms) const {
  auto const& c=constants_;
  int grid=Grid(residency),d=work.head_dim,q=work.queries;
  // Attention constants are keyed by the norm mode: 0 none, 1 per head, 2 per row.
  int norm=work.qk_norm?(work.qk_norm_row?2:1):0;

  double tasks=double(work.batch)*work.hkv*a.chunks,per_cta=std::ceil(tasks/grid);
  auto history=AttentionHistory(a);
  double kv_task=history.keys*d*4,kv_tot=double(work.batch)*work.hkv*575*d*4;
  double mma=c.AttentionBlock(d,q,norm);
  double compute=Waves(tasks,grid,kv_task)+per_cta*history.mma_blocks*mma;
  double stream=std::max(kv_tot/c.dram_gbps,compute);
  if(target_.arch_tag=="sm_80")stream=AmpereAttentionStream(work,a,grid,norm);
  // D10b: the DSMEM merge replaces the global partials and last arriver.
  double merge=a.chunks==1?0:a.dsmem?c.AttentionMergeDsmem(d,q,norm,a.chunks):
      c.AttentionMerge(d,q,norm,a.chunks);
  double total=c.Barrier(grid)+c.l2_latency+c.AttentionPrepare(d,q,norm)+stream+merge;
  if(terms)*terms=json::Object{{"barrier",c.Barrier(grid)},{"prepare",c.AttentionPrepare(d,q,norm)},
      {"dram",kv_tot/c.dram_gbps},{"compute",compute},{"merge",merge},{"dsmem",a.dsmem},
      {"history_keys",history.keys},{"history_mma_blocks",history.mma_blocks},{"history_ring_slots",history.ring_slots},
      {"request_weighted_stream_ns",stream}};
  return total;
}

Sm90DensePrediction Sm90DenseCostModel::Predict(Sm90DenseWorkload const& work,
                                                Sm90DenseCoordinates const& c) const {
  Sm90DensePrediction p;
  // DT-1 model rule (prompt §6 D6): a GEMM stage whose predecessor is fully
  // resident takes pre; full-width attention breaks both of its boundaries.
  // MoE (§8): M5 never takes pre (M4's routes are unknown to its producer).
  int arena=Sm90DenseArena(Sm90DenseSharedBudget(target_,c.residency),c.reuse_payload);
  bool wide=c.attention.FullWidth(work.head_dim,work.queries,arena);
  auto resident=[&](unsigned f) {return c.gemms[f].FullyResident(work.shapes[f][1]);};
  unsigned last=c.moe?7:3;  // the layer's final GEMM, before the next qkv
  std::array<bool,8> pre{};
  pre[0]=c.weight_stream && resident(last);
  pre[1]=c.weight_stream && !wide && c.attention.FullyResident(575);
  pre[4]=c.weight_stream && resident(last);
  if(c.moe) {pre[5]=c.weight_stream && resident(1);pre[7]=c.weight_stream && resident(6) && !c.expert_fuse;}
  else {pre[2]=c.weight_stream && resident(1);pre[3]=c.weight_stream && resident(2);}
  for(unsigned f:c.Active()) {
    // Layer 0's qkv follows the ring-less embedding.
    int instances=f==4?1:f==0 && pre[0]?work.layers-1:work.layers;
    Sm90DensePrediction::Stage s{kSm90DenseFamilies[f],instances};
    s.ns=Gemm(work,f,c.gemms[f],c.residency,&s.terms,pre[f],&c);
    p.stages.push_back(std::move(s));
    if(f==0 && pre[0]) {
      Sm90DensePrediction::Stage first{kSm90DenseFamilies[0],1};
      first.ns=Gemm(work,0,c.gemms[0],c.residency,&first.terms);
      p.stages.push_back(std::move(first));
    }
  }
  int grid=Grid(c.residency);
  // MR7: M56 runs both expert task lists in one stage. Fusion removes the M5
  // -> M6 grid barrier, and the down producer issues its first weight slots
  // before it waits on gu_done, so M6's first-byte latency overlaps M5's tail.
  if(c.expert_fuse) {
    auto at=[&](char const* kind) {
      return std::find_if(p.stages.begin(),p.stages.end(),[&](auto const& s) {return s.kind==kind;});
    };
    auto gu=at("expert_gate_up"),dn=at("expert_down");
    Sm90DensePrediction::Stage fused{"expert_fused",work.layers,
        gu->ns+dn->ns-constants_.Barrier(grid)-constants_.dram_latency,
        json::Object{{"gate_up",gu->terms},{"down",dn->terms},{"barrier_removed",constants_.Barrier(grid)},
                     {"first_byte_overlapped",constants_.dram_latency}}};
    p.stages.erase(dn);p.stages.erase(at("expert_gate_up"));
    p.stages.push_back(std::move(fused));
  }
  Sm90DensePrediction::Stage s{"attention",work.layers};
  s.ns=Attention(work,c.attention,c.residency,&s.terms);
  // D8: the boundary before attention is a group counter wait, not V3.
  if(c.group_wait)s.ns-=constants_.GroupWaitSaving(grid);
  p.stages.push_back(std::move(s));
  if(c.moe) {
    // M4's last arriver (top-k and the schedule, beyond its barrier); M6's
    // per-tile last arriver reads P and x' and writes x''.
    p.stages.push_back({"route_la",work.layers,
        constants_.RouteLa(grid,work.batch,work.experts,work.softmax_all)-constants_.Barrier(grid),
        json::Value()});
    double tile=c.gemms[7].tile_n*double(work.batch);
    p.stages.push_back({"combine_la",work.layers,
        constants_.l2_latency+(9*tile*4+tile*2)/constants_.sm_gbps,json::Value()});
  }
  // D9: the head's last arriver reduces the argmax (beyond the head's own
  // barrier) and, with emb = LA, reads B table rows and writes them to x.
  p.stages.push_back({"argmax_la",1,constants_.ArgmaxLa(grid,work.batch)-constants_.Barrier(grid)+
      (work.tp>1?constants_.HeadExchange(work.batch,work.tp):0),json::Value()});
  double row_bytes=2.0*work.shapes[0][1]*work.batch;
  p.stages.push_back(c.embed_la?
      Sm90DensePrediction::Stage{"embedding_la",1,constants_.dram_latency+2*row_bytes/constants_.sm_gbps,
          json::Value()}:
      Sm90DensePrediction::Stage{"embedding",1,constants_.Embedding(grid,work.shapes[0][1],work.batch),
          json::Value()});
  for(auto const& stage:p.stages)p.step_ns+=stage.instances*stage.ns;
  return p;
}

}  // namespace tilemega::solver
