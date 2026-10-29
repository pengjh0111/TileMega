// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <algorithm>
#include <cmath>

namespace tilemega::solver {

json::Value Sm90DensePrediction::ToJson() const {
  json::Array rows;
  for(auto const& s:stages)rows.emplace_back(json::Object{{"kind",s.kind},{"instances",s.instances},
      {"stage_ns",s.ns},{"terms",s.terms}});
  return json::Object{{"step_ns",step_ns},{"stages",rows}};
}

Sm90DenseCostModel::Sm90DenseCostModel(TargetSpec target,Sm90NativeConstants constants)
    :target_(std::move(target)),constants_(std::move(constants)) {
  if(target_.arch_tag!="sm_90" || target_.res.num_sms<=0)
    throw std::invalid_argument("native dense cost model requires the H800 target");
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
    Sm90DenseGemm const& g,int residency,json::Value* terms,bool pre) const {
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
  double stream=std::max({w_tot/c.dram_gbps,waves,(w_tot+a_tot)/c.l2_gbps,cta,ring});
  // A3: OLMoE's 64-expert stages key their own measurements; rows per slot
  // follow n_active(E), so Qwen3's E = 128 entries do not stand in for them.
  std::string geometry=family>=5 && work.experts==64?"_e64":"";
  double task_fixed=family>=6?
      per_cta*c.TaskExpert((family==6?"expert_gate_up":"expert_down")+geometry,b):0;
  static constexpr char const* kEpilogue[]={"qkv","residual","gate_up","residual","lm_head",
      "router","expert_gate_up","expert_down"};
  double epilogue=c.Epilogue(kEpilogue[family]+geometry,g.tile_n,b,g.warps_k);
  double first=pre?c.l2_latency:c.dram_latency;
  double total=c.Barrier(grid)+first+stream+epilogue+task_fixed;
  if(terms)*terms=json::Object{{"barrier",c.Barrier(grid)},{"first_byte",first},{"pre",pre},
      {"dram",w_tot/c.dram_gbps},{"waves",waves},{"l2",(w_tot+a_tot)/c.l2_gbps},{"cta",cta},{"ring",ring},
      {"epilogue",epilogue},{"task_fixed",task_fixed},{"tasks",tasks},{"multicast",g.multicast}};
  return total;
}

// Prompt §4 L0-2 attention formula for the D2 body at the midpoint past 575.
double Sm90DenseCostModel::Attention(Sm90DenseWorkload const& work,
    Sm90DenseAttention const& a,int residency,json::Value* terms) const {
  auto const& c=constants_;
  int grid=Grid(residency),d=work.head_dim,q=work.queries;bool norm=work.qk_norm;
  double past=575,tasks=double(work.batch)*work.hkv*a.chunks,per_cta=std::ceil(tasks/grid);
  double keys=std::ceil((past+1)/a.chunks),blocks=std::ceil(keys/64);
  double kv_task=keys*d*4,kv_tot=double(work.batch)*work.hkv*past*d*4;
  double compute=Waves(tasks,grid,kv_task)+per_cta*blocks*c.AttentionBlock(d,q,norm);
  double stream=std::max(kv_tot/c.dram_gbps,compute);
  // D10b: the DSMEM merge replaces the global partials and last arriver.
  double merge=a.chunks==1?0:a.dsmem?c.AttentionMergeDsmem(d,q,norm,a.chunks):
      c.AttentionMerge(d,q,norm,a.chunks);
  double total=c.Barrier(grid)+c.l2_latency+c.AttentionPrepare(d,q,norm)+stream+merge;
  if(terms)*terms=json::Object{{"barrier",c.Barrier(grid)},{"prepare",c.AttentionPrepare(d,q,norm)},
      {"dram",kv_tot/c.dram_gbps},{"compute",compute},{"merge",merge},{"dsmem",a.dsmem}};
  return total;
}

Sm90DensePrediction Sm90DenseCostModel::Predict(Sm90DenseWorkload const& work,
                                                Sm90DenseCoordinates const& c) const {
  Sm90DensePrediction p;
  // DT-1 model rule (prompt §6 D6): a GEMM stage whose predecessor is fully
  // resident takes pre; full-width attention breaks both of its boundaries.
  // MoE (§8): M5 never takes pre (M4's routes are unknown to its producer).
  int arena=Sm90DenseArena(Sm90DenseSharedBudget(target_,c.residency));
  bool wide=c.attention.FullWidth(work.head_dim,work.queries,arena);
  auto resident=[&](unsigned f) {return c.gemms[f].FullyResident(work.shapes[f][1]);};
  unsigned last=c.moe?7:3;  // the layer's final GEMM, before the next qkv
  std::array<bool,8> pre{};
  pre[0]=c.weight_stream && resident(last);
  pre[1]=c.weight_stream && !wide && c.attention.FullyResident(575);
  pre[4]=c.weight_stream && resident(last);
  if(c.moe) {pre[5]=c.weight_stream && resident(1);pre[7]=c.weight_stream && resident(6);}
  else {pre[2]=c.weight_stream && resident(1);pre[3]=c.weight_stream && resident(2);}
  for(unsigned f:c.Active()) {
    // Layer 0's qkv follows the ring-less embedding.
    int instances=f==4?1:f==0 && pre[0]?work.layers-1:work.layers;
    Sm90DensePrediction::Stage s{kSm90DenseFamilies[f],instances};
    s.ns=Gemm(work,f,c.gemms[f],c.residency,&s.terms,pre[f]);
    p.stages.push_back(std::move(s));
    if(f==0 && pre[0]) {
      Sm90DensePrediction::Stage first{kSm90DenseFamilies[0],1};
      first.ns=Gemm(work,0,c.gemms[0],c.residency,&first.terms);
      p.stages.push_back(std::move(first));
    }
  }
  int grid=Grid(c.residency);
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
  p.stages.push_back({"argmax_la",1,constants_.ArgmaxLa(grid,work.batch)-constants_.Barrier(grid),
      json::Value()});
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
