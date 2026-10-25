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

double Sm90DenseCostModel::Gemm(Sm90DenseWorkload const& work,unsigned family,
    Sm90DenseGemm const& g,int residency,json::Value* terms,bool pre) const {
  auto const& c=constants_;
  int grid=Grid(residency),n=work.shapes[family][0],k=work.shapes[family][1],b=work.batch;
  double tasks=n/g.tile_n,per_cta=std::ceil(tasks/grid);
  double w_tot=2.0*n*k,a_tot=tasks*b*k*2.0;
  double w_cta=per_cta*g.tile_n*k*2.0,a_cta=per_cta*b*k*2.0;
  // D3: weights (bulk) and activations (TMA) share one CTA's async stream.
  double cta=(w_cta+a_cta)/c.sm_gbps;
  double ring=w_cta*c.dram_latency/(double(g.slots)*g.slot_bytes);
  double stream=std::max({w_tot/c.dram_gbps,(w_tot+a_tot)/c.l2_gbps,cta,ring});
  static constexpr char const* kEpilogue[]={"qkv","residual","gate_up","residual","lm_head"};
  double epilogue=c.Epilogue(kEpilogue[family],g.tile_n,b,g.warps_k);
  double first=pre?c.l2_latency:c.dram_latency;
  double total=c.Barrier(grid)+first+stream+epilogue;
  if(terms)*terms=json::Object{{"barrier",c.Barrier(grid)},{"first_byte",first},{"pre",pre},
      {"dram",w_tot/c.dram_gbps},{"l2",(w_tot+a_tot)/c.l2_gbps},{"cta",cta},{"ring",ring},
      {"epilogue",epilogue},{"tasks",tasks}};
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
  double compute=per_cta*(kv_task/(c.dram_gbps/grid)+blocks*c.AttentionBlock(d,q,norm));
  double stream=std::max(kv_tot/c.dram_gbps,compute);
  double merge=a.chunks>1?c.AttentionMerge(d,q,norm,a.chunks):0;
  double total=c.Barrier(grid)+c.l2_latency+c.AttentionPrepare(d,q,norm)+stream+merge;
  if(terms)*terms=json::Object{{"barrier",c.Barrier(grid)},{"prepare",c.AttentionPrepare(d,q,norm)},
      {"dram",kv_tot/c.dram_gbps},{"compute",compute},{"merge",merge}};
  return total;
}

Sm90DensePrediction Sm90DenseCostModel::Predict(Sm90DenseWorkload const& work,
                                                Sm90DenseCoordinates const& c) const {
  Sm90DensePrediction p;
  static constexpr char const* kKinds[]={"qkv","o","gate_up","down","lm_head"};
  // DT-1 model rule (prompt §6 D6): a GEMM stage whose predecessor is fully
  // resident takes pre; full-width attention breaks both of its boundaries.
  int arena=Sm90DenseArena(Sm90DenseSharedBudget(target_,c.residency));
  bool wide=c.attention.FullWidth(work.head_dim,work.queries,arena);
  auto resident=[&](unsigned f) {return c.gemms[f].FullyResident(work.shapes[f][1]);};
  std::array<bool,5> pre{c.weight_stream && resident(3),
      c.weight_stream && !wide && c.attention.FullyResident(575),
      c.weight_stream && resident(1),c.weight_stream && resident(2),c.weight_stream && resident(3)};
  for(unsigned f=0;f<5;++f) {
    // Layer 0's qkv follows the ring-less embedding.
    int instances=f==4?1:f==0 && pre[0]?work.layers-1:work.layers;
    Sm90DensePrediction::Stage s{kKinds[f],instances};
    s.ns=Gemm(work,f,c.gemms[f],c.residency,&s.terms,pre[f]);
    p.stages.push_back(std::move(s));
    if(f==0 && pre[0]) {
      Sm90DensePrediction::Stage first{kKinds[0],1};
      first.ns=Gemm(work,0,c.gemms[0],c.residency,&first.terms);
      p.stages.push_back(std::move(first));
    }
  }
  Sm90DensePrediction::Stage s{"attention",work.layers};
  s.ns=Attention(work,c.attention,c.residency,&s.terms);p.stages.push_back(std::move(s));
  int grid=Grid(c.residency);
  p.stages.push_back({"embedding",1,constants_.Embedding(grid,work.shapes[0][1],work.batch),json::Value()});
  p.stages.push_back({"argmax",1,constants_.Argmax(grid,work.shapes[4][0]/c.gemms[4].tile_n,work.batch),
      json::Value()});
  for(auto const& stage:p.stages)p.step_ns+=stage.instances*stage.ns;
  return p;
}

}  // namespace tilemega::solver
