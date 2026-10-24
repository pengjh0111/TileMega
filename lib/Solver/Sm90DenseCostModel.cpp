// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCostModel.h>
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
    Sm90DenseGemm const& g,int residency,bool vector_sums,json::Value* terms) const {
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
  bool norm=family==0 || family==2 || family==4;
  double epilogue=c.Epilogue(kEpilogue[family],g.tile_n,b,norm?k:0,vector_sums);
  // Prompt §4 L0-2: no preissue in L0, so the DRAM first byte is exposed.
  double total=c.Barrier(grid)+c.l2_latency+(c.dram_latency-c.l2_latency)+stream+epilogue;
  if(terms)*terms=json::Object{{"barrier",c.Barrier(grid)},{"first_byte",c.dram_latency},
      {"dram",w_tot/c.dram_gbps},{"l2",(w_tot+a_tot)/c.l2_gbps},{"cta",cta},{"ring",ring},
      {"epilogue",epilogue},{"tasks",tasks}};
  return total;
}

std::optional<double> Sm90DenseCostModel::Attention(Sm90DenseWorkload const& work,
    Sm90DenseAttention& a,int residency,json::Value* terms) const {
  // Before D2 the stage time is the measured critical path of the R2 bodies.
  int grid=Grid(residency);
  auto entry=constants_.Attention(work.batch,work.head_dim,work.queries,work.qk_norm,
      a.chunks,grid,a.block_keys);
  if(!entry)return std::nullopt;
  if(work.batch>=8)a.slots=entry->slots;
  a.new_row_work=a.chunks==1?0:entry->new_row_work;
  double total=constants_.Barrier(grid)+entry->ns;
  if(terms)*terms=json::Object{{"barrier",constants_.Barrier(grid)},{"table",entry->ns}};
  return total;
}

Sm90DensePrediction Sm90DenseCostModel::Predict(Sm90DenseWorkload const& work,
                                                Sm90DenseCoordinates const& c) const {
  Sm90DensePrediction p;
  static constexpr char const* kKinds[]={"qkv","o","gate_up","down","lm_head"};
  for(unsigned f=0;f<5;++f) {
    Sm90DensePrediction::Stage s{kKinds[f],f==4?1:work.layers};
    s.ns=Gemm(work,f,c.gemms[f],c.residency,c.dn_vector_sums,&s.terms);
    p.stages.push_back(std::move(s));
  }
  auto attention=c.attention;
  Sm90DensePrediction::Stage s{"attention",work.layers};
  auto ns=Attention(work,attention,c.residency,&s.terms);
  if(!ns)throw Sm90NativeMissing("attention coordinate has no measured stage time");
  s.ns=*ns;p.stages.push_back(std::move(s));
  int grid=Grid(c.residency);
  p.stages.push_back({"embedding",1,constants_.Embedding(grid,work.shapes[0][1],work.batch),json::Value()});
  p.stages.push_back({"argmax",1,constants_.Argmax(grid,work.shapes[4][0]/c.gemms[4].tile_n,work.batch),
      json::Value()});
  for(auto const& stage:p.stages)p.step_ns+=stage.instances*stage.ns;
  return p;
}

}  // namespace tilemega::solver
