// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
namespace {
std::string AttentionKey(Sm90DenseWorkload const& w) {
  return std::string(w.batch<=4?"attention_simt":"attention_mma_sync")+
      "_d"+std::to_string(w.head_dim)+"_q"+std::to_string(w.queries)+
      "_norm"+std::to_string(int(w.qk_norm));
}
std::string GemmKey(Sm90DenseGemm const& g) {
  return "gemm_mma_sync_tn"+std::string(g.tile_n<=32?"8_32":g.tile_n<=64?"40_64":"72_plus");
}
double Interpolate(std::map<double,double> const& curve,double x,bool scale_tail=false) {
  auto upper=curve.lower_bound(x);
  if(upper==curve.begin())return upper->second;
  if(upper==curve.end()) {
    auto last=std::prev(upper);
    return last->second*(scale_tail?x/last->first:1);
  }
  auto lower=std::prev(upper);double fraction=(x-lower->first)/(upper->first-lower->first);
  return lower->second+(upper->second-lower->second)*fraction;
}
}
Sm90DenseCostModel::Sm90DenseCostModel(TargetSpec target,json::Value const& cal):target_(target) {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" ||
     cal.At("schema").AsNumber("native calibration schema")!=1)
    throw std::invalid_argument("native dense pricing requires calibrated sm90a schema 1");
  if(!cal.Find("flow_model") ||
     cal.At("flow_model").AsString("native flow model")!="overlapped_operands_v1")
    throw std::invalid_argument("native operand overlap requires refitted calibration");
  if(!cal.Find("issue_model") ||
     cal.At("issue_model").AsString("native issue model")!="issuer_exposure_v1")
    throw std::invalid_argument("native preissue exposure requires issuer calibration");
  for(auto const& raw:cal.At("observations").AsArray("native observations")) {
    auto v=raw;
    if(raw.kind()==json::Value::Kind::kArray) {
      auto const& columns=cal.At("observation_columns").AsArray("native observation columns");
      auto const& row=raw.AsArray("native observation row");
      if(row.size()!=columns.size())throw std::invalid_argument("native observation table width mismatch");
      json::Object fields;
      for(unsigned i=0;i<columns.size();++i)if(!row[i].IsNull())
        fields.emplace_back(columns[i].AsString("native observation column"),row[i]);
      v=fields;
    }
    if(v.At("processes").AsNumber("fresh process count")<3 ||
       v.At("median").AsNumber("measurement median")<=0)
      throw std::invalid_argument("native calibration requires positive multi-process measurements");
    if(v.At("metric").AsString("metric")=="native_ldmatrix_mma_k16" &&
       (!v.Find("production_compatible") ||
        !v.At("production_compatible").AsBool("production-compatible calibration")))
      throw std::invalid_argument("native mainloop calibration needs a matching resource audit");
    observations_[v.At("metric").AsString("native metric")].push_back(v);
    if(v.At("metric").AsString("metric")=="bulk_stream_per_active_cta" &&
       v.At("cache").AsString("cache")=="dram" &&
       v.At("grid").AsNumber("grid")>=target_.res.num_sms)
      dram_bw_=std::max(dram_bw_,v.At("median").AsNumber("bandwidth")*
          v.At("grid").AsNumber("grid"));
  }
  for(auto const& item:cal.At("body_fits").AsObject("native body fits")) {
    if(item.second.At("evidence").AsString("fit evidence")!="inferred" ||
       item.second.At("sample_count").AsNumber("fit count")<1)
      throw std::invalid_argument("native fit needs measured inputs and inferred coefficients");
    fits_.emplace(item.first,item.second.At("coefficients"));
  }
}
double Sm90DenseCostModel::Quantity(std::string const& metric,
    std::map<std::string,double> const& dims,std::map<std::string,std::string> const& filters) const {
  std::string cache_key=metric;
  for(auto const& d:dims)cache_key+='|'+d.first+'='+std::to_string(d.second);
  for(auto const& f:filters)cache_key+='|'+f.first+'='+f.second;
  auto cached=quantities_.find(cache_key);
  if(cached!=quantities_.end())return cached->second;
  auto it=observations_.find(metric);
  if(it==observations_.end())throw std::invalid_argument("missing native measurement: "+metric);
  double best=std::numeric_limits<double>::infinity(),result=0;
  std::vector<std::pair<double,double>> curve;
  for(auto const& row:it->second) {
    bool match=true;
    for(auto const& f:filters) {
      auto* value=row.Find(f.first);
      if(!value || value->AsString(f.first.c_str())!=f.second) {match=false;break;}
    }
    if(!match)continue;
    double distance=0;
    for(auto const& d:dims) {
      auto* value=row.Find(d.first);
      if(!value) {match=false;break;}
      double x=value->AsNumber(d.first.c_str());
      if((d.first=="head_dim" || d.first=="queries" || d.first=="qk_norm" ||
          d.first=="vector_sums") && x!=d.second) {match=false;break;}
      if(metric.rfind("attention_",0)==0 && d.first=="batch" &&
          (x<=4)!=(d.second<=4)) {match=false;break;}
      if(metric=="activation_inactive_rows_initialize" && d.first=="batch" &&
          x!=d.second) {match=false;break;}
      if(metric.find("_at_input_k")!=std::string::npos && d.first=="input_k" &&
          x!=d.second) {match=false;break;}
      if(d.first!="past")distance+=std::abs(std::log((x+1)/(d.second+1)));
    }
    if(match && distance<=best+1e-12) {
      if(distance<best-1e-12)curve.clear();
      best=distance;result=row.At("median").AsNumber("median");
      if(dims.count("past"))curve.emplace_back(row.At("past").AsNumber("past"),result);
    }
  }
  if(!curve.empty()) {
    std::sort(curve.begin(),curve.end());double past=dims.at("past");
    if(past<=curve.front().first)result=curve.front().second;
    else if(past>=curve.back().first)result=curve.back().second;
    else for(unsigned i=1;i<curve.size();++i)if(past<=curve[i].first) {
      double fraction=(past-curve[i-1].first)/(curve[i].first-curve[i-1].first);
      result=curve[i-1].second+(curve[i].second-curve[i-1].second)*fraction;break;
    }
  }
  if(!result)throw std::invalid_argument("no matching native measurement: "+metric);
  quantities_[cache_key]=result;return result;
}
double Sm90DenseCostModel::MainloopNs(Sm90DenseGemm const& g) const {
  std::string key="native-mainloop|"+std::to_string(g.tile_n)+'|'+std::to_string(g.TileK())+
      '|'+std::to_string(g.warps_n)+'|'+std::to_string(g.register_stages);
  auto cached=quantities_.find(key);if(cached!=quantities_.end())return cached->second;
  auto found=observations_.find("native_ldmatrix_mma_k16");
  if(found==observations_.end())throw std::invalid_argument("native mainloop calibration is missing");
  std::map<double,std::map<double,std::map<double,double>>> curves;
  for(auto const& row:found->second) {
    if(row.At("warps_n").AsNumber("warps_n")!=g.warps_n ||
       row.At("warps_k").AsNumber("warps_k")!=g.warps_k)continue;
    curves[row.At("register_stages").AsNumber("register_stages")]
        [row.At("tile_k").AsNumber("tile_k")][row.At("tile_n").AsNumber("tile_n")]=
        row.At("median").AsNumber("median");
  }
  if(curves.empty())throw std::invalid_argument("native warp partition is not calibrated");
  // These are measured shared-load + MMA chains per owned k16, not isolated
  // instruction throughput. Interpolation is inferred and checked by E-3.
  std::map<double,double> register_curve;
  for(auto const& r:curves) {
    std::map<double,double> k_curve;
    for(auto const& k:r.second)k_curve[k.first]=Interpolate(k.second,g.tile_n,true);
    register_curve[r.first]=Interpolate(k_curve,g.TileK());
  }
  double value=Interpolate(register_curve,g.register_stages);
  quantities_[key]=value;return value;
}
double Sm90DenseCostModel::Coefficient(std::string const& key,char const* name) const {
  auto found=fits_.find(key);
  if(found==fits_.end())throw std::invalid_argument("native body is not calibrated: "+key);
  double v=found->second.At(name).AsNumber(name);
  if(!std::isfinite(v) || v<0)throw std::invalid_argument("invalid native fitted coefficient");
  return v;
}
Sm90DensePrediction Sm90DenseCostModel::Predict(Sm90DenseWorkload const& w,
    Sm90DenseCoordinates const& c,int past) const {
  c.ValidateSchedule(w.shapes,w.batch,w.head_dim,w.queries,1088,
      std::min(target_.res.max_dynamic_smem_per_cta,target_.res.max_smem_per_sm/c.residency));
  if(w.layers<1 || past<64 || past>1086)throw std::invalid_argument("invalid native prediction workload");
  int grid=target_.res.num_sms*c.residency;
  double barrier=Quantity("v3_last_arrival_to_departure",{{"grid",grid}});
  // Saturating DRAM service is measured, not the stated H800 peak.
  double global_bw=dram_bw_;
  if(!global_bw)throw std::invalid_argument("native DRAM streaming calibration is missing");
  Sm90DensePrediction result;
  for(unsigned family:{0u,5u,1u,2u,3u,4u}) {
    Sm90DensePrediction::Stage s;s.instances=family==4?1:w.layers;s.barrier=barrier;
    if(family<5) {
      auto const& g=c.gemms[family];int n=w.shapes[family][0],k=w.shapes[family][1];
      s.kind=kSm90DenseFamilies[family];s.key=GemmKey(g);s.tasks=n/g.tile_n;
      s.waves=(s.tasks+grid-1)/grid;int active=std::min(grid,s.tasks);
      double bw=Quantity("bulk_stream_per_active_cta",{{"grid",active},
          {"slot_bytes",g.slot_bytes},{"slots",g.slots}},{{"cache","dram"}});
      s.dram=2.*n*k/global_bw;
      double tile_stream=2.*g.tile_n*k/bw;
      s.stream=s.waves*tile_stream;
      s.activation=s.waves*Quantity("activation_wait_and_shared_read",{{"batch",w.batch},
          {"tile_k",g.TileK()}})*k/g.TileK();
      // Bulk weights and cp.async activations occupy the same multi-slot ring.
      // Their calibrated service rates constrain throughput concurrently;
      // serially adding both charges an already overlapped operand twice.
      s.flow=std::max({s.dram,s.stream*Coefficient(s.key,"memory_scale"),
          Coefficient(s.key,"activation_scale")*s.activation});
      s.compute=s.waves*double(k/(16*g.warps_k))*MainloopNs(g)*
          Coefficient(s.key,"mma_scale");
      char const* op=family==0?"epilogue_store_return":family==2?"epilogue_swiglu_return":
          family==4?"epilogue_argmax_return":"epilogue_residual_return";
      std::map<std::string,double> epi_shape={{"batch",w.batch},{"tile_n",g.tile_n},
          {"vector_sums",c.dn_vector_sums}};
      // Native o/down consume already-formed attention/SwiGLU activations.
      // Their output sum-of-squares is included in the plain residual price;
      // only qkv, gate_up and lm_head receive an input norm-sums pointer.
      double epilogue;
      if(family==1 || family==3)epilogue=Quantity(op,epi_shape,{{"cache","plain"}});
      else {
        epi_shape["input_k"]=k;
        // Input reduction also has fixed shuffle and barrier costs. A measured
        // K-specific total must not be scaled as if all of it were a K loop.
        epilogue=Quantity(std::string(op)+"_at_input_k",epi_shape,{{"cache","dn"}});
      }
      s.epilogue=s.waves*epilogue*Coefficient(s.key,"epilogue_scale");
      // Run initializes all slots once per stage, even when this CTA has
      // several tasks. The producer never executes this branch at B16.
      s.initialization=w.batch<16?Quantity("activation_inactive_rows_initialize",
          {{"grid",grid},{"batch",w.batch},{"tile_k",g.TileK()},{"slots",g.slots}}):0;
      s.startup=Coefficient(s.key,"startup_ns")+s.initialization+
          Quantity("bulk_single_completion",{{"slot_bytes",g.slot_bytes}},{{"cache","dram"}});
      double fraction=double(g.preissue_smem_bytes)/(2.*g.tile_n*k);
      auto payload=backend::Sm90DenseLayout::GemmPayload(g.tile_n,g.TileK(),g.slots,k,
          g.preissue_smem_bytes).Shift(g.workspace_offset);
      backend::Sm90DenseInterval previous;
      if(family==1)previous=backend::Sm90DenseLayout::Attention(w.batch,w.head_dim,w.queries,
          c.attention.chunks,c.attention.block_keys,c.attention.slots,c.attention.new_row_work,
          1088).Live(c.attention.workspace_offset);
      else {
        auto const& prev=c.gemms[family==0 || family==4?3:family==2?1:2];
        previous=backend::Sm90DenseLayout::Gemm(prev.tile_n,prev.TileK(),prev.slots,
            prev.warps_k).Live(prev.workspace_offset);
      }
      // Incoming data can move DRAM service into the preceding stage, but
      // cannot remove this stage's MMA chain or inactive-row initialization.
      s.overlap_limit=s.startup-s.initialization+std::max(s.flow-s.compute,0.);
      (g.preissue_late || payload.Overlaps(previous)?s.late_service:s.preissue_service)=
          fraction*tile_stream;
      s.l2_service=std::min(double(g.prefetch_l2_bytes),
          2.*g.tile_n*k-g.preissue_smem_bytes)/bw;
      s.issue_count=(g.preissue_smem_bytes>0)+(g.prefetch_l2_bytes>0);
    }else {
      auto const& a=c.attention;s.kind="attention";s.key=AttentionKey(w);
      s.tasks=w.batch*8*a.chunks;s.waves=(s.tasks+grid-1)/grid;
      std::map<std::string,double> dimensions={{"batch",w.batch},{"head_dim",w.head_dim},
          {"queries",w.queries},{"qk_norm",w.qk_norm},{"chunks",a.chunks},
          {"block_keys",a.block_keys},{"slots",a.slots},{"past",past}};
      // The whole-stage calibration includes deterministic LA and wave costs.
      s.flow=Quantity("attention_critical_before_barrier",dimensions)*
          Coefficient(s.key,"body_scale");
      s.dram=4.*w.batch*8*past*w.head_dim/global_bw;
      s.compute=s.flow;s.startup=Coefficient(s.key,"startup_ns");
      double keys=double(past+1)/a.chunks;
      double fraction=std::min(1.,a.preissue_smem_bytes/(4.*keys*w.head_dim));
      auto payload=backend::Sm90DenseLayout::AttentionPayload(w.batch,w.head_dim,w.queries,
          a.chunks,a.block_keys,a.slots,a.new_row_work,1088,a.preissue_smem_bytes)
          .Shift(a.workspace_offset);
      auto const& prev=c.gemms[0];
      auto previous=backend::Sm90DenseLayout::Gemm(prev.tile_n,prev.TileK(),prev.slots,
          prev.warps_k).Live(prev.workspace_offset);
      s.overlap_limit=std::max(s.flow-s.dram,0.);
      (a.preissue_late || payload.Overlaps(previous)?s.late_service:s.preissue_service)=s.flow*fraction;
      double history_bytes=4.*past*w.head_dim/a.chunks;
      s.l2_service=std::min(double(a.prefetch_l2_bytes),
          std::max(0.,history_bytes-a.preissue_smem_bytes))/global_bw*std::min(grid,s.tasks);
      s.issue_count=(a.preissue_smem_bytes>0)+(a.prefetch_l2_bytes>0);
    }
    s.overlap=std::min(s.overlap_limit,
        s.preissue_service*Coefficient(s.key,"preissue_efficiency")+
        s.late_service*Coefficient(s.key,"tail_efficiency")+
        s.l2_service*Coefficient(s.key,"l2_efficiency"));
    s.ns=barrier+s.startup+std::max(s.flow,s.compute)+s.epilogue-s.overlap;
    result.stages.push_back(std::move(s));
  }
  for(auto kind:{"embedding","argmax"}) {
    Sm90DensePrediction::Stage s;s.kind=kind;s.key=std::string("native_")+kind;
    s.tasks=w.batch;s.waves=1;s.barrier=barrier;
    s.startup=Coefficient(s.key,"startup_ns");
    s.compute=Coefficient(s.key,"per_element_ns")*(s.kind=="embedding"?
        w.shapes[0][1]:(w.shapes[4][0]+c.gemms[4].tile_n-1)/c.gemms[4].tile_n);
    s.ns=s.barrier+s.startup+s.compute;result.stages.push_back(s);
  }
  // A stage's incoming transfer is issued by its predecessor. The first qkv
  // follows embedding; the final down issues lm_head rather than another qkv.
  // Class means retain both boundary cases without charging the receiver.
  auto& stages=result.stages;
  std::array<double,6> incoming{};
  for(unsigned i=0;i<incoming.size();++i)incoming[i]=stages[i].issue_count;
  for(unsigned i=0;i<4;++i)stages[i].issue_count=incoming[i+1];
  stages[4].issue_count=(incoming[0]*(w.layers-1)+incoming[5])/w.layers;
  stages[5].issue_count=0;stages[6].issue_count=incoming[0];
  for(auto& s:stages) {
    // An attention producer can hide weight issue behind compute, while a
    // streaming GEMM exposes competing traffic. Price the issuing body.
    s.issue=s.issue_count? s.issue_count*Coefficient(s.key,"issue_ns"):0;
    s.ns+=s.issue;result.step_ns+=s.ns*s.instances;
  }
  return result;
}
Sm90DensePrediction Sm90DenseCostModel::Request(Sm90DenseWorkload const& w,
    Sm90DenseCoordinates const& c) const {
  Sm90DensePrediction result;
  // Midpoints integrate all 1023 steady decode positions without treating
  // capacity as the amount of work. Class totals and ranking use this same integral.
  constexpr int intervals=16;
  for(int i=0;i<intervals;++i) {
    int past=64+int((i+.5)*1023/intervals);auto sample=Predict(w,c,past);
    if(!i) {
      result=sample;result.step_ns=0;
      for(auto& s:result.stages)
        s.ns=s.barrier=s.startup=s.initialization=s.dram=s.stream=s.flow=
            s.activation=s.compute=s.epilogue=s.overlap=s.issue=
            s.overlap_limit=s.preissue_service=s.late_service=s.l2_service=s.issue_count=0;
    }
    for(unsigned j=0;j<sample.stages.size();++j) {
      auto& out=result.stages[j];auto const& in=sample.stages[j];
      out.ns+=in.ns/intervals;
      out.barrier+=in.barrier/intervals;out.startup+=in.startup/intervals;
      out.initialization+=in.initialization/intervals;
      out.dram+=in.dram/intervals;out.flow+=in.flow/intervals;
      out.stream+=in.stream/intervals;
      out.activation+=in.activation/intervals;
      out.compute+=in.compute/intervals;out.epilogue+=in.epilogue/intervals;
      out.overlap+=in.overlap/intervals;
      out.issue+=in.issue/intervals;
      out.overlap_limit+=in.overlap_limit/intervals;
      out.preissue_service+=in.preissue_service/intervals;
      out.late_service+=in.late_service/intervals;
      out.l2_service+=in.l2_service/intervals;
      out.issue_count+=in.issue_count/intervals;
    }
    result.step_ns+=sample.step_ns/intervals;
  }
  return result;
}
json::Value Sm90DensePrediction::ToJson() const {
  json::Array values;
  for(auto const& s:stages)values.emplace_back(json::Object{{"kind",s.kind},{"key",s.key},
      {"instances",s.instances},{"tasks",s.tasks},{"waves",s.waves},
      {"stage_ns",s.ns},{"class_ns",s.ns*s.instances},{"barrier_ns",s.barrier},
      {"startup_ns",s.startup},{"initialization_ns",s.initialization},
      {"dram_ns",s.dram},{"flow_ns",s.flow},
      {"stream_ns",s.stream},
      {"activation_ns",s.activation},
      {"compute_ns",s.compute},{"epilogue_ns",s.epilogue},{"overlap_ns",s.overlap},
      {"issue_ns",s.issue},{"overlap_limit_ns",s.overlap_limit},
      {"preissue_service_ns",s.preissue_service},{"late_service_ns",s.late_service},
      {"l2_service_ns",s.l2_service},{"issue_count",s.issue_count}});
  return json::Object{{"step_ns",step_ns},{"evidence","inferred prediction from native calibration"},
      {"stages",values}};
}
}  // namespace tilemega::solver
