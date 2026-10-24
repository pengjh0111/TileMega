// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
namespace {
std::map<std::string,double> AttentionDimensions(Sm90DenseWorkload const& w,
    Sm90DenseCoordinates const& c,int grid,int past) {
  auto const& a=c.attention;
  return {{"batch",w.batch},{"grid",grid},{"head_dim",w.head_dim},
      {"queries",w.queries},{"qk_norm",w.qk_norm},{"chunks",a.chunks},
      {"block_keys",w.batch<=4?64:a.block_keys},{"slots",w.batch<=4?3:a.slots},
      {"past",past},{"new_row_work",a.chunks==1?0:a.new_row_work}};
}
std::string AttentionKey(Sm90DenseWorkload const& w) {
  return std::string(w.batch<=4?"attention_simt":"attention_mma_sync")+
      "_d"+std::to_string(w.head_dim)+"_q"+std::to_string(w.queries)+
      "_norm"+std::to_string(int(w.qk_norm));
}
std::string GemmKey(Sm90DenseGemm const& g) {
  return "gemm_mma_sync_tn"+std::string(g.tile_n<=32?"8_32":g.tile_n<=64?"40_64":"72_plus")+
      "_wk"+std::to_string(g.warps_k);
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
double Median(std::vector<double> values) {
  std::sort(values.begin(),values.end());auto n=values.size();
  return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;
}
}
Sm90DenseMissingMeasurement::Sm90DenseMissingMeasurement(std::string const& message,
    std::string metric,std::map<std::string,double> dimensions,
    std::map<std::string,std::string> labels)
    :std::invalid_argument(message),metric(std::move(metric)),dimensions(std::move(dimensions)),
     labels(std::move(labels)) {}
json::Value Sm90DenseMissingMeasurement::ToJson() const {
  json::Object dims,filters;
  for(auto const& value:dimensions)dims.emplace_back(value.first,value.second);
  for(auto const& value:labels)filters.emplace_back(value.first,value.second);
  return json::Object{{"metric",metric},{"dimensions",dims},{"labels",filters}};
}
Sm90DenseCostModel::Sm90DenseCostModel(TargetSpec target,json::Value const& cal):target_(target) {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" ||
     cal.At("schema").AsNumber("native calibration schema")!=1)
    throw std::invalid_argument("native dense pricing requires calibrated sm90a schema 1");
  auto flow=cal.Find("flow_model");
  bool joint=flow && flow->AsString("native flow model")=="native_joint_ring_v1";
  if(!flow || (!joint && flow->AsString("native flow model")!="overlapped_operands_v1"))
    throw std::invalid_argument("native operand overlap requires refitted calibration");
  if(joint) {
    if(!cal.At("development_only").AsBool("joint-ring development flag") ||
       cal.At("e3_validated").AsBool("joint-ring E3 flag") ||
       cal.At("fit_status").AsString("joint-ring fit status")!=
           "joint_ring_diagnostic_pending_refit" ||
       cal.At("joint_ring_residual_model").AsString("joint-ring residual model")!=
           "none_pending_independent_refit")
      throw std::invalid_argument("native joint-ring flow requires development-only pending refit");
    joint_ring_cache_state_=cal.At("joint_ring_cache_state").AsString("joint-ring cache state");
    joint_ring_profiles_.emplace(cal.At("joint_ring_profiles"),
        cal.At("native_gemm_header_sha256").AsString("native GEMM calibration header"));
  }
  if(!cal.Find("issue_model") ||
     cal.At("issue_model").AsString("native issue model")!="issuer_packets_v3")
    throw std::invalid_argument("native preissue exposure requires issuer calibration");
  if(!cal.Find("overlap_model") ||
     cal.At("overlap_model").AsString("native overlap model")!="first_task_transfer_v3")
    throw std::invalid_argument("native preissue credit requires first-task transfer calibration");
  if(!cal.Find("attention_grid_model") ||
     cal.At("attention_grid_model").AsString("attention grid model")!="explicit_geometry_cache_v3")
    throw std::invalid_argument("native attention pricing requires exact geometry and cache calibration");
  if(!cal.Find("attention_cache_state"))
    throw std::invalid_argument("native attention pricing requires an explicit attention_cache_state");
  attention_cache_state_=cal.At("attention_cache_state").AsString("attention cache state");
  if(attention_cache_state_!="warm_reuse" && attention_cache_state_!="l2_scrub_surrogate")
    throw std::invalid_argument("unsupported native attention_cache_state: "+attention_cache_state_);
  if(!cal.Find("barrier_model") ||
     cal.At("barrier_model").AsString("native barrier model")!="active_arrival_v1")
    throw std::invalid_argument("native barriers require active-arrival calibration");
  if(!cal.Find("epilogue_model") ||
     cal.At("epilogue_model").AsString("native epilogue model")!="operation_shared_v1")
    throw std::invalid_argument("native epilogues require operation-specific calibration");
  if(!cal.Find("receiver_model") ||
     cal.At("receiver_model").AsString("native receiver model")!="prefix_receiver_v1")
    throw std::invalid_argument("native incoming transfers require receiver protocol calibration");
  if(!cal.Find("protocol_key_model") ||
     cal.At("protocol_key_model").AsString("protocol key model")!="stage_batch_v1")
    throw std::invalid_argument("native transfer prices require stage and batch protocol keys");
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
    if(v.At("metric").AsString("metric").rfind("attention_",0)==0) {
      // Runtime grid alone cannot identify the launch-bounds specialization.
      auto compiled=v.Find("compiled_residency");auto grid=v.Find("grid");
      double residency=compiled?compiled->AsNumber("compiled attention residency"):0;
      double blocks=grid?grid->AsNumber("attention grid"):0;
      if(!compiled || !grid || (residency!=1 && residency!=2) ||
          target_.res.num_sms<=0 || blocks!=residency*target_.res.num_sms)
        throw std::invalid_argument("attention observation requires explicit compiled_residency matching grid: "+
            v.At("metric").AsString("metric")+" compiled_residency="+
            (compiled?std::to_string(residency):"missing")+" grid="+
            (grid?std::to_string(blocks):"missing")+" num_sms="+std::to_string(target_.res.num_sms));
      if(!v.Find("cache_state") ||
          (v.At("cache_state").AsString("attention observation cache state")!="warm_reuse" &&
           v.At("cache_state").AsString("attention observation cache state")!="l2_scrub_surrogate"))
        throw std::invalid_argument("attention observation needs explicit cache_state: "+
            v.At("metric").AsString("metric"));
      // Low-batch attention does not instantiate streaming keys/slots. A
      // single chunk has no movable new-row boundary in either body family.
      if(v.Find("batch") && v.At("batch").AsNumber("batch")<=4) {
        v.Set("block_keys",64);v.Set("slots",3);
      }
      if(v.Find("chunks") && v.At("chunks").AsNumber("chunks")==1)v.Set("new_row_work",0);
    }
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
  auto missing=missing_quantities_.find(cache_key);
  if(missing!=missing_quantities_.end())throw missing->second;
  auto fail=[&](char const* reason) {
    auto entry=missing_quantities_.emplace(cache_key,
        Sm90DenseMissingMeasurement(std::string(reason)+cache_key,metric,dims,filters));
    throw entry.first->second;
  };
  auto it=observations_.find(metric);
  if(it==observations_.end())fail("missing native measurement: ");
  double best=std::numeric_limits<double>::infinity(),result=0;
  std::map<double,std::vector<double>> samples;
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
      // Whole-stage observations include waves, ring synchronization and LA.
      // Only past is interpolated; a neighboring geometry is another program.
      if(metric.rfind("attention_",0)==0 && d.first!="past" &&
          x!=d.second) {match=false;break;}
      if(metric.rfind("v3_sparse_",0)==0 && d.first=="grid" && x!=d.second) {
        match=false;break;
      }
      if(metric=="activation_inactive_rows_initialize" && d.first=="batch" &&
          x!=d.second) {match=false;break;}
      if(metric.find("_at_input_k")!=std::string::npos && d.first=="input_k" &&
          x!=d.second) {match=false;break;}
      if(d.first!="past")distance+=std::abs(std::log((x+1)/(d.second+1)));
    }
    if(match && distance<=best+1e-12) {
      if(distance<best-1e-12)samples.clear();
      best=distance;result=row.At("median").AsNumber("median");
      samples[dims.count("past")?row.At("past").AsNumber("past"):0].push_back(result);
    }
  }
  if(!samples.empty()) {
    // Repeated equally close measurements must not depend on file order.
    std::map<double,double> curve;
    for(auto const& point:samples)curve.emplace(point.first,Median(point.second));
    result=Interpolate(curve,dims.count("past")?dims.at("past"):0);
  }
  if(!result)fail("no matching native measurement: ");
  quantities_[cache_key]=result;return result;
}
void Sm90DenseCostModel::RequireAttentionCalibration(Sm90DenseWorkload const& w,
    Sm90DenseCoordinates const& c) const {
  c.ValidateSchedule(w.shapes,w.batch,w.head_dim,w.queries,1088,
      Sm90DenseSharedBudget(target_,c.residency));
  if(w.layers<1)throw std::invalid_argument("invalid native prediction workload");
  // Only past is interpolated. One request midpoint therefore proves exact
  // geometry coverage for every position without pricing the other bodies.
  Quantity("attention_critical_before_barrier",
      AttentionDimensions(w,c,target_.res.num_sms*c.residency,95),
      {{"cache_state",attention_cache_state_}});
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
      Sm90DenseSharedBudget(target_,c.residency));
  if(w.layers<1 || past<64 || past>1086)throw std::invalid_argument("invalid native prediction workload");
  int grid=target_.res.num_sms*c.residency;
  bool activation_reuse=c.ActivationReuse(w.batch,w.head_dim,w.queries,1088).valid;
  int joint_shared=0;
  if(joint_ring_profiles_) {
    // NativeProgram reserves the maximum arena of the entire generated kernel,
    // not this GEMM's payload or the target's maximum available shared memory.
    using Layout=backend::Sm90DenseLayout;
    joint_shared=Layout::Argmax().end;
    for(auto const& g:c.gemms)joint_shared=std::max(joint_shared,
        g.workspace_offset+Layout::Gemm(g.tile_n,g.TileK(),g.slots,g.warps_k).end);
    auto const& a=c.attention;
    joint_shared=std::max(joint_shared,a.workspace_offset+
        Layout::Attention(w.batch,w.head_dim,w.queries,a.chunks,a.block_keys,
            a.slots,a.new_row_work,1088).end);
  }
  auto barrier=[&](int tasks) {
    // Empty CTAs arrive before the working wave. The 2 us delayed-arrival
    // probe models this steady wave; nearest active count is inferred.
    return Quantity("v3_sparse_last_arrival_to_last_departure",
        {{"grid",grid},{"active_ctas",std::min(grid,tasks)},{"delay_ns",2000}});
  };
  // Saturating DRAM service is measured, not the stated H800 peak.
  double global_bw=dram_bw_;
  if(!global_bw)throw std::invalid_argument("native DRAM streaming calibration is missing");
  Sm90DensePrediction result;
  for(unsigned family:{0u,5u,1u,2u,3u,4u}) {
    Sm90DensePrediction::Stage s;s.instances=family==4?1:w.layers;
    if(family<5) {
      auto const& g=c.gemms[family];int n=w.shapes[family][0],k=w.shapes[family][1];
      s.kind=kSm90DenseFamilies[family];s.key=GemmKey(g);s.tasks=n/g.tile_n;
      s.protocol_key=s.key+":"+s.kind+(w.batch<=4?"_low":"_high");
      s.receive_count=g.preissue_smem_bytes>0;
      s.waves=(s.tasks+grid-1)/grid;int active=std::min(grid,s.tasks);
      double bw=0,tile_stream=0;
      s.dram=2.*n*k/global_bw;
      if(!joint_ring_profiles_) {
        bw=Quantity("bulk_stream_per_active_cta",{{"grid",active},
            {"slot_bytes",g.slot_bytes},{"slots",g.slots}},{{"cache","dram"}});
        tile_stream=2.*g.tile_n*k/bw;
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
      }
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
      std::string epilogue_key=family==0?"store":family==2?"swiglu":family==4?"argmax":"residual";
      s.epilogue=s.waves*epilogue*Coefficient("native_epilogue_"+epilogue_key,"scale");
      // Run initializes all slots once per stage, even when this CTA has
      // several tasks. The producer never executes this branch at B16.
      s.initialization=w.batch<16 && !activation_reuse?Quantity("activation_inactive_rows_initialize",
          {{"grid",grid},{"batch",w.batch},{"tile_k",g.TileK()},{"slots",g.slots}}):0;
      if(joint_ring_profiles_) {
        std::map<std::string,double> dimensions={{"batch",w.batch},{"tile_n",g.tile_n},
            {"tile_k",g.TileK()},{"slots",g.slots},{"warps_n",g.warps_n},
            {"warps_k",g.warps_k},{"register_stages",g.register_stages},{"grid",grid},
            {"active_ctas",active},{"compiled_residency",c.residency},{"input_k",k},{"tasks_per_cta",s.waves},
            {"prefix_bytes",g.preissue_smem_bytes},{"prefetch_l2_bytes",g.prefetch_l2_bytes},
            {"dynamic_shared_bytes",joint_shared},{"workspace_offset",g.workspace_offset}};
        std::string mma=g.mma==Sm90DenseMma::kMmaSync?"mma_sync":"wgmma";
        auto profile=joint_ring_profiles_->Find(dimensions,joint_ring_cache_state_,mma);
        if(!profile)throw Sm90DenseMissingMeasurement("missing exact native joint-ring profile; calibration pending",
            "native_joint_ring_profile",dimensions,{{"cache_state",joint_ring_cache_state_},{"mma",mma}});
        s.joint_ring_profile=*profile;
        s.joint_ring_profile.Set("residual_model","none_pending_independent_refit");
        s.joint_ring_profile.Set("diagnostic_only",true);
        s.joint_ring_profile.Set("fit_status","joint_ring_diagnostic_pending_refit");
        // The paired total already contains operand first-ready, real MMA/
        // reader, full/empty protocol and finalize. Marginal component medians
        // need not add; none is appended to the paired total or old startup.
        s.flow=profile->At("issue_to_finalize_ns").AsNumber("joint-ring finite span");
        s.startup=s.initialization;
      }else {
        s.read_startup=Quantity("bulk_single_completion",{{"slot_bytes",g.slot_bytes}},{{"cache","dram"}});
        s.startup=Coefficient(s.key,"startup_ns")+s.initialization+s.read_startup;
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
        // Only weights were preissued. Activation copies still execute here,
        // even when every weight slot is already full at the stage boundary.
        double unmoved=std::max(s.compute,Coefficient(s.key,"activation_scale")*s.activation);
        s.overlap_limit=s.read_startup+std::max(s.flow-unmoved,0.);
        (g.preissue_late || payload.Overlaps(previous)?s.late_service:s.preissue_service)=
            fraction*tile_stream;
        s.l2_service=std::min(double(g.prefetch_l2_bytes),
            2.*g.tile_n*k-g.preissue_smem_bytes)/bw;
        // A prefix only belongs to the first task. Even fitted efficiencies
        // above one must not credit the remaining waves' weight service.
        s.overlap_limit=std::min(s.overlap_limit,
            s.read_startup+s.preissue_service+s.late_service+s.l2_service);
        // Each prefixed slot executes expect_tx and one bulk instruction. A
        // six-slot prefix cannot share the fixed issue price of a one-slot prefix.
        s.issue_count=2*((g.preissue_smem_bytes+g.slot_bytes-1)/g.slot_bytes)+(s.l2_service>0);
        s.issue_service=fraction*tile_stream+s.l2_service;
      }
    }else {
      auto const& a=c.attention;s.kind="attention";s.key=AttentionKey(w);
      s.protocol_key=s.key+":attention"+(w.batch<=4?std::string("_low"):std::string("_high"));
      s.tasks=w.batch*8*a.chunks;s.waves=(s.tasks+grid-1)/grid;
      // The whole-stage calibration includes deterministic LA and wave costs.
      s.flow=Quantity("attention_critical_before_barrier",AttentionDimensions(w,c,grid,past),
          {{"cache_state",attention_cache_state_}})*
          Coefficient(s.key,"body_scale");
      s.dram=4.*w.batch*8*past*w.head_dim/global_bw;
      s.compute=s.flow;s.startup=Coefficient(s.key,"startup_ns");
      auto payload=backend::Sm90DenseLayout::AttentionPayload(w.batch,w.head_dim,w.queries,
          a.chunks,a.block_keys,a.slots,a.new_row_work,1088,a.preissue_smem_bytes)
          .Shift(a.workspace_offset);
      auto const& prev=c.gemms[0];
      auto previous=backend::Sm90DenseLayout::Gemm(prev.tile_n,prev.TileK(),prev.slots,
          prev.warps_k).Live(prev.workspace_offset);
      int active=std::min(grid,s.tasks),windows=0,max_history=0;
      double prefix_bytes=0,l2_bytes=0,max_prefix=0,max_l2=0;
      for(int chunk=0;chunk<a.chunks;++chunk) {
        int owners=active/a.chunks+(chunk<active%a.chunks);
        if(!owners)continue;
        // This is Partition::Range clipped to history (<past), with the same
        // weighted new-row boundary as the actual first task of each CTA.
        long long weighted=static_cast<long long>(past)+1+a.new_row_work;
        int begin=int(std::min<long long>(past,weighted*chunk/a.chunks));
        int end=int(std::min<long long>(past,weighted*(chunk+1)/a.chunks));
        int history=end-begin,copied=std::min(history,a.preissue_smem_bytes/(4*w.head_dim));
        int prefixed=4*copied*w.head_dim;
        // Prefetch divides its budget between K/V and rounds each to 16 B.
        int prefetched=2*(std::min(a.prefetch_l2_bytes/2,
            2*w.head_dim*(history-copied))&~15);
        prefix_bytes+=double(owners)*prefixed;l2_bytes+=double(owners)*prefetched;
        max_prefix=std::max(max_prefix,double(prefixed));max_l2=std::max(max_l2,double(prefetched));
        max_history=std::max(max_history,history);
        windows=std::max(windows,w.batch<=4?int(copied>0):(copied+a.block_keys-1)/a.block_keys);
      }
      if(prefix_bytes || l2_bytes) {
        int slot_bytes=2*w.head_dim*(w.batch<=4?max_history:std::min(a.block_keys,max_history));
        double bw=Quantity("bulk_stream_per_active_cta",{{"grid",active},
            {"slot_bytes",slot_bytes},{"slots",w.batch<=4?1:a.slots}},{{"cache","dram"}});
        double prefixed=std::max(prefix_bytes/global_bw,max_prefix/bw);
        (a.preissue_late || payload.Overlaps(previous)?s.late_service:s.preissue_service)=prefixed;
        s.l2_service=std::max(l2_bytes/global_bw,max_l2/bw);
      }
      // Inferred transfer-service bound: one first-task history read can be
      // hidden, not the other task waves' query/MMA/partial/LA work. These
      // byte-derived services do not scale with the whole-body fitted price.
      s.overlap_limit=std::min(s.flow/s.waves,s.preissue_service+s.late_service+s.l2_service);
      // KV slots execute one expect_tx and separate K/V bulks; L2 prefetch
      // likewise emits distinct instructions for the two cache arrays.
      s.issue_count=3*windows+2*(s.l2_service>0);
      s.receive_count=a.preissue_smem_bytes>0;
      // Outgoing contention prices transferred KV bytes, not the receiver's
      // query/softmax work that also contributes to its overlap feature.
      s.issue_service=s.preissue_service+s.late_service+s.l2_service;
    }
    s.overlap=std::min(s.overlap_limit,
        s.preissue_service*Coefficient(s.protocol_key,"preissue_efficiency")+
        s.late_service*Coefficient(s.protocol_key,"tail_efficiency")+
        s.l2_service*Coefficient(s.protocol_key,"l2_efficiency"));
    s.barrier=barrier(s.tasks);
    s.ns=s.barrier+s.startup+std::max(s.flow,s.compute)+s.epilogue-s.overlap;
    result.stages.push_back(std::move(s));
  }
  for(auto kind:{"embedding","argmax"}) {
    Sm90DensePrediction::Stage s;s.kind=kind;s.key=std::string("native_")+kind;
    s.protocol_key=s.key+":"+kind+(w.batch<=4?"_low":"_high");
    s.tasks=w.batch;s.waves=1;s.barrier=barrier(s.tasks);
    s.startup=Coefficient(s.key,"startup_ns");
    s.compute=Coefficient(s.key,"per_element_ns")*(s.kind=="embedding"?
        w.shapes[0][1]:(w.shapes[4][0]+c.gemms[4].tile_n-1)/c.gemms[4].tile_n);
    s.ns=s.barrier+s.startup+s.compute;result.stages.push_back(s);
  }
  // A stage's incoming transfer is issued by its predecessor. The first qkv
  // follows embedding; the final down issues lm_head rather than another qkv.
  // Class means retain both boundary cases without charging the receiver.
  auto& stages=result.stages;
  std::array<double,6> incoming{},service{};
  for(unsigned i=0;i<incoming.size();++i) {
    incoming[i]=stages[i].issue_count;
    service[i]=stages[i].issue_service;
  }
  for(unsigned i=0;i<4;++i) {
    stages[i].issue_count=incoming[i+1];stages[i].issue_service=service[i+1];
  }
  stages[4].issue_count=(incoming[0]*(w.layers-1)+incoming[5])/w.layers;
  stages[4].issue_service=(service[0]*(w.layers-1)+service[5])/w.layers;
  stages[5].issue_count=0;stages[6].issue_count=incoming[0];
  stages[5].issue_service=0;stages[6].issue_service=service[0];
  for(auto& s:stages) {
    // An attention producer can hide weight issue behind compute, while a
    // streaming GEMM exposes competing traffic. Price the issuing body.
    s.issue=s.issue_count? s.issue_count*Coefficient(s.protocol_key,"issue_ns")+
        s.issue_service*Coefficient(s.protocol_key,"issue_service_scale"):0;
    // Incoming prefix setup runs once, even when the stage has many task waves.
    s.receive=s.receive_count? s.receive_count*Coefficient(s.protocol_key,"receive_ns"):0;
    s.ns+=s.issue+s.receive;result.step_ns+=s.ns*s.instances;
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
        s.ns=s.barrier=s.startup=s.read_startup=s.initialization=s.dram=s.stream=s.flow=
            s.activation=s.compute=s.epilogue=s.overlap=s.issue=
            s.overlap_limit=s.preissue_service=s.late_service=s.l2_service=s.issue_count=s.issue_service=
            s.receive_count=s.receive=0;
    }
    for(unsigned j=0;j<sample.stages.size();++j) {
      auto& out=result.stages[j];auto const& in=sample.stages[j];
      out.ns+=in.ns/intervals;
      out.barrier+=in.barrier/intervals;out.startup+=in.startup/intervals;
      out.read_startup+=in.read_startup/intervals;
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
      out.issue_service+=in.issue_service/intervals;
      out.receive_count+=in.receive_count/intervals;out.receive+=in.receive/intervals;
    }
    result.step_ns+=sample.step_ns/intervals;
  }
  return result;
}
json::Value Sm90DensePrediction::ToJson() const {
  json::Array values;
  for(auto const& s:stages) {
    json::Value value=json::Object{{"kind",s.kind},{"key",s.key},
      {"protocol_key",s.protocol_key},
      {"instances",s.instances},{"tasks",s.tasks},{"waves",s.waves},
      {"stage_ns",s.ns},{"class_ns",s.ns*s.instances},{"barrier_ns",s.barrier},
      {"startup_ns",s.startup},{"initialization_ns",s.initialization},
      {"dram_ns",s.dram},{"flow_ns",s.flow},
      {"stream_ns",s.stream},
      {"activation_ns",s.activation},
      {"compute_ns",s.compute},{"epilogue_ns",s.epilogue},{"overlap_ns",s.overlap},
      {"issue_ns",s.issue},{"read_startup_ns",s.read_startup},{"overlap_limit_ns",s.overlap_limit},
      {"preissue_service_ns",s.preissue_service},{"late_service_ns",s.late_service},
      {"l2_service_ns",s.l2_service},{"issue_count",s.issue_count},{"issue_service_ns",s.issue_service},
      {"receive_count",s.receive_count},{"receive_ns",s.receive}};
    if(!s.joint_ring_profile.IsNull())value.Set("joint_ring_profile",s.joint_ring_profile);
    values.push_back(std::move(value));
  }
  json::Value result=json::Object{{"step_ns",step_ns},{"evidence","inferred prediction from native calibration"},
      {"stages",values}};
  if(std::any_of(stages.begin(),stages.end(),[](auto const& s) {return !s.joint_ring_profile.IsNull();})) {
    result.Set("diagnostic_only",true);
    result.Set("flow_model","native_joint_ring_v1");
    result.Set("fit_status","joint_ring_diagnostic_pending_refit");
    result.Set("pricing_scope","typical-CTA profile diagnostic; not stage-critical or E2E validation");
  }
  return result;
}
}  // namespace tilemega::solver
