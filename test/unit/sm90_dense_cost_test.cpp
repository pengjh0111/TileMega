// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <cassert>
#include <set>
using namespace tilemega;
using namespace tilemega::solver;

// Deliberately synthetic units test missing-calibration rejection, physical
// accounting and native-only search. These coefficients are never installed.
json::Value Fixture() {
  json::Array observations;
  auto point=[&](char const* metric,double median,char const* cache) {
    observations.emplace_back(json::Object{{"metric",metric},{"median",median},{"processes",5},
        {"grid",132},{"slot_bytes",16384},{"slots",3},{"batch",1},{"tile_n",32},
        {"vector_sums",1},{"head_dim",128},{"queries",2},{"qk_norm",1},
        {"warps_n",1},{"warps_k",4},{"register_stages",1},
        {"production_compatible",true},
        {"chunks",16},{"block_keys",64},{"past",575},{"tile_k",256},{"cache",cache}});
  };
  point("v3_last_arrival_to_departure",768,"na");
  point("bulk_stream_per_active_cta",24,"dram");point("bulk_single_completion",600,"dram");
  point("activation_wait_and_shared_read",200,"l2");
  point("activation_inactive_rows_initialize",500,"na");
  for(int wn:{1,2,4}) {
    point("native_ldmatrix_mma_k16",wn==1?40:wn==2?24:16,"na");
    observations.back().Set("warps_n",wn);observations.back().Set("warps_k",4/wn);
  }
  for(auto op:{"epilogue_store_return","epilogue_residual_return","epilogue_swiglu_return",
               "epilogue_argmax_return"}) {point(op,200,"plain");point(op,300,"dn");}
  point("attention_critical_before_barrier",6000,"na");
  point("attention_critical_before_barrier",8000,"na");observations.back().Set("batch",16);
  json::Object fits;
  auto fit=[&](char const* key,json::Object coefficients) {
    fits.emplace_back(key,json::Object{{"evidence","inferred"},{"sample_count",5},
        {"coefficients",coefficients}});
  };
  for(auto key:{"gemm_mma_sync_tn8_32","gemm_mma_sync_tn40_64","gemm_mma_sync_tn72_plus"})
    fit(key,{{"startup_ns",300},{"mma_scale",1},{"memory_scale",1},{"activation_scale",.2},
        {"epilogue_scale",1},{"preissue_efficiency",.5},{"tail_efficiency",.1},
        {"l2_efficiency",.1},{"issue_ns",20}});
  for(auto key:{"attention_simt_d128_q2_norm1","attention_mma_sync_d128_q2_norm1"})
    fit(key,{{"body_scale",1},{"startup_ns",100},
      {"preissue_efficiency",.2},{"tail_efficiency",.1},{"l2_efficiency",.1},{"issue_ns",20}});
  for(auto key:{"native_embedding","native_argmax"})
    fit(key,{{"startup_ns",100},{"per_element_ns",.1}});
  return json::Object{{"schema",1},{"observations",observations},{"body_fits",fits}};
}
int main() {
  TargetSpec t;t.arch_tag="sm_90";t.nvcc_arch="sm_90a";
  t.res.num_sms=132;t.res.max_smem_per_sm=233472;t.res.max_dynamic_smem_per_cta=232448;
  t.res.l2_bytes=52428800;
  Sm90DenseWorkload w{{{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{151936,1024}}},
      1,28,128,2,true};
  auto c=Sm90DenseCoordinates::Exact(w.shapes,w.batch,t.res.max_dynamic_smem_per_cta);
  Sm90DenseCostModel model(t,Fixture());auto prediction=model.Request(w,c);
  auto tabular=Fixture();json::Array columns,rows;
  for(auto const& p:tabular.At("observations").AsArray("observations")[0].AsObject("point"))
    columns.emplace_back(p.first);
  for(auto const& point:tabular.At("observations").AsArray("observations")) {
    json::Array values;for(auto const& key:columns)values.push_back(point.At(key.AsString("column")));
    rows.emplace_back(values);
  }
  tabular.Set("observation_columns",columns);tabular.Set("observations",rows);
  assert(Sm90DenseCostModel(t,tabular).Request(w,c).step_ns==prediction.step_ns);
  assert(prediction.stages.size()==8 && prediction.step_ns>0);
  double total=0;
  for(auto const& p:prediction.stages) {
    assert(p.ns>=p.dram && p.ns>=p.barrier && p.ns>0);
    total+=p.ns*p.instances;
  }
  assert(std::abs(total-prediction.step_ns)<1e-6);
  for(auto const& p:prediction.stages)
    assert(p.initialization==(p.key.rfind("gemm_",0)==0?500:0));
  assert(prediction.stages[5].waves>1 && prediction.stages[5].initialization==500);
  auto full_batch=w;full_batch.batch=16;
  auto full_coordinates=Sm90DenseCoordinates::Exact(w.shapes,16,t.res.max_dynamic_smem_per_cta);
  for(auto const& p:model.Request(full_batch,full_coordinates).stages)assert(p.initialization==0);
  auto issue_only=Fixture();
  auto issue_fits=issue_only.At("body_fits").AsObject("body fits");
  for(auto& fit:issue_fits) {
    auto coefficients=fit.second.At("coefficients");
    if(coefficients.Find("preissue_efficiency")) {
      coefficients.Set("preissue_efficiency",0);coefficients.Set("tail_efficiency",0);
      coefficients.Set("l2_efficiency",0);
      fit.second.Set("coefficients",coefficients);
    }
  }
  issue_only.Set("body_fits",issue_fits);Sm90DenseCostModel issue_model(t,issue_only);
  auto baseline=issue_model.Predict(w,c);auto prefetch=c;
  prefetch.gemms[0].prefetch_l2_bytes=16384;
  prefetch.gemms[4].prefetch_l2_bytes=16384;
  auto issued=issue_model.Predict(w,prefetch);
  assert(issued.stages[0].issue==0 && issued.stages[5].issue==0);
  assert(issued.stages[6].issue==20 && issued.stages[4].issue==20);
  assert(std::abs(issued.step_ns-baseline.step_ns-20*(w.layers+1))<1e-6);
  auto one_layer=w;one_layer.layers=1;prefetch.gemms[4].prefetch_l2_bytes=0;
  auto first_only=issue_model.Predict(one_layer,prefetch);
  assert(first_only.stages[6].issue==20 && first_only.stages[4].issue==0);
  prefetch=c;prefetch.attention.prefetch_l2_bytes=16384;
  auto attention_issue=issue_model.Request(w,prefetch);
  assert(attention_issue.stages[0].issue==20 && attention_issue.stages[1].issue==0);
  auto partition=c;partition.gemms[0].warps_n=1;partition.gemms[0].warps_k=4;
  auto split_prediction=model.Predict(w,partition);
  assert(split_prediction.stages[0].compute==double(w.shapes[0][1]/64)*40);
  partition.gemms[0].warps_n=4;partition.gemms[0].warps_k=1;
  assert(model.Predict(w,partition).stages[0].compute==double(w.shapes[0][1]/16)*16);
  auto incomplete=Fixture();incomplete.Set("body_fits",json::Object{});bool rejected=false;
  try {Sm90DenseCostModel(t,incomplete).Request(w,c);}catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
  auto no_initialization=Fixture();json::Array retained;
  for(auto const& point:no_initialization.At("observations").AsArray("observations"))
    if(point.At("metric").AsString("metric")!="activation_inactive_rows_initialize")
      retained.push_back(point);
  no_initialization.Set("observations",retained);rejected=false;
  try {Sm90DenseCostModel(t,no_initialization).Request(w,c);}
  catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
  auto bare_only=Fixture();json::Array without_mainloop;
  for(auto const& point:bare_only.At("observations").AsArray("observations"))
    if(point.At("metric").AsString("metric")!="native_ldmatrix_mma_k16")
      without_mainloop.push_back(point);
  bare_only.Set("observations",without_mainloop);rejected=false;
  try {Sm90DenseCostModel(t,bare_only).Request(w,c);}catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
  auto surrogate=Fixture();auto audit_rows=surrogate.At("observations").AsArray("observations");
  for(auto& row:audit_rows)if(row.At("metric").AsString("metric")=="native_ldmatrix_mma_k16")
    row.Set("production_compatible",false);
  surrogate.Set("observations",audit_rows);rejected=false;
  try {Sm90DenseCostModel(t,surrogate).Request(w,c);}catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
  Sm90DenseSearchSpace space{w.shapes,w.batch,w.head_dim,w.queries,t};
  Sm90DenseSearchOptions options;options.passes=2;options.beam_width=2;options.retained=8;
  auto search=SearchSm90Dense(space,w,model,options);
  assert(search.evaluated>space.Seeds().size() && search.ranked.size()>=8);
  double previous=0;std::set<std::string> unique;
  for(auto const& p:search.ranked) {
    assert(space.Legal(p.coordinates) && p.prediction.step_ns>=previous);previous=p.prediction.step_ns;
    assert(unique.insert(p.coordinates.ToJson().Dump(0)).second);
    for(auto const& g:p.coordinates.gemms)assert(g.mma==Sm90DenseMma::kMmaSync);
  }
}
