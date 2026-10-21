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
        {"chunks",16},{"block_keys",64},{"past",575},{"tile_k",256},{"cache",cache}});
  };
  point("v3_last_arrival_to_departure",768,"na");
  point("bulk_stream_per_active_cta",24,"dram");point("bulk_single_completion",600,"dram");
  point("activation_wait_and_shared_read",200,"l2");
  point("mma_sync_per_instruction",4,"na");
  for(auto op:{"epilogue_store_return","epilogue_residual_return","epilogue_swiglu_return",
               "epilogue_argmax_return"}) {point(op,200,"plain");point(op,300,"dn");}
  point("attention_critical_before_barrier",6000,"na");
  json::Object fits;
  auto fit=[&](char const* key,json::Object coefficients) {
    fits.emplace_back(key,json::Object{{"evidence","inferred"},{"sample_count",5},
        {"coefficients",coefficients}});
  };
  for(auto key:{"gemm_mma_sync_tn8_32","gemm_mma_sync_tn40_64","gemm_mma_sync_tn72_plus"})
    fit(key,{{"startup_ns",300},{"mma_scale",1},{"memory_scale",1},{"activation_scale",.2},
        {"epilogue_scale",1},{"preissue_efficiency",.5},{"tail_efficiency",.1},
        {"l2_efficiency",.1},{"issue_ns",20}});
  fit("attention_simt_d128_q2_norm1",{{"body_scale",1},{"startup_ns",100},
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
  auto incomplete=Fixture();incomplete.Set("body_fits",json::Object{});bool rejected=false;
  try {Sm90DenseCostModel(t,incomplete).Request(w,c);}catch(std::invalid_argument const&) {rejected=true;}
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
