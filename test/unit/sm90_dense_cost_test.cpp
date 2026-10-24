// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <tilemega/Solver/Sm90DenseResources.h>
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
        {"grid",132},{"active_ctas",128},{"delay_ns",2000},
        {"slot_bytes",16384},{"slots",3},{"batch",1},{"tile_n",32},
        {"vector_sums",1},{"head_dim",128},{"queries",2},{"qk_norm",1},
        {"warps_n",1},{"warps_k",4},{"register_stages",1},
        {"cache_state","warm_reuse"},{"compiled_residency",1},
        {"production_compatible",true},
        {"chunks",16},{"block_keys",64},{"new_row_work",0},{"past",575},{"tile_k",256},{"input_k",4096},{"cache",cache}});
  };
  point("v3_last_arrival_to_departure",768,"na");
  point("v3_sparse_last_arrival_to_last_departure",1000,"na");
  point("v3_sparse_last_arrival_to_last_departure",1400,"na");observations.back().Set("grid",264);
  point("bulk_stream_per_active_cta",24,"dram");point("bulk_single_completion",600,"dram");
  point("activation_wait_and_shared_read",200,"l2");
  point("activation_inactive_rows_initialize",500,"na");
  for(int wn:{1,2,4}) {
    point("native_ldmatrix_mma_k16",wn==1?40:wn==2?24:16,"na");
    observations.back().Set("warps_n",wn);observations.back().Set("warps_k",4/wn);
  }
  for(auto op:{"epilogue_store_return","epilogue_residual_return","epilogue_swiglu_return",
               "epilogue_argmax_return"}) {point(op,200,"plain");point(op,300,"dn");}
  for(auto op:{"epilogue_store_return_at_input_k","epilogue_swiglu_return_at_input_k",
               "epilogue_argmax_return_at_input_k"})for(int k:{1024,2048,3072,4096}) {
    point(op,std::string(op)=="epilogue_store_return_at_input_k" && k==1024?450:300,"dn");
    observations.back().Set("input_k",k);
  }
  point("attention_critical_before_barrier",6000,"na");
  point("attention_critical_before_barrier",8000,"na");observations.back().Set("batch",16);
  point("attention_critical_before_barrier",7000,"na");observations.back().Set("grid",264);
  observations.back().Set("compiled_residency",2);
  point("attention_critical_before_barrier",9000,"na");
  observations.back().Set("batch",16);observations.back().Set("grid",264);
  observations.back().Set("compiled_residency",2);
  auto attention_rows=observations;
  for(auto const& row:attention_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier")
      for(int chunks:{1,2,4,8,32,64}) {
        auto value=row;value.Set("chunks",chunks);
        value.Set("median",row.At("median").AsNumber("median")*std::max(1,chunks/16));
        observations.push_back(value);
      }
  attention_rows=observations;
  for(auto const& row:attention_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier" &&
        row.At("chunks").AsNumber("chunks")>1)
      for(int work:{8,16}) {
        auto value=row;value.Set("new_row_work",work);observations.push_back(value);
      }
  json::Object fits;
  auto fit=[&](char const* key,json::Object coefficients) {
    coefficients.emplace_back("issue_service_scale",0);
    coefficients.emplace_back("receive_ns",0);
    fits.emplace_back(key,json::Object{{"evidence","inferred"},{"sample_count",5},
        {"coefficients",coefficients}});
  };
  for(auto key:{"gemm_mma_sync_tn8_32","gemm_mma_sync_tn40_64","gemm_mma_sync_tn72_plus"})
    for(int wk:{1,2,4})fit((std::string(key)+"_wk"+std::to_string(wk)).c_str(),
        {{"startup_ns",300},{"mma_scale",1},{"memory_scale",1},{"activation_scale",.2},
        {"epilogue_scale",1},{"preissue_efficiency",.5},{"tail_efficiency",.1},
        {"l2_efficiency",.1},{"issue_ns",20}});
  for(auto key:{"attention_simt_d128_q2_norm1","attention_mma_sync_d128_q2_norm1"})
    fit(key,{{"body_scale",1},{"startup_ns",100},
      {"preissue_efficiency",.2},{"tail_efficiency",.1},{"l2_efficiency",.1},{"issue_ns",20}});
  for(auto key:{"native_embedding","native_argmax"})
    fit(key,{{"startup_ns",100},{"per_element_ns",.1},{"issue_ns",20}});
  for(auto op:{"store","residual","swiglu","argmax"})
    fit((std::string("native_epilogue_")+op).c_str(),{{"scale",1}});
  auto physical=fits;
  for(auto const& body:physical)for(auto kind:{"qkv","o","gate_up","down","lm_head","attention","embedding","argmax"})
    for(auto band:{"low","high"})fits.emplace_back(body.first+":"+kind+"_"+band,body.second);
  return json::Object{{"schema",1},{"flow_model","overlapped_operands_v1"},
      {"issue_model","issuer_packets_v3"},
      {"overlap_model","first_task_transfer_v3"},
      {"attention_grid_model","explicit_geometry_cache_v3"},
      {"attention_cache_state","warm_reuse"},
      {"barrier_model","active_arrival_v1"},
      {"epilogue_model","operation_shared_v1"},
      {"receiver_model","prefix_receiver_v1"},
      {"protocol_key_model","stage_batch_v1"},
      {"observations",observations},{"body_fits",fits}};
}

void JointRingContracts(TargetSpec const& t) {
  // Explicit synthetic CPU data, not installed calibration observations.
  Sm90DenseWorkload w{{{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{16384,1024}}},
      1,28,128,2,true};
  auto c=Sm90DenseCoordinates::Exact(w.shapes,w.batch,t.res.max_dynamic_smem_per_cta);
  using Layout=backend::Sm90DenseLayout;
  int shared=c.attention.workspace_offset+Layout::Attention(w.batch,w.head_dim,w.queries,
      c.attention.chunks,c.attention.block_keys,c.attention.slots,c.attention.new_row_work,1088).end;
  for(auto const& g:c.gemms)shared=std::max(shared,
      g.workspace_offset+Layout::Gemm(g.tile_n,g.TileK(),g.slots,g.warps_k).end);
  json::Object binding;
  for(auto name:{"source_digest","producer_header_sha256","probe_sha256","binary_sha256","collector_sha256"})
    binding.emplace_back(name,std::string(64,'a'));
  json::Array profiles;
  for(unsigned f=0;f<5;++f) {
    auto const& g=c.gemms[f];json::Array prefix;
    for(int q=0;q<std::min(g.slots+1,w.shapes[f][1]/g.TileK());++q)
      prefix.emplace_back(json::Object{{"sequence",q},{"full_ns",1000+200*q},{"release_ns",1100+200*q}});
    profiles.emplace_back(json::Object{{"profile_id",std::string("synthetic_")+kSm90DenseFamilies[f]},
        {"batch",w.batch},{"tile_n",g.tile_n},{"tile_k",g.TileK()},{"slots",g.slots},
        {"warps_n",g.warps_n},{"warps_k",g.warps_k},{"register_stages",g.register_stages},
        {"grid",132},{"active_ctas",128},{"compiled_residency",1},{"mma","mma_sync"},
        {"input_k",w.shapes[f][1]},
        {"tasks_per_cta",1},{"prefix_bytes",0},{"prefetch_l2_bytes",0},
        {"dynamic_shared_bytes",shared},{"workspace_offset",g.workspace_offset},
        {"continuous_task_drain",false},{"cache_state","four_shape_replay_no_scrub"},
        {"evidence","synthetic_cpu_fixture"},{"aggregation","per_cta_sample_median"},
        {"processes",5},{"finite_prefix",prefix},
        // Independent marginal medians intentionally do not sum to total.
        {"first_full_ns",1000},{"main_span_ns",8000},{"finalize_tail_ns",3000},
        {"issue_to_finalize_ns",10000}});
  }
  json::Value table=json::Object{{"schema",1},{"endpoint_model","producer_call_to_finalize_observed_v1"},
      {"source_binding",binding},{"profiles",profiles}};
  auto cal=Fixture();cal.Set("flow_model","native_joint_ring_v1");
  cal.Set("development_only",true);cal.Set("e3_validated",false);
  cal.Set("fit_status","joint_ring_diagnostic_pending_refit");
  cal.Set("native_gemm_header_sha256",std::string(64,'a'));
  cal.Set("joint_ring_residual_model","none_pending_independent_refit");
  cal.Set("joint_ring_cache_state","four_shape_replay_no_scrub");cal.Set("joint_ring_profiles",table);
  Sm90DenseCostModel model(t,cal);auto p=model.Predict(w,c),request=model.Request(w,c);
  auto covered=model.Preflight(w,c);
  assert(covered.missing.empty() && covered.prediction->ToJson().Dump(0)==request.ToJson().Dump(0));
  for(unsigned i:{0u,2u,3u,4u,5u}) {
    auto const& s=p.stages[i];
    assert(s.flow==10000 && s.compute==0 && s.activation==0 && s.read_startup==0);
    assert(s.startup==s.initialization && s.overlap==0);
    assert(s.ns==s.barrier+s.initialization+10000+s.epilogue+s.issue+s.receive);
    assert(s.joint_ring_profile.At("evidence").AsString("evidence")=="synthetic_cpu_fixture");
    assert(request.stages[i].joint_ring_profile.Dump(0)==s.joint_ring_profile.Dump(0));
    assert(request.stages[i].flow==s.flow);
  }
  assert(!p.ToJson().At("stages").AsArray("stages")[0].At("joint_ring_profile").IsNull());
  assert(p.ToJson().At("diagnostic_only").AsBool("diagnostic prediction"));
  auto error=[&](json::Value const& changed,Sm90DenseWorkload const& work,
      Sm90DenseCoordinates const& coords) {
    try {Sm90DenseCostModel(t,changed).Predict(work,coords);}
    catch(std::exception const& e) {return std::string(e.what());}
    assert(false && "unsupported joint-ring input was accepted");return std::string{};
  };
  auto bad=cal;bad.Set("development_only",false);
  assert(error(bad,w,c).find("development-only")!=std::string::npos);
  bad=cal;bad.Set("e3_validated",true);
  assert(error(bad,w,c).find("development-only")!=std::string::npos);
  bad=cal;bad.Set("fit_status","initial_overlap_fitted");
  assert(error(bad,w,c).find("pending refit")!=std::string::npos);
  bad=cal;bad.Set("native_gemm_header_sha256",std::string(64,'b'));
  assert(error(bad,w,c).find("producer header differs")!=std::string::npos);
  for(auto dimension:{"active_ctas","grid","tile_k","warps_k","slots","dynamic_shared_bytes"}) {
    auto wrong=profiles;wrong[0].Set(dimension,dimension==std::string("active_ctas")?64:
        wrong[0].At(dimension).AsNumber(dimension)+1);
    auto changed=table;changed.Set("profiles",wrong);bad=cal;bad.Set("joint_ring_profiles",changed);
    assert(!error(bad,w,c).empty());
  }
  auto missing=table;auto short_rows=profiles;short_rows.erase(short_rows.begin());missing.Set("profiles",short_rows);
  bad=cal;bad.Set("joint_ring_profiles",missing);
  try {Sm90DenseCostModel(t,bad).Predict(w,c);assert(false);}
  catch(Sm90DenseMissingMeasurement const& e) {
    assert(e.metric=="native_joint_ring_profile" && e.dimensions.at("active_ctas")==128);
  }
  auto empty_profiles=table;empty_profiles.Set("profiles",json::Array{});
  auto missing_all=cal;missing_all.Set("joint_ring_profiles",empty_profiles);
  Sm90DenseCostModel uncovered(t,missing_all);auto all_profiles=uncovered.Preflight(w,c);
  assert(!all_profiles.prediction && all_profiles.missing.size()==5);
  for(auto const& error:all_profiles.missing)assert(error.metric=="native_joint_ring_profile");
  try {uncovered.Request(w,c);assert(false);}
  catch(Sm90DenseMissingMeasurement const& error) {assert(error.metric=="native_joint_ring_profile");}
  auto multiple=w;multiple.shapes[0][0]*=2;
  assert(error(cal,multiple,c).find("calibration pending")!=std::string::npos);
  auto prefix=c;prefix.gemms[0].preissue_smem_bytes=prefix.gemms[0].slot_bytes;
  assert(!error(cal,w,prefix).empty());
  auto partial=profiles;auto cold=partial[0].At("finite_prefix").AsArray("prefix");cold.pop_back();
  partial[0].Set("finite_prefix",cold);bad=table;bad.Set("profiles",partial);
  bool rejected=false;try {Sm90DenseJointRingProfiles reader(bad,std::string(64,'a'));}
  catch(std::invalid_argument const&) {rejected=true;}assert(rejected);
  // I=1 and I<S use their measured finite prefix, never a steady extrapolation.
  for(int iterations:{1,2}) {
    auto short_profile=profiles[0];int tk=int(short_profile.At("tile_k").AsNumber("tk"));
    short_profile.Set("input_k",iterations*tk);
    cold=short_profile.At("finite_prefix").AsArray("prefix");cold.resize(iterations);
    short_profile.Set("finite_prefix",cold);bad=table;bad.Set("profiles",json::Array{short_profile});
    Sm90DenseJointRingProfiles reader(bad,std::string(64,'a'));std::map<std::string,double> dims;
    for(auto const& name:Sm90DenseJointRingProfiles::Dimensions())dims[name]=short_profile.At(name).AsNumber(name.c_str());
    assert(reader.Find(dims,"four_shape_replay_no_scrub"));
    assert(!reader.Find(dims,"four_shape_replay_no_scrub","wgmma"));
    dims["active_ctas"]=64;assert(!reader.Find(dims,"four_shape_replay_no_scrub"));
  }
}
int main() {
  TargetSpec t;t.arch_tag="sm_90";t.nvcc_arch="sm_90a";
  t.res.num_sms=132;t.res.max_smem_per_sm=233472;t.res.max_dynamic_smem_per_cta=232448;
  t.res.l2_bytes=52428800;
  JointRingContracts(t);
  Sm90DenseWorkload w{{{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{151936,1024}}},
      1,28,128,2,true};
  auto c=Sm90DenseCoordinates::Exact(w.shapes,w.batch,t.res.max_dynamic_smem_per_cta);
  Sm90DenseCostModel model(t,Fixture());auto prediction=model.Request(w,c);
  auto coverage=model.Preflight(w,c);
  assert(coverage.missing.empty() && coverage.prediction->ToJson().Dump(0)==prediction.ToJson().Dump(0));
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
  assert(prediction.stages[0].barrier==1000);
  auto resident_two=Sm90DenseCoordinates::Exact(w.shapes,w.batch,Sm90DenseSharedBudget(t,2));
  resident_two.residency=2;
  assert(model.Predict(w,resident_two).stages[1].flow==7000);
  assert(model.Predict(w,resident_two).stages[0].barrier==1400);
  auto missing_grid=Fixture();json::Array single_grid;
  for(auto const& row:missing_grid.At("observations").AsArray("observations"))
    if(row.At("metric").AsString("metric").rfind("attention_",0)!=0 ||
       row.At("grid").AsNumber("grid")==132)single_grid.push_back(row);
  missing_grid.Set("observations",single_grid);bool missing_attention_grid=false;
  try {Sm90DenseCostModel(t,missing_grid).Predict(w,resident_two);}
  catch(std::invalid_argument const& e) {
    missing_attention_grid=std::string(e.what()).find("attention_critical_before_barrier")!=std::string::npos;
  }
  assert(missing_attention_grid);
  auto more_chunks=c;more_chunks.attention.chunks=32;
  assert(model.Predict(w,more_chunks).stages[1].flow==12000);
  more_chunks.attention.chunks=64;
  assert(model.Predict(w,more_chunks).stages[1].flow==24000);
  auto missing_chunks=Fixture();json::Array without_sixty_four;
  for(auto const& row:missing_chunks.At("observations").AsArray("observations"))
    if(row.At("metric").AsString("metric")!="attention_critical_before_barrier" ||
        row.At("chunks").AsNumber("chunks")!=64)without_sixty_four.push_back(row);
  missing_chunks.Set("observations",without_sixty_four);bool rejected_chunks=false;
  try {Sm90DenseCostModel(t,missing_chunks).Predict(w,more_chunks);}
  catch(std::invalid_argument const& e) {
    rejected_chunks=std::string(e.what()).find("attention_critical_before_barrier")!=std::string::npos;
  }
  assert(rejected_chunks);
  auto rejects=[&](json::Value const& calibration,Sm90DenseWorkload const& work,
      Sm90DenseCoordinates const& coordinates) {
    try {Sm90DenseCostModel(t,calibration).Predict(work,coordinates);}
    catch(std::invalid_argument const& error) {return std::string(error.what());}
    assert(false && "missing exact attention calibration must be rejected");
    return std::string{};
  };
  auto without_field=[](json::Value value,char const* name) {
    auto fields=value.AsObject("object");
    fields.erase(std::remove_if(fields.begin(),fields.end(),[&](auto const& field) {
      return field.first==name;
    }),fields.end());
    return json::Value(fields);
  };
  auto old_attention=Fixture();old_attention.Set("attention_grid_model","explicit_grid_chunks_v2");
  assert(rejects(old_attention,w,c).find("exact geometry and cache")!=std::string::npos);
  for(int bad:{0,1,3}) {
    auto wrong_residency=Fixture();auto bad_rows=wrong_residency.At("observations").AsArray("observations");
    for(auto& row:bad_rows)
      if(row.At("metric").AsString("metric")=="attention_critical_before_barrier" &&
          row.At("grid").AsNumber("grid")==264) {
        if(bad==0)row=without_field(row,"compiled_residency");
        else row.Set("compiled_residency",bad);
      }
    wrong_residency.Set("observations",bad_rows);
    auto error=rejects(wrong_residency,w,resident_two);
    for(auto key:{"compiled_residency", "attention_critical_before_barrier", "grid=264", "num_sms=132"})
      assert(error.find(key)!=std::string::npos);
  }
  assert(rejects(without_field(Fixture(),"attention_cache_state"),w,c)
      .find("attention_cache_state")!=std::string::npos);
  auto unknown_cache=Fixture();unknown_cache.Set("attention_cache_state","unspecified");
  assert(rejects(unknown_cache,w,c).find("unsupported native attention_cache_state")!=std::string::npos);
  auto missing_cache=Fixture();auto uncached_rows=missing_cache.At("observations").AsArray("observations");
  for(auto& row:uncached_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier")
      row=without_field(row,"cache_state");
  missing_cache.Set("observations",uncached_rows);
  assert(rejects(missing_cache,w,c).find("explicit cache_state")!=std::string::npos);
  auto wrong_cache=Fixture();wrong_cache.Set("attention_cache_state","l2_scrub_surrogate");
  auto cache_error=rejects(wrong_cache,w,c);
  for(auto key:{"attention_critical_before_barrier","cache_state=l2_scrub_surrogate",
      "batch=1","chunks=16","grid=132"})assert(cache_error.find(key)!=std::string::npos);
  auto mixed_cache=Fixture();auto mixed_rows=mixed_cache.At("observations").AsArray("observations");
  auto warm_rows=mixed_rows;
  for(auto row:warm_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier") {
      row.Set("cache_state","l2_scrub_surrogate");
      row.Set("median",row.At("median").AsNumber("median")*10);mixed_rows.push_back(row);
    }
  mixed_cache.Set("observations",mixed_rows);
  assert(Sm90DenseCostModel(t,mixed_cache).Predict(w,c).stages[1].flow==6000);
  mixed_cache.Set("attention_cache_state","l2_scrub_surrogate");
  assert(Sm90DenseCostModel(t,mixed_cache).Predict(w,c).stages[1].flow==60000);
  std::reverse(mixed_rows.begin(),mixed_rows.end());mixed_cache.Set("observations",mixed_rows);
  assert(Sm90DenseCostModel(t,mixed_cache).Predict(w,c).stages[1].flow==60000);
  auto unused_axes=c;unused_axes.attention.block_keys=128;unused_axes.attention.slots=8;
  assert(model.Predict(w,unused_axes).step_ns==model.Predict(w,c).step_ns);
  auto unused_labels=Fixture();auto label_rows=unused_labels.At("observations").AsArray("observations");
  for(auto& row:label_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier" &&
        row.At("batch").AsNumber("batch")==1) {
      row.Set("block_keys",16);row.Set("slots",8);
    }
  unused_labels.Set("observations",label_rows);
  assert(Sm90DenseCostModel(t,unused_labels).Predict(w,c).step_ns==model.Predict(w,c).step_ns);
  auto missing_work=Fixture();json::Array row_work_zero;
  for(auto const& row:missing_work.At("observations").AsArray("observations"))
    if(row.At("metric").AsString("metric")!="attention_critical_before_barrier" ||
        row.At("new_row_work").AsNumber("new row work")==0)row_work_zero.push_back(row);
  missing_work.Set("observations",row_work_zero);
  auto row_work=c;row_work.attention.new_row_work=8;
  auto work_error=rejects(missing_work,w,row_work);
  assert(work_error.find("new_row_work=8")!=std::string::npos &&
      work_error.find("chunks=16")!=std::string::npos);
  double total=0;
  for(auto const& p:prediction.stages) {
    assert(p.ns>=p.dram && p.ns>=p.barrier && p.ns>0);
    total+=p.ns*p.instances;
  }
  assert(std::abs(total-prediction.step_ns)<1e-6);
  for(auto const& p:prediction.stages)
    assert(p.initialization==(p.key.rfind("gemm_",0)==0?500:0));
  assert(prediction.stages[5].waves>1 && prediction.stages[5].initialization==500);
  assert(prediction.stages[0].epilogue==450);
  auto repeated=Fixture();auto repeated_rows=repeated.At("observations").AsArray("observations");
  auto original_rows=repeated_rows;
  for(auto row:original_rows)
    if(row.At("metric").AsString("metric")=="bulk_single_completion") {
      row.Set("median",1000);repeated_rows.push_back(row);
    }
  repeated.Set("observations",repeated_rows);
  assert(Sm90DenseCostModel(t,repeated).Predict(w,c).stages[0].read_startup==800);
  std::reverse(repeated_rows.begin(),repeated_rows.end());repeated.Set("observations",repeated_rows);
  assert(Sm90DenseCostModel(t,repeated).Predict(w,c).stages[0].read_startup==800);
  auto slow_activation=Fixture();auto activation_rows=slow_activation.At("observations").AsArray("observations");
  for(auto& row:activation_rows)
    if(row.At("metric").AsString("metric")=="activation_wait_and_shared_read")
      row.Set("median",1e6);
  slow_activation.Set("observations",activation_rows);
  auto limited=Sm90DenseCostModel(t,slow_activation).Predict(w,c);
  auto const& qkv=limited.stages[0];
  assert(qkv.flow==qkv.activation*.2 && qkv.flow>qkv.dram && qkv.flow>qkv.stream);
  auto weight_prefix=c;
  weight_prefix.gemms[0].workspace_offset=115712;
  weight_prefix.gemms[0].preissue_smem_bytes=weight_prefix.gemms[0].slot_bytes;
  auto activation_bound=Sm90DenseCostModel(t,slow_activation).Predict(w,weight_prefix).stages[0];
  assert(activation_bound.overlap_limit==activation_bound.read_startup);
  assert(activation_bound.ns+1e-6>=activation_bound.barrier+activation_bound.startup-
      activation_bound.read_startup+activation_bound.activation*.2+activation_bound.epilogue);
  auto stale_overlap=Fixture();stale_overlap.Set("overlap_model","operand_service_only_v1");
  bool stale_overlap_rejected=false;
  try {Sm90DenseCostModel(t,stale_overlap);}catch(std::invalid_argument const&) {
    stale_overlap_rejected=true;
  }
  assert(stale_overlap_rejected);
  auto stale=Fixture();stale.Set("flow_model","serial_operands");bool stale_rejected=false;
  try {Sm90DenseCostModel old(t,stale);}catch(std::invalid_argument const&) {stale_rejected=true;}
  assert(stale_rejected);
  auto full_batch=w;full_batch.batch=16;
  auto balanced=Fixture();auto balanced_rows=balanced.At("observations").AsArray("observations");
  for(auto& row:balanced_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier" &&
        row.At("batch").AsNumber("batch")==1 &&
        row.At("new_row_work").AsNumber("new row work")==8) {
      row.Set("median",5000);
    }
  balanced.Set("observations",balanced_rows);auto balanced_c=c;balanced_c.attention.new_row_work=8;
  assert(Sm90DenseCostModel(t,balanced).Predict(w,balanced_c).stages[1].flow==5000);
  auto missing_batch=w;missing_batch.batch=2;bool missing_attention_batch=false;
  for(auto row:balanced.At("observations").AsArray("observations"))
    if(row.At("metric").AsString("metric")=="activation_inactive_rows_initialize") {
      row.Set("batch",2);balanced_rows.push_back(row);
    }
  balanced.Set("observations",balanced_rows);
  try {Sm90DenseCostModel(t,balanced).Predict(missing_batch,c);}
  catch(std::invalid_argument const& e) {
    missing_attention_batch=std::string(e.what()).find("attention_critical_before_barrier")!=std::string::npos;
  }
  assert(missing_attention_batch);
  auto full_coordinates=Sm90DenseCoordinates::Exact(w.shapes,16,t.res.max_dynamic_smem_per_cta);
  for(auto const& p:model.Request(full_batch,full_coordinates).stages)assert(p.initialization==0);
  auto different_keys=full_coordinates;different_keys.attention.block_keys=32;
  auto key_error=rejects(Fixture(),full_batch,different_keys);
  assert(key_error.find("block_keys=32")!=std::string::npos && key_error.find("batch=16")!=std::string::npos);
  auto different_slots=full_coordinates;different_slots.attention.slots=4;
  auto slot_error=rejects(Fixture(),full_batch,different_slots);
  assert(slot_error.find("slots=4")!=std::string::npos && slot_error.find("batch=16")!=std::string::npos);
  auto single_chunk=full_coordinates;single_chunk.attention.new_row_work=16;
  assert(single_chunk.attention.chunks==1);
  assert(model.Predict(full_batch,single_chunk).step_ns==model.Predict(full_batch,full_coordinates).step_ns);
  auto irrelevant_work=Fixture();auto irrelevant_rows=irrelevant_work.At("observations").AsArray("observations");
  for(auto& row:irrelevant_rows)
    if(row.At("metric").AsString("metric")=="attention_critical_before_barrier" &&
        row.At("chunks").AsNumber("chunks")==1)row.Set("new_row_work",16);
  irrelevant_work.Set("observations",irrelevant_rows);
  assert(Sm90DenseCostModel(t,irrelevant_work).Predict(full_batch,full_coordinates).step_ns==
      model.Predict(full_batch,full_coordinates).step_ns);
  auto overlap_calibration=Fixture();auto overlap_fits=overlap_calibration.At("body_fits").AsObject("fits");
  for(auto& fit:overlap_fits) {
    auto coefficients=fit.second.At("coefficients");
    if(coefficients.Find("preissue_efficiency")) {
      coefficients.Set("preissue_efficiency",1e6);coefficients.Set("tail_efficiency",1e6);
      coefficients.Set("l2_efficiency",1e6);
    }
    if(coefficients.Find("body_scale"))coefficients.Set("body_scale",20);
    fit.second.Set("coefficients",coefficients);
  }
  overlap_calibration.Set("body_fits",overlap_fits);
  Sm90DenseCostModel transfer_model(t,overlap_calibration);
  auto many_tasks=full_coordinates;many_tasks.attention.chunks=16;
  many_tasks.attention.preissue_smem_bytes=32768;
  auto first_wave=transfer_model.Predict(full_batch,many_tasks).stages[1];
  assert(first_wave.tasks==2048 && first_wave.waves==16);
  assert(first_wave.preissue_service+first_wave.late_service==768);
  assert(first_wave.overlap==768 && first_wave.overlap_limit==768);
  assert(first_wave.ns>=first_wave.barrier+first_wave.startup+
      first_wave.flow*(first_wave.waves-1)/first_wave.waves);
  auto slower_attention=overlap_calibration;auto slower_fits=overlap_fits;
  for(auto& fit:slower_fits) {
    auto coefficients=fit.second.At("coefficients");
    if(coefficients.Find("body_scale"))coefficients.Set("body_scale",40);
    fit.second.Set("coefficients",coefficients);
  }
  slower_attention.Set("body_fits",slower_fits);
  auto slower_wave=Sm90DenseCostModel(t,slower_attention).Predict(full_batch,many_tasks).stages[1];
  assert(slower_wave.flow==2*first_wave.flow && slower_wave.overlap==first_wave.overlap);
  assert(slower_wave.preissue_service==first_wave.preissue_service &&
      slower_wave.late_service==first_wave.late_service);
  many_tasks.attention.new_row_work=8;
  auto weighted=transfer_model.Predict(full_batch,many_tasks,64).stages[1];
  double first_task_bytes=0,max_task_bytes=0;
  for(int task=0;task<132;++task) {
    int chunk=task%16;
    int begin=std::min(64,73*chunk/16),end=std::min(64,73*(chunk+1)/16);
    double bytes=4.*128*(end-begin);
    first_task_bytes+=bytes;max_task_bytes=std::max(max_task_bytes,bytes);
  }
  assert(weighted.preissue_service+weighted.late_service==
      std::max(first_task_bytes/(132*24.),max_task_bytes/24.));
  auto short_l2=full_coordinates;short_l2.attention.prefetch_l2_bytes=32;
  auto short_prefetch=transfer_model.Predict(full_batch,short_l2);
  assert(short_prefetch.stages[1].l2_service==32./24. &&
      short_prefetch.stages[1].overlap==short_prefetch.stages[1].l2_service);
  assert(short_prefetch.stages[0].issue_count==2);
  auto head_prefix=c;head_prefix.gemms[4].preissue_smem_bytes=8192;
  auto head=transfer_model.Predict(w,head_prefix).stages[5];
  assert(head.waves>1 && head.overlap_limit==head.read_startup+
      head.preissue_service+head.late_service+head.l2_service);
  assert(head.overlap==head.overlap_limit && head.overlap<head.stream/head.waves);
  auto reused=c;reused.attention.workspace_offset=32768;
  assert(reused.ActivationReuse(w.batch,w.head_dim,w.queries,1088).valid);
  auto reused_prediction=model.Predict(w,reused);
  for(unsigned i:{0u,2u,3u,4u,5u})assert(reused_prediction.stages[i].initialization==0);
  assert(!c.ActivationReuse(w.batch,w.head_dim,w.queries,1088).valid);
  auto other_slots=reused;other_slots.gemms[0].slots=2;
  assert(!other_slots.ActivationReuse(w.batch,w.head_dim,w.queries,1088).valid);
  auto other_offset=reused;other_offset.gemms[0].workspace_offset=1024;
  assert(!other_offset.ActivationReuse(w.batch,w.head_dim,w.queries,1088).valid);
  assert(!reused.ActivationReuse(16,w.head_dim,w.queries,1088).valid);
  using AStage=backend::Sm90DenseActivationStage;
  std::array<AStage,3> lifetime{{{256,3,0,{1024,80000},false},
      {0,0,0,{25600,60000},false},{0,0,0,{1024,1056},true}}};
  assert(backend::Sm90DenseActivationReuse::Prove(lifetime,1).valid);
  lifetime[1].written.begin=25599;
  assert(!backend::Sm90DenseActivationReuse::Prove(lifetime,1).valid);
  lifetime[1].written.begin=25600;lifetime[2].written.end=80000;
  assert(!backend::Sm90DenseActivationReuse::Prove(lifetime,1).valid);
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
  auto stale_receiver=Fixture();stale_receiver.Set("receiver_model","old");
  bool rejected_receiver=false;
  try {Sm90DenseCostModel old(t,stale_receiver);}catch(std::invalid_argument const& e) {
    rejected_receiver=std::string(e.what()).find("receiver protocol")!=std::string::npos;
  }
  assert(rejected_receiver);
  auto stale_protocol=Fixture();stale_protocol.Set("protocol_key_model","shared_body_v0");
  bool rejected_protocol=false;
  try {Sm90DenseCostModel old(t,stale_protocol);}catch(std::invalid_argument const& e) {
    rejected_protocol=std::string(e.what()).find("stage and batch protocol keys")!=std::string::npos;
  }
  assert(rejected_protocol);
  auto receive_coordinates=Sm90DenseSearchSpace{w.shapes,w.batch,w.head_dim,w.queries,t}
      .MaxPreissue(c,false);
  auto zero_receive=issue_model.Request(w,receive_coordinates);
  auto receive_calibration=issue_only;
  auto receive_fits=receive_calibration.At("body_fits").AsObject("fits");
  for(auto& fit:receive_fits) {
    auto coefficients=fit.second.At("coefficients");coefficients.Set("receive_ns",170);
    fit.second.Set("coefficients",coefficients);
  }
  receive_calibration.Set("body_fits",receive_fits);
  auto receiving=Sm90DenseCostModel(t,receive_calibration).Request(w,receive_coordinates);
  assert(receiving.stages[5].waves>1 && receiving.stages[5].receive_count==1);
  auto packets=c;packets.gemms[0].workspace_offset=96*1024;
  packets.gemms[0].preissue_smem_bytes=packets.gemms[0].slot_bytes;
  auto one_packet=issue_model.Predict(w,packets);
  packets.gemms[0].preissue_smem_bytes*=2;
  auto two_packets=issue_model.Predict(w,packets);
  assert(one_packet.stages[6].issue_count==2 && two_packets.stages[6].issue_count==4);
  assert(two_packets.stages[6].issue==2*one_packet.stages[6].issue);
  for(unsigned i=0;i<receiving.stages.size();++i) {
    auto const& s=receiving.stages[i];
    assert(s.receive==170*s.receive_count);
    assert(std::abs(s.ns-zero_receive.stages[i].ns-s.receive)<1e-6);
  }
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
  for(auto const& s:attention_issue.stages)assert(s.receive_count==0 && s.receive==0);
  assert(attention_issue.stages[0].issue==40 && attention_issue.stages[1].issue==0);
  auto issuer=issue_only;auto issuer_fits=issuer.At("body_fits").AsObject("body fits");
  for(auto& fit:issuer_fits) {
    auto coefficients=fit.second.At("coefficients");
    if(fit.first=="gemm_mma_sync_tn8_32_wk1:qkv_low")coefficients.Set("issue_ns",200);
    if(fit.first=="attention_simt_d128_q2_norm1:attention_low")coefficients.Set("issue_ns",9999);
    fit.second.Set("coefficients",coefficients);
  }
  issuer.Set("body_fits",issuer_fits);
  auto issuer_price=Sm90DenseCostModel(t,issuer).Predict(w,prefetch);
  assert(issuer_price.stages[0].issue==400 && issuer_price.stages[0].issue_count==2);
  auto isolated=issuer;auto isolated_fits=isolated.At("body_fits").AsObject("body fits");
  for(auto& fit:isolated_fits)
    if(fit.first=="gemm_mma_sync_tn8_32_wk1:qkv_high" ||
       fit.first=="gemm_mma_sync_tn8_32_wk1:o_low") {
      auto coefficients=fit.second.At("coefficients");coefficients.Set("issue_ns",1e6);
      fit.second.Set("coefficients",coefficients);
    }
  isolated.Set("body_fits",isolated_fits);
  assert(Sm90DenseCostModel(t,isolated).Predict(w,prefetch).stages[0].issue==400);
  assert(issuer_price.stages[0].protocol_key=="gemm_mma_sync_tn8_32_wk1:qkv_low");
  for(auto& fit:issuer_fits) {
    auto coefficients=fit.second.At("coefficients");
    if(fit.first=="gemm_mma_sync_tn8_32_wk1:qkv_low")coefficients.Set("issue_service_scale",.5);
    fit.second.Set("coefficients",coefficients);
  }
  issuer.Set("body_fits",issuer_fits);
  auto transfer_price=Sm90DenseCostModel(t,issuer).Predict(w,prefetch);
  assert(transfer_price.stages[0].issue_service>0 &&
      transfer_price.stages[0].issue==400+.5*transfer_price.stages[1].l2_service);
  auto prefetched=c;prefetched.gemms[0].prefetch_l2_bytes=2*c.gemms[0].tile_n*w.shapes[0][1];
  auto clipped=model.Predict(w,prefetched);
  prefetched.gemms[0].prefetch_l2_bytes*=4;
  assert(model.Predict(w,prefetched).step_ns==clipped.step_ns);
  for(auto const& s:clipped.stages)if(s.key.rfind("gemm_",0)==0)
    assert(s.ns+1e-6>=s.barrier+s.initialization+s.compute+s.epilogue+s.issue);
  prefetched=c;prefetched.attention.prefetch_l2_bytes=4*575*w.head_dim;
  auto kv_clipped=model.Predict(w,prefetched);
  prefetched.attention.prefetch_l2_bytes*=4;
  assert(model.Predict(w,prefetched).step_ns==kv_clipped.step_ns);
  auto expensive_dispatch=Fixture();auto dispatch_fits=expensive_dispatch.At("body_fits").AsObject("body fits");
  for(auto& fit:dispatch_fits) {
    auto coefficients=fit.second.At("coefficients");
    coefficients.Set("startup_ns",1e6);fit.second.Set("coefficients",coefficients);
  }
  expensive_dispatch.Set("body_fits",dispatch_fits);
  auto dispatched=Sm90DenseCostModel(t,expensive_dispatch).Predict(w,prefetched);
  for(unsigned i=0;i<6;++i) {
    assert(dispatched.stages[i].overlap_limit==kv_clipped.stages[i].overlap_limit);
    assert(dispatched.stages[i].overlap==kv_clipped.stages[i].overlap);
  }
  auto residual_dn=Fixture();auto epi_rows=residual_dn.At("observations").AsArray("observations");
  for(auto& row:epi_rows)
    if(row.At("metric").AsString("metric")=="epilogue_residual_return" &&
        row.At("cache").AsString("cache")=="dn")row.Set("median",1e9);
  residual_dn.Set("observations",epi_rows);
  auto ignored_dn=Sm90DenseCostModel(t,residual_dn).Predict(w,c);
  for(unsigned stage:{2u,4u})assert(ignored_dn.stages[stage].epilogue==200);
  auto argmax_price=Fixture();auto op_fits=argmax_price.At("body_fits").AsObject("fits");
  for(auto& fit:op_fits)if(fit.first=="native_epilogue_argmax")
    fit.second.Set("coefficients",json::Object{{"scale",2}});
  argmax_price.Set("body_fits",op_fits);
  auto argmax_prediction=Sm90DenseCostModel(t,argmax_price).Predict(w,c);
  auto plain_prediction=model.Predict(w,c);
  for(unsigned i=0;i<argmax_prediction.stages.size();++i)
    assert(argmax_prediction.stages[i].epilogue==plain_prediction.stages[i].epilogue*(i==5?2:1));
  auto unmeasured=w;unmeasured.shapes[0][1]=512;bool missing_width=false;
  try {model.Predict(unmeasured,c);}catch(std::invalid_argument const&) {missing_width=true;}
  assert(missing_width);
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
  auto without_observations=[](auto remove) {
    auto fixture=Fixture();json::Array rows;
    for(auto const& row:fixture.At("observations").AsArray("observations"))
      if(!remove(row))rows.push_back(row);
    fixture.Set("observations",rows);return fixture;
  };
  auto no_attention=without_observations([](auto const& row) {
    return row.At("metric").AsString("metric")=="attention_critical_before_barrier";
  });
  Sm90DenseCostModel missing_attention(t,no_attention);
  for(int repeat=0;repeat<2;++repeat) {
    bool typed=false;
    try {missing_attention.RequireAttentionCalibration(w,unused_axes);}
    catch(Sm90DenseMissingMeasurement const& error) {
      typed=error.metric=="attention_critical_before_barrier" &&
          error.dimensions.at("block_keys")==64 && error.dimensions.at("slots")==3 &&
          error.labels.at("cache_state")=="warm_reuse";
      assert(error.ToJson().At("dimensions").At("chunks").AsNumber("chunks")==16);
    }
    assert(typed);
  }
  auto protected_seeds=space.Seeds();auto activation_seeds=space.ActivationReuseSeeds();
  protected_seeds.insert(protected_seeds.end(),activation_seeds.begin(),activation_seeds.end());
  std::set<std::string> seed_coordinates;
  for(auto const& seed:protected_seeds)seed_coordinates.insert(seed.ToJson().Dump(0));
  bool pending_seeds=false;
  try {SearchSm90Dense(space,w,missing_attention,options);}
  catch(Sm90DenseCalibrationPending const& pending) {
    auto const& report=pending.report;
    pending_seeds=report.At("status").AsString("status")=="calibration_pending" &&
        report.At("frontier").AsString("frontier")=="seeds";
    assert(report.At("evaluated").AsNumber("evaluated")==0);
    assert(!report.At("converged").AsBool("converged") &&
        !report.At("selection_performed").AsBool("selected"));
    assert(!report.Find("ranked") && !report.Find("winner"));
    auto const& missing=report.At("missing").AsArray("missing");assert(missing.size()==32);
    std::set<std::string> recorded;
    std::map<std::string,int> occurrences;
    for(auto const& row:missing)for(auto const& candidate:row.At("candidates").AsArray("candidates"))
      {auto key=candidate.At("coordinates").Dump(0);recorded.insert(key);++occurrences[key];}
    assert(recorded==seed_coordinates);
    for(auto const& count:occurrences)assert(count.second==16);
  }
  assert(pending_seeds);
  auto missing_neighbours=without_observations([](auto const& row) {
    return row.At("metric").AsString("metric")=="attention_critical_before_barrier" &&
        (row.At("chunks").AsNumber("chunks")==4 || row.At("chunks").AsNumber("chunks")==32);
  });
  Sm90DenseCostModel incomplete_frontier(t,missing_neighbours);
  std::set<std::string> expected_coordinates;
  std::set<std::array<int,3>> expected_tuples;
  for(auto const& seed:protected_seeds)for(auto const& candidate:space.Neighbours(seed))
    if(candidate.attention.chunks==4 || candidate.attention.chunks==32) {
      expected_coordinates.insert(candidate.ToJson().Dump(0));
      expected_tuples.insert({candidate.residency*132,candidate.attention.chunks,
          candidate.attention.new_row_work});
    }
  assert(expected_tuples.size()>2);
  bool pending_beam=false;auto tiny_budget=options;tiny_budget.budget_ms=1;
  try {SearchSm90Dense(space,w,incomplete_frontier,tiny_budget);}
  catch(Sm90DenseCalibrationPending const& pending) {
    auto const& report=pending.report;
    pending_beam=report.At("frontier").AsString("frontier")=="beam" &&
        report.At("pass").AsNumber("pass")==0;
    assert(report.At("evaluated").AsNumber("evaluated")==seed_coordinates.size());
    assert(!report.Find("ranked") && !report.Find("winner"));
    std::set<std::string> recorded;std::set<std::array<int,3>> tuples;
    for(auto const& row:report.At("missing").AsArray("missing")) {
      auto const& dims=row.At("dimensions");
      assert(row.At("labels").At("cache_state").AsString("cache")=="warm_reuse");
      tuples.insert({int(dims.At("grid").AsNumber("grid")),
          int(dims.At("chunks").AsNumber("chunks")),
          int(dims.At("new_row_work").AsNumber("work"))});
      for(auto const& candidate:row.At("candidates").AsArray("candidates")) {
        assert(space.Legal(Sm90DenseCoordinates::FromJson(candidate.At("coordinates"))));
        recorded.insert(candidate.At("coordinates").Dump(0));
      }
    }
    assert(tuples==expected_tuples && recorded==expected_coordinates);
  }
  assert(pending_beam);
  bool non_attention_error=false;
  try {SearchSm90Dense(space,w,Sm90DenseCostModel(t,no_initialization),options);}
  catch(Sm90DenseCalibrationPending const& pending) {
    assert(!pending.report.At("selection_performed").AsBool("selected"));
    for(auto const& error:pending.report.At("missing").AsArray("missing")) {
      assert(error.At("metric").AsString("metric")=="activation_inactive_rows_initialize");
      non_attention_error=true;
    }
  }
  assert(non_attention_error);
  auto several_missing=without_observations([](auto const& row) {
    auto metric=row.At("metric").AsString("metric");
    return metric=="activation_inactive_rows_initialize" || metric=="bulk_single_completion" ||
        metric=="native_ldmatrix_mma_k16" || metric=="epilogue_argmax_return_at_input_k";
  });
  Sm90DenseCostModel several(t,several_missing);auto complete_missing=several.Preflight(w,c);
  assert(!complete_missing.prediction);
  std::set<std::string> missing_metrics;
  for(auto const& error:complete_missing.missing)missing_metrics.insert(error.metric);
  assert(missing_metrics==std::set<std::string>({"activation_inactive_rows_initialize",
      "bulk_single_completion","native_ldmatrix_mma_k16","epilogue_argmax_return_at_input_k"}));
  assert(several.Preflight(w,c).missing.size()==complete_missing.missing.size());
  try {several.Request(w,c);assert(false);}
  catch(Sm90DenseMissingMeasurement const&) {}
  bool aggregated_services=false;
  try {SearchSm90Dense(space,w,several,options);}
  catch(Sm90DenseCalibrationPending const& pending) {
    std::set<std::string> metrics;
    assert(pending.report.At("evaluated").AsNumber("evaluated")==0);
    assert(!pending.report.Find("ranked") && !pending.report.Find("winner"));
    for(auto const& error:pending.report.At("missing").AsArray("missing"))
      metrics.insert(error.At("metric").AsString("metric"));
    aggregated_services=metrics==missing_metrics;
  }
  assert(aggregated_services);
  auto no_capacity=without_observations([](auto const& row) {
    return row.At("metric").AsString("metric")=="bulk_stream_per_active_cta";
  });
  Sm90DenseCostModel capacity_missing(t,no_capacity);auto capacity_coverage=capacity_missing.Preflight(w,c);
  std::set<std::string> capacity_metrics;
  for(auto const& error:capacity_coverage.missing)capacity_metrics.insert(error.metric);
  assert(!capacity_coverage.prediction && capacity_metrics==std::set<std::string>({
      "bulk_stream_per_active_cta","native_dram_streaming_capacity"}));
  try {capacity_missing.Request(w,c);assert(false);}
  catch(Sm90DenseMissingMeasurement const& error) {assert(error.metric=="native_dram_streaming_capacity");}
  bool missing_coefficients=false;
  try {SearchSm90Dense(space,w,Sm90DenseCostModel(t,incomplete),options);}
  catch(Sm90DenseCalibrationPending const& pending) {
    std::set<std::string> names;
    for(auto const& error:pending.report.At("missing").AsArray("missing")) {
      assert(error.At("metric").AsString("metric")=="native_fitted_coefficient");
      names.insert(error.At("labels").At("coefficient").AsString("coefficient"));
    }
    missing_coefficients=names.count("startup_ns") && names.count("mma_scale") && names.count("receive_ns");
  }
  assert(missing_coefficients);
  bool other_error=false;
  auto invalid_fit=Fixture();auto invalid_fits=invalid_fit.At("body_fits").AsObject("fits");
  for(auto& fit:invalid_fits) {
    auto coefficients=fit.second.At("coefficients");
    if(coefficients.Find("startup_ns"))coefficients.Set("startup_ns",-1);
    fit.second.Set("coefficients",coefficients);
  }
  invalid_fit.Set("body_fits",invalid_fits);Sm90DenseCostModel malformed(t,invalid_fit);
  try {SearchSm90Dense(space,w,malformed,options);}
  catch(Sm90DenseCalibrationPending const&) {assert(false && "invalid body fit was swallowed");}
  catch(std::invalid_argument const& error) {
    other_error=std::string(error.what()).find("invalid native fitted coefficient")!=std::string::npos;
  }
  assert(other_error);
  try {malformed.Request(w,c);assert(false);}
  catch(std::invalid_argument const& error) {
    assert(std::string(error.what()).find("invalid native fitted coefficient")!=std::string::npos);
  }
  auto wrong_type=Fixture();auto wrong_type_fits=wrong_type.At("body_fits").AsObject("fits");
  for(auto& fit:wrong_type_fits)fit.second.Set("coefficients",json::Array{});
  wrong_type.Set("body_fits",wrong_type_fits);bool wrong_type_rejected=false;
  try {Sm90DenseCostModel(t,wrong_type).Preflight(w,c);}
  catch(Sm90DenseMissingMeasurement const&) {assert(false && "malformed coefficients became missing calibration");}
  catch(std::runtime_error const&) {wrong_type_rejected=true;}
  assert(wrong_type_rejected);
  auto search=SearchSm90Dense(space,w,model,options);
  assert(search.evaluated>space.Seeds().size() && search.ranked.size()>=8);
  for(int field=0;field<8;++field) {
    auto base=c;
    if(field==2)base.gemms[0].prefetch_l2_bytes=32;
    if(field==5)base.attention.prefetch_l2_bytes=32;
    auto changed=base;
    if(field==0)changed.gemms[0].workspace_offset+=1024;
    if(field==1)changed.gemms[0].register_stages=2;
    if(field==2)changed.gemms[0].preissue_late=true;
    if(field==3)changed.gemms[0].prefetch_l2_bytes=32;
    if(field==4)changed.attention.workspace_offset+=1024;
    if(field==5)changed.attention.preissue_late=true;
    if(field==6)changed.attention.prefetch_l2_bytes=32;
    if(field==7)changed.gemms[0].slots+=1;
    assert(space.Legal(base) && space.Legal(changed));
    assert(Sm90DenseScheduleKey(changed,prediction)!=Sm90DenseScheduleKey(base,prediction));
  }
  auto masked_compute=prediction;masked_compute.stages[0].compute+=1;
  assert(Sm90DenseScheduleKey(c,masked_compute)==Sm90DenseScheduleKey(c,prediction));
  auto different_price=prediction;different_price.step_ns+=1;
  assert(Sm90DenseScheduleKey(c,different_price)!=Sm90DenseScheduleKey(c,prediction));
  double previous=0;std::set<std::string> unique,schedules;
  for(auto const& p:search.ranked) {
    assert(space.Legal(p.coordinates) && p.prediction.step_ns>=previous);previous=p.prediction.step_ns;
    assert(unique.insert(p.coordinates.ToJson().Dump(0)).second);
    bool fresh=schedules.insert(p.schedule_key).second;
    assert(fresh || p.origin!="search");
    for(auto const& g:p.coordinates.gemms)assert(g.mma==Sm90DenseMma::kMmaSync);
  }
  assert(schedules.size()>=8);
}
