// SPDX-License-Identifier: BSD-3-Clause
// Synthetic CPU contracts only; no values here are calibration data.
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <cassert>
#include <fstream>
#include <iostream>
using namespace tilemega;
using namespace tilemega::solver;

namespace {
json::Value Row(double value,char const* unit="ns") {
  return json::Object{{"median",value},{"unit",unit},{"processes",5}};
}
void Save(char const* path,json::Value const& target) {
  std::ofstream out(path);out<<target.Dump(0);
}
template<class F> void Missing(F action) {
  bool rejected=false;try{action();}catch(Sm90ForwardMissing const&){rejected=true;}
  assert(rejected);
}
bool Near(double a,double b){return std::abs(a-b)<1e-8*std::max(1.,std::abs(b));}
}

int main(int argc,char** argv) {
  assert(argc==3);
  auto target_json=json::ParseFile(argv[1]);
  TargetSpec target;target.arch_tag="sm_90";
  target.res.num_sms=target_json.At("resources").At("num_sms").AsNumber("SMs");
  target.res.max_dynamic_smem_per_cta=target_json.At("resources").At(
      "max_dynamic_smem_per_cta").AsNumber("dynamic shared budget");
  auto native=Sm90NativeConstants::FromTarget(target_json,"");
  json::Object k64,epilogue,split;
  for(int tm:{8,16,64,128,256})for(int tn:{32,64,128,256}) {
    auto tile="tm="+std::to_string(tm)+";tn="+std::to_string(tn);
    for(auto mode:{"normal","gather","swapab"})
      k64.emplace_back(std::string("mode=")+mode+";"+tile,Row(100));
    for(auto cls:kSm90F3ChainClasses)
      epilogue.emplace_back(std::string("class=")+cls+";"+tile,Row(50));
    for(int slices=2;slices<=32;++slices)
      split.emplace_back(tile+";slices="+std::to_string(slices),Row(30));
  }
  json::Object small;
  for(auto kind:{"layout","pool2","pool3","global_pool","layernorm","embedding"})
    small.emplace_back(std::string("kind=")+kind,Row(3,"ns/work_item"));
  json::Value section=json::Object{{"schema",1},{"variant","SYNTHETIC_CPU_ONLY"},
      {"synthetic_cpu_fixture",true},{"method","CPU formula and search contracts"},
      {"header_sha256",json::Object{}},
      {"by_gpu",json::Object{{"1",json::Object{{"f3_k64_ns",k64},
          {"f3_epilogue_ns",epilogue},{"f3_splitk_ns",split},
          {"f3_attention_task_ns",json::Object{{"",Row(900,"ns/task")}}},
          {"f3_small_ns",small}}}}}};
  target_json.Set("sm90_f3_constants",section);Save(argv[2],target_json);
  bool rejected=false;
  try{Sm90ForwardConstants::FromFile(argv[2],1,Sm90ForwardAbi::kF3);}
  catch(std::invalid_argument const&){rejected=true;}assert(rejected);
  setenv("DNN3_CPU_CONTRACT_TEST","1",1);
  setenv("TILEMEGA_FORWARD_CONSTANTS_FILE",argv[2],1);
  // Execution GPU selection must not select a different calibration table.
  setenv("TILEMEGA_DEVICE_INDEX","3",1);
  Sm90ForwardCostModel cost(target,native,Sm90ForwardAbi::kF3);
  auto constants=cost.ForwardConstants();
  assert(constants.Receipt().At("gpu").AsNumber("CPU F3 contract")==1);
  Missing([&]{constants.Get("f3_k64_ns",{{"mode","normal"}},"ns");});
  Missing([&]{constants.Get("f3_k64_ns",{{"mode","absent"},{"tm","128"},{"tn","64"}},"ns");});

  Sm90ForwardShape s;Sm90ForwardStage c;
  s.body=c.body=Sm90ForwardBody::kGemm;s.m=512;s.n=256;s.k=256;
  s.weight_bytes=2*s.n*s.k;
  s.dnn1_geometry=json::Object{{"f3_vectors",0},{"f3_residual",false},
      {"needs_stats",false},{"has_channel_pool",false},{"f3_chain_class","plain"},
      {"epilogue_steps",json::Array{}},{"has_row_stats",false}};
  c.tile_m=128;c.tile_n=64;c.slots=2;c.split_k=2;
  json::Value terms;
  double actual=cost.F3Stage(s,c,false,&terms);
  double service=std::max(2.*(128+64)*128/native.sm_gbps,200.)+50+30;
  double expected=std::max(native.Barrier(132)+native.dram_latency+service,
      s.weight_bytes/native.dram_gbps);
  assert(Near(actual,expected));
  assert(terms.At("tasks").AsNumber("CPU F3 contract")==32 && terms.At("k_iter").AsNumber("CPU F3 contract")==2);
  assert(Near(cost.F3Stage(s,c,true),expected-native.dram_latency+native.l2_latency));

  Sm90ForwardCoordinates coords;coords.abi=Sm90ForwardAbi::kF3;
  coords.stages={c,c};auto prediction=cost.Predict({s,s},coords);
  assert(!prediction.stages[0].terms.At("preissued").AsBool("CPU F3 contract"));
  assert(prediction.stages[1].terms.At("preissued").AsBool("CPU F3 contract"));
  coords.stages[0].slots=8;
  assert(!cost.Predict({s,s},coords).stages[1].terms.At("preissued").AsBool("CPU F3 contract"));
  auto gather=s;gather.mode=Sm90ForwardMode::kGather;
  coords.stages={c,c};coords.stages[1].mode=Sm90ForwardMode::kGather;
  assert(!cost.Predict({s,gather},coords).stages[1].terms.At("preissued").AsBool("CPU F3 contract"));
  auto swap=s;swap.m=1;swap.mode=Sm90ForwardMode::kSwapAB;
  swap.dnn1_geometry.Set("f3_chain_class","swapab_bias_act");
  coords.stages[1].mode=Sm90ForwardMode::kSwapAB;coords.stages[1].tile_n=128;
  prediction=cost.Predict({s,swap},coords);
  assert(prediction.stages[1].terms.At("preissued").AsBool("CPU F3 contract"));
  assert(prediction.stages[1].terms.At("a_task_bytes").AsNumber("CPU F3 contract")==2*8*128);
  swap.m=16;
  assert(cost.Predict({s,swap},coords).stages[1].terms.At("a_task_bytes").AsNumber("CPU F3 contract")==2*16*128);
  auto floor=s;floor.weight_bytes=1000000000000LL;
  assert(Near(cost.F3Stage(floor,c,false),floor.weight_bytes/native.dram_gbps));

  Sm90ForwardShape small_shape;Sm90ForwardStage small_c;
  small_shape.body=small_c.body=Sm90ForwardBody::kLayerNorm;
  small_shape.dnn1_geometry=json::Object{{"rows",2048}};
  assert(Near(cost.F3Stage(small_shape,small_c,false),native.Barrier(132)+native.dram_latency+2*8*3));
  small_shape.body=small_c.body=Sm90ForwardBody::kLayout;
  small_shape.dnn1_geometry=json::Object{{"batch",16},{"width",224},{"height",224}};
  assert(Near(cost.F3Stage(small_shape,small_c,false),native.Barrier(132)+native.dram_latency+3*256*3));
  small_shape.body=small_c.body=Sm90ForwardBody::kEncoderAttention;
  small_shape.dnn1_geometry=json::Object{{"batch",16},{"heads",12}};
  assert(Near(cost.F3Stage(small_shape,small_c,false),native.Barrier(132)+native.dram_latency+2*900));

  auto producer=s;producer.m=16*196;
  producer.dnn1_geometry.Set("batch",16);producer.dnn1_geometry.Set("rows",16*196);
  Sm90ForwardShape gap;gap.body=Sm90ForwardBody::kGlobalPool;gap.channels=256;
  gap.dnn1_geometry=json::Object{{"pool_producer_stage",0},{"batch",16},{"channels",256}};
  auto linked=Sm90ForwardCostModel::PoolShape({producer,gap},1,c,Sm90ForwardAbi::kF3);
  assert(linked.k==5);
  auto narrow=c;narrow.tile_m=64;
  assert(Sm90ForwardCostModel::PoolShape({producer,gap},1,narrow,Sm90ForwardAbi::kF3).k==linked.k);
  assert(Sm90ForwardCostModel::PoolShape({producer,gap},1,c).k==3);

  Sm90ForwardSearchSpace space{{s,s},target,Sm90ForwardAbi::kF3};
  auto search=SearchSm90Forward(space,cost);
  assert(search.ranked.size()==3 && search.receipt.At("residencies").AsArray("CPU F3 contract").size()==1);
  for(auto const& candidate:search.ranked) {
    assert(candidate.coordinates.abi==Sm90ForwardAbi::kF3 && candidate.coordinates.residency==1);
    auto const& rows=candidate.search_receipt.At("stages").AsArray("CPU F3 contract");
    assert(rows[0].At("remaining_candidate_count").AsNumber("CPU F3 contract")==16);
    assert(rows[0].At("ranked_candidate_count").AsNumber("CPU F3 contract")>16);
    assert(!candidate.prediction.stages[0].terms.At("preissued").AsBool("CPU F3 contract"));
  }
  s.dnn1_geometry.Set("f3_chain_class","missing_class");
  space.shapes={s};Missing([&]{SearchSm90Forward(space,cost);});
  Missing([&]{Sm90ForwardConstants::FromFile(argv[1],1,Sm90ForwardAbi::kF3);});
  std::cout<<"F3 strict constants, formula, preissue and top16/top4 search PASS\n";
}
