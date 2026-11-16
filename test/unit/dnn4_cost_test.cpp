// SPDX-License-Identifier: BSD-3-Clause
// Synthetic values exercise the formula; they are never calibration data.
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
bool Near(double a,double b){return std::abs(a-b)<1e-8*std::max(1.,std::abs(b));}
}
int main(int argc,char** argv) {
 assert(argc==3);auto source=json::ParseFile(argv[1]);
 TargetSpec target;target.arch_tag="sm_90";
 target.res.num_sms=132;target.res.max_dynamic_smem_per_cta=232448;
 auto native=Sm90NativeConstants::FromTarget(source,"");
 json::Object mma,ring,epilogue,split,rates,small;
 for(int tm:{64,128,256})for(int tn:{32,64,96,128,192,256}) {
  std::string tile="tm="+std::to_string(tm)+";tn="+std::to_string(tn);
  for(int tk:{64,128})for(auto mode:{"normal","gather"}) {
   auto key="mode="+std::string(mode)+";"+tile+";tk="+std::to_string(tk);
   mma.emplace_back(key,Row(tk==64?100:150));
   for(auto src:{"l2","dram"})ring.emplace_back(key+";src="+src,
       Row((tk==64?1200.:1800.)*(tm+tn)/256*(std::string(src)=="l2"?1:2)));
  }
  for(auto cls:kSm90F3ChainClasses)for(auto store:{"reg","tma"})
   epilogue.emplace_back("class="+std::string(cls)+";"+tile+";store="+store,Row(std::string(store)=="reg"?100:70));
  for(int s=2;s<=16;++s)split.emplace_back(tile+";slices="+std::to_string(s),Row(30*s));
 }
 for(auto src:{"l2","dram"})for(int n:{33,66,99,132})rates.emplace_back("src="+std::string(src)+";n="+std::to_string(n),Row(std::string(src)=="l2"?100:50,"GB/s"));
 for(auto kind:{"layout","pool2","pool3","global_pool","layernorm","embedding"})small.emplace_back("kind="+std::string(kind),Row(3,"ns/work_item"));
 json::Value section=json::Object{{"schema",1},{"variant","SYNTHETIC_CPU_ONLY"},{"synthetic_cpu_fixture",true},
  {"method","F4 formula sensitivity and candidate-audit CPU contracts"},{"header_sha256",json::Object{}},
  {"by_gpu",json::Object{{"1",json::Object{{"f4_gate_ns",json::Object{{"",Row(20)}}},
   {"f4_first_ns",json::Object{{"src=l2",Row(100)},{"src=dram",Row(300)}}},
   {"f4_mma_ns",mma},{"f4_ring_ns",ring},{"f4_epilogue_ns",epilogue},
   {"f4_splitk_ns",split},{"f4_rate_gbps",rates},{"f4_small_ns",small}}}}}};
 source.Set("sm90_f4_constants",section);{std::ofstream out(argv[2]);out<<source.Dump(0);}
 unsetenv("DNN3_CPU_CONTRACT_TEST");bool rejected=false;
 try{Sm90ForwardConstants::FromFile(argv[2],1,Sm90ForwardAbi::kF3);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
 setenv("DNN3_CPU_CONTRACT_TEST","1",1);setenv("TILEMEGA_FORWARD_CONSTANTS_FILE",argv[2],1);
 Sm90ForwardCostModel cost(target,native,Sm90ForwardAbi::kF3);assert(cost.ForwardConstants().F4());
 Sm90ForwardShape s;Sm90ForwardStage c;s.body=c.body=Sm90ForwardBody::kGemm;
 s.m=1024;s.n=768;s.k=2048;s.weight_bytes=2*s.n*s.k;
 s.dnn1_geometry=json::Object{{"f3_vectors",0},{"f3_residual",false},{"needs_stats",false},
  {"has_channel_pool",false},{"has_row_stats",false},{"f3_chain_class","plain"},
  {"epilogue_steps",json::Array{}},{"f4_model_weight_bytes",double(s.weight_bytes)},
  {"f4_activation_resident_bytes",double(2*s.m*s.k)},{"f4_prefetch_weight_bytes",0}};
 c.tile_m=128;c.tile_n=128;c.slots=4;
 json::Value terms;double base=cost.F4Stage(s,c,false,&terms);
 assert(Near(base,20+100+32*400+100));
 assert(terms.At("k_iter").AsNumber("k iterations")==32);
 auto original=c;json::Object sensitivity;
 auto check=[&](char const* field,Sm90ForwardStage altered) {
  double value=cost.F4Stage(s,altered,false);
  assert(!Near(value,cost.F4Stage(s,original,false)));
  sensitivity.emplace_back(field,true);
 };
 c.tile_m=64;check("TM",c);c=original;c.tile_n=64;check("TN",c);
 c=original;c.tk=128;c.slots=2;original.slots=2;check("TK",c);
 original=c=Sm90ForwardStage{};original.body=c.body=s.body;
 original.tile_m=c.tile_m=128;original.tile_n=c.tile_n=128;original.slots=c.slots=4;
 c.split_k=2;check("S",c);c=original;c.slots=3;check("Slots",c);
 c=original;c.store=Sm90ForwardStore::kTma;check("store",c);
 original.slots=6;c=original;c.mc=1;check("mc",c);
 assert(sensitivity.size()==7);
 original.slots=3;c=original;Sm90ForwardCoordinates coords;coords.abi=Sm90ForwardAbi::kF3;coords.stages={c,c};
 auto p=cost.Predict({s,s},coords);assert(p.stages[1].terms.At("preissued").AsBool("preissued"));
 coords.stages[0].slots=4;p=cost.Predict({s,s},coords);assert(!p.stages[1].terms.At("preissued").AsBool("spanning"));
 auto dram=s;dram.dnn1_geometry.Set("f4_model_weight_bytes",double(64LL<<20));
 assert(cost.F4Stage(dram,c,false)>cost.F4Stage(s,c,false));
 Sm90ForwardSearchResult audit;Sm90ForwardCandidate a,b;
 a.coordinates=coords;b.coordinates=coords;a.prediction.total_ns=10000;b.prediction.total_ns=10001;
 b.coordinates.stages[1].slots=2;audit.ranked={a,b};
 assert(!AuditSm90ForwardF4Candidates(audit).At("passed").AsBool("near tie"));
 audit.ranked[1].prediction.total_ns=10100;
 assert(AuditSm90ForwardF4Candidates(audit).At("passed").AsBool("separated"));
 audit.ranked[1]=a;assert(AuditSm90ForwardF4Candidates(audit).At("passed").AsBool("identical"));
 auto searched=SearchSm90Forward(Sm90ForwardSearchSpace{{s,s},target,Sm90ForwardAbi::kF3},cost);
 assert(!searched.ranked.empty());
 for(auto const& candidate:searched.ranked)for(auto const& stage:candidate.search_receipt.At("stages").AsArray("stages")) {
  assert(stage.At("remaining_candidate_count").AsNumber("top8")<=8);
  auto const& tuning=stage.At("tuning_candidates").AsArray("tuning choices");
  assert(!tuning.empty() && tuning.size()<=6);
  for(std::size_t i=0;i<tuning.size();++i)for(std::size_t j=0;j<i;++j) {
   double a=tuning[i].At("prediction_ns").AsNumber("prediction");
   double b=tuning[j].At("prediction_ns").AsNumber("prediction");
   assert(std::abs(a-b)>=.005*std::min(a,b));
  }
 }
 assert(searched.receipt.Find("f4_candidate_selfcheck"));
 std::cout<<"F4 formula, seven coordinate sensitivities, spanning and near-tie audit PASS\n";
}
