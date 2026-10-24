// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseFlowEvents.h>
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <cassert>
#include <map>
#include <set>

using namespace tilemega;
using namespace tilemega::solver;
using Kind=Sm90DenseFlowEventKind;
using Id=Sm90DenseEventId;
namespace {
struct Reach {
  Sm90DenseFlowEvents const& graph;
  std::vector<std::vector<Id>> parents;
  explicit Reach(Sm90DenseFlowEvents const& g):graph(g),parents(g.events.size()) {
    for(auto const& e:g.edges)if(e.condition<0)parents[e.after].push_back(e.before);
  }
  bool Before(Id from,Id to) const {
    std::set<Id> visited;std::vector<Id> todo{to};
    while(!todo.empty()) {
      auto id=todo.back();todo.pop_back();if(id==from)return true;
      if(visited.insert(id).second)for(auto p:parents[id])todo.push_back(p);
    }
    return false;
  }
};
std::vector<Id> Find(Sm90DenseFlowEvents const& g,Kind kind,int step=-2,int stage=-2,int cta=-2,int sequence=-2) {
  std::vector<Id> ids;
  for(Id i=0;i<g.events.size();++i) {
    auto const& e=g.events[i];
    if(e.kind==kind && (step==-2 || e.step==step) && (stage==-2 || e.stage==stage) &&
        (cta==-2 || e.cta==cta) && (sequence==-2 || e.sequence==sequence))ids.push_back(i);
  }
  return ids;
}
template<class F>void Reject(F f) {bool rejected=false;try {f();}catch(std::invalid_argument const&) {rejected=true;}assert(rejected);}
void Conservation(Sm90DenseWorkGraph const& w,Sm90DenseFlowEvents const& g) {
  std::int64_t expected_bulk=0,expected_a=0,expected_l2=0,issued_bulk=0,issued_a=0,issued_l2=0;
  int blocks=0,queries=0,current=0;
  for(auto const& s:w.stages)for(auto const& c:s.ctas) {
    expected_l2+=c.l2_hint_bytes;
    for(auto const& t:c.tasks) {
      if(s.kind=="attention") {++queries;current+=t.contains_current_row;}
      for(auto const& b:t.blocks) {++blocks;expected_bulk+=b.weight_bytes+b.kv_bytes;expected_a+=b.activation_bytes;}
    }
  }
  int query_current=0;
  for(auto const& e:g.events) {
    if(e.kind==Kind::kBulkIssue)issued_bulk+=e.bytes;
    if(e.kind==Kind::kActivationIssue)issued_a+=e.bytes;
    if(e.kind==Kind::kPrefetchL2Issue)issued_l2+=e.bytes;
    if(e.kind==Kind::kQueryCurrentRow)query_current+=e.current_rows;
    if(e.kind==Kind::kFullReady)assert(e.arrivals==33);
    if(e.kind==Kind::kEmptyArrivals)assert(e.arrivals==128);
    if(e.kind==Kind::kActivationComplete)assert(e.arrivals==32);
  }
  assert(expected_bulk==issued_bulk && expected_a==issued_a && expected_l2==issued_l2);
  assert(Find(g,Kind::kFullReady).size()==unsigned(blocks));
  assert(Find(g,Kind::kExpectTransaction).size()+Find(g,Kind::kBulkArrival).size()==unsigned(blocks));
  assert(Find(g,Kind::kQueryCurrentRow).size()==unsigned(queries) && query_current==current);
  assert(Find(g,Kind::kCtaArrival).size()==w.stages.size()*w.grid);
  auto topology=g.UnconditionalOrder();assert(topology.size()==g.events.size());
  std::vector<std::vector<Id>> children(g.events.size());
  for(auto const& edge:g.edges)if(edge.condition<0)children[edge.before].push_back(edge.after);
  std::vector<bool> reachable(g.events.size());reachable.front()=true;
  for(auto id:topology) {assert(reachable[id]);for(auto child:children[id])reachable[child]=true;}
  for(auto const& edge:g.edges)if(g.events[edge.before].kind==Kind::kPrefetchL2Issue)
    assert(g.events[edge.after].kind!=Kind::kFullReady && g.events[edge.after].kind!=Kind::kBulkComplete);
}
}
int main() {
  TargetSpec target;target.arch_tag="sm_90";target.nvcc_arch="sm_90a";target.res.num_sms=132;
  target.res.max_smem_per_sm=233472;target.res.max_dynamic_smem_per_cta=232448;target.res.l2_bytes=52428800;
  Sm90DenseWorkload w{{{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{32768,1024}}},1,2,128,2,true};
  auto c=Sm90DenseCoordinates::Exact(w.shapes,w.batch,target.res.max_dynamic_smem_per_cta);
  auto low=BuildSm90DenseWorkGraph(w,c,target,64);auto plain=BuildSm90DenseFlowEvents(low);Conservation(low,plain);
  assert(plain.merge_conditions.size()==16);
  assert(Find(plain,Kind::kRingSlotInitialize).size()==unsigned(16*low.grid));
  assert(Find(plain,Kind::kRingInvalidate).size()==unsigned(2*low.grid));
  assert(Find(plain,Kind::kStageInitialize).size()==low.stages.size()*low.grid);
  Reach order(plain);
  for(auto const& condition:plain.merge_conditions) {
    assert(condition.alternatives.size()==16);std::set<int> owners;
    for(unsigned i=0;i<condition.alternatives.size();++i) {
      auto const& a=condition.alternatives[i];owners.insert(a.cta);
      assert(order.Before(a.publication,condition.all_published));
      assert(!order.Before(condition.all_published,a.continuation));
      assert(plain.events[a.merge].condition>=0 && plain.events[a.merge].owner==int(i));
      bool conditional=false;
      for(auto const& e:plain.edges)if(e.before==a.merge && e.after==a.continuation) {
        assert(e.condition>=0 && e.owner==int(i));conditional=true;
      }
      assert(conditional);
    }
    assert(owners.size()>1);
  }
  // Each CTA can begin its next stage after its own departure; the stage
  // envelope is an observation, not a second global synchronization.
  auto next=Find(plain,Kind::kStageBegin,0,1,0).at(0);
  auto own_release=Find(plain,Kind::kV3Release,0,0,0).at(0);
  auto other_release=Find(plain,Kind::kV3Release,0,0,1).at(0);
  assert(order.Before(own_release,next) && !order.Before(other_release,next));
  assert(!order.Before(plain.stage_envelope_ends.front(),next));
  auto inactive=Find(plain,Kind::kCtaArrival,0,1,131).at(0);
  auto release=Find(plain,Kind::kV3Release,0,1,0).at(0);
  assert(order.Before(inactive,release));
  auto setup=Find(plain,Kind::kProducerSetup,0,1,0).at(0);
  assert(plain.events[setup].local_tasks==1);
  assert(order.Before(Find(plain,Kind::kStageInitialize,0,1,0).at(0),setup));
  assert(order.Before(setup,Find(plain,Kind::kEmptyWait,0,1,0,0).at(0)));
  auto idle_setup=Find(plain,Kind::kProducerSetup,0,1,131).at(0);
  assert(plain.events[idle_setup].local_tasks==0 && order.Before(idle_setup,inactive));
  assert(Find(plain,Kind::kProducerSetup,0,0).empty());
  assert(Find(plain,Kind::kProducerSetup,0,2).empty());
  auto producer_join=Find(plain,Kind::kProducerRendezvous,0,1,0).at(0);
  auto producer_publish=Find(plain,Kind::kProducerGenerations,0,1,0).at(0);
  assert(plain.events[producer_join].arrivals==32 && order.Before(producer_join,producer_publish));
  auto idle_join=Find(plain,Kind::kProducerRendezvous,0,1,131).at(0);
  assert(order.Before(idle_setup,idle_join) && order.Before(idle_join,inactive));
  for(int token_stage:{0,int(low.stages.size())-1}) {
    assert(Find(plain,Kind::kProducerGenerations,0,token_stage).empty());
    assert(Find(plain,Kind::kConsumerGenerations,0,token_stage).empty());
    assert(Find(plain,Kind::kProducerRendezvous,0,token_stage).empty());
  }

  int head=int(low.stages.size())-2;
  auto const& first=low.stages[head].ctas[0].tasks.front();int iterations=int(first.blocks.size());
  assert(Find(plain,Kind::kProducerSetup,0,head,0).size()==1);
  assert(plain.events[Find(plain,Kind::kProducerSetup,0,head,0).at(0)].local_tasks==
      int(low.stages[head].ctas[0].tasks.size()));
  auto refill=Find(plain,Kind::kEmptyWait,0,head,0,iterations).at(0);
  auto retired=Find(plain,Kind::kEmptyArrivals,0,head,0,iterations-low.stages[head].ring_slots).at(0);
  auto finalize=Find(plain,Kind::kFinalize,0,head,0).front();
  assert(order.Before(retired,refill) && !order.Before(finalize,refill));
  assert(order.Before(Find(plain,Kind::kEmptyArrivals,0,head,0,iterations-1).at(0),finalize));
  auto later=BuildSm90DenseWorkGraph(w,c,target,575);
  auto repeated=BuildSm90DenseFlowEvents(std::vector<Sm90DenseWorkGraph>{low,later});
  assert(repeated.steps==2 && Find(repeated,Kind::kRingSlotInitialize).size()==unsigned(16*low.grid));
  for(int owner=0;owner<low.grid;++owner)for(int bank=0;bank<2;++bank)for(int slot=0;slot<8;++slot)
    assert(repeated.final_generations[owner][bank][slot]>=plain.final_generations[owner][bank][slot]);
  auto first_generation=plain.events[Find(plain,Kind::kEmptyWait,0,1,0,0).at(0)].generation;
  auto next_generation=repeated.events[Find(repeated,Kind::kEmptyWait,1,1,0,0).at(0)].generation;
  assert(next_generation>first_generation);
  auto corrupted=low;++corrupted.stages[1].ctas[0].tasks.front().blocks.front().generation;
  Reject([&]{BuildSm90DenseFlowEvents(corrupted);});

  Sm90DenseSearchSpace space{w.shapes,w.batch,w.head_dim,w.queries,target};
  auto prefixed=BuildSm90DenseWorkGraph(w,space.Seeds()[2],target,575);
  auto pref=BuildSm90DenseFlowEvents(prefixed);Conservation(prefixed,pref);Reach pref_order(pref);
  bool pending_transfer=false,retire_seen=false;
  for(auto const& e:pref.events)if(e.kind==Kind::kRetireSlot)retire_seen=true;
  for(Id i=0;i<pref.events.size();++i) {
    auto const& e=pref.events[i];
    if(e.kind==Kind::kBulkComplete && e.target_stage==e.stage+1) {
      auto earlier=Find(pref,Kind::kV3Release,0,e.stage,e.cta).at(0);
      assert(!pref_order.Before(i,earlier));pending_transfer=true;
    }
  }
  assert(pending_transfer && retire_seen);
  for(unsigned s=0;s<prefixed.stages.size();++s) {
    auto const& stage=prefixed.stages[s];
    if(stage.incoming_order!=Sm90DenseTailOrder::kRetireOperands)continue;
    auto const& previous_stage=prefixed.stages[s-1];
    assert(Find(pref,Kind::kRetireSlot,0,s-1).size()==unsigned(previous_stage.ring_slots*low.grid));
  }
  auto no_prefix=c;no_prefix.gemms[0].prefetch_l2_bytes=32;
  auto hints=BuildSm90DenseFlowEvents(BuildSm90DenseWorkGraph(w,no_prefix,target));
  assert(!Find(hints,Kind::kPrefetchL2Issue).empty());
  for(auto const& e:hints.events)if(e.kind==Kind::kBulkIssue)assert(e.target_stage==e.stage);

  auto high_work=w;high_work.batch=16;high_work.layers=1;
  auto high_c=Sm90DenseCoordinates::Exact(high_work.shapes,16,target.res.max_dynamic_smem_per_cta);
  auto high_descriptor=BuildSm90DenseWorkGraph(high_work,high_c,target,64);
  auto high=BuildSm90DenseFlowEvents(high_descriptor);Conservation(high_descriptor,high);
  assert(high.merge_conditions.empty());
  bool padding=false,current_only=false;
  for(auto const& e:high.events)if(e.kind==Kind::kPatchAndPad) {
    padding|=e.valid_rows<e.compute_rows;
    current_only|=e.history_rows==0 && e.current_rows==1;
  }
  assert(padding && current_only);
  auto reuse=space.ActivationReuseSeeds();assert(!reuse.empty());
  auto reuse_work=BuildSm90DenseWorkGraph(w,reuse.front(),target);
  auto initialized=BuildSm90DenseFlowEvents(reuse_work);
  for(auto const& e:initialized.events)if(e.kind==Kind::kStageInitialize)assert(!e.bytes);
  for(auto const& e:initialized.events)if(e.kind==Kind::kLaunchInitialize)assert(e.bytes==reuse_work.launch_inactive_zero_bytes_per_cta);

  assert(!high.pending_stage_bindings.empty());
  Sm90DenseResidentEvidence fixture_resident{high.grid,high.shared_bytes,1,std::string(64,'a'),std::string(64,'b')};
  std::vector<Sm90DenseFlowServiceEvidence> fixture_records;
  for(auto const& req:high.requirements)fixture_records.push_back({req.key,"CPU metadata fixture only",std::string(64,'c'),
      fixture_resident.binary_sha256,fixture_resident.source_digest,"fixture_cache_protocol",5,true});
  auto missing_binding=high.Coverage(fixture_resident,"fixture_cache_protocol",fixture_records);
  assert(missing_binding.pending.size()==high.pending_stage_bindings.size());
  fixture_records.clear();
  for(auto& stage:high_descriptor.stages)if(stage.kind!="attention" && stage.kind!="embedding" && stage.kind!="argmax")
    stage.plan_binding=Sm90DenseStagePlanBinding{"store",false,false};
  auto bound=BuildSm90DenseFlowEvents(high_descriptor);assert(bound.pending_stage_bindings.empty());
  auto altered=high_descriptor;altered.stages[1].plan_binding->input_norm=true;
  altered.stages[1].plan_binding->output_sums=true;
  auto differing=BuildSm90DenseFlowEvents(altered);
  Reject([&]{BuildSm90DenseFlowEvents(std::vector<Sm90DenseWorkGraph>{high_descriptor,altered});});
  auto requirement=[](Sm90DenseFlowEvents const& graph) {
    for(auto const& r:graph.requirements)if(r.service==Sm90DenseFlowService::kConsumer && graph.events[r.events.front()].stage==1)return r.key;
    throw std::invalid_argument("missing consumer requirement");
  };
  assert(requirement(bound)!=requirement(differing));
  auto different_registers=high_descriptor;
  auto domain=different_registers.stages[1].service_descriptor.At("domain");
  auto coordinates=domain.At("coordinates");auto families=coordinates.At("gemms");auto qkv=families.At("qkv");
  qkv.Set("register_stages",2);families.Set("qkv",qkv);coordinates.Set("gemms",families);
  domain.Set("coordinates",coordinates);different_registers.stages[1].service_descriptor.Set("domain",domain);
  assert(requirement(bound)!=requirement(BuildSm90DenseFlowEvents(different_registers)));
  auto missing_domain=high_descriptor;missing_domain.stages[1].service_descriptor=json::Value();
  Reject([&]{BuildSm90DenseFlowEvents(missing_domain);});
  high=std::move(bound);
  auto pending=high.Coverage({},"unknown",{});assert(!pending.pending.empty());
  assert(!pending.ToJson().At("prediction_available").AsBool("prediction"));
  Sm90DenseResidentEvidence resident{high.grid,high.shared_bytes,1,std::string(64,'a'),std::string(64,'b')};
  std::vector<Sm90DenseFlowServiceEvidence> evidence;
  for(auto const& req:high.requirements)evidence.push_back({req.key,"CPU metadata fixture only",std::string(64,'c'),
      resident.binary_sha256,resident.source_digest,"fixture_cache_protocol",5,true});
  assert(high.Coverage(resident,"fixture_cache_protocol",evidence).pending.empty());
  evidence.front().verified_measurement=false;
  assert(!high.Coverage(resident,"fixture_cache_protocol",evidence).pending.empty());
  evidence.front().verified_measurement=true;evidence.front().measured_binary_sha256=std::string(64,'d');
  assert(!high.Coverage(resident,"fixture_cache_protocol",evidence).pending.empty());
  assert(!plain.Coverage({plain.grid,plain.shared_bytes,1,std::string(64,'a'),std::string(64,'b')},"explicit_cache",{}).pending.empty());
  auto json=high.ToJson();assert(!json.At("prediction_available").AsBool("prediction"));
  assert(!json.At("latency_or_default_rates_supplied").AsBool("rates"));
}
