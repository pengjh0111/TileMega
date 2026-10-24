// SPDX-License-Identifier: BSD-3-Clause
// Supplied numbers, placement and arbitration below are fictional CPU fixtures.
#include <tilemega/Solver/Sm90DenseFlowSchedule.h>
#include <tilemega/Solver/Sm90DenseFlowEvents.h>
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <cassert>
#include <iostream>
#include <numeric>

using namespace tilemega;using namespace tilemega::solver;
using Phase=Sm90DenseSchedulePhase;using Kind=Sm90DenseFlowEventKind;
using Anchor=Sm90DenseScheduleAnchor;using Id=std::size_t;
namespace {
template<class F>void Reject(F f) {bool bad=false;try {f();}catch(std::exception const&) {bad=true;}assert(bad);}
struct Fixture {
  Sm90DenseFlowEvents graph;Sm90DenseScheduleMapping map;Sm90DenseSchedulePolicies policy;
  Id Add(std::string name,std::string service,std::string resource,std::vector<Anchor> after={},
      Kind kind=Kind::kOperandReadersDone) {
    Id event=graph.events.size(),id=map.operations.size();graph.events.push_back({kind});map.source_scope.push_back(event);
    Sm90DenseScheduleOperation op;op.name=name;op.service=service;op.resource=resource;op.source_event=event;
    op.after=after;op.structural_join=service.empty();op.tie_priority=int(id);map.operations.push_back(op);return id;
  }
  void Resource(std::string name,int capacity=1) {
    policy.resources[name]=Sm90DenseScheduleResource{capacity,"fifo_explicit_ties"};map.initial_resource_available[name]=std::vector<double>(capacity,0);
  }
  void Service(std::string name,double delay,double occupancy=0,std::optional<double> rmw={}) {
    policy.services[name]=Sm90DenseScheduleService{delay,occupancy,rmw};
  }
  Sm90DenseScheduleResult Run() {return ScheduleSm90DenseFlowScope(graph,map,policy);}
};
// Real WorkGraph task/block order drives the motif, including continuous q%S.
// A joint transfer point starts at packet START, not packet-end. Store issue
// retires shared scratch before asynchronous output completion gates departure.
Fixture Gemm(Sm90DenseStageWork const& stage,std::vector<int> const& ctas,bool same_sm) {
  Fixture f;
  auto residency=stage.ServiceDescriptor().At("domain").At("coordinates").At("residency").AsNumber("residency");
  if(same_sm && ctas.size()>residency)f.map.unpriced.push_back("placement exceeds resident CTA capacity");
  f.Resource("hierarchy");f.Service("packet",2,1);f.Service("joint_transfer",7,1);
  f.Service("readers",3,3);f.Service("empty",1,1);f.Service("finalize",5,5);
  f.Service("epilogue_issue",2,2);f.Service("output",4,1);f.Service("consumer_state",1,1);
  std::vector<Anchor> departures;
  for(auto cta:ctas) {
    auto sm="sm"+std::to_string(same_sm?0:cta);f.Resource(sm);
    auto entry=f.Add("explicit_stage_entry","","",{},Kind::kStageBegin);
    Id producer=entry,consumer=entry;std::vector<Id> empty(stage.ring_slots,entry);std::vector<Anchor> outputs;
    for(auto const& task:stage.ctas.at(cta).tasks) {
      consumer=f.Add("task_state","consumer_state",sm,{{consumer}},Kind::kTaskBegin);
      for(auto const& block:task.blocks) {
        if(block.prefix_bytes || stage.ctas.at(cta).l2_hint_bytes)f.map.unpriced.push_back("prefix/L2 policy needs predecessor mapping");
        auto wait=f.Add("empty_q"+std::to_string(block.sequence),"","",{{producer},{empty.at(block.slot)}},Kind::kEmptyWait);
        auto issue=f.Add("packet_q"+std::to_string(block.sequence),"packet",sm,{{wait}},Kind::kBulkIssue);
        auto transfer=f.Add("full_q"+std::to_string(block.sequence),"joint_transfer","hierarchy",{{issue,Phase::kStart}},Kind::kFullReady);
        auto read=f.Add("read_q"+std::to_string(block.sequence),"readers",sm,{{consumer},{transfer}},Kind::kOperandReadersDone);
        auto retire=f.Add("empty_publish_q"+std::to_string(block.sequence),"empty",sm,{{read}},Kind::kEmptyArrivals);
        for(auto op:{wait,issue,transfer,read,retire}) {
          auto& e=f.graph.events[f.map.operations[op].source_event];e.cta=cta;e.task=task.task;e.sequence=block.sequence;
          e.bank=stage.ring_bank;e.slot=block.slot;e.generation=block.generation;
        }
        empty.at(block.slot)=retire;producer=issue;consumer=retire;
      }
      auto finalize=f.Add("finalize","finalize",sm,{{consumer}},Kind::kFinalize);
      consumer=f.Add("epilogue_issue","epilogue_issue",sm,{{finalize}},Kind::kEpilogue);
      auto write=f.Add("output_complete","output","hierarchy",{{consumer}},Kind::kEpilogue);outputs.push_back({write});
    }
    outputs.push_back({producer});outputs.push_back({consumer});
    auto depart=f.Add("explicit_departure_boundary","","",outputs,Kind::kCtaArrival);departures.push_back({depart});
  }
  f.map.finish=f.Add("scope_end","","",departures,Kind::kStageEnvelopeEnd);return f;
}
}
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  Fixture f;f.Resource("shared_memory_hierarchy");f.Service("packet",4,1);f.Service("async",9,2);
  auto issue=f.Add("packet","packet","shared_memory_hierarchy");
  auto copy=f.Add("copy","async","shared_memory_hierarchy",{{issue,Phase::kStart}});
  f.map.finish=f.Add("join","","",{{issue},{copy}});auto r=f.Run();assert(r.scope_time && *r.starts[copy]==1 && *r.completions[issue]==4 && *r.completions[copy]==10);
  auto broken=f;broken.policy.services["async"].reset();assert(!broken.Run().scope_time);
  broken=f;broken.map.initial_resource_available.clear();assert(!broken.Run().scope_time);
  broken=f;broken.policy.services["async"]->completion_delay=-1;Reject([&]{broken.Run();});
  Fixture issue_share;issue_share.Resource("one_SM");issue_share.Service("issue",4,4);
  auto p=issue_share.Add("producer_warp","issue","one_SM");auto q=issue_share.Add("consumer_warps","issue","one_SM");
  issue_share.map.finish=issue_share.Add("joined","","",{{p},{q}});assert(*issue_share.Run().scope_time==8);
  issue_share.policy.resources["one_SM"]->capacity=2;issue_share.map.initial_resource_available["one_SM"]={0,0};assert(*issue_share.Run().scope_time==4);
  auto carried=f;carried.map.initial_resource_available["shared_memory_hierarchy"]={6};auto c=carried.Run();assert(c.scope_time && *c.starts[issue]==6);

  Fixture connected;connected.Resource("consumer");connected.Resource("producer");
  connected.Service("whole_consume",12,8);connected.policy.services["whole_consume"]->progress_offsets[0]=4;
  auto body=connected.Add("connected_consume","whole_consume","consumer");
  connected.Service("refill",3,1);auto refill=connected.Add("refill_before_finalize","refill","producer",{{body,Phase::kProgress0}});
  connected.map.finish=connected.Add("connected_boundary","","",{{body},{refill}});auto response=connected.Run();
  assert(response.scope_time && *response.starts[refill]==4 && *response.completions[refill]==7 && *response.scope_time==12);
  Fixture atomic;atomic.Resource("rmw_counter");atomic.Resource("return0");atomic.Resource("return1");atomic.Resource("merge");
  atomic.Service("rmw",2,1,1);atomic.Service("slow_return",20);atomic.Service("fast_return",1);atomic.Service("merge",3,3);
  auto a=atomic.Add("pub0.rmw","rmw","rmw_counter");auto b=atomic.Add("pub1.rmw","rmw","rmw_counter");
  auto ar=atomic.Add("pub0.return","slow_return","return0",{{a}});auto br=atomic.Add("pub1.return","fast_return","return1",{{b}});
  auto all=atomic.Add("all_rmw_visible","","",{{a,Phase::kRmw},{b,Phase::kRmw}});
  auto am=atomic.Add("merge0","merge","merge",{{all},{ar}},Kind::kMerge);
  auto bm=atomic.Add("merge1","merge","merge",{{all},{br}},Kind::kMerge);
  atomic.map.groups={{{a,ar,am},{b,br,bm}}};
  atomic.map.finish=atomic.Add("continuations","","",{{a,Phase::kContinue},{b,Phase::kContinue}});
  auto ordered=atomic.Run();assert(ordered.scope_time && ordered.owners[0]==b && !ordered.starts[am]);
  assert(*ordered.starts[bm]<*ordered.completions[ar]);assert(*ordered.continuations[b]==*ordered.completions[bm]);
  auto reverse=atomic;reverse.map.operations[a].tie_priority=1;reverse.map.operations[b].tie_priority=0;
  auto swapped=reverse.Run();assert(swapped.scope_time && swapped.owners[0]==a && !swapped.starts[bm]);
  auto tie=atomic;tie.map.operations[a].tie_priority.reset();tie.map.operations[b].tie_priority.reset();assert(!tie.Run().scope_time);
  tie.policy.resources["rmw_counter"]->arbitration="fifo_source_order_model";assert(tie.Run().owners[0]==b);
  tie.policy.resources["rmw_counter"]->arbitration="fifo_reverse_source_order_model";assert(tie.Run().owners[0]==a);

  // Only source geometry is exercised here; epilogue/operand services are
  // fictional, so no runtime-plan binding is asserted by this fixture.
  Sm90DenseWorkload w{{{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{151936,1024}}},16,28,128,2,true};
  auto target=TargetSpec::FromJson(argv[1]);auto coordinates=Sm90DenseCoordinates::FromJson(json::ParseFile(argv[2]));
  auto graph=BuildSm90DenseWorkGraph(w,coordinates,target);auto const& head=graph.stages[graph.stages.size()-2];
  assert(head.ctas[0].tasks.size()>1);auto motif=Gemm(head,{0,1},false);auto shared=motif.Run();assert(shared.scope_time);
  assert(!Gemm(head,{0,1},true).Run().scope_time);
  auto slower=motif;slower.policy.services["joint_transfer"]->admission_occupancy=5;auto separate=slower.Run();
  assert(separate.scope_time && *separate.scope_time>*shared.scope_time);
  unsigned full=0;for(auto const& e:motif.graph.events)if(e.kind==Kind::kFullReady)++full;
  unsigned blocks=0;for(int cta:{0,1})for(auto const& task:head.ctas[cta].tasks)blocks+=task.blocks.size();assert(full==blocks);
  for(Id i=0;i<motif.map.operations.size();++i) {
    auto const& e=motif.graph.events[motif.map.operations[i].source_event];
    if(e.kind==Kind::kEmptyWait && e.sequence>=head.ring_slots) {
      bool previous=false;for(auto const& gate:motif.map.operations[i].after) {
        auto const& p=motif.graph.events[motif.map.operations[gate.operation].source_event];
        previous|=p.kind==Kind::kEmptyArrivals && p.sequence==e.sequence-head.ring_slots && p.cta==e.cta;
      }
      assert(previous);
    }
  }
  auto same_source=motif;for(auto& p:same_source.policy.services)p.second.reset();assert(!same_source.Run().scope_time);
  auto prefixed=graph.stages[3];assert(prefixed.ctas[0].tasks.front().blocks.front().prefix_bytes);
  assert(!Gemm(prefixed,{0,1},false).Run().scope_time);
  std::cout<<"synthetic schedule passed; real Qwen head tasks/CTA="<<head.ctas[0].tasks.size()<<" blocks="<<blocks
      <<" operations="<<motif.map.operations.size()<<" hierarchy_fast="<<*shared.scope_time<<" hierarchy_slow="<<*separate.scope_time
      <<" (fictional time units; not physical predictions)\n";
}
