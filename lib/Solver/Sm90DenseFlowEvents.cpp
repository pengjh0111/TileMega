// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseFlowEvents.h>
#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
namespace {
using Id=Sm90DenseEventId;
using Kind=Sm90DenseFlowEventKind;
using Service=Sm90DenseFlowService;
using Resource=Sm90DenseFlowResource;
using Slots=std::array<std::array<Id,8>,2>;
using Counts=std::array<std::array<std::uint64_t,8>,2>;
bool Hex(std::string const& s) {return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;}
bool Gemm(Sm90DenseStageWork const& s) {return s.kind!="attention" && s.kind!="embedding" && s.kind!="argmax";}
struct Block {
  Sm90DenseBlockWork const* work=nullptr;
  Id wait,expect,full,read,empty;
  std::vector<Id> prefix_completions;
};
struct Task {
  Sm90DenseTaskWork const* work=nullptr;
  Id begin,query=kSm90DenseNoEvent,finalize,partial,publish=kSm90DenseNoEvent,continuation;
  std::vector<Block> blocks;
};
struct Cta {
  Id begin,initialize,produced,tail_begin,tail_done,arrival,release;
  std::vector<Task> tasks;
};
class Builder {
 public:
  Sm90DenseFlowEvents graph;
  explicit Builder(Sm90DenseWorkGraph const& work):work_(work),current_work_(&work) {
    if(work.grid<1 || work.num_sms<1 || work.stages.empty())throw std::invalid_argument("empty native flow work graph");
    graph.grid=work.grid;graph.num_sms=work.num_sms;graph.shared_bytes=work.shared_bytes;
    graph.requested_residency=work.requested_residency;
    empty_.resize(graph.grid);generations_.resize(graph.grid);previous_.resize(graph.grid);
    auto root=Add(Kind::kKernelBegin);
    for(int c=0;c<graph.grid;++c) {
      auto launch=Add(Kind::kLaunchInitialize,-1,-1,c);Edge(root,launch);
      for(int bank=0;bank<2;++bank)for(int slot=0;slot<8;++slot) {
        auto init=Add(Kind::kRingSlotInitialize,-1,-1,c);auto& e=graph.events[init];
        e.bank=bank;e.slot=slot;e.arrivals=128;e.expected_transactions=33;
        Edge(root,init);Edge(init,launch);empty_[c][bank][slot]=init;
        Need(init,Service::kRingInitialization,Resource::kSharedArena,"kernel_ring");
      }
      graph.events[launch].bytes=work.launch_inactive_zero_bytes_per_cta;
      if(work.launch_inactive_zero_bytes_per_cta)Need(launch,Service::kInactiveRows,Resource::kSharedArena,"launch_inactive_rows");
      previous_[c]=launch;
    }
    Need(root,Service::kCtaPlacement,Resource::kResidentGrid,"all_grid_ctas");
    Need(root,Service::kSharedSmArbitration,Resource::kSmConsumerIssue,"shared_SM_placement_and_issue");
    Need(root,Service::kMemoryHierarchyArbitration,Resource::kDeviceMemoryHierarchy,"shared_device_memory_hierarchy");
  }
  void Step(Sm90DenseWorkGraph const& work) {
    if(work.grid!=graph.grid || work.num_sms!=graph.num_sms || work.shared_bytes!=graph.shared_bytes ||
        work.requested_residency!=graph.requested_residency || work.activation_reuse!=work_.activation_reuse ||
        work.launch_inactive_zero_bytes_per_cta!=work_.launch_inactive_zero_bytes_per_cta ||
        work.stages.size()!=work_.stages.size())throw std::invalid_argument("native flow steps require the same launch");
    for(unsigned s=0;s<work.stages.size();++s) {
      auto const& a=work.stages[s];auto const& b=work_.stages[s];
      if(a.ring_bank<0 || a.ring_bank>1 || a.kind!=b.kind || a.tasks!=b.tasks || a.merge_chunks!=b.merge_chunks || a.ring_slots!=b.ring_slots || a.ring_bank!=b.ring_bank ||
          a.workspace_offset!=b.workspace_offset || a.layout.end!=b.layout.end ||
          a.incoming_order!=b.incoming_order || a.ctas.size()!=unsigned(graph.grid) ||
          a.ServiceDescriptor().Dump(0)!=b.ServiceDescriptor().Dump(0))
        throw std::invalid_argument("native flow step layout changed");
    }
    current_work_=&work;
    int step=graph.steps++;auto base=generations_;
    for(unsigned s=0;s<work.stages.size();++s) {
      auto const& stage=work.stages[s];
      stage.ServiceDescriptor();
      if(Gemm(stage) && !stage.plan_binding)
        graph.pending_stage_bindings.push_back("step:"+std::to_string(step)+":stage:"+std::to_string(s));
    }
    std::vector<std::vector<Cta>> handles(work.stages.size());
    for(unsigned s=0;s<work.stages.size();++s) {
      auto const& stage=work.stages[s];handles[s].resize(graph.grid);
      for(int c=0;c<graph.grid;++c) {
        auto& h=handles[s][c];auto const& owner=stage.ctas[c];
        if(owner.cta!=c)throw std::invalid_argument("native flow CTA identity differs");
        h.begin=Add(Kind::kStageBegin,step,s,c);h.initialize=Add(Kind::kStageInitialize,step,s,c);
        h.produced=Add(Kind::kProduceDone,step,s,c);h.tail_begin=Add(Kind::kTailBegin,step,s,c);
        h.tail_done=Add(Kind::kTailIssueDone,step,s,c);h.arrival=Add(Kind::kCtaArrival,step,s,c);
        h.release=Add(Kind::kV3Release,step,s,c);
        for(auto const& task:owner.tasks) {
          Task t;t.work=&task;t.begin=Add(Kind::kTaskBegin,step,s,c,task.task);
          t.finalize=stage.kind=="attention" || Gemm(stage)?Add(Kind::kFinalize,step,s,c,task.task):kSm90DenseNoEvent;
          t.partial=Add(stage.kind=="attention"?Kind::kPartial:Gemm(stage)?Kind::kEpilogue:Kind::kTokenWork,step,s,c,task.task);
          t.continuation=Add(Kind::kTaskContinue,step,s,c,task.task);
          if(stage.kind=="attention") {
            t.query=Add(Kind::kQueryCurrentRow,step,s,c,task.task);
            graph.events[t.query].current_rows=task.contains_current_row;
            if(stage.merge_chunks>1)t.publish=Add(Kind::kAtomicPublish,step,s,c,task.task);
          }
          for(auto const& b:task.blocks) {
            if(stage.ring_slots<1 || stage.ring_slots>8 || b.slot<0 || b.slot>=stage.ring_slots ||
                b.sequence<0 || b.slot!=b.sequence%stage.ring_slots || b.activation_arrivals!=32 || b.empty_arrivals!=128 ||
                b.prefix_bytes<0 || b.prefix_bytes>b.weight_bytes+b.kv_bytes ||
                b.expected_transactions+b.explicit_bulk_arrivals!=1)
              throw std::invalid_argument("native flow block contract differs");
            Block p;p.work=&b;p.wait=Part(Kind::kEmptyWait,step,s,c,task.task,b,base[c]);
            p.expect=Part(b.expected_transactions?Kind::kExpectTransaction:Kind::kBulkArrival,step,s,c,task.task,b,base[c]);
            graph.events[p.expect].bytes=b.weight_bytes+b.kv_bytes;graph.events[p.expect].arrivals=1;
            graph.events[p.expect].expected_transactions=b.expected_transactions;
            if(b.prefix_bytes) {graph.events[p.expect].stage=int(s)-1;graph.events[p.expect].target_stage=s;}
            p.full=Part(Kind::kFullReady,step,s,c,task.task,b,base[c]);
            graph.events[p.full].bytes=b.weight_bytes+b.kv_bytes;graph.events[p.full].arrivals=33;
            p.read=Part(Kind::kOperandReadersDone,step,s,c,task.task,b,base[c]);
            p.empty=Part(Kind::kEmptyArrivals,step,s,c,task.task,b,base[c]);graph.events[p.empty].arrivals=128;
            t.blocks.push_back(p);
          }
          h.tasks.push_back(std::move(t));
        }
      }
    }
    for(unsigned s=0;s<work.stages.size();++s) {
      auto const& stage=work.stages[s];
      for(int c=0;c<graph.grid;++c) {
        auto& h=handles[s][c];auto const& owner=stage.ctas[c];
        Edge(previous_[c],h.begin);Edge(h.begin,h.initialize);
        graph.events[h.initialize].bytes=stage.inactive_zero_bytes_per_cta;
        if(stage.inactive_zero_bytes_per_cta)Need(h.initialize,Service::kInactiveRows,Resource::kSharedArena,stage.kind);
        Id producer=h.initialize;
        bool before=s+1<work.stages.size() && work.stages[s+1].incoming_order==Sm90DenseTailOrder::kBeforeProduce;
        if(before)producer=h.tail_done;
        if(Gemm(stage)) {
          // Produce's invariant operand loads/address setup precede its first
          // empty wait. Packet-internal issue measurements do not cover them;
          // even a zero-task CTA enters this outlined function.
          auto setup=Add(Kind::kProducerSetup,step,s,c);
          graph.events[setup].local_tasks=int(owner.tasks.size());
          Edge(producer,setup);producer=setup;
          Need(setup,Service::kProducerSetup,Resource::kCtaProducerIssue,
              stage.kind+":outlined_producer_setup");
        }
        int sequence=0;
        for(auto& t:h.tasks)for(auto& p:t.blocks) {
          auto const& b=*p.work;
          if(b.sequence!=sequence++ || b.previous_reader!=(b.sequence<stage.ring_slots?-1:b.sequence-stage.ring_slots) ||
              b.generation+base[c][stage.ring_bank][b.slot]!=generations_[c][stage.ring_bank][b.slot])
            throw std::invalid_argument("native continuous ring generation differs");
          Edge(producer,p.wait);Edge(empty_[c][stage.ring_bank][b.slot],p.wait);
          ++generations_[c][stage.ring_bank][b.slot];
          empty_[c][stage.ring_bank][b.slot]=p.empty;
          if(!b.prefix_bytes)Edge(p.wait,p.expect);
          producer=p.wait;
          if(b.prefix_bytes) {
            if(!s || stage.prefix_predecessor!=int(s)-1 || stage.incoming_order==Sm90DenseTailOrder::kNone || p.prefix_completions.empty())
              throw std::invalid_argument("native prefix lacks predecessor requests");
          }
          Need(p.wait,Service::kBarrierWait,Resource::kCtaProducerIssue,stage.kind+":empty_wait");
          Need(p.expect,Service::kProducerIssue,Resource::kCtaProducerIssue,stage.kind+":expect_or_bulk_arrive");
          Edge(p.expect,p.full);
          auto remainder=b.weight_bytes+b.kv_bytes-b.prefix_bytes;
          producer=Copies(work,h,producer,p,remainder,b.remainder_requests,step,s,c,t.work->task,false);
          for(auto done:p.prefix_completions)Edge(done,p.full);
          auto issue=Part(Kind::kActivationIssue,step,s,c,t.work->task,b,base[c]);
          graph.events[issue].bytes=b.activation_bytes;Edge(producer,issue);Edge(p.expect,issue);
          Need(issue,Service::kActivationIssue,Resource::kCtaProducerIssue,stage.kind);
          auto done=Part(Kind::kActivationComplete,step,s,c,t.work->task,b,base[c]);
          graph.events[done].bytes=b.activation_bytes;graph.events[done].arrivals=32;
          Edge(issue,done);Edge(done,p.full);
          Need(done,Service::kActivationCompletion,b.activation_bytes?Resource::kDeviceMemoryHierarchy:Resource::kCtaProducerIssue,stage.kind);
          producer=issue;
        }
        if(Gemm(stage)) {
          auto join=Add(Kind::kProducerRendezvous,step,s,c);
          graph.events[join].arrivals=32;
          Edge(producer,join);producer=join;
          Need(join,Service::kBarrierPublication,Resource::kCtaProducerIssue,
              stage.kind+":producer_warp_join_32");
        }
        if(stage.ring_slots>0 && !h.tasks.empty()) {
          auto publish=Add(Kind::kProducerGenerations,step,s,c);Edge(producer,publish);producer=publish;
          Need(publish,Service::kGenerationPublication,Resource::kSharedArena,stage.kind);
        }
        Edge(producer,h.produced);
        if(Gemm(stage)) {
          graph.events[h.produced].local_tasks=int(owner.tasks.size());
          Need(h.produced,Service::kProducerReturn,Resource::kCtaProducerIssue,
              stage.kind+":outlined_producer_return");
        }
        Id consumer=h.initialize;
        for(auto& t:h.tasks) {
          Edge(consumer,t.begin);
          if(stage.kind=="attention" || Gemm(stage))Need(t.begin,Service::kConsumerState,Resource::kSmConsumerIssue,stage.kind);
          consumer=t.begin;
          if(t.query!=kSm90DenseNoEvent) {
            Edge(consumer,t.query);Need(t.query,Service::kQueryCurrentRow,Resource::kSmConsumerIssue,stage.kind);consumer=t.query;
          }
          for(auto& p:t.blocks) {
            auto wait=Add(Kind::kFullWait,step,s,c,t.work->task);
            graph.events[wait]=graph.events[p.read];graph.events[wait].kind=Kind::kFullWait;
            Edge(consumer,wait);Edge(p.full,wait);Edge(wait,p.read);
            Need(wait,Service::kBarrierWait,Resource::kSmConsumerIssue,stage.kind+":full_wait");
            if(stage.kind=="attention" && stage.ring_slots>1) {
              auto patch=Add(Kind::kPatchAndPad,step,s,c,t.work->task);auto const& b=*p.work;
              graph.events[patch]=graph.events[p.read];graph.events[patch].kind=Kind::kPatchAndPad;
              Edge(wait,patch);Edge(patch,p.read);
              Need(patch,Service::kPatchAndPad,Resource::kSharedArena,stage.kind);
            }
            Need(p.read,Service::kConsumer,Resource::kSmConsumerIssue,stage.kind);
            Edge(p.read,p.empty);consumer=p.empty;
            Need(p.empty,Service::kBarrierPublication,Resource::kSharedArena,stage.kind+":empty_128");
          }
          if(t.finalize!=kSm90DenseNoEvent) {
            Edge(consumer,t.finalize);Need(t.finalize,Service::kFinalize,Resource::kSmConsumerIssue,stage.kind);
            Edge(t.finalize,t.partial);
          }else Edge(consumer,t.partial);
          Need(t.partial,stage.kind=="attention"?Service::kPartial:Gemm(stage)?Service::kEpilogue:Service::kTokenWork,
              Resource::kSmConsumerIssue,stage.kind);
          if(t.publish!=kSm90DenseNoEvent) {
            Edge(t.partial,t.publish);Edge(t.publish,t.continuation);
            Need(t.publish,Service::kAtomicPublication,Resource::kAtomicPublication,stage.kind);
          }else Edge(t.partial,t.continuation);
          consumer=t.continuation;
        }
        if(stage.ring_slots>0 && !h.tasks.empty()) {
          auto publish=Add(Kind::kConsumerGenerations,step,s,c);Edge(consumer,publish);consumer=publish;
          Need(publish,Service::kGenerationPublication,Resource::kSharedArena,stage.kind);
        }
        Tail(work,handles,s,c,step,h,before);
        auto join=Add(Kind::kCtaJoin,step,s,c);Edge(consumer,join);Edge(h.produced,join);Edge(h.tail_done,join);
        graph.events[join].arrivals=160;Need(join,Service::kBarrierPublication,Resource::kSharedArena,stage.kind+":CTA_join_160");
        Id arrival=join;
        if(s+1<work.stages.size() && (Gemm(work.stages[s+1]) || work.stages[s+1].kind=="attention")) {
          arrival=Add(Kind::kOperandRecordPublish,step,s,c);Edge(join,arrival);
          Need(arrival,Service::kGenerationPublication,Resource::kSharedArena,"next_operand_bank");
        }
        Edge(arrival,h.arrival);Need(h.arrival,Service::kGridArrival,Resource::kAtomicPublication,stage.kind);
        Need(h.release,Service::kGridRelease,Resource::kGridSynchronization,stage.kind);
      }
      if(stage.kind=="attention" && stage.merge_chunks>1)Merges(handles[s],stage,step,s);
      auto all=Add(Kind::kAllCtasArrived,step,s);auto envelope=Add(Kind::kStageEnvelopeEnd,step,s);
      for(int c=0;c<graph.grid;++c) {
        Edge(handles[s][c].arrival,all);Edge(all,handles[s][c].release);
        Edge(handles[s][c].release,envelope);previous_[c]=handles[s][c].release;
      }
      graph.stage_envelope_ends.push_back(envelope);
    }
  }
  Sm90DenseFlowEvents Finish() {
    auto end=Add(Kind::kKernelEnd);
    for(int c=0;c<graph.grid;++c) {
      Id p=previous_[c];
      for(int bank=0;bank<2;++bank) {
        auto invalid=Add(Kind::kRingInvalidate,-1,-1,c);graph.events[invalid].bank=bank;
        Edge(p,invalid);p=invalid;
      }
      Edge(p,end);
    }
    graph.kernel_end=end;graph.final_generations=generations_;
    graph.UnconditionalOrder();return std::move(graph);
  }
 private:
  Sm90DenseWorkGraph const& work_;
  Sm90DenseWorkGraph const* current_work_;
  std::vector<Slots> empty_;
  std::vector<Counts> generations_;
  std::vector<Id> previous_;
  std::map<std::string,std::size_t> requirement_index_;
  Id Add(Kind kind,int step=-1,int stage=-1,int cta=-1,int task=-1) {
    Sm90DenseFlowEvent e{kind};e.step=step;e.stage=stage;e.cta=cta;e.task=task;
    graph.events.push_back(e);return graph.events.size()-1;
  }
  Id Part(Kind kind,int step,int stage,int cta,int task,Sm90DenseBlockWork const& b,Counts const& base) {
    auto id=Add(kind,step,stage,cta,task);auto& e=graph.events[id];
    e.sequence=b.sequence;e.bank=work_.stages[stage].ring_bank;e.slot=b.slot;
    e.generation=base[e.bank][e.slot]+b.generation;e.history_rows=b.history_rows;
    e.valid_rows=b.valid_rows;e.compute_rows=b.compute_rows;e.current_rows=b.current_rows;
    return id;
  }
  void Edge(Id a,Id b,int condition=-1,int owner=-1) {graph.edges.push_back({a,b,condition,owner});}
  void Need(Id event,Service service,Resource resource,std::string const& operation) {
    auto const& e=graph.events[event];auto const& work=*current_work_;
    // Producer and consumer generation stores share a service family, but
    // execute on different roles and must never alias one measurement key.
    json::Object fields{{"schema",3},{"service",int(service)},{"resource",int(resource)},
        {"event_kind",int(e.kind)},{"operation",operation},
        {"sender_descriptor",e.stage>=0?work.stages[e.stage].ServiceDescriptor():json::Value()},
        {"receiver_descriptor",e.target_stage>=0?work.stages[e.target_stage].ServiceDescriptor():json::Value()},
        {"past",work.past},
        {"operation",operation},{"grid",graph.grid},{"num_sms",graph.num_sms},
        {"residency",graph.requested_residency},{"shared_bytes",graph.shared_bytes},
        {"sender_tasks",e.stage>=0?work.stages[e.stage].tasks:0},
        {"sender_active_ctas",e.stage>=0?std::min(graph.grid,work.stages[e.stage].tasks):0},
        {"receiver_active_ctas",e.target_stage>=0?std::min(graph.grid,work.stages[e.target_stage].tasks):0},
        {"ring_slots",e.stage>=0?work.stages[e.stage].ring_slots:0},
        {"sequence",e.sequence},{"slot",e.slot},{"bank",e.bank},
        {"request_index",e.request_index},{"request_count",e.request_count},
        {"incoming_order",e.stage>=0?int(work.stages[e.stage].incoming_order):0},
        {"bytes",double(e.bytes)},{"arrivals",e.arrivals},{"expected_transactions",e.expected_transactions},{"history_rows",e.history_rows},
        {"valid_rows",e.valid_rows},{"compute_rows",e.compute_rows},{"current_rows",e.current_rows}};
    if(e.local_tasks>=0)fields.emplace_back("local_tasks",e.local_tasks);
    auto key=json::Value(std::move(fields)).Dump(0);
    auto found=requirement_index_.emplace(key,graph.requirements.size());
    if(found.second)graph.requirements.push_back({service,resource,key,{}});
    graph.requirements[found.first->second].events.push_back(event);
  }
  Id Copies(Sm90DenseWorkGraph const& work,Cta&,Id issue,Block& p,std::int64_t bytes,int requests,
      int step,int stage,int cta,int task,bool prefix) {
    if((bytes==0)!=(requests==0) || requests<0 || requests>2 || (requests && bytes%requests))
      throw std::invalid_argument("native bulk request count/byte conservation differs");
    for(int request=0;request<requests;++request) {
      auto next=Add(Kind::kBulkIssue,step,stage,cta,task);auto& i=graph.events[next];
      i.bank=graph.events[p.expect].bank;i.slot=p.work->slot;i.sequence=p.work->sequence;
      i.generation=graph.events[p.expect].generation;i.bytes=bytes/requests;
      i.target_stage=prefix?stage+1:stage;i.request_index=request;i.request_count=requests;
      i.history_rows=p.work->history_rows;i.valid_rows=p.work->valid_rows;
      i.compute_rows=p.work->compute_rows;i.current_rows=p.work->current_rows;
      Edge(issue,next);Edge(p.expect,next);Need(next,Service::kProducerIssue,Resource::kCtaProducerIssue,work.stages[stage].kind);
      auto done=Add(Kind::kBulkComplete,step,stage,cta,task);graph.events[done]=graph.events[next];graph.events[done].kind=Kind::kBulkComplete;
      Edge(next,done);Need(done,Service::kBulkCompletion,Resource::kDeviceMemoryHierarchy,work.stages[stage].kind);
      if(prefix)p.prefix_completions.push_back(done);else Edge(done,p.full);
      issue=next;
    }
    return issue;
  }
  void Tail(Sm90DenseWorkGraph const& work,std::vector<std::vector<Cta>>& handles,unsigned s,int c,int step,Cta& h,bool before) {
    if(s+1==work.stages.size() || work.stages[s+1].incoming_order==Sm90DenseTailOrder::kNone) {
      Edge(h.produced,h.tail_begin);Edge(h.tail_begin,h.tail_done);return;
    }
    auto const& next=work.stages[s+1];
    if(next.prefix_predecessor!=int(s))throw std::invalid_argument("native flow predecessor is not adjacent");
    Edge(before?h.initialize:h.produced,h.tail_begin);Id issue=h.tail_begin;
    if(next.incoming_order==Sm90DenseTailOrder::kRetireOperands) {
      auto const& current=work.stages[s];
      for(int slot=0;slot<current.ring_slots;++slot) {
        auto retire=Add(Kind::kRetireSlot,step,s,c);auto& e=graph.events[retire];
        e.bank=current.ring_bank;e.slot=slot;e.generation=generations_[c][e.bank][slot];
        Edge(issue,retire);Edge(empty_[c][e.bank][slot],retire);issue=retire;
      }
    }
    auto& target=handles[s+1][c];
    if(!target.tasks.empty())for(auto& p:target.tasks.front().blocks)if(p.work->prefix_bytes) {
      Edge(issue,p.expect);
      issue=Copies(work,h,p.expect,p,p.work->prefix_bytes,p.work->prefix_requests,step,s,c,target.tasks.front().work->task,true);
    }
    auto const& owner=next.ctas[c];
    if((owner.l2_hint_bytes==0)!=(owner.l2_hint_requests==0) || owner.l2_hint_requests<0 || owner.l2_hint_requests>2 ||
        (owner.l2_hint_requests && owner.l2_hint_bytes%owner.l2_hint_requests))throw std::invalid_argument("native L2 request count differs");
    for(int request=0;request<owner.l2_hint_requests;++request) {
      auto hint=Add(Kind::kPrefetchL2Issue,step,s,c);auto& e=graph.events[hint];
      e.target_stage=s+1;e.bytes=owner.l2_hint_bytes/owner.l2_hint_requests;
      e.request_index=request;e.request_count=owner.l2_hint_requests;
      Edge(issue,hint);Need(hint,Service::kPrefetchIssue,Resource::kCtaProducerIssue,work.stages[s].kind);issue=hint;
    }
    Edge(issue,h.tail_done);
  }
  void Merges(std::vector<Cta>& handles,Sm90DenseStageWork const& stage,int step,int index) {
    std::map<int,std::vector<std::pair<int,Task*>>> groups;
    for(int c=0;c<graph.grid;++c)for(auto& t:handles[c].tasks)groups[t.work->group].push_back({c,&t});
    for(auto const& group:groups) {
      if(group.first<0 || group.second.size()!=unsigned(stage.merge_chunks))throw std::invalid_argument("native split-attention publication group differs");
      Sm90DenseMergeCondition condition;condition.step=step;condition.stage=index;condition.group=group.first;
      int condition_id=int(graph.merge_conditions.size());
      condition.all_published=Add(Kind::kGroupAllPublished,step,index);
      for(unsigned owner=0;owner<group.second.size();++owner) {
        auto& t=*group.second[owner].second;int c=group.second[owner].first;
        Edge(t.publish,condition.all_published);
        auto merge=Add(Kind::kMerge,step,index,c,t.work->task);auto& e=graph.events[merge];e.condition=condition_id;e.owner=owner;
        Edge(condition.all_published,merge);Edge(t.publish,merge);
        Edge(merge,t.continuation,condition_id,owner);
        Need(merge,Service::kMerge,Resource::kSmConsumerIssue,"attention_last_acq_rel_publisher");
        condition.alternatives.push_back({c,t.work->task,t.publish,merge,t.continuation});
      }
      graph.merge_conditions.push_back(std::move(condition));
    }
  }
};
}

std::vector<Sm90DenseEventId> Sm90DenseFlowEvents::UnconditionalOrder() const {
  std::vector<std::vector<Id>> next(events.size());std::vector<unsigned> degree(events.size());
  for(auto const& edge:edges) {
    if(edge.before>=events.size() || edge.after>=events.size())throw std::invalid_argument("native flow edge is out of bounds");
    if(edge.condition>=0) {
      if(unsigned(edge.condition)>=merge_conditions.size() || edge.owner<0 ||
          unsigned(edge.owner)>=merge_conditions[edge.condition].alternatives.size())
        throw std::invalid_argument("native conditional merge edge differs");
      continue;
    }
    next[edge.before].push_back(edge.after);++degree[edge.after];
  }
  std::deque<Id> ready;for(Id id=0;id<degree.size();++id)if(!degree[id])ready.push_back(id);
  std::vector<Id> result;
  while(!ready.empty()) {
    auto id=ready.front();ready.pop_front();result.push_back(id);
    for(auto after:next[id])if(!--degree[after])ready.push_back(after);
  }
  if(result.size()!=events.size())throw std::invalid_argument("native unconditional source-order cycle");
  return result;
}
Sm90DenseFlowCoverage Sm90DenseFlowEvents::Coverage(Sm90DenseResidentEvidence const& resident,
    std::string const& cache,std::vector<Sm90DenseFlowServiceEvidence> const& evidence) const {
  Sm90DenseFlowCoverage result;
  if(!Hex(resident.binary_sha256) || !Hex(resident.source_digest) || resident.grid!=grid ||
      resident.dynamic_shared_bytes!=shared_bytes || resident.resident_ctas_per_sm<requested_residency)
    result.pending.push_back({"resident_grid","bound final-binary occupancy does not cover this persistent launch"});
  if(cache.empty() || cache=="unknown" || cache=="unclassified")
    result.pending.push_back({"cache_state","explicit observed cache protocol required"});
  std::map<std::string,Sm90DenseFlowServiceEvidence const*> matches;
  for(auto const& item:evidence) {
    if(!matches.emplace(item.requirement_key,&item).second)throw std::invalid_argument("duplicate native flow service witness");
  }
  for(auto const& requirement:requirements) {
    auto match=matches.find(requirement.key);
    auto p=match==matches.end()?nullptr:match->second;
    if(!p || !p->verified_measurement || p->processes<3 || p->measurement_id.empty() ||
        !Hex(p->record_sha256) || !Hex(p->measured_source_digest) || !Hex(p->measured_binary_sha256) ||
        p->measured_source_digest!=resident.source_digest || p->measured_binary_sha256!=resident.binary_sha256 ||
        p->cache_state!=cache)
      result.pending.push_back({requirement.key,"missing or mismatched caller-provided service/resource witness"});
  }
  for(auto const& stage:pending_stage_bindings)
    result.pending.push_back({stage,"actual emitted GEMM epilogue/input_norm/output_sums binding required"});
  for(unsigned i=0;i<merge_conditions.size();++i)
    result.pending.push_back({"merge_owner:"+std::to_string(i),
        "last acq_rel publication owner requires a future arrival-order/resource resolver"});
  return result;
}
json::Value Sm90DenseFlowCoverage::ToJson() const {
  json::Array rows;for(auto const& p:pending)rows.emplace_back(json::Object{{"requirement",p.requirement},{"reason",p.reason}});
  return json::Object{{"status",pending.empty()?"referenced_services_covered":"service_pending"},
      {"prediction_available",false},{"solver_gate_pass",false},{"E3_validated",false},
      {"evidence_authenticated_here",false},{"coverage_scope","caller_provided_witness_metadata_only"},{"pending",rows}};
}
json::Value Sm90DenseFlowEvents::ToJson(bool include_events) const {
  json::Array needed,conditions,generations,rows,dependencies;
  for(auto const& r:requirements)needed.emplace_back(json::Object{{"service",int(r.service)},
      {"resource",int(r.resource)},{"key",r.key},{"event_count",double(r.events.size())}});
  for(auto const& c:merge_conditions) {
    json::Array alternatives;
    for(auto const& a:c.alternatives)alternatives.emplace_back(json::Object{{"cta",a.cta},{"task",a.task},
        {"publication",double(a.publication)},{"merge",double(a.merge)},{"continuation",double(a.continuation)}});
    conditions.emplace_back(json::Object{{"step",c.step},{"stage",c.stage},{"group",c.group},
        {"owner","runtime_last_acq_rel_publication"},{"all_published",double(c.all_published)},{"alternatives",alternatives}});
  }
  for(int c=0;c<grid;++c)for(int bank=0;bank<2;++bank) {
    json::Array slots;for(auto generation:final_generations[c][bank])slots.emplace_back(double(generation));
    generations.emplace_back(json::Object{{"cta",c},{"bank",bank},{"next_generation_ordinals",slots}});
  }
  if(include_events) {
    for(auto const& e:events)rows.emplace_back(json::Object{{"kind",int(e.kind)},{"step",e.step},{"stage",e.stage},
        {"target_stage",e.target_stage},{"cta",e.cta},{"task",e.task},{"sequence",e.sequence},
        {"bank",e.bank},{"slot",e.slot},{"generation_ordinal",double(e.generation)},
        {"bytes",double(e.bytes)},{"arrivals",e.arrivals},{"expected_transactions",e.expected_transactions},
        {"history_rows",e.history_rows},{"valid_rows",e.valid_rows},{"compute_rows",e.compute_rows},
        {"current_rows",e.current_rows},{"request_index",e.request_index},{"request_count",e.request_count},
        {"local_tasks",e.local_tasks},{"condition",e.condition},{"owner",e.owner}});
    for(auto const& e:edges)dependencies.emplace_back(json::Object{{"before",double(e.before)},{"after",double(e.after)},
        {"condition",e.condition},{"owner",e.owner}});
  }
  return json::Object{{"schema",1},{"status","source_order_events_only"},{"prediction_available",false},
      {"E3_validated",false},{"grid",grid},{"steps",steps},{"shared_bytes",shared_bytes},
      {"event_count",double(events.size())},{"edge_count",double(edges.size())},
      {"generation_semantics","persistent use ordinal; not a hardware parity-equality test"},
      {"events_are_service_start_constraints",false},{"latency_or_default_rates_supplied",false},
      {"event_count_includes_mutually_exclusive_alternatives",true},
      {"conditional_owner_resolution_available",false},{"logical_bytes_are_HBM_transactions",false},
      {"L2_hints_guarantee_readiness",false},{"per_CTA_release_skew_preserved",true},
      {"service_requirements",needed},{"merge_conditions",conditions},{"final_generations",generations},
      {"events",rows},{"edges",dependencies}};
}
Sm90DenseFlowEvents BuildSm90DenseFlowEvents(Sm90DenseWorkGraph const& work) {
  Builder builder(work);builder.Step(work);return builder.Finish();
}
Sm90DenseFlowEvents BuildSm90DenseFlowEvents(std::vector<Sm90DenseWorkGraph> const& steps) {
  if(steps.empty())throw std::invalid_argument("native flow needs a program step");
  Builder builder(steps.front());for(auto const& step:steps)builder.Step(step);return builder.Finish();
}
}  // namespace tilemega::solver
