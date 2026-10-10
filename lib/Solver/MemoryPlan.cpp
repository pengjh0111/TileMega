// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MemoryPlan.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tilemega::solver {
namespace {
using namespace analysis;
struct Lifetime {
  unsigned buffer, first, last;
  std::uint64_t bytes;
  std::vector<StorageTaskAccess> writes, reads;
};
unsigned Width(frontend::PlanBuffer const& buffer) {
  if(buffer.dtype=="bf16")return 2;
  if(buffer.dtype=="f32" || buffer.dtype=="i32")return 4;
  if(buffer.dtype=="i64")return 8;
  throw std::invalid_argument("memory planner has an unknown buffer dtype");
}
std::uint64_t Align(std::uint64_t bytes) {
  if(bytes>std::numeric_limits<std::uint64_t>::max()-255)
    throw std::overflow_error("memory arena size overflow");
  return (bytes+255)/256*256;
}
std::uint64_t AddBytes(std::uint64_t a,std::uint64_t b) {
  if(a>std::numeric_limits<std::uint64_t>::max()-b)
    throw std::overflow_error("memory allocation size overflow");
  return a+b;
}
std::uint64_t AllocationBytes(frontend::PlanBuffer const& buffer,
    frontend::ModelPlan const& plan,ParamBinding const& known) {
  auto const& layout=buffer.layout;
  std::uint64_t elements=0;
  if(layout.rank) {
    if(layout.physical[0] && layout.strides[0]>
        std::numeric_limits<std::uint64_t>::max()/layout.physical[0])
      throw std::overflow_error("memory allocation extent overflow");
    elements=layout.physical[0]*layout.strides[0];
  }else {
    if(buffer.per_past || buffer.per_total)
      throw std::invalid_argument("stateless memory planner found a state allocation");
    elements=AddBytes(buffer.constant,std::uint64_t(buffer.per_seq)*plan.serving_seq);
    if(buffer.per_batch) {
      if(!known.Contains("B") || known.At("B")<=0)
        throw std::invalid_argument("memory allocation needs the bound DNN batch");
      auto batch=std::uint64_t(known.At("B"));
      if(batch>std::numeric_limits<std::uint64_t>::max()/buffer.per_batch)
        throw std::overflow_error("memory batch allocation overflow");
      elements=AddBytes(elements,batch*buffer.per_batch);
    }
  }
  if(elements>std::numeric_limits<std::uint64_t>::max()/Width(buffer))
    throw std::overflow_error("memory allocation byte extent overflow");
  return elements*Width(buffer);
}
bool Compatible(frontend::PlanBuffer const& a,frontend::PlanBuffer const& b) {
  if(a.dtype!=b.dtype || a.layout.fill!=b.layout.fill)return false;
  auto const& x=a.layout;auto const& y=b.layout;
  auto halo=[](auto const& l){return l.halo_top||l.halo_bottom||l.halo_left||l.halo_right;};
  if(!halo(x) && !halo(y))return true;
  if(x.kind!=y.kind || x.rank!=y.rank || x.halo_top!=y.halo_top ||
      x.halo_bottom!=y.halo_bottom || x.halo_left!=y.halo_left || x.halo_right!=y.halo_right)
    return false;
  for(unsigned i=0;i<4;++i)
    if(x.logical[i]!=y.logical[i] || x.physical[i]!=y.physical[i] || x.strides[i]!=y.strides[i])
      return false;
  return true;
}
bool SameLayout(frontend::PlanBuffer const& a,frontend::PlanBuffer const& b) {
  auto const& x=a.layout;auto const& y=b.layout;
  if(x.kind!=y.kind || x.rank!=y.rank)return false;
  for(unsigned i=0;i<4;++i)
    if(x.physical[i]!=y.physical[i] || x.strides[i]!=y.strides[i])return false;
  return true;
}
CouplingRelation Physical(CouplingRelation const& access,
    frontend::PlanBuffer const& buffer,ParamBinding const& known) {
  auto const& layout=buffer.layout;auto rank=layout.rank;
  if(rank!=access.RangeDimNames().size())
    throw std::invalid_argument("memory planner tensor/layout rank mismatch: "+buffer.name);
  std::ostringstream map;map<<"{ [";
  for(unsigned axis=0;axis<rank;++axis)map<<(axis?",":"")<<"x"<<axis;
  map<<"] -> [byte] : byte = 0";
  for(unsigned axis=0;axis<rank;++axis)
    map<<" + "<<Width(buffer)*layout.strides[axis]<<"*x"<<axis;
  map<<" }";
  return access.BindParams(known).ApplyRange(CouplingRelation::FromIslText(map.str()));
}
}
MemoryPlan PlanBufferReuse(frontend::ModelPlan const& plan,OperatorGraph const& graph,
    ParamBinding const& known,std::string const& policy,std::uint64_t budget) {
  MemoryPlan result;
  if(policy=="none")return result;
  if(policy!="greedy" && policy!="l2")throw std::invalid_argument("invalid memory reuse policy");
  if(!plan.dm || !plan.forward || plan.forward_token_axis)
    throw std::invalid_argument("DNN buffer reuse requires a stateless DNN plan");
  if(policy=="l2" && !budget)throw std::invalid_argument("L2 reuse needs a target-derived budget");
  std::map<std::string,Lifetime> lives;
  std::map<unsigned,std::uint64_t> internal_allocations;
  // Every internal allocation has already been bound by storage materialization.
  for(unsigned id=0;id<plan.buffers.size();++id) {
    auto const& b=plan.buffers[id];
    if(b.role=="internal" && b.source==frontend::PlanBuffer::Source::kZero)
      internal_allocations.emplace(id,AllocationBytes(b,plan,known));
    if(b.role!="internal" || b.source!=frontend::PlanBuffer::Source::kZero ||
        !b.pack_json.empty() || !b.file.empty())continue;
    if(std::any_of(plan.outputs.begin(),plan.outputs.end(),[&](auto const& output) {
        return output.buffer==id;}))continue;
    auto const& l=b.layout;
    if(!l.rank)continue; // A buffer without a physical layout has no alias proof.
    std::uint64_t elements=l.physical[0]*l.strides[0];
    if(!elements || elements>std::numeric_limits<std::uint64_t>::max()/Width(b))
      throw std::invalid_argument("memory planner invalid allocation extent");
    lives.emplace(b.name,Lifetime{id,~0u,0,elements*Width(b),{}, {}});
  }
  for(unsigned index=0;index<graph.nodes.size();++index) {
    auto const& task=graph.nodes[index];
    if(!task.element_access) {
      lives.erase(task.output.name);
      for(auto const& input:task.operands)lives.erase(input.tensor.name);
      continue;
    }
    auto const& access=*task.element_access;auto const& sem=access.semantic;
    auto project=[&](TensorSpace const& tensor,IndexingMap const& map,
        std::vector<IndexResult> const& predicates,bool write) {
      auto found=lives.find(tensor.name);if(found==lives.end())return;
      if(std::any_of(map.results.begin(),map.results.end(),[](auto const& x) {
          return x.kind==IndexResult::Kind::kDataDependent;})) {
        lives.erase(found);return;
      }
      auto& life=found->second;auto const& b=plan.buffers[life.buffer];
      auto relation=write?ProjectTaskWrite(sem,task,access.partition,tensor,map,predicates,known):
          ProjectTaskRead(sem,task,access.partition,tensor,map,predicates,known);
      if(tensor.axes.size()!=b.layout.rank)
        throw std::invalid_argument("memory planner tensor/layout rank mismatch: "+b.name);
      StorageTaskAccess physical{&task,relation.BindParams(known)};
      (write?life.writes:life.reads).push_back(std::move(physical));
      if(write)life.first=std::min(life.first,index);
      life.last=std::max(life.last,index);
    };
    project(sem.result,sem.result_map,{},true);
    for(auto const& side:sem.additional_writes)project(side.tensor,side.map,side.nonnegative,true);
    if(sem.element_reads.empty()) {
      for(auto const& input:sem.operands)project(input.tensor,input.map,{},false);
    }else {
      for(auto const& input:sem.element_reads)project(input.tensor,input.map,input.nonnegative,false);
    }
  }
  std::vector<Lifetime const*> ordered;
  for(auto const& [name,life]:lives) {
    // Repeated writers or overlapping reduction ownership need a separate
    // storage contract; never infer that contract from stage order alone.
    if(life.writes.size()==1 && life.first!=~0u && life.writes[0].elements.Reverse().IsSingleValued())
      ordered.push_back(&life);
  }
  std::sort(ordered.begin(),ordered.end(),[](auto* a,auto* b) {
    return a->first!=b->first?a->first<b->first:a->bytes>b->bytes;});
  std::set<unsigned> aliased;
  for(auto* life:ordered)aliased.insert(life->buffer);
  // Unproved layouts, retained outputs and multiple/data-dependent writers
  // still allocate storage. Conservatively keep them live throughout the plan.
  for(auto const& [id,bytes]:internal_allocations)if(!aliased.count(id))
    result.retained_internal_bytes=AddBytes(result.retained_internal_bytes,Align(bytes));
  for(unsigned stage=0;stage<graph.nodes.size();++stage) {
    std::uint64_t active=result.retained_internal_bytes;
    for(auto* life:ordered)if(life->first<=stage && stage<=life->last)
      active=AddBytes(active,life->bytes);
    result.live_peak_bytes=std::max(result.live_peak_bytes,active);
  }
  struct Slot {
    std::uint64_t offset,bytes;
    Lifetime const* occupant;
    // A smaller allocation can leave an older allocation's tail untouched.
    // Preserve those readers/writers until a later write covers all of their
    // observed elements; the newest occupant alone cannot protect that tail.
    std::vector<Lifetime const*> history;
  };
  std::vector<Slot> slots;
  for(auto* life:ordered) {
    Slot* selected=nullptr;
    for(auto& slot:slots)if(slot.occupant->last<life->first && slot.bytes>=life->bytes &&
        Compatible(plan.buffers[slot.occupant->buffer],plan.buffers[life->buffer])) {
      if(!selected || (policy=="l2" && slot.bytes<selected->bytes))selected=&slot;
      if(policy=="greedy")break;
    }
    if(!selected) {
      auto bytes=Align(life->bytes);
      if(result.arena_bytes>std::numeric_limits<std::uint64_t>::max()-bytes)
        throw std::overflow_error("memory arena size overflow");
      slots.push_back({result.arena_bytes,bytes,life,{}});selected=&slots.back();
      result.arena_bytes+=bytes;
    }else {
      std::vector<Lifetime const*> retained;
      for(auto* previous:selected->history) {
        auto old_write=previous->writes[0],new_write=life->writes[0];
        auto reads=previous->reads;
        if(!SameLayout(plan.buffers[previous->buffer],plan.buffers[life->buffer])) {
          old_write.elements=Physical(old_write.elements,plan.buffers[previous->buffer],known);
          new_write.elements=Physical(new_write.elements,plan.buffers[life->buffer],known);
          for(auto& read:reads)read.elements=Physical(read.elements,plan.buffers[previous->buffer],known);
        }
        auto hazards=DeriveStorageReuseHazards(old_write,reads,new_write,known);
        result.hazards.insert(result.hazards.end(),hazards.begin(),hazards.end());
        auto covered=new_write.elements.Image();
        bool complete=old_write.elements.Image().IsSubset(covered);
        for(auto const& read:reads)complete&=read.elements.Image().IsSubset(covered);
        if(!complete)retained.push_back(previous);
      }
      selected->history=std::move(retained);
      selected->occupant=life;
    }
    selected->history.push_back(life);
    result.aliases.push_back({life->buffer,selected->offset,life->bytes});
  }
  result.total_internal_bytes=AddBytes(result.arena_bytes,result.retained_internal_bytes);
  result.live_peak_bytes=std::max(result.live_peak_bytes,result.retained_internal_bytes);
  result.fits_l2_budget=budget && result.total_internal_bytes<=budget;
  return result;
}
} // namespace tilemega::solver
