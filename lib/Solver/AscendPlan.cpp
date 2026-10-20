// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/AscendPlan.h>
#include <tilemega/Support/Json.h>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
AscendPlan MaterializeAscendPlan(SymbolicProblem const& problem,PlanSkeleton const& skeleton,
    ascend::TargetCapability const& target,std::vector<ascend::TaskImplementation> const& impls,
    unsigned epochs) {
  if(!skeleton.ascend_workers || skeleton.costs_calibrated || skeleton.kappa!=1 ||
     impls.size()!=skeleton.spaces.size() || epochs<1 || epochs>16)
    throw std::invalid_argument("AS-1 requires a bound fixed skeleton, kappa=1 and epochs=1..16");
  AscendPlan plan;plan.workers=*skeleton.ascend_workers;plan.stage_implementations=impls;
  auto total=problem.offsets.back();
  if(total<=0 || total>4096)throw std::invalid_argument("AS-1 bounded plan requires 1..4096 tasks");
  plan.tasks.resize(total);plan.task_metadata.resize(total);
  std::vector<ascend::TaskAssignment> assignments(total);
  std::vector<std::vector<unsigned>> queues(plan.workers.workers.size());
  for(auto s:skeleton.stage_order) {
    auto const& space=skeleton.spaces.at(s);auto const& impl=impls.at(s);
    impl.Validate(target);
    if(problem.projection.stages.at(s).combine ||
       impl.kind!=static_cast<codegen::TaskKind>(problem.model.stages.at(problem.projection.stages[s].logical_stage).kind))
      throw std::invalid_argument("Native implementation disagrees with projected TaskKind");
    if(space.width!=1 || !space.ascend_role || *space.ascend_role!=impl.required_role)
      throw std::invalid_argument("Stage implementation lacks its legal worker domain");
    for(int tile=0;tile<space.count;++tile) {
      unsigned id=space.offset+tile;int owner=skeleton.Home(s,tile);
      std::set<unsigned> predecessors;
      for(int e:skeleton.incoming.at(s)) {
        auto const& edge=skeleton.edges.at(e);auto const& producer=skeleton.spaces.at(edge.producer);
        edge.oracle->reverse.Query({tile},skeleton.theta).ForEach([&](auto const& p) {
          if(p.size()!=1 || p[0]<0 || p[0]>=producer.count)
            throw std::invalid_argument("Oracle predecessor lies outside its task domain");
          predecessors.insert(producer.offset+p[0]);
        });
      }
      auto begin=uint32_t(plan.dependencies.size());
      plan.dependencies.insert(plan.dependencies.end(),predecessors.begin(),predecessors.end());
      auto count=uint32_t(predecessors.size());
      plan.tasks[id]={unsigned(s),unsigned(tile),begin,count,begin,count,0};
      plan.task_metadata[id]={unsigned(owner),id};
      auto& a=assignments[id];a.owner=owner;a.implementation=impl;
      a.dependencies.assign(predecessors.begin(),predecessors.end());a.publishes={id};a.participants={owner};
      if(impl.required_role==ascend::ImplementationRole::kMix1_2) {
        int group=plan.workers.At(owner).group;
        for(auto const& w:plan.workers.workers)
          if(w.group==group && w.global_worker_id!=owner)a.participants.push_back(w.global_worker_id);
      }
      for(int worker:a.participants)queues.at(worker).push_back(id);
    }
  }
  plan.audit=ascend::ValidatePlan(target,plan.workers,assignments,queues,std::vector<unsigned>(total,1));
  plan.schedule_offsets.push_back(0);
  for(auto const& q:queues) {
    plan.schedule.insert(plan.schedule.end(),q.begin(),q.end());
    plan.schedule_offsets.push_back(plan.schedule.size());
  }
  plan.workers.Validate(target,plan.schedule_offsets,plan.schedule.size());
  plan.metadata={ascend::abi::kVersion,unsigned(plan.workers.launch.num_groups),unsigned(queues.size()),
    unsigned(total),unsigned(plan.dependencies.size()),epochs,unsigned(impls.size()),unsigned(problem.model.physical_buffers.size())};
  plan.storage_bytes["plan_tables"]=sizeof(ascend::abi::Metadata)+plan.tasks.size()*sizeof(ascend::abi::TaskRef)+
    plan.task_metadata.size()*sizeof(ascend::abi::TaskMeta)+4*(plan.schedule_offsets.size()+plan.schedule.size()+
        std::max<std::size_t>(1,plan.dependencies.size()));
  plan.storage_bytes["single_owner_events"]=std::uint64_t(total)*512;
  plan.storage_bytes["epoch_control"]=std::uint64_t(queues.size()+2)*512;
  plan.storage_bytes["trace_with_abi_roundtrip"]=std::uint64_t(total+1)*512;
  if(problem.model.gemms.size()==1 && problem.model.gemm_access.size()==1) {
    auto const& g=problem.model.gemms[0];
    auto rows=std::uint64_t(problem.model.gemm_access[0].rows_per_batch)*problem.model.dims.batch;
    auto padded_k=(g.k+15)/16*16;
    plan.storage_bytes["logical_input"]=rows*g.k*2;
    plan.storage_bytes["packed_input"]=rows*padded_k*2;
    plan.storage_bytes["logical_weight"]=std::uint64_t(g.n)*g.k*2;
    plan.storage_bytes["packed_weight"]=std::uint64_t(g.n)*padded_k*2;
    plan.storage_bytes["output"]=rows*g.n*2;
    plan.storage_bytes["data_guards"]=3*512;
    auto const& bytes=plan.storage_bytes;
    plan.storage_bytes["total_device_allocation"]=bytes.at("plan_tables")+bytes.at("single_owner_events")+
      bytes.at("epoch_control")+bytes.at("trace_with_abi_roundtrip")+bytes.at("packed_input")+
      bytes.at("packed_weight")+bytes.at("output")+bytes.at("data_guards");
  }
  return plan;
}
std::string AscendPlan::ToJson() const {
  using namespace json;
  Array workers,stages,tasks_json,offsets,refs,topology;
  Value storage(Object{});
  for(auto const& [name,bytes]:storage_bytes)storage.Set(name,double(bytes));
  for(auto const& w:this->workers.workers)workers.emplace_back(Object{
    {"worker",w.global_worker_id},{"group",w.group},{"role",w.role==ascend::TrackRole::kAic?"aic":"aiv"},
    {"local_block_id",w.local_block_id},{"sub_block_id",w.sub_block_id}});
  for(auto const& impl:stage_implementations)stages.emplace_back(Object{
    {"impl_id",impl.impl_id},{"task_kind",int(impl.kind)},
    {"role",impl.required_role==ascend::ImplementationRole::kAicOnly?"AIC_ONLY":
      impl.required_role==ascend::ImplementationRole::kAivOnly?"AIV_ONLY":"MIX_1_2"},
    {"input_layout",impl.input_layout},{"output_layout",impl.output_layout}});
  for(std::size_t i=0;i<tasks.size();++i) {
    auto const& t=tasks[i];Array deps;
    for(unsigned j=0;j<t.dependency_count;++j)deps.emplace_back(int(dependencies.at(t.dependency_begin+j)));
    tasks_json.emplace_back(Object{{"id",int(i)},{"stage",int(t.stage)},{"logical_task",int(t.logical_task)},
      {"owner",int(task_metadata[i].owner)},{"event",int(task_metadata[i].event)},
      {"triggers",1},{"dependencies",std::move(deps)}});
  }
  for(auto i:schedule_offsets)offsets.emplace_back(int(i));
  for(auto i:schedule)refs.emplace_back(int(i));
  for(auto i:audit.topological_order)topology.emplace_back(int(i));
  return Value(Object{{"schema","tilemega.ascend.plan.v1"},{"cost_status","not_calibrated"},
    {"selection","fixed_legal_whitelist"},{"launch_groups",int(metadata.groups)},
    {"launch_blocks",this->workers.launch.num_blocks},{"residency",1},{"epochs",int(metadata.epochs)},
    {"workers",std::move(workers)},{"stage_implementations",std::move(stages)},
    {"tasks",std::move(tasks_json)},{"schedule_offsets",std::move(offsets)},
    {"schedule",std::move(refs)},{"wait_graph_acyclic",true},
    {"wait_graph_topological_order",std::move(topology)},
    {"event_triggers_single_owner",true},{"storage_bytes",std::move(storage)}}).Dump();
}
} // namespace tilemega::solver
