// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/FusionResources.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/TaskModel.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
namespace {
analysis::ParamBinding Coordinate(std::vector<std::string> const& names,std::vector<long> const& values) {
  if (names.size()!=values.size()) throw std::invalid_argument("fusion coordinate rank mismatch");
  analysis::ParamBinding point;
  for (std::size_t i=0;i<names.size();++i) point.Bind(names[i],values[i]);
  return point;
}

using Counts=std::map<std::string,std::vector<long>>;
Counts AccessCounts(std::map<std::string,analysis::CouplingRelation> const& accesses,
    analysis::ParamBinding const& theta,std::vector<std::vector<long>> const& coordinates) {
  Counts result;
  for (auto const& [name,relation]:accesses) {
    std::vector<analysis::ParamBinding> points;
    auto names=relation.DomainDimNames();
    for (auto const& coordinate:coordinates) points.push_back(Coordinate(names,coordinate));
    result.emplace(name,relation.Card().EvalPoints(theta,points));
  }
  return result;
}
}  // namespace

FusionTaskPrice PriceFusionTasks(ModelFusionCandidate const& candidate,
    CostModel const& cost,BackendTraits const& producer,BackendTraits const& consumer,
    Residency residency,ModelDescription const& model) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_FUSION_TASK_COST
  throw std::runtime_error("fusion task pricing disabled");
#endif
  if (model.dims.IsSymbolic() || residency.ctas_per_sm<=0 || producer.threads<=0 ||
      producer.threads!=consumer.threads || producer.stages<0 || consumer.stages!=0)
    throw std::invalid_argument("fusion price requires bound theta, common CTA and a SIMT consumer");
  auto theta=model.MetricBindings();
  auto relation=candidate.accesses.consumer_to_producer.BindParams(theta);
  if (!relation.IsSingleValued()) throw std::invalid_argument("fusion price violates tile constraint");
  auto pairs=relation.Points();
  std::sort(pairs.begin(),pairs.end());
  std::vector<std::vector<long>> pc,cc;
  std::map<std::vector<long>,long> fanouts;
  for (auto const& [c,p]:pairs) { cc.push_back(c); pc.push_back(p); ++fanouts[p]; }
  FusionTaskPrice result;
  result.producer_tasks=candidate.producer.work.task_count.Eval(theta);
  result.consumer_tasks=candidate.accesses.task_count.Eval(theta);
  result.recomputed_tasks=candidate.accesses.recompute_tasks.Eval(theta);
  if (result.consumer_tasks!=static_cast<long>(pairs.size()) ||
      result.producer_tasks!=static_cast<long>(fanouts.size()))
    throw std::invalid_argument("fusion price requires complete producer/consumer coverage");
  if (result.recomputed_tasks!=result.consumer_tasks-result.producer_tasks)
    throw std::invalid_argument("fusion recomputation metric differs from coupling");
  auto pr=AccessCounts(candidate.producer_accesses.reads,theta,pc);
  auto pw=AccessCounts(candidate.producer_accesses.writes,theta,pc);
  auto cr=AccessCounts(candidate.consumer_accesses.reads,theta,cc);
  auto cw=AccessCounts(candidate.consumer_accesses.writes,theta,cc);
  double bytes=model.dtype==ScalarType::kBF16 ? 2 : 4;
  long grid=static_cast<long>(cost.target().res.num_sms)*residency.ctas_per_sm;
  if (grid<=0) throw std::invalid_argument("fusion wave grid is empty");
  auto occupancy=[&](long active) { return cost.options().wave_tail
      ? std::max(1.0,double(active)/cost.target().res.num_sms) : residency.ctas_per_sm; };
  std::set<std::vector<long>> seen;
  std::vector<std::vector<long>> unique;
  for (auto const& p:pc) if (seen.insert(p).second) unique.push_back(p);
  std::sort(unique.begin(),unique.end());
  std::vector<analysis::ParamBinding> producer_points;
  for (auto const& p:unique) producer_points.push_back(Coordinate(relation.RangeDimNames(),p));
  auto checked_fanout=candidate.accesses.fanout.EvalPoints(theta,producer_points);
  for (std::size_t i=0;i<unique.size();++i)
    if (checked_fanout[i]!=fanouts.at(unique[i]))
      throw std::invalid_argument("fusion per-producer fanout differs from CG metric");
  auto separate=[&](DerivedTaskInput const& input,BackendTraits const& traits,
                    std::vector<std::vector<long>> const& coordinates,bool is_producer) {
    double total=0;
    for (long first=0;first<static_cast<long>(coordinates.size());first+=grid) {
      long active=std::min(grid,static_cast<long>(coordinates.size())-first);
      double wave=0,o=occupancy(active);
      for (long i=first;i<first+active;++i) {
        auto point=Coordinate(input.cost_coordinates,coordinates[i]);
        double ns=cost.TaskInstanceNs(input,traits,residency,model,1,point,o);
        wave=std::max(wave,ns);
        if (is_producer) result.recompute_ns+=(fanouts.at(coordinates[i])-1)*ns;
      }
      total+=wave;
    }
    return total;
  };
  result.separate_ns=separate(candidate.producer,producer,unique,true)+
      separate(candidate.consumer,consumer,cc,false);
  result.producer_waves=(result.producer_tasks+grid-1)/grid;
  result.consumer_waves=(result.consumer_tasks+grid-1)/grid;
  seen.clear();
  for (long first=0;first<result.consumer_tasks;first+=grid) {
    long active=std::min(grid,result.consumer_tasks-first);
    double wave=0,o=occupancy(active);
    for (long i=first;i<first+active;++i) {
      TaskMemoryTraffic pm,cm;
      for (auto const& [name,counts]:pr) pm.global_read_bytes+=bytes*counts[i];
      for (auto const& [name,counts]:pw) {
        bool local=candidate.accesses.intermediate_tiles.count(name);
        if (local) pm.local_write_bytes+=bytes*counts[i];
        if (!local || candidate.accesses.retained_intermediates.count(name))
          pm.global_write_bytes+=bytes*counts[i];
      }
      for (auto const& [name,counts]:cr) {
        bool local=candidate.accesses.intermediate_tiles.count(name);
        (local ? cm.local_read_bytes : cm.global_read_bytes)+=bytes*counts[i];
      }
      for (std::size_t operand=0;operand<candidate.consumer.task.operands.size();++operand)
        if (candidate.accesses.intermediate_tiles.count(candidate.consumer.task.operands[operand].tensor.name))
          cm.local_read_operands.insert(operand);
      for (auto const& [name,counts]:cw) cm.global_write_bytes+=bytes*counts[i];
      double p=cost.TaskInstanceNs(candidate.producer,producer,residency,model,1,
          Coordinate(candidate.producer.cost_coordinates,pc[i]),o,&pm);
      double c=cost.TaskInstanceNs(candidate.consumer,consumer,residency,model,1,
          Coordinate(candidate.consumer.cost_coordinates,cc[i]),o,&cm);
      wave=std::max(wave,p+c+cost.target().calib.syncthreads_ns);
      result.global_bytes_after+=pm.global_read_bytes+pm.global_write_bytes+cm.global_read_bytes+cm.global_write_bytes;
      result.local_bytes+=pm.local_write_bytes+cm.local_read_bytes;
      if (seen.insert(pc[i]).second) {
        for (auto const& [name,counts]:pr) result.global_bytes_before+=bytes*counts[i];
        for (auto const& [name,counts]:pw) result.global_bytes_before+=bytes*counts[i];
      }
      for (auto const& [name,counts]:cr) result.global_bytes_before+=bytes*counts[i];
      for (auto const& [name,counts]:cw) result.global_bytes_before+=bytes*counts[i];
    }
    result.fused_ns+=wave;
  }
  return result;
}

double FusionRecomputeNs(analysis::FusionAccesses const& accesses,
    CostModel const& cost, DerivedTaskInput const& producer, BackendTraits const& traits,
    Residency residency, ModelDescription const& model, int chunks,
    double active_ctas_per_sm) {
  analysis::IslReferenceAudit audit(__func__);
  auto theta=model.MetricBindings();
  auto relation=accesses.consumer_to_producer.BindParams(theta);
  auto names=relation.RangeDimNames();
  if (names!=producer.cost_coordinates)
    throw std::invalid_argument("fusion producer cost coordinates differ from coupling");
  std::map<std::vector<long>,long> fanout;
  for (auto const& [consumer,task]:relation.Points()) ++fanout[task];
  double total=0;
  for (auto const& [task,count]:fanout) {
    analysis::ParamBinding point;
    for (std::size_t axis=0;axis<names.size();++axis) point.Bind(names[axis],task[axis]);
    auto exact=accesses.fanout.BindCoordinates(point).SubstituteParams(theta).Eval({});
    if (exact!=count) throw std::runtime_error("fusion fanout differs from exact CG metric");
    if (count>1)
      total+=static_cast<double>(count-1)*cost.TaskInstanceNs(producer,traits,residency,
          model,chunks,point,active_ctas_per_sm);
  }
  return total;
}

int FusionCtasPerSm(FusionResources const& r, TargetSpec const& target,
                    int register_allocation_per_warp, int static_shared_bytes) {
  auto const& res=target.res;
  if (r.threads<=0 || r.registers<=0 || r.shared_bytes<0 ||
      static_shared_bytes<0 || register_allocation_per_warp<=0 ||
      res.warp_size<=0 || r.threads%res.warp_size!=0 ||
      res.regs_per_sm<=0 || res.max_threads_per_sm<=0 ||
      res.max_smem_per_sm<=0 || res.max_dynamic_smem_per_cta<=0)
    throw std::invalid_argument("fusion residency requires complete resource budgets");
  if (r.shared_bytes>res.max_dynamic_smem_per_cta) return 0;
  long long warp_regs=static_cast<long long>(r.registers)*res.warp_size;
  long long allocated=((warp_regs+register_allocation_per_warp-1)/
      register_allocation_per_warp)*register_allocation_per_warp;
  long long cta_regs=allocated*(r.threads/res.warp_size);
  long long shared=static_cast<long long>(r.shared_bytes)+static_shared_bytes;
  long long limit=std::min(res.regs_per_sm/cta_regs,
      static_cast<long long>(res.max_threads_per_sm/r.threads));
  if (shared) limit=std::min(limit,res.max_smem_per_sm/shared);
  return static_cast<int>(limit);
}

FusionResources DeriveFusionResources(analysis::FusionAccesses const& accesses,
    analysis::ParamBinding const& theta, std::map<std::string,int> const& element_bytes,
    BackendTraits const& producer, int producer_registers,
    BackendTraits const& consumer, int consumer_registers) {
  analysis::IslReferenceAudit audit(__func__);
  if (producer.threads<=0 || producer.threads!=consumer.threads ||
      producer.smem_bytes<0 || consumer.smem_bytes<0 ||
      producer_registers<=0 || consumer_registers<=0)
    throw std::invalid_argument("fusion requires compatible threads and measured resources");
  std::map<std::vector<long>,long> live;
  for (auto const& [tensor,relation]:accesses.intermediate_tiles) {
    auto bytes=element_bytes.find(tensor);
    if (bytes==element_bytes.end() || bytes->second<=0)
      throw std::invalid_argument("fusion intermediate dtype is missing");
    // Concrete tier-3 resource query; each unique element comes from the
    // exact access set. It does not sample theta or fit a symbolic footprint.
    for (auto const& [task,element]:relation.BindParams(theta).Points()) {
      auto& total=live[task];
      if (total>std::numeric_limits<int>::max()-bytes->second)
        throw std::overflow_error("fusion intermediate exceeds resource representation");
      total+=bytes->second;
    }
  }
  long peak=0;
  for (auto const& [task,bytes]:live) peak=std::max(peak,bytes);
  long scratch=std::max(producer.smem_bytes,consumer.smem_bytes);
  if (peak>std::numeric_limits<int>::max()-scratch)
    throw std::overflow_error("fusion shared resource overflow");
  return {peak,static_cast<int>(scratch+peak),
          std::max(producer_registers,consumer_registers),producer.threads};
}
}  // namespace tilemega::solver
