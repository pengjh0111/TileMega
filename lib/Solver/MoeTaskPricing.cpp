// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeTaskPricing.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace tilemega::solver {
MoeTaskPrice PriceMoeVirtualTask(CostModel const& cost,DerivedTaskInput const& input,
    ModelTaskSemantics const& semantic,BackendTraits const& traits,Residency residency,
    ModelDescription const& model,int chunks,analysis::DramFloor const& floor,
    MoeRoutingPoint const& routing,bool token_slot,
    analysis::ParamBinding const& coordinate) {
  using namespace analysis;
  if(!model.dm || model.dtype!=ScalarType::kBF16 || !cost.options().regime_a || !cost.options().physical_traffic ||
     !input.task.element_access || !routing.windows || traits.stages<2 ||
     residency.ctas_per_sm<1)
    throw std::invalid_argument("MoE expectation needs physical BF16 regime-A work");
  auto theta=model.MetricBindings();auto const& op=input.task.element_access->semantic;
  if(op.task_space.axes.size()!=3 || input.task.output.axes.size()<3)
    throw std::invalid_argument("MoE expectation needs virtual/row/column task axes");
  auto const* vd=op.Dim(UnitTaskOwnershipDimension(op,0));
  auto const* rd=op.Dim(UnitTaskOwnershipDimension(op,1));
  if(!vd || !vd->runtime || !vd->capacity || !rd || rd->runtime ||
     !rd->origin.IsLiteral(0))
    throw std::invalid_argument("MoE expectation lacks static row bounds");
  auto bm=rd->BoundExtent().Eval(theta,{});
  auto axis=[&](unsigned a) {return input.task.IsTiled(a)?coordinate.At(input.task.output.axes[a].name):0;};
  auto v=axis(0),row=axis(1),row_tile=input.task.tile[1].Eval(theta,{});
  if(bm<=0 || bm>UINT32_MAX || row_tile<=0 || row<0 || row>(bm-1)/row_tile ||
     semantic.stage<0 || semantic.stage>=int(model.stages.size()))
    throw std::invalid_argument("MoE expectation has invalid row ownership");
  auto gemm=model.stages.at(semantic.stage).gemm;
  if(gemm<0 || gemm>=int(model.gemm_access.size()))
    throw std::invalid_argument("MoE expectation lacks a GEMM access descriptor");
  auto const& access=model.gemm_access.at(gemm);
  if(access.b!=codegen::DmBAccess::kExpertIndirect || access.block_rows!=bm ||
     access.experts!=routing.experts || model.dims.seq<=0 || model.dims.batch<=0 ||
     std::uint64_t(model.dims.seq)*model.dims.batch!=routing.tokens)
    throw std::invalid_argument("MoE expectation differs from the model routing coordinate");
  auto capacity=token_slot?routing.SlotCapacity():routing.GroupCapacity(bm);
  if((token_slot && bm!=1) || v<0 ||
     std::uint64_t(v)>=capacity || vd->BoundExtent().Eval(theta,{})!=long(capacity))
    throw std::invalid_argument("MoE expectation differs from virtual capacity");
  std::map<std::uint32_t,std::uint64_t> histogram=token_slot?
      std::map<std::uint32_t,std::uint64_t>{{1,routing.windows}}:
      routing.VirtualRowHistogram(bm,v);
  auto const& cal=cost.target().CalibrationFor("bf16");
  double fair=cal.dram_gbps/(cost.target().res.num_sms*residency.ctas_per_sm);
  MoeTaskPrice result;long double dram_service=0;std::uint64_t active=0;
  for(auto const& [rows,count]:histogram) {
    if(!rows || rows>std::min<long>(bm,routing.tokens) || !count || count>routing.windows-active)
      throw std::invalid_argument("invalid binding occupancy histogram");
    active+=count;
  }
  auto add=[&](TaskPriceParts const& parts,double probability) {
    result.parts.fixed_ns+=probability*parts.fixed_ns;
    result.parts.compute_ns+=probability*parts.compute_ns;
    result.parts.dram_bytes+=probability*parts.dram_bytes;
    result.parts.no_producer_dram_bytes+=probability*parts.no_producer_dram_bytes;
    result.parts.inflight_bytes+=probability*parts.inflight_bytes;
    if(parts.dram_bytes>0) {
      if(!(parts.dram_rate_cap>0))throw std::invalid_argument("profiled DRAM work lacks a service cap");
      dram_service+=probability*parts.dram_bytes/parts.dram_rate_cap;
    }
    result.expected_isolated_ns+=probability*IsolatedNs(parts,fair);
  };
  std::uint64_t empty=routing.windows-active;
  for(auto const& [rows,count]:histogram) {
    auto live=std::max<long>(0,std::min<long>(row_tile,long(rows)-row*row_tile));
    if(!live){empty+=count;continue;}
    auto conditioned=RestrictVirtualTaskRows(input,rows,traits,theta);
    BindTaskDramProvenance(conditioned,semantic,floor,theta,model.serving);
    auto parts=cost.PriceParts(conditioned,traits,residency,model,chunks,
        coordinate,residency.ctas_per_sm);
    auto probability=double(count)/routing.windows;
    add(parts,probability);result.active_probability+=probability;
    result.expected_rows+=probability*live;
  }
  if(empty) {
    auto fit=cal.task_body.serving.find(input.serving_body_kind+"_empty");
    if(input.serving_body_kind.empty() || fit==cal.task_body.serving.end() ||
       fit->second.samples<=0 || fit->second.fixed_ns<0 ||
       !std::isfinite(fit->second.fixed_ns))
      throw std::invalid_argument("empty MoE task body has no calibrated binding-check cost");
    TaskPriceParts parts;parts.fixed_ns=fit->second.fixed_ns;
    add(parts,double(empty)/routing.windows);
  }
  // Service time is averaged before recovering a rate. Averaging a zero cap
  // for empty tasks would incorrectly cancel the occupancy traffic reduction.
  result.parts.dram_rate_cap=dram_service>0?result.parts.dram_bytes/double(dram_service):0;
  return result;
}
} // namespace tilemega::solver
