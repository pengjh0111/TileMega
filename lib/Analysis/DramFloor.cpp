// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include "IslUtil.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
QuasiPolynomial Cardinality(CouplingRelation const& image) {
  if(image.empty())return QuasiPolynomial::Constant(0);
  auto map=isl_util::ReadMap(SharedIslContext().raw(),image.ToString());
  isl_util::Set set(isl_map_range(map.release()));
  isl_util::PwQPolynomial count(isl_set_card(set.release()));
  if(!count)throw std::runtime_error("DramFloor: uncountable access image");
  return QuasiPolynomial::FromIslText(isl_util::ToString(count.get()));
}
std::string Reciprocal(double rate) {
  if(!(rate>0) || !std::isfinite(rate))throw std::invalid_argument("DramFloor requires calibrated positive rates");
  // Decimal rates are represented as exact rationals by ISL.
  std::ostringstream text;text<<std::fixed<<std::setprecision(9)<<rate;
  auto s=text.str();auto decimal=s.find('.');s.erase(decimal,1);
  return "1000000000/"+s;
}
QuasiPolynomial Polynomial(ClosedForm const& value,ParamBinding const& fixed) {
  auto v=value.Substitute(fixed);auto parameters=v.FreeSymbols();std::string prefix;
  for(auto const& p:parameters){if(!prefix.empty())prefix+=",";prefix+=p;}
  return QuasiPolynomial::FromIslText((prefix.empty()?"":"["+prefix+"] -> ")+"{ "+v.ToIslText()+" }");
}
}
DramFloor DeriveDramFloor(SemanticGraph const& semantics,DramFloorOptions const& options,
                          ParamBinding const& fixed) {
  IslReferenceAudit audit(__func__);DramFloor result;
  result.dram_rate=options.dram_gbps;result.compute_rate=options.tc_gflops;
  auto graph=Instantiate(semantics,{});std::set<std::string> consumers;
  auto tensor=[&](std::string const& name,ScalarType dtype)->DramTensorFootprint& {
    auto& t=result.tensors[name];auto supplied=options.element_bytes.find(name);
    int bytes=supplied==options.element_bytes.end()?(dtype==ScalarType::kBF16?2:4):supplied->second;
    if(bytes<=0 || (t.element_bytes && t.element_bytes!=bytes))throw std::invalid_argument("inconsistent tensor storage width: "+name);
    t.element_bytes=bytes;return t;
  };
  for(auto const& op:semantics.ops) {
    auto const* task=graph.Find(op.name);if(!task)throw std::invalid_argument("missing semantic task "+op.name);
    auto& dst=tensor(op.result.name,op.dtype);
    auto writes=ElementAccess(*task,BuildWriteMap(*task),fixed,AccessDomain::kPhysicalTensor).Image();
    dst.writes=dst.writes.Union(writes);dst.state|=!op.result_effect.state_object.empty();
    for(auto const& operand:op.operands)consumers.insert(operand.tensor.name);
    auto append=[&](std::string const& name,CouplingRelation const& relation){
      auto& src=tensor(name,op.dtype);src.reads=src.reads.Union(relation.Image());
    };
    std::set<std::string> indirect;
    for(auto const& operand:op.operands)for(auto const& index:operand.map.results)
      if(index.kind==IndexResult::Kind::kDataDependent)indirect.insert(operand.tensor.name);
    for(auto const& name:indirect) {
      auto actual=options.indirect_read_images.find(name);
      if(actual==options.indirect_read_images.end())throw std::invalid_argument("DramFloor needs the physical runtime read image for "+name);
      append(name,actual->second);
    }
    if(!op.element_reads.empty()) {
      for(auto const& read:op.element_reads)if(!indirect.count(read.tensor.name))
        append(read.tensor.name,ExactElementRead(op,*task,read,fixed));
    } else {
      for(std::size_t i=0;i<task->operands.size();++i)if(!indirect.count(task->operands[i].tensor.name))
        append(task->operands[i].tensor.name,ElementAccess(*task,BuildReadMap(*task,i),fixed,AccessDomain::kPhysicalTensor));
    }
    if(op.kind==OperatorKind::kMatmul) {
      ClosedForm work=ClosedForm::Constant(2);
      for(auto const& axis:op.domain)work=work*axis.extent;
      result.matmul_flops=result.matmul_flops.Add(Polynomial(work,fixed));
    }
  }
  std::vector<QuasiPolynomial> reads,writes;
  for(auto& [name,t]:result.tensors) {
    t.no_producer=t.reads.Subtract(t.writes);
    t.output=options.outputs.count(name) || (!t.writes.empty() && !consumers.count(name));
    if(t.state || t.output)t.external_writes=t.writes;
    t.read_bytes=Cardinality(t.no_producer).Scale(t.element_bytes);
    t.write_bytes=Cardinality(t.external_writes).Scale(t.element_bytes);
    reads.push_back(t.read_bytes);writes.push_back(t.write_bytes);
  }
  result.no_producer_bytes=QuasiPolynomial::Sum(reads);result.output_state_bytes=QuasiPolynomial::Sum(writes);
  result.dram_ns=result.no_producer_bytes.Add(result.output_state_bytes).ScaleRational(Reciprocal(options.dram_gbps));
  result.compute_ns=result.matmul_flops.ScaleRational(Reciprocal(options.tc_gflops));
  return result;
}
DramFloor::Value DramFloor::Evaluate(ParamBinding const& theta) const {
  // QuasiPolynomial::Eval is integral; retain fractional ns by evaluating
  // integral work first, then dividing by the calibrated physical rate.
  Value value;value.read_bytes=no_producer_bytes.Eval(theta);value.write_bytes=output_state_bytes.Eval(theta);
  value.flops=matmul_flops.Eval(theta);value.dram_ns=(value.read_bytes+value.write_bytes)/dram_rate;
  value.compute_ns=value.flops/compute_rate;value.floor_ns=std::max(value.dram_ns,value.compute_ns);return value;
}
} // namespace tilemega::analysis
