// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskElementRelation.h>
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
  std::map<std::string,CouplingRelation> read_envelopes,write_envelopes,expected_envelopes;
  std::set<std::string> affine_readers;
  std::map<std::string,std::set<std::string>> binding_writers;
  std::set<std::string> active_rows_used;
  for(auto const& source:semantics.ops) {
    auto const* task=graph.Find(source.name);if(!task)throw std::invalid_argument("missing semantic task "+source.name);
    auto const& op=task->element_access?task->element_access->semantic:source;
    auto dependent=[](IndexingMap const& map) {
      for(auto const& index:map.results)
        if(index.kind==IndexResult::Kind::kDataDependent)return true;
      return false;
    };
    auto witness=[&](TensorSpace const& space,CouplingRelation const& actual) {
      auto image=actual.BindParams(fixed);
      if(!image.DomainDimNames().empty() || image.RangeDimNames().size()!=space.axes.size())
        throw std::invalid_argument("DramFloor runtime image violates its physical envelope: "+space.name);
      return image;
    };
    auto store=[&](TensorSpace const& space,IndexingMap const& map,
                   std::vector<IndexResult> const& predicates) {
      if(auto contract=options.binding_internal.find(space.name);contract!=options.binding_internal.end()) {
        if(!task->element_access || options.outputs.count(space.name) ||
            !contract->second.writers.count(op.name) || contract->second.source.empty())
          throw std::invalid_argument("DramFloor invalid binding producer contract: "+space.name);
        auto image=ProjectTaskWrite(op,*task,task->element_access->partition,
            space,map,predicates,fixed).Image();
        if(!image.IsSubset(contract->second.envelope.BindParams(fixed)))
          throw std::invalid_argument("DramFloor binding store exceeds capacity: "+space.name);
        tensor(space.name,op.dtype).binding_producer=contract->second;
        binding_writers[space.name].insert(op.name);
        return CouplingRelation{};
      }
      if(dependent(map)) {
        auto actual=options.indirect_write_images.find(space.name);
        if(!task->element_access || actual==options.indirect_write_images.end())
          throw std::invalid_argument("DramFloor needs the physical runtime write image for "+space.name);
        auto envelope=ProjectTaskWrite(op,*task,task->element_access->partition,
            space,map,predicates,fixed);
        write_envelopes[space.name]=write_envelopes[space.name].Union(envelope.Image());
        return witness(space,actual->second);
      }
      if(task->element_access)
        return ProjectTaskElements(op,*task,task->element_access->partition,
            space,map,predicates,fixed).Image();
      ElementRead indexed{space,map,predicates};
      return ExactElementRead(op,*task,indexed,fixed).Image();
    };
    auto& dst=tensor(op.result.name,op.dtype);
    auto writes = task->element_access || dependent(op.result_map)
        ? store(op.result,op.result_map,{})
        : ElementAccess(*task,BuildWriteMap(*task),fixed,AccessDomain::kPhysicalTensor).Image();
    dst.writes=dst.writes.Union(writes);dst.state|=!op.result_effect.state_object.empty();
    for (auto const& write:op.additional_writes) {
      auto& extra=tensor(write.tensor.name,op.dtype);
      extra.writes=extra.writes.Union(store(write.tensor,write.map,write.nonnegative));
      extra.state|=!write.effect.state_object.empty();
    }
    for(auto const& operand:op.operands)consumers.insert(operand.tensor.name);
    auto append=[&](std::string const& name,CouplingRelation const& relation){
      auto& src=tensor(name,op.dtype);src.reads=src.reads.Union(relation.Image());
    };
    std::set<std::string> indirect;
    std::map<std::string,CouplingRelation> indirect_envelopes;
    auto indirect_read=[&](TensorSpace const& space,IndexingMap const& map,
                           std::vector<IndexResult> const& predicates) {
      if(options.binding_internal.count(space.name)) {
        auto const& contract=options.binding_internal.at(space.name);
        if(!task->element_access || contract.source.empty())
          throw std::invalid_argument("DramFloor binding read lacks exact task ownership: "+space.name);
        auto image=ProjectTaskRead(op,*task,task->element_access->partition,
            space,map,predicates,fixed).Image();
        if(!image.IsSubset(contract.envelope.BindParams(fixed)))
          throw std::invalid_argument("DramFloor binding read exceeds capacity: "+space.name);
        auto producer=std::find_if(op.operands.begin(),op.operands.end(),[&](auto const& input) {
          return input.tensor.name==space.name && contract.writers.count(input.producer);
        });
        if(producer==op.operands.end())
          throw std::invalid_argument("DramFloor binding read has no declared producer: "+space.name);
        tensor(space.name,op.dtype).binding_producer=contract;
        indirect.insert(space.name);
        return;
      }
      if(!dependent(map)){affine_readers.insert(space.name);return;}
      indirect.insert(space.name);
      if(task->element_access)
        indirect_envelopes[space.name]=indirect_envelopes[space.name].Union(
            ProjectTaskRead(op,*task,task->element_access->partition,space,map,predicates,fixed));
    };
    if(op.element_reads.empty())for(auto const& operand:op.operands)
      indirect_read(operand.tensor,operand.map,{});
    else for(auto const& read:op.element_reads)
      indirect_read(read.tensor,read.map,read.nonnegative);
    for(auto const& name:indirect) {
      if(options.binding_internal.count(name))continue;
      auto expected=options.expected_indirect_reads.find(name);
      if(expected!=options.expected_indirect_reads.end()) {
        if(options.indirect_read_images.count(name) || expected->second.source.empty() ||
           (expected->second.kind!="expectation" && expected->second.kind!="lower_bound") ||
           !task->element_access || !indirect_envelopes.count(name))
          throw std::invalid_argument("DramFloor expectation lacks a unique bounded source: "+name);
        expected_envelopes[name]=expected_envelopes[name].Union(indirect_envelopes.at(name).Image());
        auto& src=tensor(name,op.dtype);
        src.expected_read_elements=expected->second.elements;
        src.expectation_source=expected->second.source;
        src.cardinality_kind=expected->second.kind;
        continue;
      }
      auto actual=options.indirect_read_images.find(name);
      if(actual==options.indirect_read_images.end())throw std::invalid_argument("DramFloor needs the physical runtime read image for "+name);
      auto image=actual->second;
      if(auto envelope=indirect_envelopes.find(name);envelope!=indirect_envelopes.end()) {
        TensorSpace space;space.name=name;
        space.axes.resize(envelope->second.RangeDimNames().size());
        image=witness(space,image);
        read_envelopes[name]=read_envelopes[name].Union(envelope->second.Image());
      }
      append(name,image);
    }
    if(!op.element_reads.empty()) {
      for(auto const& read:op.element_reads)if(!indirect.count(read.tensor.name)) {
        affine_readers.insert(read.tensor.name);
        append(read.tensor.name,ExactElementRead(op,*task,read,fixed));
      }
    } else {
      for(std::size_t i=0;i<task->operands.size();++i)if(!indirect.count(task->operands[i].tensor.name)) {
        affine_readers.insert(task->operands[i].tensor.name);
        append(task->operands[i].tensor.name, task->element_access
            ? ProjectTaskRead(op, *task, task->element_access->partition, op.operands[i].tensor, op.operands[i].map, {}, fixed)
            : ElementAccess(*task,BuildReadMap(*task,i),fixed,AccessDomain::kPhysicalTensor));
      }
    }
    if(op.kind==OperatorKind::kMatmul) {
      // The interleaved gate/up epilogue computes two independent dots for
      // each logical activation element.  Its result has half the packed
      // projection width, so multiply the physical MMA work by two here.
      ClosedForm work=ClosedForm::Constant(
          op.arithmetic=="swiglu_gemm" ? 4 : 2);
      if(auto rows=options.active_runtime_rows.find(op.name);rows!=options.active_runtime_rows.end()) {
        active_rows_used.insert(op.name);
        if(op.domain.size()!=4 || !op.domain[0].runtime || !op.domain[0].capacity ||
            op.domain[1].runtime || !op.exact_task_access)
          throw std::invalid_argument("DramFloor active row contract lacks virtual ownership");
        for(unsigned axis:{2u,3u})work=work*op.domain[axis].extent;
        auto active=rows->second;
        auto capacity=Polynomial(op.domain[0].BoundExtent()*op.domain[1].BoundExtent(),fixed);
        if(active.CompareScalar(QuasiPolynomial::Constant(0),fixed)<0 ||
            active.CompareScalar(capacity,fixed)>0)
          throw std::invalid_argument("DramFloor active row count exceeds virtual capacity");
        result.matmul_flops=result.matmul_flops.Add(active.Multiply(Polynomial(work,fixed)));
      } else {
        for(auto const& axis:op.domain)work=work*axis.extent;
        result.matmul_flops=result.matmul_flops.Add(Polynomial(work,fixed));
      }
    }
  }
  auto validate_images=[&](auto const& supplied,auto const& envelopes) {
    for(auto const& [name,envelope]:envelopes)
      if(!supplied.at(name).BindParams(fixed).IsSubset(envelope))
        throw std::invalid_argument("DramFloor runtime image violates its physical envelope: "+name);
  };
  validate_images(options.indirect_read_images,read_envelopes);
  validate_images(options.indirect_write_images,write_envelopes);
  if(active_rows_used.size()!=options.active_runtime_rows.size())
    throw std::invalid_argument("DramFloor unused active runtime row contract");
  for(auto const& [name,contract]:options.binding_internal) {
    auto found=result.tensors.find(name);
    if(found==result.tensors.end() || !found->second.binding_producer ||
        found->second.state || options.outputs.count(name) ||
        binding_writers[name]!=contract.writers || contract.writers.empty() ||
        options.indirect_read_images.count(name) || options.indirect_write_images.count(name) ||
        options.expected_indirect_reads.count(name))
      throw std::invalid_argument("DramFloor incomplete internal binding contract: "+name);
  }
  for(auto const& [name,expected]:options.expected_indirect_reads) {
    auto footprint=result.tensors.find(name);
    auto envelope=expected_envelopes.find(name);
    if(footprint==result.tensors.end() || envelope==expected_envelopes.end() ||
       !footprint->second.writes.empty() || footprint->second.state || affine_readers.count(name))
      throw std::invalid_argument("DramFloor expectation requires an indirect-only external tensor: "+name);
    if(expected.elements.CompareScalar(QuasiPolynomial::Constant(0),fixed)<0 ||
       expected.elements.CompareScalar(Cardinality(envelope->second),fixed)>0)
      throw std::invalid_argument("DramFloor expectation exceeds its access envelope: "+name);
  }
  std::vector<QuasiPolynomial> reads,writes;
  for(auto& [name,t]:result.tensors) {
    if(t.binding_producer) {
      t.read_bytes=t.write_bytes=QuasiPolynomial::Constant(0);
      reads.push_back(t.read_bytes);writes.push_back(t.write_bytes);continue;
    }
    t.no_producer=t.reads.Subtract(t.writes);
    t.output=options.outputs.count(name) || (options.infer_leaf_outputs && !t.writes.empty() && !consumers.count(name));
    if(t.state || t.output)t.external_writes=t.writes;
    t.read_bytes=(t.expected_read_elements?*t.expected_read_elements:Cardinality(t.no_producer)).Scale(t.element_bytes);
    t.write_bytes=Cardinality(t.external_writes).Scale(t.element_bytes);
    reads.push_back(t.read_bytes);writes.push_back(t.write_bytes);
  }
  result.no_producer_bytes=QuasiPolynomial::Sum(reads);result.output_state_bytes=QuasiPolynomial::Sum(writes);
  result.dram_ns=result.no_producer_bytes.Add(result.output_state_bytes).ScaleRational(Reciprocal(options.dram_gbps));
  result.compute_ns=result.matmul_flops.ScaleRational(Reciprocal(options.tc_gflops));
  return result;
}
DramFloor::Value DramFloor::Evaluate(ParamBinding const& theta) const {
  // Profile means may be fractional bytes; integer address cardinalities keep
  // their exact representation until this physical-rate evaluation.
  Value value;value.read_bytes=no_producer_bytes.EvalReal(theta);value.write_bytes=output_state_bytes.EvalReal(theta);
  value.flops=matmul_flops.Eval(theta);value.dram_ns=(value.read_bytes+value.write_bytes)/dram_rate;
  value.compute_ns=value.flops/compute_rate;value.floor_ns=std::max(value.dram_ns,value.compute_ns);return value;
}
} // namespace tilemega::analysis
