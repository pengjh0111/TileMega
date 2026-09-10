// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/ISLContext.h>
#include <set>
#include <map>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

namespace tilemega::analysis {
namespace {
std::string Join(std::vector<std::string> const& values, char const* separator) {
  std::string text;
  for (auto const& v: values) { if (!text.empty()) text+=separator; text+=v; }
  return text;
}
std::string Prefix(std::set<std::string> const& parameters) {
  if (parameters.empty()) return "";
  return "["+Join({parameters.begin(),parameters.end()},",")+"] -> ";
}
QuasiPolynomial Polynomial(ClosedForm const& value, ParamBinding const& known) {
  auto expression = value.Substitute(known);
  auto symbols = expression.FreeSymbols();
  return QuasiPolynomial::FromIslText(Prefix({symbols.begin(),symbols.end()})+
                                     "{ "+expression.ToIslText()+" }");
}
}  // namespace

CouplingRelation ExactElementRead(SemanticOp const& semantic, OperatorNode const& task,
                                 ElementRead const& read, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  bool split=task.output.axes.size()==semantic.result_map.results.size()+1 && semantic.reduction.splittable;
  if ((!split && task.output.axes.size()!=semantic.result_map.results.size()) ||
      read.tensor.axes.size()!=read.map.results.size())
    throw std::invalid_argument("exact element indexing rank mismatch or unprojected split");
  std::set<std::string> parameters;
  auto expression=[&](ClosedForm const& value) {
    auto bound=value.Substitute(known);
    for (auto const& symbol:bound.FreeSymbols()) parameters.insert(symbol);
    return bound.ToIslText();
  };
  std::map<std::string,std::string> variables;
  std::vector<std::string> bounds, existential, elements;
  for (std::size_t i=0;i<semantic.domain.size();++i) {
    auto const& dim=semantic.domain[i];
    std::string variable="iteration"+std::to_string(i);
    if (!variables.emplace(dim.name,variable).second)
      throw std::invalid_argument("duplicate semantic iteration axis");
    existential.push_back(variable);
    auto origin=expression(dim.origin),extent=expression(dim.extent);
    bounds.push_back("("+origin+") <= "+variable+" < ("+origin+")+("+extent+")");
  }
  auto index=[&](IndexResult const& result) {
    if (result.kind!=IndexResult::Kind::kAffine)
      throw std::invalid_argument("exact element read requires affine indexing");
    std::string value="("+expression(result.offset)+")";
    for (auto const& term:result.terms) {
      auto dim=variables.find(term.dim);
      if (dim==variables.end()) throw std::invalid_argument("unknown element indexing axis: "+term.dim);
      auto divisor=term.group.Eval(known,known);
      if (divisor<=0) throw std::invalid_argument("nonpositive element indexing divisor");
      long scale=term.coefficient.Eval(known,known);
      value+=" + "+std::to_string(scale)+"*floord("+dim->second+", "+std::to_string(divisor)+")";
    }
    return value;
  };
  auto writes=BuildWriteMap(task);
  for (std::size_t i=0;i<task.output.axes.size();++i) {
    if (task.IsTiled(i)) bounds.push_back("0 <= "+task.output.axes[i].name+" < ("+
                                         expression(task.CoordinateExtent(i))+")");
    if (i>=semantic.result_map.results.size()) continue;
    // The output indexing map is relative to the tensor's semantic origin.
    auto point="("+index(semantic.result_map.results[i])+")+("+
               expression(semantic.result.axes[i].origin)+")";
    auto const& interval=writes.index[i];
    for (auto const& symbol:interval.base.FreeSymbols()) if (!known.Contains(symbol)) parameters.insert(symbol);
    auto base=interval.base.ToIslText(known);
    bounds.push_back("("+base+") <= ("+point+") < ("+base+")+("+expression(interval.span)+")");
  }
  if (split) {
    bool found=false;
    if (semantic.operands.size()!=task.operands.size())
      throw std::invalid_argument("split element read requires matching semantic operands");
    for (std::size_t operand=0;operand<semantic.operands.size();++operand) {
      auto access=BuildReadMap(task,operand);
      auto const& map=semantic.operands[operand].map.results;
      if (map.size()!=access.index.size()) throw std::invalid_argument("split read indexing rank mismatch");
      for (std::size_t axis=0;axis<map.size();++axis) {
        auto const& index=map[axis];
        if (index.kind!=IndexResult::Kind::kAffine || index.terms.size()!=1 ||
            index.terms[0].dim!=semantic.reduction.dim) continue;
        if (!index.terms[0].coefficient.IsLiteral(1) || !index.terms[0].group.IsLiteral(1))
          throw std::invalid_argument("split reduction requires direct iteration indexing");
        auto const& interval=access.index[axis];
        for (auto const& symbol:interval.base.FreeSymbols()) if (!known.Contains(symbol)) parameters.insert(symbol);
        auto base=interval.base.ToIslText(known),span=expression(interval.span);
        auto variable=variables.at(semantic.reduction.dim);
        bounds.push_back("("+base+") <= "+variable+" < ("+base+")+("+span+")");
        found=true;
      }
    }
    if (!found) throw std::invalid_argument("split reduction has no indexed read span");
  }
  for (std::size_t i=0;i<read.map.results.size();++i) {
    auto element="element"+std::to_string(i); elements.push_back(element);
    bounds.push_back(element+" = "+index(read.map.results[i]));
    auto const& axis=read.tensor.axes[i];
    auto origin=expression(axis.origin),extent=expression(axis.extent);
    bounds.push_back("("+origin+") <= "+element+" < ("+origin+")+("+extent+")");
  }
  for (auto const& predicate:read.nonnegative) bounds.push_back("("+index(predicate)+") >= 0");
  std::string condition=Join(bounds," and ");
  if (!existential.empty()) condition="exists ("+Join(existential,",")+" : "+condition+")";
  return CouplingRelation::FromIslText(Prefix(parameters)+"{ ["+Join(task.Coordinates(),",")+
      "] -> ["+Join(elements,",")+"] : "+condition+" }");
}

CouplingRelation ElementAccess(OperatorNode const& task, AccessRelation const& access,
                               ParamBinding const& known, AccessDomain domain) {
  IslReferenceAudit audit(__func__);
  if (access.data_dependent) throw std::invalid_argument("uncountable tensor access: "+task.name);
  if (access.index.size()!=access.tensor.axes.size()) throw std::invalid_argument("access rank mismatch");
  std::set<std::string> parameters;
  auto collect = [&](std::vector<std::string> const& names) {
    for (auto const& name:names) if (!known.Contains(name)) parameters.insert(name);
  };
  std::vector<std::string> bounds, elements;
  for (std::size_t a=0;a<task.output.axes.size();++a) if (task.IsTiled(a)) {
    auto extent = task.CoordinateExtent(a).Substitute(known);
    collect(extent.FreeSymbols());
    bounds.push_back("0 <= "+task.output.axes[a].name+" < ("+extent.ToIslText()+")");
  }
  for (std::size_t a=0;a<access.index.size();++a) {
    std::string element = "element"+std::to_string(a);
    elements.push_back(element);
    auto const& interval = access.index[a];
    collect(interval.base.FreeSymbols()); collect(interval.span.FreeSymbols());
    std::string base = interval.base.ToIslText(known);
    bounds.push_back("("+base+") <= "+element+" < ("+base+")+("+
                     interval.span.Substitute(known).ToIslText()+")");
    if (domain==AccessDomain::kPhysicalTensor) {
      auto const& axis = access.tensor.axes[a];
      collect(axis.origin.FreeSymbols()); collect(axis.extent.FreeSymbols());
      auto origin = axis.origin.Substitute(known).ToIslText();
      bounds.push_back("("+origin+") <= "+element+" < ("+origin+")+("+
                       axis.extent.Substitute(known).ToIslText()+")");
    }
  }
  return CouplingRelation::FromIslText(Prefix(parameters)+"{ ["+
      Join(task.Coordinates(),",")+"] -> ["+Join(elements,",")+"]"+
      (bounds.empty() ? "" : " : "+Join(bounds," and "))+" }");
}

TaskWork DeriveTaskWork(SemanticOp const& semantic, OperatorNode const& task,
                       ParamBinding const& known, TaskWorkOptions const& options) {
  IslReferenceAudit audit(__func__);
#if defined(TILEMEGA_TASK_WORK) && !TILEMEGA_TASK_WORK
  throw std::runtime_error("access-derived TaskWork is disabled");
#endif
  std::set<std::string> output_axes;
  for (auto const& result:semantic.result_map.results) {
    if (result.kind!=IndexResult::Kind::kAffine)
      throw std::invalid_argument("non-affine output indexing cannot identify reduction axes");
    for (auto const& term:result.terms) if (!term.coefficient.IsLiteral(0)) output_axes.insert(term.dim);
  }
  for (auto const& [axis,tile]:options.reduction_tiles) {
    if (!semantic.Dim(axis) || output_axes.count(axis))
      throw std::invalid_argument("inner tile does not name an output-absent iteration axis");
    if (tile.Eval(known,known)<=0)
      throw std::invalid_argument("inner reduction tile must be a positive concrete extent");
  }
  TaskWork work;
  auto write = BuildWriteMap(task);
  auto physical = ElementAccess(task,write,known,AccessDomain::kPhysicalTensor);
  work.task_count = physical.Reverse().ImageCard();
  work.write_elements = physical.Card();
  work.nominal_write_elements = ElementAccess(task,write,known,AccessDomain::kNominalTile).Card();
  std::map<std::string, std::pair<CouplingRelation, CouplingRelation>> tensor_reads;
  std::map<std::string, std::string> layouts;
  for (std::size_t i=0;i<task.operands.size();++i) {
    auto read = BuildReadMap(task,i);
    auto nominal_read=read;
    if (!options.reduction_tiles.empty()) {
      if (semantic.operands.size()!=task.operands.size() ||
          semantic.operands[i].map.results.size()!=read.index.size())
        throw std::invalid_argument("inner tile requires matching semantic indexing");
      for (std::size_t axis=0;axis<read.index.size();++axis) {
        auto const& index=semantic.operands[i].map.results[axis];
        for (auto const& term:index.terms) {
          auto tile=options.reduction_tiles.find(term.dim);
          if (tile==options.reduction_tiles.end()) continue;
          if (index.kind!=IndexResult::Kind::kAffine || index.terms.size()!=1 ||
              !term.coefficient.IsLiteral(1) || !term.group.IsLiteral(1))
            throw std::invalid_argument("inner tile requires an exact unit indexing axis");
          auto extent=tile->second.Substitute(known);
          nominal_read.index[axis].span=read.index[axis].span.CeilDiv(extent)*extent;
        }
      }
    }
    if (read.tensor.name.empty()) throw std::invalid_argument("read tensor identity missing");
    auto [layout, inserted] = layouts.emplace(read.tensor.name, read.tensor.layout_id);
    if (!inserted && layout->second != read.tensor.layout_id)
      throw std::invalid_argument("read union requires a common tensor layout");
    auto& relations = tensor_reads[read.tensor.name];
    relations.first = relations.first.Union(
        ElementAccess(task,read,known,AccessDomain::kPhysicalTensor));
    relations.second = relations.second.Union(
        ElementAccess(task,nominal_read,known,AccessDomain::kNominalTile));
    if (std::getenv("TILEMEGA_TASK_WORK_TRACE"))
      std::cerr << "READ " << i << " physical=" << relations.first.ToString()
                << " nominal=" << relations.second.ToString() << '\n';
  }
  // Tensor identity tags disjoint address spaces; repeated reads of one tensor
  // are a set union, not a second copy of its physical footprint.
  std::vector<QuasiPolynomial> reads, nominal_reads;
  for (auto const& [name, relations] : tensor_reads) {
    reads.push_back(relations.first.Card());
    nominal_reads.push_back(relations.second.Card());
  }
  work.read_elements = QuasiPolynomial::Sum(reads);
  work.nominal_read_elements = QuasiPolynomial::Sum(nominal_reads);
  if (TILEMEGA_EXACT_ELEMENT_WORK && !semantic.element_reads.empty()) {
    std::map<std::string,CouplingRelation> exact;
    std::map<std::string,std::string> exact_layouts;
    for (auto const& read:semantic.element_reads) {
      if (read.tensor.name.empty()) throw std::invalid_argument("exact read tensor identity missing");
      auto [layout,inserted]=exact_layouts.emplace(read.tensor.name,read.tensor.layout_id);
      if (!inserted && layout->second!=read.tensor.layout_id)
        throw std::invalid_argument("exact read union has conflicting layouts");
      exact[read.tensor.name]=exact[read.tensor.name].Union(ExactElementRead(semantic,task,read,known));
    }
    std::vector<QuasiPolynomial> counts;
    for (auto const& [name,relation]:exact) counts.push_back(relation.Card());
    work.read_elements=QuasiPolynomial::Sum(counts);
  }
  ClosedForm reduce=ClosedForm::Constant(1), parallel=ClosedForm::Constant(1);
  ClosedForm local_reduce=ClosedForm::Constant(1);
  ClosedForm nominal_reduce=ClosedForm::Constant(1);
  for (auto const& dim:semantic.domain) {
    if (output_axes.count(dim.name)) parallel=parallel*dim.extent;
    else {
      reduce=reduce*dim.extent;
      if (semantic.operands.size()!=task.operands.size())
        throw std::invalid_argument("local reduction requires matching semantic operands");
      bool found=false;
      ClosedForm local;
      for (std::size_t operand=0;operand<semantic.operands.size();++operand) {
        auto access=BuildReadMap(task,operand);
        auto const& indices=semantic.operands[operand].map.results;
        if (indices.size()!=access.index.size())
          throw std::invalid_argument("local reduction indexing rank mismatch");
        for (std::size_t axis=0;axis<indices.size();++axis) {
          auto const& index=indices[axis];
          for (auto const& term:index.terms) if (term.dim==dim.name) {
            if (index.kind!=IndexResult::Kind::kAffine || index.terms.size()!=1 ||
                !term.coefficient.IsLiteral(1) || !term.group.IsLiteral(1))
              throw std::invalid_argument("local reduction requires an exact unit indexing axis");
            auto span=access.index[axis].span.Substitute(known);
            if (found && span.ToString()!=local.ToString())
              throw std::invalid_argument("reduction operands disagree on the local span");
            local=span; found=true;
          }
        }
      }
      if (!found) throw std::invalid_argument("reduction axis has no indexed read: "+dim.name);
      local_reduce=local_reduce*local;
      auto tile=options.reduction_tiles.find(dim.name);
      if (tile!=options.reduction_tiles.end()) {
        auto extent=tile->second.Substitute(known);
        local=local.CeilDiv(extent)*extent;
      }
      nominal_reduce=nominal_reduce*local;
    }
  }
  work.reduce_extent=Polynomial(reduce,known);
  work.parallel_extent=Polynomial(parallel,known);
  work.task_reduce_extent=Polynomial(local_reduce,known);
  work.nominal_task_reduce_extent=Polynomial(nominal_reduce,known);
  return work;
}
}  // namespace tilemega::analysis
