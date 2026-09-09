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
                       ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
#if defined(TILEMEGA_TASK_WORK) && !TILEMEGA_TASK_WORK
  throw std::runtime_error("access-derived TaskWork is disabled");
#endif
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
    if (read.tensor.name.empty()) throw std::invalid_argument("read tensor identity missing");
    auto [layout, inserted] = layouts.emplace(read.tensor.name, read.tensor.layout_id);
    if (!inserted && layout->second != read.tensor.layout_id)
      throw std::invalid_argument("read union requires a common tensor layout");
    auto& relations = tensor_reads[read.tensor.name];
    relations.first = relations.first.Union(
        ElementAccess(task,read,known,AccessDomain::kPhysicalTensor));
    relations.second = relations.second.Union(
        ElementAccess(task,read,known,AccessDomain::kNominalTile));
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
  std::set<std::string> output_axes;
  for (auto const& result:semantic.result_map.results) {
    if (result.kind!=IndexResult::Kind::kAffine)
      throw std::invalid_argument("non-affine output indexing cannot identify reduction axes");
    for (auto const& term:result.terms) if (!term.coefficient.IsLiteral(0)) output_axes.insert(term.dim);
  }
  ClosedForm reduce=ClosedForm::Constant(1), parallel=ClosedForm::Constant(1);
  for (auto const& dim:semantic.domain) {
    if (output_axes.count(dim.name)) parallel=parallel*dim.extent;
    else reduce=reduce*dim.extent;
  }
  work.reduce_extent=Polynomial(reduce,known);
  work.parallel_extent=Polynomial(parallel,known);
  return work;
}
}  // namespace tilemega::analysis
