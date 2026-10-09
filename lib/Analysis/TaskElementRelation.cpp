// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/VirtualTaskBinding.h>
#include <tilemega/Analysis/ISLContext.h>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tilemega::analysis {
namespace {
std::string Join(std::vector<std::string> const& values, char const* separator) {
  std::string out;
  for (auto const& value : values) { if (!out.empty()) out += separator; out += value; }
  return out;
}
}

CouplingRelation ProjectTaskElements(SemanticOp const& semantic,
    OperatorNode const& task, TaskElementPartition const& partition,
    TensorSpace const& tensor, IndexingMap const& indexing,
    std::vector<IndexResult> const& nonnegative, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  for (auto const& binding : VirtualBindings(semantic)) {
    auto capacity = binding.capacity.Substitute(known);
    if (capacity.IsConstant() && capacity.Eval({}, {}) <= 0)
      throw std::invalid_argument("virtual capacity must be positive after binding");
  }
  bool split = !partition.reduction_chunk.IsLiteral(0);
  if (indexing.results.size() != tensor.axes.size() ||
      task.output.axes.size() != partition.ownership.results.size() + unsigned(split))
    throw std::invalid_argument("task element partition rank mismatch");
  std::set<std::string> parameters;
  std::vector<std::string> bounds, variables, elements;
  std::map<std::string, std::string> iteration;
  auto expression = [&](ClosedForm const& value) {
    auto bound = value.Substitute(known);
    for (auto const& symbol : bound.FreeSymbols()) parameters.insert(symbol);
    return bound.ToIslText();
  };
  for (unsigned i = 0; i < semantic.domain.size(); ++i) {
    auto const& dim = semantic.domain[i];
    auto variable = "_tm_i" + std::to_string(i);
    if (!iteration.emplace(dim.name, variable).second)
      throw std::invalid_argument("duplicate task element iteration dimension");
    variables.push_back(variable);
    auto origin = expression(dim.origin);
    bounds.push_back("(" + origin + ") <= " + variable + " < (" + origin +
                     ") + (" + expression(dim.BoundExtent()) + ")");
  }
  auto index = [&](IndexResult const& value) {
    if (value.kind != IndexResult::Kind::kAffine)
      throw std::invalid_argument("task element projection requires affine indexing");
    std::string out = "(" + expression(value.offset) + ")";
    for (auto const& term : value.terms) {
      auto variable = iteration.find(term.dim);
      if (variable == iteration.end())
        throw std::invalid_argument("unknown task element indexing dimension: " + term.dim);
      auto group = term.group.Eval(known, known);
      if (group <= 0) throw std::invalid_argument("task element group must be positive");
      out += " + " + std::to_string(term.coefficient.Eval(known, known)) +
          " * floord(" + variable->second + " + (" + expression(term.shift) + "), " +
          std::to_string(group) + ")";
    }
    auto divisor=value.outer_divisor.Eval(known,known);
    if(divisor<=0)throw std::invalid_argument("task element outer divisor must be positive");
    return divisor==1?out:"floord(("+out+"), "+std::to_string(divisor)+")";
  };
  for (unsigned axis = 0; axis < task.output.axes.size(); ++axis) {
    auto const& shape = task.output.axes[axis];
    if (task.IsTiled(axis))
      bounds.push_back("0 <= " + shape.name + " < (" +
                       expression(task.CoordinateExtent(axis)) + ")");
    if (axis == partition.ownership.results.size()) continue;
    auto origin = expression(shape.origin);
    auto base = "(" + origin + ")";
    auto span = expression(shape.extent);
    if (task.IsTiled(axis)) {
      span = expression(task.tile[axis]);
      auto tile = task.tile[axis].Eval(known, known);
      if (tile <= 0) throw std::invalid_argument("task tile must be positive");
      base += " + " + std::to_string(tile) + " * " + shape.name;
    }
    auto point = "(" + index(partition.ownership.results[axis]) + ")";
    bounds.push_back("(" + base + ") <= " + point + " < (" + base + ") + (" + span + ")");
  }
  if (split) {
    auto const* dim = semantic.Dim(semantic.reduction.dim);
    if (!semantic.reduction.splittable || !dim ||
        !task.tile.back().IsLiteral(1) || partition.reduction_chunk.Eval(known, known) <= 0)
      throw std::invalid_argument("invalid task element reduction partition");
    auto coordinate = task.IsTiled(task.output.axes.size() - 1)
        ? task.output.axes.back().name : "0";
    auto base = "(" + expression(dim->origin) + ") + " +
        std::to_string(partition.reduction_chunk.Eval(known, known)) + " * " + coordinate;
    auto reduced=iteration.at(dim->name);
    if(partition.reduction_index) {
      base=std::to_string(partition.reduction_chunk.Eval(known,known))+" * "+coordinate;
      reduced=index(partition.reduction_index->index);
    }
    bounds.push_back("(" + base + ") <= (" + reduced + ") < (" +
                     base + ") + (" + expression(partition.reduction_chunk) + ")");
  }
  if(partition.reduction_index) {
    auto const& reduction=*partition.reduction_index;
    auto point=index(reduction.index);
    bounds.push_back("0 <= ("+point+") < ("+expression(reduction.capacity)+")");
  }
  for (unsigned axis = 0; axis < tensor.axes.size(); ++axis) {
    auto element = "_tm_e" + std::to_string(axis);
    elements.push_back(element);
    bounds.push_back(element + " = " + index(indexing.results[axis]));
    auto origin = expression(tensor.axes[axis].origin);
    bounds.push_back("(" + origin + ") <= " + element + " < (" + origin + ") + (" +
                     expression(tensor.axes[axis].extent) + ")");
  }
  for (auto const& predicate : nonnegative) bounds.push_back("(" + index(predicate) + ") >= 0");
  std::string condition = Join(bounds, " and ");
  if (!variables.empty()) condition = "exists (" + Join(variables, ",") + " : " + condition + ")";
  std::string prefix;
  if (!parameters.empty()) prefix = "[" + Join({parameters.begin(), parameters.end()}, ",") + "] -> ";
  return CouplingRelation::FromIslText(prefix + "{ [" + Join(task.Coordinates(), ",") +
      "] -> [" + Join(elements, ",") + "] : " + condition + " }");
}
void ValidateTaskReductionIndex(OperatorNode const& task) {
  if(!task.element_access || !task.element_access->partition.reduction_index)return;
  auto const& access=*task.element_access;
  auto const& semantic=access.semantic;
  auto const& reduction=*access.partition.reduction_index;
  std::string key;
  auto append=[&](std::string const& text) {key+=std::to_string(text.size())+':'+text;};
  append(semantic.Serialize());append(access.partition.ownership.Serialize());
  append(access.partition.reduction_chunk.ToString());append(reduction.index.Serialize());
  append(reduction.capacity.ToString());append(reduction.issued_width.ToString());
  append(std::to_string(task.output.axes.size()));
  for(unsigned i=0;i<task.output.axes.size();++i) {
    auto const& axis=task.output.axes[i];
    append(axis.name);append(axis.extent.ToString());append(axis.origin.ToString());
    append(task.tile.at(i).ToString());append(axis.runtime?"1":"0");
  }
  // Coordinate descent revisits unchanged stages. Cache only successful
  // symbolic coverage proofs, keyed by the complete semantic/partition input.
  static thread_local std::unordered_set<std::string> verified;
  if(verified.count(key))return;
  auto original=task;
  bool split=!access.partition.reduction_chunk.IsLiteral(0);
  bool chunk_coordinate=split && task.IsTiled(task.output.axes.size()-1);
  if(split) {original.output.axes.pop_back();original.tile.pop_back();}
  auto partition=access.partition;
  partition.reduction_chunk=ClosedForm::Constant(0);partition.reduction_index.reset();
  TensorSpace iterations;iterations.name=semantic.name+".iterations";
  IndexingMap points;
  for(auto const& dim:semantic.domain) {
    iterations.axes.push_back({dim.name,dim.BoundExtent(),dim.origin,false});
    points.results.push_back(IndexResult::Dim(dim.name));
  }
  auto expected=ProjectTaskElements(semantic,original,partition,iterations,points,{},{});
  auto assigned=ProjectTaskElements(semantic,task,access.partition,iterations,points,{},{});
  if(!assigned.Reverse().IsSingleValued())
    throw std::invalid_argument("reduction index assigns an iteration to several tasks");
  if(chunk_coordinate)
    assigned=assigned.Reverse().ProjectRange(task.Coordinates().size()-1,1).Reverse();
  if(!Contains(assigned,expected) || !Contains(expected,assigned))
    throw std::invalid_argument("reduction index does not cover every semantic iteration");
  if(verified.size()>=4096)verified.clear();
  verified.insert(std::move(key));
}
CouplingRelation TaskElementBoxEnvelope(CouplingRelation const& exact) {
  IslReferenceAudit audit(__func__);
  auto rank = exact.RangeDimNames().size();
  if (!rank) return exact;
  CouplingRelation result;
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto projection = exact.ProjectRange(axis + 1, rank - axis - 1).ProjectRange(0, axis);
    auto endpoints = projection.LexMin().RangeProduct(projection.LexMax());
    auto interval = CouplingRelation::FromIslText("{ [_tm_lo, _tm_hi] -> [_tm_box_" +
        std::to_string(axis) + "] : _tm_lo <= _tm_box_" + std::to_string(axis) + " <= _tm_hi }");
    auto box = endpoints.ApplyRange(interval);
    result = axis ? result.RangeProduct(box) : box;
  }
  if (!exact.IsSubset(result)) throw std::logic_error("axis envelope violated I2");
  return result;
}
TaskElementEnvelope DescribeTaskElementBox(CouplingRelation const& exact) {
  return {TaskElementBoxEnvelope(exact)};
}

CouplingRelation ProjectTaskRead(SemanticOp const& semantic, OperatorNode const& task,
    TaskElementPartition const& partition, TensorSpace const& tensor,
    IndexingMap const& indexing, std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known) {
  if (indexing.results.size() != tensor.axes.size())
    throw std::invalid_argument("task read indexing rank mismatch");
  SemanticOp expanded = semantic;
  auto map = indexing;
  for (unsigned axis = 0; axis < map.results.size(); ++axis) {
    auto& index = map.results[axis];
    if (index.kind == IndexResult::Kind::kAffine) continue;
    auto name = "_tm_access_dim_" + std::to_string(axis);
    while (expanded.Dim(name)) name += "_";
    IterationDim window; window.name = name; window.type = IteratorType::kReduction;
    window.extent = tensor.axes[axis].extent;
    window.origin = tensor.axes[axis].origin;
    if (index.kind == IndexResult::Kind::kBroadcast) {
      window.origin = index.offset; window.extent = index.span;
    } else if (index.kind == IndexResult::Kind::kFullRange) window.origin = index.offset;
    // A value-dependent coordinate ranges over its whole physical axis. All
    // other affine coordinates remain coupled exactly, as required by I2.
    expanded.domain.push_back(std::move(window));
    index = IndexResult::Dim(name);
  }
  return ProjectTaskElements(expanded, task, partition, tensor, map, nonnegative, known);
}

CouplingRelation ProjectTaskWrite(SemanticOp const& semantic, OperatorNode const& task,
    TaskElementPartition const& partition, TensorSpace const& tensor,
    IndexingMap const& indexing, std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known) {
  bool dependent=false;
  for(auto const& axis:indexing.results) {
    if(axis.kind==IndexResult::Kind::kAffine)continue;
    if(axis.kind!=IndexResult::Kind::kDataDependent || axis.binding_source.empty())
      throw std::invalid_argument("non-affine store lacks a runtime binding source");
    dependent=true;
  }
  return dependent?ProjectTaskRead(semantic,task,partition,tensor,indexing,nonnegative,known):
      ProjectTaskElements(semantic,task,partition,tensor,indexing,nonnegative,known);
}

bool HasBindingRequests(IndexingMap const& indexing) {
  for(auto const& index:indexing.results)if(!index.request_dims.empty())return true;
  return false;
}
CouplingRelation ProjectTaskRequests(SemanticOp const& semantic, OperatorNode const& task,
    TaskElementPartition const& partition, TensorSpace const& tensor,
    IndexingMap const& indexing, std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  if(indexing.results.size()!=tensor.axes.size())
    throw std::invalid_argument("logical request indexing rank mismatch");
  TensorSpace requests;requests.name=tensor.name+".requests";
  IndexingMap map;
  std::set<std::pair<std::string,std::string>> keys;
  for(unsigned axis=0;axis<indexing.results.size();++axis) {
    auto const& index=indexing.results[axis];
    if(index.kind!=IndexResult::Kind::kDataDependent) {
      if(!index.request_dims.empty())throw std::invalid_argument("affine index has binding requests");
      requests.axes.push_back(tensor.axes[axis]);map.results.push_back(index);continue;
    }
    if(index.binding_source.empty() || index.request_dims.empty())
      throw std::invalid_argument("data-dependent access lacks logical binding requests");
    std::set<std::string> dimensions;
    for(auto const& name:index.request_dims) {
      auto dim=semantic.Dim(name);
      if(!dim || !dimensions.insert(name).second)
        throw std::invalid_argument("invalid logical binding request dimension");
      if(keys.emplace(index.binding_source,name).second) {
        requests.axes.push_back({"request"+std::to_string(requests.axes.size()),dim->BoundExtent(),dim->origin,false});
        map.results.push_back(IndexResult::Dim(name));
      }
    }
  }
  return ProjectTaskRead(semantic,task,partition,requests,map,nonnegative,known);
}

ExactTaskCoupling DeriveExactTaskCoupling(CouplingRelation const& writes,
    CouplingRelation const& reads, OperatorNode const& consumer,
    ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  ExactTaskCoupling result;
  result.relation = reads.ApplyRange(writes.Reverse());
  result.shared_elements = reads.Reverse().RangeProduct(writes.Reverse()).Reverse();
  result.metrics.wait = result.relation.BoundTaskCard();
  result.metrics.fanout = result.relation.Reverse().BoundTaskCard();
  result.metrics.volume = result.shared_elements.BoundTaskCard();
  result.metrics.count = QuasiPolynomial::FromClosedForm(consumer.Count(), known);
  result.consumer_read_elements = reads.ApplyRange(writes.ImageIdentity()).BoundTaskCard();
  return result;
}
}  // namespace tilemega::analysis
