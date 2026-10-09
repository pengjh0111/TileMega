// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ExactTaskWork.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <set>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
QuasiPolynomial Polynomial(ClosedForm const& value, ParamBinding const& known) {
  return QuasiPolynomial::FromClosedForm(value, known);
}
}
TaskWork DeriveExactTaskWork(OperatorNode const& task, ParamBinding const& known,
                            TaskWorkOptions const& options) {
  IslReferenceAudit audit(__func__);
  if (!task.element_access) throw std::invalid_argument("missing exact task access");
  auto const& access = *task.element_access;
  auto const& sem = access.semantic;
  bool indirect_write=HasBindingRequests(sem.result_map);
  auto writes = indirect_write?
      ProjectTaskRequests(sem,task,access.partition,sem.result,sem.result_map,{},known):
      ProjectTaskElements(sem, task, access.partition, sem.result, sem.result_map, {}, known);
  TaskWork work;
  work.write_elements = writes.BoundTaskCard();
  work.task_count = writes.Reverse().Image().BoundTaskCard();
  std::map<std::string, CouplingRelation> reads;
  std::set<std::string> produced;
  for (auto const& input : sem.operands) if (!input.producer.empty()) produced.insert(input.tensor.name);
  auto append = [&](TensorSpace const& tensor, IndexingMap const& map,
                    std::vector<IndexResult> const& predicates) {
    auto relation=HasBindingRequests(map)?
        ProjectTaskRequests(sem,task,access.partition,tensor,map,predicates,known):
        ProjectTaskRead(sem,task,access.partition,tensor,map,predicates,known);
    reads[tensor.name] = reads[tensor.name].Union(relation);
  };
  if (sem.element_reads.empty()) {
    for (auto const& input : sem.operands) append(input.tensor, input.map, {});
  } else {
    for (auto const& read : sem.element_reads) append(read.tensor, read.map, read.nonnegative);
  }
  std::vector<QuasiPolynomial> counts, frontier;
  for (auto const& [name, relation] : reads) {
    counts.push_back(relation.BoundTaskCard());
    if (!produced.count(name)) frontier.push_back(relation.BoundTaskCard());
  }
  work.read_elements = QuasiPolynomial::Sum(counts);
  work.frontier_read_elements = QuasiPolynomial::Sum(frontier);
  // cp.async predicates replace absent elements with zero. Padding changes
  // issued MMA work, not the physical global-memory read set.
  work.nominal_read_elements = work.read_elements;
  std::set<std::string> output_dims;
  for (auto const& index : (indirect_write?sem.task_map:sem.result_map).results)
    for (auto const& term : index.terms) if (!term.coefficient.IsLiteral(0)) output_dims.insert(term.dim);
  if (!access.partition.reduction_chunk.IsLiteral(0)) {
    output_dims.erase(sem.reduction.dim);
    if(access.partition.reduction_index)
      for(auto const& term:access.partition.reduction_index->index.terms)
        output_dims.erase(term.dim);
  }
  ClosedForm parallel = ClosedForm::Constant(1), reduction = ClosedForm::Constant(1);
  ClosedForm issued = ClosedForm::Constant(1);
  TensorSpace reduction_space; reduction_space.name = sem.name + ".reduction_domain";
  IndexingMap reduction_map;
  for (auto const& dim : sem.domain) {
    if (output_dims.count(dim.name)) parallel = parallel * dim.BoundExtent();
    else {
      reduction = reduction * dim.BoundExtent();
      auto extent = !access.partition.reduction_chunk.IsLiteral(0) && dim.name == sem.reduction.dim
          ? access.partition.reduction_chunk : dim.BoundExtent();
      if (auto tile = options.reduction_tiles.find(dim.name); tile != options.reduction_tiles.end()) {
        if (tile->second.Eval(known, known) <= 0) throw std::invalid_argument("nonpositive exact reduction tile");
        extent = extent.CeilDiv(tile->second) * tile->second;
      }
      issued = issued * extent;
      reduction_space.axes.push_back({dim.name, dim.BoundExtent(), dim.origin, dim.runtime && !dim.capacity});
      reduction_map.results.push_back(IndexResult::Dim(dim.name));
    }
  }
  auto local_reduction = ProjectTaskElements(sem, task, access.partition,
      reduction_space, reduction_map, {}, known).BoundTaskCard();
  work.parallel_extent = Polynomial(parallel, known);
  work.reduce_extent = Polynomial(reduction, known);
  work.task_reduce_extent = local_reduction;
  for (auto const& [dim, tile] : options.reduction_tiles)
    if (!sem.Dim(dim) || output_dims.count(dim))
      throw std::invalid_argument("invalid exact task reduction tile");
  if(access.partition.reduction_index) {
    auto const& index=*access.partition.reduction_index;
    issued=(access.partition.reduction_chunk.IsLiteral(0)?index.capacity:
        access.partition.reduction_chunk)*index.issued_width;
  }
  auto tasks = writes.Reverse().ImageIdentity();
  work.nominal_task_reduce_extent = Polynomial(issued, known).SumAlong(tasks);
  if(access.partition.reduction_index && access.partition.reduction_index->chunks &&
      !access.partition.reduction_chunk.IsLiteral(0)) {
    auto const& geometry=*access.partition.reduction_index;
    auto coordinates=task.Coordinates();std::string tuple;
    for(auto const& name:coordinates) {if(!tuple.empty())tuple+=",";tuple+=name;}
    auto chunk=task.IsTiled(task.output.axes.size()-1)?task.output.axes.back().name:"0";
    auto capacity=std::to_string(geometry.capacity.Eval(known,known));
    auto count=std::to_string(geometry.chunks);
    auto relation=CouplingRelation::FromIslText("{ ["+tuple+"] -> [_tm_issued_k] : "+
        "floord("+capacity+"*("+chunk+"),"+count+") <= _tm_issued_k < "+
        "floord("+capacity+"*("+chunk+"+1),"+count+") }");
    work.nominal_task_reduce_extent=tasks.ApplyRange(relation).BoundTaskCard().Scale(
        geometry.issued_width.Eval(known,known));
  }
  if(!sem.domain_nonnegative.empty() && options.reduction_tiles.empty() &&
     access.partition.reduction_chunk.IsLiteral(0))
    work.nominal_task_reduce_extent=local_reduction;
  ClosedForm owned = ClosedForm::Constant(1), tiled = ClosedForm::Constant(1);
  for (unsigned axis = 0; axis < access.partition.ownership.results.size(); ++axis) {
    owned = owned * task.output.axes[axis].extent;
    tiled = tiled * task.tile[axis];
  }
  auto output_width = indirect_write?parallel:sem.result.Volume();
  auto chunk_count=[&]() {
    auto const* reduced=sem.Dim(sem.reduction.dim);
    if(!reduced)throw std::invalid_argument("partials lack their reduction dimension");
    if(access.partition.reduction_index && access.partition.reduction_index->chunks)
      return ClosedForm::Constant(access.partition.reduction_index->chunks);
    return (access.partition.reduction_index?access.partition.reduction_index->capacity:
        reduced->BoundExtent()).CeilDiv(access.partition.reduction_chunk);
  };
  if(indirect_write && !access.partition.reduction_chunk.IsLiteral(0)) {
    output_width=output_width*chunk_count();
  }
  if (!access.partition.reduction_chunk.IsLiteral(0)) {
    ClosedForm per_chunk;
    if (!output_width.TryExactDivide(chunk_count(), &per_chunk))
      throw std::invalid_argument("partial tensor volume does not factor by its chunks");
    output_width = per_chunk;
  }
  ClosedForm ratio;
  if(output_width.Substitute(known).IsLiteral(0))
    work.nominal_write_elements=work.write_elements;
  else if (output_width.TryExactDivide(owned, &ratio))
    work.nominal_write_elements = Polynomial(tiled * ratio, known).SumAlong(tasks);
  else if (owned.TryExactDivide(output_width, &ratio) && ratio.Substitute(known).IsConstant()) {
    auto inverse = ratio.Eval(known, {});
    if (inverse <= 0) throw std::invalid_argument("invalid exact task output ratio");
    work.nominal_write_elements = Polynomial(tiled, known).SumAlong(tasks)
        .ScaleRational("1/" + std::to_string(inverse));
  } else {
    auto total = output_width.Eval(known, {}), ownership = owned.Eval(known, {});
    if (ownership <= 0 || total <= 0) throw std::invalid_argument("invalid exact task output volume");
    work.nominal_write_elements = Polynomial(tiled, known).SumAlong(tasks)
        .ScaleRational(std::to_string(total) + "/" + std::to_string(ownership));
  }
  return work;
}
}  // namespace tilemega::analysis
