// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/VirtualTaskBinding.h>

#include <map>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tilemega::analysis {
namespace {

void ValidateReductionIndex(TaskReductionIndex const& index) {
  if(index.index.kind!=IndexResult::Kind::kAffine ||
     (index.capacity.IsConstant() && index.capacity.Eval({},{})<=0) ||
     (index.issued_width.IsConstant() && index.issued_width.Eval({},{})<=0) ||
     (index.index.outer_divisor.IsConstant() && index.index.outer_divisor.Eval({},{})<=0))
    throw std::invalid_argument("invalid indexed reduction geometry");
}

int ResultAxisOf(SemanticOp const& op, std::string const& dim) {
  auto const& mapping = op.exact_task_access ? op.task_map : op.result_map;
  for (std::size_t i = 0; i < mapping.results.size(); ++i) {
    auto const& result = mapping.results[i];
    if (result.kind != IndexResult::Kind::kAffine || !result.outer_divisor.IsLiteral(1)) continue;
    if (result.terms.size() == 1 && result.terms.front().dim == dim &&
        result.terms.front().coefficient.IsLiteral(1) &&
        result.terms.front().group.IsLiteral(1))
      return static_cast<int>(i);
  }
  return -1;
}

/// One indexing-map result becomes one OperandAxisMap. A term on a reduction
/// dim that was not split is read in full: the whole reduction lives inside a
/// single task, which is exactly a full-range access along that axis.
OperandAxisMap LowerResult(SemanticOp const& op, IndexResult const& result,
                           std::map<std::string, int> const& extra_axis) {
  // Exact element projection retains nested floors. The older rectangular
  // operand summary can only enclose that address set conservatively.
  if(!result.outer_divisor.IsLiteral(1)) {
    if(!op.exact_task_access)
      throw std::invalid_argument("outer floor requires exact task access");
    return OperandAxisMap::FullRange();
  }
  switch (result.kind) {
    case IndexResult::Kind::kFullRange:
      return OperandAxisMap::FullRange(result.offset);
    case IndexResult::Kind::kBroadcast:
      return OperandAxisMap::Broadcast(result.span);
    case IndexResult::Kind::kDataDependent:
      {
        auto map = OperandAxisMap::DataDependent(); map.binding_source = result.binding_source; return map;
      }
    case IndexResult::Kind::kAffine:
      break;
  }
  std::vector<OperandAxisMap::Term> terms;
  for (auto const& term : result.terms) {
    auto extra = extra_axis.find(term.dim);
    int axis = extra != extra_axis.end() ? extra->second
                                         : ResultAxisOf(op, term.dim);
    // A dim with no task coordinate is either an unsplit reduction or a
    // parallel dim the result does not expose; either way the whole axis is
    // inside one task, which is a full-range access and I2-safe.
    if (axis < 0) {
      if (!op.exact_task_access) return OperandAxisMap::FullRange(result.offset);
      auto const* dim = op.Dim(term.dim);
      if (!dim) throw std::invalid_argument("window names an unknown iteration dim");
      OperandAxisMap::Term window;
      window.scale = term.coefficient; window.group = term.group;
      window.window_dim = dim->name; window.window_extent = dim->BoundExtent();
      window.window_origin = dim->origin;
      window.shift = term.shift;
      terms.push_back(std::move(window));
      continue;
    }
    OperandAxisMap::Term lowered{axis, term.coefficient, term.group};
    lowered.shift = term.shift;
    terms.push_back(std::move(lowered));
  }
  if (terms.empty()) return OperandAxisMap::Broadcast(result.span);
  OperandAxisMap map = OperandAxisMap::Packed(std::move(terms));
  map.offset = result.offset;
  return map;
}

Operand LowerOperand(SemanticOp const& op, SemanticOperand const& operand,
                     std::map<std::string, int> const& extra_axis) {
  Operand lowered;
  lowered.producer = operand.producer;
  lowered.tensor = operand.tensor;
  for (auto const& result : operand.map.results)
    lowered.axes.push_back(LowerResult(op, result, extra_axis));
  return lowered;
}

std::vector<ClosedForm> LowerTiles(SemanticOp const& op,
                                   Granularity const& g) {
  std::vector<ClosedForm> tiles;
  auto space = op.exact_task_access ? BindCapacityTaskSpace(op) : op.result;
  for (std::size_t axis = 0; axis < space.axes.size(); ++axis) {
    ClosedForm tile = space.axes[axis].extent;
    for (auto const& dim : op.domain) {
      if (ResultAxisOf(op, dim.name) != static_cast<int>(axis)) continue;
      ClosedForm chosen;
      if (g.TileOf(op.name, dim.name, &chosen)) tile = chosen;
    }
    tiles.push_back(tile);
  }
  return tiles;
}

}  // namespace

Granularity& Granularity::Tile(std::string op, std::string dim,
                               ClosedForm value) {
  tiles[std::move(op)][std::move(dim)] = std::move(value);
  return *this;
}

Granularity& Granularity::Split(std::string op, ClosedForm chunk) {
  reduction_chunk[std::move(op)] = std::move(chunk);
  return *this;
}

Granularity& Granularity::IndexReduction(std::string op,TaskReductionIndex index) {
  ValidateReductionIndex(index);
  reduction_index[std::move(op)]=std::move(index);
  return *this;
}

bool Granularity::TileOf(std::string const& op, std::string const& dim,
                         ClosedForm* value) const {
  auto found = tiles.find(op);
  if (found == tiles.end()) return false;
  auto entry = found->second.find(dim);
  if (entry == found->second.end()) return false;
  if (value) *value = entry->second;
  return true;
}

bool Granularity::ChunkOf(std::string const& op, ClosedForm* value) const {
  auto found = reduction_chunk.find(op);
  if (found == reduction_chunk.end()) return false;
  if (value) *value = found->second;
  return true;
}

std::string Granularity::Serialize() const {
  std::ostringstream out;
  for (auto const& [op, dims] : tiles)
    for (auto const& [dim, tile] : dims)
      out << op << "." << dim << " = " << tile.ToString() << "\n";
  for (auto const& [op, chunk] : reduction_chunk)
    out << op << ".split = " << chunk.ToString() << "\n";
  for(auto const& [op,index]:reduction_index)
    out<<op<<".reduction_index = "<<index.index.Serialize()<<" capacity="
       <<index.capacity.ToString()<<" width="<<index.issued_width.ToString()<<"\n";
  return out.str();
}

OperatorGraph Instantiate(SemanticGraph const& graph, Granularity const& g) {
  OperatorGraph result;
  // Splitting inserts a combiner between an op and its consumers, so which
  // node carries the final result is a function of g. Consumers name the
  // producer in L-sem, where that choice does not exist yet; this map applies
  // it, in both directions, so the graph is well formed under either g.
  std::map<std::string, std::string> producer_of;
  for (auto const& op : graph.ops) {
    if (!op.reduction.splittable) continue;
    if (g.ChunkOf(op.name, nullptr))
      producer_of[op.name] = op.reduction.combiner;
    else
      producer_of[op.reduction.combiner] = op.name;
  }
  auto resolve = [&](std::string const& name) {
    auto found = producer_of.find(name);
    return found == producer_of.end() ? name : found->second;
  };
  for (auto const& op : graph.ops) {
    ClosedForm chunk;
    bool split = op.reduction.splittable && g.ChunkOf(op.name, &chunk);
    auto index=g.reduction_index.find(op.name);
    bool indexed=index!=g.reduction_index.end();
    if(indexed) {
      ValidateReductionIndex(index->second);
      if(!op.exact_task_access || !op.reduction.splittable)
        throw std::invalid_argument("indexed reduction needs exact splittable semantics");
      for(auto const& term:index->second.index.terms) {
        auto dim=op.Dim(term.dim);
        if(!dim || dim->type!=IteratorType::kReduction)
          throw std::invalid_argument("reduction index names a non-reduction dimension");
      }
    }
    if (!split) {
      OperatorNode node;
      node.name = op.name;
      node.kind = op.kind;
      node.output = op.exact_task_access ? BindCapacityTaskSpace(op) : op.result;
      node.tile = LowerTiles(op, g);
      for (auto const& operand : op.operands) {
        node.operands.push_back(LowerOperand(op, operand, {}));
        node.operands.back().producer = resolve(operand.producer);
      }
      if (op.exact_task_access) {
        TaskElementAccess access; access.semantic = op;
        access.partition.ownership = op.task_map;
        if(indexed)access.partition.reduction_index=index->second;
        node.element_access = std::make_shared<TaskElementAccess>(std::move(access));
        if(indexed)ValidateTaskReductionIndex(node);
      }
      result.nodes.push_back(std::move(node));
      continue;
    }

    // §2.4 Split: the reduction dim becomes a parallel chunk axis on a
    // materialized partial tensor, and the declared combiner becomes its own
    // task space that reduces that axis away. The combiner is explicit CG,
    // not a TaskBody detail, so the coupling derivation sees both halves.
    IterationDim const* reduced = op.Dim(op.reduction.dim);
    if (!reduced)
      throw std::invalid_argument("split reduction names an unknown dim: " +
                                  op.reduction.dim);
    if (op.exact_task_access) {
      if (reduced->type != IteratorType::kReduction ||
          (chunk.IsConstant() && chunk.Eval({}, {}) <= 0))
        throw std::invalid_argument("invalid exact split reduction dimension or chunk");
      TensorAxis chunk_axis; chunk_axis.name = "j";
      auto has_axis = [&](std::string const& name) {
        return op.Dim(name) || std::any_of(op.task_space.axes.begin(), op.task_space.axes.end(),
            [&](auto const& axis) { return axis.name == name; });
      };
      while (has_axis(chunk_axis.name)) chunk_axis.name += "_";
      chunk_axis.extent = (indexed?index->second.capacity:reduced->BoundExtent()).CeilDiv(chunk);
      chunk_axis.runtime = !indexed && reduced->runtime && !reduced->capacity;
      auto partial = op.result; partial.name = op.reduction.partial_tensor;
      partial.axes.push_back(chunk_axis);
      auto contribution_sem = op; contribution_sem.result = partial;
      contribution_sem.additional_writes.clear();
      if(indexed) {
        auto partial_index=index->second.index;
        // floor(floor(x/a)/b) == floor(x/(a*b)) for positive a,b.
        partial_index.outer_divisor=partial_index.outer_divisor*chunk;
        contribution_sem.result_map.results.push_back(std::move(partial_index));
      } else contribution_sem.result_map.results.push_back(IndexResult::Dim(reduced->name,
          ClosedForm::Constant(1), chunk, ClosedForm::Constant(-1) * reduced->origin));
      OperatorNode contribution;
      contribution.name = op.name; contribution.kind = op.kind;
      contribution.output = BindCapacityTaskSpace(op); contribution.output.axes.push_back(chunk_axis);
      contribution.tile = LowerTiles(op, g); contribution.tile.push_back(ClosedForm::Constant(1));
      std::map<std::string, int> extra;
      if(!indexed)extra.emplace(reduced->name,int(op.task_space.axes.size()));
      for (auto const& operand : op.operands) {
        auto lowered = LowerOperand(op, operand, extra);
        for (auto& axis : lowered.axes)
          for (auto& term : axis.terms)
            if (!indexed && term.output_axis == extra.at(reduced->name)) term.scale = term.scale * chunk;
        lowered.producer = resolve(operand.producer);
        contribution.operands.push_back(std::move(lowered));
      }
      TaskElementAccess access; access.semantic = std::move(contribution_sem);
      access.partition.ownership = op.task_map; access.partition.reduction_chunk = chunk;
      if(indexed)access.partition.reduction_index=index->second;
      contribution.element_access = std::make_shared<TaskElementAccess>(std::move(access));
      if(indexed)ValidateTaskReductionIndex(contribution);
      result.nodes.push_back(std::move(contribution));
      SemanticOp combine_sem = op;
      combine_sem.name = op.reduction.combiner; combine_sem.kind = OperatorKind::kReduction;
      combine_sem.arithmetic = "sum"; combine_sem.reduction = {};
      combine_sem.operands.clear(); combine_sem.element_reads.clear();
      combine_sem.domain.erase(std::remove_if(combine_sem.domain.begin(), combine_sem.domain.end(),
          [](auto const& dim) { return dim.type == IteratorType::kReduction; }), combine_sem.domain.end());
      combine_sem.domain.push_back({chunk_axis.name, chunk_axis.extent, ClosedForm::Constant(0), IteratorType::kReduction, chunk_axis.runtime});
      auto map = op.result_map; map.results.push_back(IndexResult::Dim(chunk_axis.name));
      combine_sem.operands.push_back({op.name, partial, map, {}});
      OperatorNode combine;
      combine.name = combine_sem.name; combine.kind = combine_sem.kind;
      combine.output = BindCapacityTaskSpace(op); combine.tile = LowerTiles(op, g);
      combine.operands.push_back(LowerOperand(combine_sem, combine_sem.operands.front(), {}));
      TaskElementAccess final_access; final_access.semantic = std::move(combine_sem);
      final_access.partition.ownership = op.task_map;
      combine.element_access = std::make_shared<TaskElementAccess>(std::move(final_access));
      result.nodes.push_back(std::move(combine));
      continue;
    }
    TensorSpace partial = op.result;
    partial.name = op.reduction.partial_tensor;
    TensorAxis chunk_axis;
    chunk_axis.name = "j";
    chunk_axis.extent = reduced->extent.CeilDiv(chunk);
    chunk_axis.runtime = reduced->runtime;
    partial.axes.push_back(chunk_axis);

    OperatorNode contribution;
    contribution.name = op.name;
    contribution.kind = op.kind;
    contribution.output = partial;
    contribution.tile = LowerTiles(op, g);
    contribution.tile.push_back(ClosedForm::Constant(1));
    std::map<std::string, int> extra;
    extra[op.reduction.dim] = static_cast<int>(partial.axes.size()) - 1;
    for (auto const& operand : op.operands) {
      Operand lowered = LowerOperand(op, operand, extra);
      // The chunk axis addresses elements, so a term on it is scaled by the
      // chunk size: element = chunk * j.
      for (std::size_t i = 0; i < lowered.axes.size(); ++i) {
        auto& axis = lowered.axes[i];
        if (axis.kind != OperandAxisMap::Kind::kIndexed) continue;
        for (auto& term : axis.terms)
          if (term.output_axis == extra[op.reduction.dim])
            term.scale = term.scale * chunk;
      }
      lowered.producer = resolve(operand.producer);
      contribution.operands.push_back(std::move(lowered));
    }
    result.nodes.push_back(std::move(contribution));

    OperatorNode combine;
    combine.name = op.reduction.combiner;
    combine.kind = OperatorKind::kReduction;
    combine.output = op.result;
    combine.tile = LowerTiles(op, g);
    Operand read;
    read.producer = op.name;
    read.tensor = partial;
    for (std::size_t axis = 0; axis < op.result.axes.size(); ++axis)
      read.axes.push_back(OperandAxisMap::Indexed(static_cast<int>(axis)));
    read.axes.push_back(OperandAxisMap::FullRange());
    combine.operands.push_back(std::move(read));
    result.nodes.push_back(std::move(combine));
  }
  return result;
}

}  // namespace tilemega::analysis
