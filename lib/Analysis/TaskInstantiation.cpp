// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskInstantiation.h>

#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tilemega::analysis {
namespace {

int ResultAxisOf(SemanticOp const& op, std::string const& dim) {
  for (std::size_t i = 0; i < op.result_map.results.size(); ++i) {
    auto const& result = op.result_map.results[i];
    if (result.kind != IndexResult::Kind::kAffine) continue;
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
  switch (result.kind) {
    case IndexResult::Kind::kFullRange:
      return OperandAxisMap::FullRange(result.offset);
    case IndexResult::Kind::kBroadcast:
      return OperandAxisMap::Broadcast(result.span);
    case IndexResult::Kind::kDataDependent:
      return OperandAxisMap::DataDependent();
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
    if (axis < 0) return OperandAxisMap::FullRange(result.offset);
    terms.push_back({axis, term.coefficient, term.group});
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
  for (std::size_t axis = 0; axis < op.result.axes.size(); ++axis) {
    ClosedForm tile = op.result.axes[axis].extent;
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
    if (!split) {
      OperatorNode node;
      node.name = op.name;
      node.kind = op.kind;
      node.output = op.result;
      node.tile = LowerTiles(op, g);
      for (auto const& operand : op.operands) {
        node.operands.push_back(LowerOperand(op, operand, {}));
        node.operands.back().producer = resolve(operand.producer);
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
