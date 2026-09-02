// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/AccessRelation.h>

#include <sstream>
#include <stdexcept>

namespace tilemega::analysis {

ClosedForm TensorSpace::Volume() const {
  ClosedForm result = ClosedForm::Constant(1);
  for (auto const& axis : axes) result = result * axis.extent;
  return result;
}

OperandAxisMap OperandAxisMap::Indexed(int output_axis, ClosedForm scale,
                                       ClosedForm group) {
  OperandAxisMap map;
  map.kind = Kind::kIndexed;
  map.terms.push_back({output_axis, std::move(scale), std::move(group)});
  return map;
}

OperandAxisMap OperandAxisMap::Packed(std::vector<Term> terms) {
  OperandAxisMap map;
  map.kind = Kind::kIndexed;
  map.terms = std::move(terms);
  return map;
}

OperandAxisMap OperandAxisMap::FullRange(ClosedForm offset) {
  OperandAxisMap map;
  map.kind = Kind::kFullRange;
  map.offset = std::move(offset);
  return map;
}

OperandAxisMap OperandAxisMap::Broadcast(ClosedForm span) {
  OperandAxisMap map;
  map.kind = Kind::kBroadcast;
  map.span = std::move(span);
  return map;
}

OperandAxisMap OperandAxisMap::DataDependent() {
  OperandAxisMap map;
  map.kind = Kind::kDataDependent;
  return map;
}

std::string ToString(OperatorKind kind) {
  switch (kind) {
    case OperatorKind::kPointwise: return "pointwise";
    case OperatorKind::kReduction: return "reduction";
    case OperatorKind::kMatmul: return "matmul";
    case OperatorKind::kBroadcast: return "broadcast";
    case OperatorKind::kConcat: return "concat";
    case OperatorKind::kSlice: return "slice";
    case OperatorKind::kTranspose: return "transpose";
    case OperatorKind::kView: return "view";
    case OperatorKind::kGather: return "gather";
  }
  return "unknown";
}

bool OperatorNode::IsTiled(std::size_t axis) const {
  if (axis >= output.axes.size()) throw std::out_of_range("output axis");
  if (axis >= tile.size()) return false;
  return tile[axis].ToString() != output.axes[axis].extent.ToString();
}

std::vector<std::string> OperatorNode::Coordinates() const {
  std::vector<std::string> result;
  for (std::size_t i = 0; i < output.axes.size(); ++i)
    if (IsTiled(i)) result.push_back(output.axes[i].name);
  return result;
}

ClosedForm OperatorNode::CoordinateExtent(std::size_t axis) const {
  if (!IsTiled(axis)) return ClosedForm::Constant(1);
  return output.axes[axis].extent.CeilDiv(tile[axis]);
}

ClosedForm OperatorNode::Count() const {
  ClosedForm result = ClosedForm::Constant(1);
  for (std::size_t i = 0; i < output.axes.size(); ++i)
    if (IsTiled(i)) result = result * CoordinateExtent(i);
  return result;
}

bool OperatorNode::HasRuntimeTaskSpace() const {
  for (auto const& axis : output.axes)
    if (axis.runtime) return true;
  return false;
}

OperatorNode const* OperatorGraph::Find(std::string const& name) const {
  for (auto const& node : nodes)
    if (node.name == name) return &node;
  return nullptr;
}

std::string AccessRelation::ToString() const {
  std::ostringstream out;
  out << op << (is_write ? ":W" : ":R") << "[";
  for (std::size_t i = 0; i < coordinates.size(); ++i) {
    if (i) out << ",";
    out << coordinates[i];
  }
  out << "] -> " << tensor.name << "{";
  for (std::size_t i = 0; i < index.size(); ++i) {
    if (i) out << ", ";
    out << "[" << index[i].base.ToString() << " +: " << index[i].span.ToString()
        << ")";
  }
  out << "}";
  if (data_dependent) out << " data_dependent";
  return out.str();
}

AccessRelation BuildWriteMap(OperatorNode const& op) {
  AccessRelation relation;
  relation.op = op.name;
  relation.producer = op.name;
  relation.tensor = op.output;
  relation.coordinates = op.Coordinates();
  relation.is_write = true;
  for (std::size_t i = 0; i < op.output.axes.size(); ++i) {
    ElementInterval interval;
    if (op.IsTiled(i)) {
      interval.base = AffineExpr::Variable(op.output.axes[i].name, op.tile[i]) +
                      AffineExpr::Constant(op.output.axes[i].origin);
      interval.span = op.tile[i];
    } else {
      interval.base = AffineExpr::Constant(op.output.axes[i].origin);
      interval.span = op.output.axes[i].extent;
    }
    relation.index.push_back(interval);
  }
  return relation;
}

AccessRelation BuildReadMap(OperatorNode const& op, std::size_t operand_index) {
  if (operand_index >= op.operands.size())
    throw std::out_of_range("operand index");
  Operand const& operand = op.operands[operand_index];
  if (operand.axes.size() != operand.tensor.axes.size())
    throw std::invalid_argument("operand axis map must cover every tensor axis");

  AccessRelation relation;
  relation.op = op.name;
  relation.producer = operand.producer;
  relation.tensor = operand.tensor;
  relation.coordinates = op.Coordinates();

  for (std::size_t a = 0; a < operand.axes.size(); ++a) {
    OperandAxisMap const& map = operand.axes[a];
    ClosedForm const& axis_extent = operand.tensor.axes[a].extent;
    ElementInterval interval;
    switch (map.kind) {
      case OperandAxisMap::Kind::kIndexed: {
        if (map.terms.empty())
          throw std::invalid_argument("kIndexed needs at least one term");
        interval.base = AffineExpr::Constant(map.offset);
        ClosedForm span = ClosedForm::Constant(1);
        for (auto const& term : map.terms) {
          if (term.output_axis < 0 ||
              static_cast<std::size_t>(term.output_axis) >=
                  op.output.axes.size())
            throw std::invalid_argument("kIndexed needs a valid output axis");
          auto out_axis = static_cast<std::size_t>(term.output_axis);
          if (op.IsTiled(out_axis)) {
            // The tile of the consumer axis multiplies the element scale: one
            // step of the task coordinate advances a whole tile of elements.
            ClosedForm stride = op.tile[out_axis] * term.scale;
            interval.base = interval.base +
                            AffineExpr::Variable(op.output.axes[out_axis].name,
                                                 stride, term.group);
            span = stride;
          } else {
            // A whole axis contributes no coordinate; the access covers it all.
            span = op.output.axes[out_axis].extent * term.scale;
          }
        }
        interval.span = span;
        break;
      }
      case OperandAxisMap::Kind::kFullRange:
        interval.base = AffineExpr::Constant(map.offset);
        interval.span = axis_extent;
        break;
      case OperandAxisMap::Kind::kBroadcast:
        interval.base = AffineExpr::Constant(map.offset);
        interval.span = map.span;
        break;
      case OperandAxisMap::Kind::kDataDependent:
        // The index is a runtime permutation.  Recording the whole axis is the
        // only sound rectangular cover; the tier classifier turns this into the
        // I2 relaxation rather than pretending an affine index exists.
        interval.base = AffineExpr::Constant(ClosedForm::Constant(0));
        interval.span = axis_extent;
        relation.data_dependent = true;
        break;
    }
    relation.index.push_back(interval);
  }
  return relation;
}

}  // namespace tilemega::analysis
