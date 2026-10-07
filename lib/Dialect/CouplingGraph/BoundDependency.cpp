// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/BoundDependency.h>
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <limits>
#include <stdexcept>

namespace tilemega::dialect {
mlir::DictionaryAttr EncodeBoundTaskGeometry(mlir::OpBuilder& builder,
    BoundTaskGeometry const& geometry, analysis::CouplingRelation const& producer_coordinates,
    analysis::CouplingRelation const& consumer_coordinates, analysis::ParamBinding const& binding) {
  llvm::SmallVector<mlir::NamedAttribute> values;
  for (auto const& [name, value] : binding.values)
    values.push_back(builder.getNamedAttr(name, builder.getI64IntegerAttr(value)));
  auto* context = builder.getContext();
  return builder.getDictionaryAttr({
      builder.getNamedAttr("producers", builder.getI64IntegerAttr(geometry.producers)),
      builder.getNamedAttr("consumers", builder.getI64IntegerAttr(geometry.consumers)),
      builder.getNamedAttr("relation", CouplingMapAttr::get(context, geometry.relation)),
      builder.getNamedAttr("producer_coordinates", CouplingMapAttr::get(context, producer_coordinates)),
      builder.getNamedAttr("consumer_coordinates", CouplingMapAttr::get(context, consumer_coordinates)),
      builder.getNamedAttr("binding", builder.getDictionaryAttr(values))});
}
mlir::DictionaryAttr EncodeBoundDependencyTable(mlir::OpBuilder& builder,
    analysis::DependencyTable const& table, analysis::CouplingRelation const& producer_coordinates,
    analysis::CouplingRelation const& consumer_coordinates, analysis::ParamBinding const& binding) {
  auto geometry = EncodeBoundTaskGeometry(builder, {table.linear_relation, table.producers, table.consumers},
                                         producer_coordinates, consumer_coordinates, binding);
  llvm::SmallVector<mlir::NamedAttribute> attrs(geometry.getValue());
  llvm::SmallVector<std::int64_t> intervals;
  for (auto const& interval : table.intervals) {
    intervals.push_back(interval.first); intervals.push_back(interval.count);
  }
  attrs.push_back(builder.getNamedAttr("stride", builder.getI64IntegerAttr(table.stride)));
  attrs.push_back(builder.getNamedAttr("intervals", builder.getDenseI64ArrayAttr(intervals)));
  return builder.getDictionaryAttr(attrs);
}
namespace {
BoundTaskGeometry ReadGeometry(mlir::Operation* op, mlir::DictionaryAttr table,
                              analysis::ParamBinding const& binding) {
  auto coupling = mlir::dyn_cast<CouplingOp>(op);
  if (!table || !coupling) throw std::invalid_argument("invalid bound task geometry owner");
  auto integer = [&](char const* name) {
    auto attr = table.getAs<mlir::IntegerAttr>(name);
    if (!attr || attr.getInt() <= 0 || attr.getInt() > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid bound task geometry dimension");
    return std::uint32_t(attr.getInt());
  };
  auto producers = integer("producers"), consumers = integer("consumers");
  auto relation = table.getAs<CouplingMapAttr>("relation");
  auto p = table.getAs<CouplingMapAttr>("producer_coordinates");
  auto c = table.getAs<CouplingMapAttr>("consumer_coordinates");
  auto values = table.getAs<mlir::DictionaryAttr>("binding");
  if (!relation || !p || !c || !values)
    throw std::invalid_argument("incomplete bound task geometry");
  auto known = binding;
  for (auto value : values) {
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value.getValue());
    if (!integer) throw std::invalid_argument("table shape binding must be integral");
    auto name = value.getName().str();
    if (known.Contains(name) && known.At(name) != integer.getInt())
      throw std::invalid_argument("table shape binding differs from the plan");
    known.Bind(name, integer.getInt());
  }
  if (!p.getMap().IsSingleValued() || !p.getMap().Reverse().IsSingleValued() ||
      !c.getMap().IsSingleValued() || !c.getMap().Reverse().IsSingleValued())
    throw std::invalid_argument("table task linearizations must be injective functions");
  auto exact = c.getMap().Reverse().ApplyRange(coupling.getRelation().getMap().BindParams(known)).ApplyRange(p.getMap());
  if (!analysis::Contains(exact, relation.getMap()) || !analysis::Contains(relation.getMap(), exact))
    throw std::invalid_argument("table relation differs from the bound coupling");
  if (p.getMap().Image().BoundTaskCard().Eval({}) != producers ||
      c.getMap().Image().BoundTaskCard().Eval({}) != consumers)
    throw std::invalid_argument("bound task counts differ from their linearizations");
  auto p_domain = analysis::CouplingRelation::FromIslText("{ [] -> [p] : 0 <= p < " + std::to_string(producers) + " }");
  auto c_domain = analysis::CouplingRelation::FromIslText("{ [] -> [c] : 0 <= c < " + std::to_string(consumers) + " }");
  if (!analysis::Contains(p.getMap().Image(), p_domain) || !analysis::Contains(p_domain, p.getMap().Image()) ||
      !analysis::Contains(c.getMap().Image(), c_domain) || !analysis::Contains(c_domain, c.getMap().Image()))
    throw std::invalid_argument("bound task linearization IDs are not contiguous");
  return {relation.getMap(), producers, consumers};
}
}
std::optional<BoundTaskGeometry> ReadBoundTaskGeometry(mlir::Operation* op, analysis::ParamBinding const& binding) {
  if (!op->hasAttr("dependency_geometry")) return {};
  return ReadGeometry(op, op->getAttrOfType<mlir::DictionaryAttr>("dependency_geometry"), binding);
}
std::optional<analysis::DependencyTable> ReadBoundDependencyTable(mlir::Operation* op,
    analysis::ParamBinding const& binding) {
  if (!op->hasAttr("dependency_table")) return {};
  auto table = op->getAttrOfType<mlir::DictionaryAttr>("dependency_table");
  auto geometry = ReadGeometry(op, table, binding);
  auto stride = table.getAs<mlir::IntegerAttr>("stride");
  auto intervals = table.getAs<mlir::DenseI64ArrayAttr>("intervals");
  if (!stride || stride.getInt() < 0 || stride.getInt() > std::numeric_limits<std::uint32_t>::max() ||
      !intervals || std::uint64_t(geometry.consumers) * stride.getInt() * 2 != intervals.size())
    throw std::invalid_argument("incomplete bound dependency intervals");
  auto expected = analysis::BuildDependencyTableLinear(geometry.relation, geometry.producers, geometry.consumers);
  if (stride.getInt() != expected.stride) throw std::invalid_argument("table stride differs from its exact intervals");
  auto data = intervals.asArrayRef();
  for (unsigned i = 0; i < expected.intervals.size(); ++i)
    if (data[i*2] != expected.intervals[i].first || data[i*2+1] != expected.intervals[i].count)
      throw std::invalid_argument("table intervals differ from their exact relation");
  return expected;
}
}  // namespace tilemega::dialect
