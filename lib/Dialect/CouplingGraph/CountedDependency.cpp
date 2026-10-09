// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Analysis/ISLContext.h>
#include <mlir/IR/SymbolTable.h>
#include <limits>
#include <stdexcept>

namespace tilemega::dialect {
namespace {
analysis::OperatorNode Task(TileSpaceOp space) {
  if(!space || !space.getSemantic())
    throw std::invalid_argument("counted scatter needs L-sem on both tasks");
  auto semantic=analysis::DecodeSemanticOp(space.getSemantic()->str());
  auto ownership=space.getGranularity().getAs<mlir::StringAttr>("ownership");
  if(!semantic.exact_task_access || !ownership || ownership.getValue()!="tile_per_block")
    throw std::invalid_argument("counted scatter needs exact tile ownership");
  analysis::Granularity geometry;
  for(unsigned i=0;i<semantic.task_space.axes.size();++i) {
    auto const& axis=semantic.task_space.axes[i];
    auto tile=space.getGranularity().getAs<mlir::StringAttr>(axis.name);
    if(!tile)throw std::invalid_argument("counted scatter lacks its task granularity");
    geometry.Tile(semantic.name,analysis::UnitTaskOwnershipDimension(semantic,i),
                  analysis::ClosedForm::Parse(tile.getValue().str()));
  }
  auto graph=analysis::Instantiate({{semantic}},geometry);
  if(graph.nodes.size()!=1 || graph.nodes[0].name!=space.getOperatorName())
    throw std::invalid_argument("counted scatter requires one unsplit semantic task");
  return graph.nodes.front();
}
analysis::CountedDependencyForm Contract(TileSpaceOp producer,TileSpaceOp consumer,
    std::string const& tensor,std::vector<unsigned> const& axes,std::string const& source,
    analysis::ParamBinding const& binding) {
  if(producer.getStage()>=consumer.getStage())
    throw std::invalid_argument("counted scatter producer must precede consumer");
  return analysis::BindAlignedCountedScatterDependency(Task(producer),Task(consumer),
      tensor,axes,source,binding);
}
}
mlir::DictionaryAttr EncodeBoundCountedScatter(mlir::OpBuilder& builder,
    TileSpaceOp producer,TileSpaceOp consumer,std::string const& tensor,
    std::vector<unsigned> const& unit_axes,std::string const& binding_source,
    analysis::ParamBinding const& binding) {
  analysis::IslReferenceAudit audit(__func__);
  auto contract=Contract(producer,consumer,tensor,unit_axes,binding_source,binding);
  llvm::SmallVector<std::int64_t> axes(unit_axes.begin(),unit_axes.end());
  llvm::SmallVector<std::int64_t> counts(contract.expected.begin(),contract.expected.end());
  llvm::SmallVector<mlir::NamedAttribute> values;
  for(auto const& [name,value]:binding.values)
    values.push_back(builder.getNamedAttr(name,builder.getI64IntegerAttr(value)));
  return builder.getDictionaryAttr({
    builder.getNamedAttr("tensor",builder.getStringAttr(tensor)),
    builder.getNamedAttr("binding_source",builder.getStringAttr(binding_source)),
    builder.getNamedAttr("binding_contract",builder.getStringAttr("unit_permutation")),
    builder.getNamedAttr("unit_axes",builder.getDenseI64ArrayAttr(axes)),
    builder.getNamedAttr("expected",builder.getDenseI64ArrayAttr(counts)),
    builder.getNamedAttr("target_units",CouplingMapAttr::get(builder.getContext(),contract.target_units)),
    builder.getNamedAttr("binding",builder.getDictionaryAttr(values))});
}
std::optional<analysis::CountedDependencyForm> ReadBoundCountedScatter(
    mlir::Operation* op,analysis::ParamBinding const& binding) {
  analysis::IslReferenceAudit audit(__func__);
  if(!op->hasAttr("dependency_counted"))return {};
  auto coupling=mlir::dyn_cast<CouplingOp>(op);
  auto attr=op->getAttrOfType<mlir::DictionaryAttr>("dependency_counted");
  if(!coupling || !attr || op->hasAttr("dependency_table"))
    throw std::invalid_argument("invalid counted scatter owner or competing table");
  auto string=[&](char const* name) {
    auto value=attr.getAs<mlir::StringAttr>(name);
    if(!value || value.getValue().empty())throw std::invalid_argument("incomplete counted scatter metadata");
    return value.getValue().str();
  };
  if(string("binding_contract")!="unit_permutation")
    throw std::invalid_argument("unknown counted scatter binding contract");
  auto units=attr.getAs<mlir::DenseI64ArrayAttr>("unit_axes");
  auto expected=attr.getAs<mlir::DenseI64ArrayAttr>("expected");
  auto relation=attr.getAs<CouplingMapAttr>("target_units");
  auto shape=attr.getAs<mlir::DictionaryAttr>("binding");
  if(!units || !expected || !relation || !shape)
    throw std::invalid_argument("incomplete counted scatter domain");
  auto known=binding;
  for(auto item:shape) {
    auto value=mlir::dyn_cast<mlir::IntegerAttr>(item.getValue());
    auto name=item.getName().str();
    if(!value || (known.Contains(name) && known.At(name)!=value.getInt()))
      throw std::invalid_argument("counted scatter shape binding differs from plan");
    known.Bind(name,value.getInt());
  }
  std::vector<unsigned> axes;
  for(auto axis:units.asArrayRef()) {
    if(axis<0 || std::uint64_t(axis)>std::numeric_limits<unsigned>::max())
      throw std::invalid_argument("counted scatter unit axis overflows");
    axes.push_back(unsigned(axis));
  }
  auto p=mlir::SymbolTable::lookupNearestSymbolFrom<TileSpaceOp>(coupling,coupling.getSrcAttr());
  auto c=mlir::SymbolTable::lookupNearestSymbolFrom<TileSpaceOp>(coupling,coupling.getDstAttr());
  if(!p || !c)throw std::invalid_argument("counted scatter references unknown tasks");
  auto result=Contract(p,c,string("tensor"),axes,string("binding_source"),known);
  if(expected.size()!=result.expected.size())
    throw std::invalid_argument("counted scatter thresholds differ from consumer count");
  for(unsigned target=0;target<result.expected.size();++target)
    if(expected[target]!=result.expected[target])
      throw std::invalid_argument("counted scatter threshold differs from L-sem");
  if(!analysis::Contains(relation.getMap(),result.target_units) ||
     !analysis::Contains(result.target_units,relation.getMap()))
    throw std::invalid_argument("counted scatter contribution image differs from L-sem");
  auto pn=Task(p),cn=Task(c);
  auto edges=analysis::CouplingDerivation{}.Derive({{pn,cn}},known);
  if(edges.empty())throw std::invalid_argument("counted scatter has no producer operand coupling");
  for(auto const& edge:edges)
    if(!analysis::Contains(coupling.getRelation().getMap().BindParams(known),edge.C))
      throw std::invalid_argument("counted scatter coupling violates I2");
  return result;
}
} // namespace tilemega::dialect
