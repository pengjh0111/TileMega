// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <mlir/IR/SymbolTable.h>
#include <limits>
#include <stdexcept>

namespace tilemega::dialect {
namespace {
analysis::OperatorNode Task(analysis::SemanticOp const& semantic,mlir::DictionaryAttr tiles) {
  auto ownership=tiles.getAs<mlir::StringAttr>("ownership");
  if(!semantic.exact_task_access || !ownership || ownership.getValue()!="tile_per_block")
    throw std::invalid_argument("counted scatter needs exact tile ownership");
  analysis::Granularity geometry;
  for(unsigned i=0;i<semantic.task_space.axes.size();++i) {
    auto const& axis=semantic.task_space.axes[i];
    auto tile=tiles.getAs<mlir::StringAttr>(axis.name);
    if(!tile)throw std::invalid_argument("counted scatter lacks its task granularity");
    geometry.Tile(semantic.name,analysis::UnitTaskOwnershipDimension(semantic,i),
                  analysis::ClosedForm::Parse(tile.getValue().str()));
  }
  auto graph=analysis::Instantiate({{semantic}},geometry);
  if(graph.nodes.size()!=1 || graph.nodes[0].name!=semantic.name)
    throw std::invalid_argument("counted scatter requires one unsplit semantic task");
  return graph.nodes.front();
}
analysis::OperatorNode Task(TileSpaceOp space) {
  if(!space || !space.getSemantic())
    throw std::invalid_argument("counted scatter needs L-sem on both tasks");
  auto task=Task(analysis::DecodeSemanticOp(space.getSemantic()->str()),space.getGranularity());
  if(task.name!=space.getOperatorName())
    throw std::invalid_argument("counted scatter requires one unsplit semantic task");
  return task;
}
analysis::CouplingRelation Identity(analysis::OperatorNode const& task,
                                   analysis::ParamBinding const& binding) {
  auto linear=analysis::LinearizeTaskCoordinates(task,task.Coordinates(),binding,"_tm_phase");
  return linear.ApplyRange(linear.Reverse());
}
bool Equal(analysis::CouplingRelation const& a,analysis::CouplingRelation const& b) {
  return analysis::Contains(a,b) && analysis::Contains(b,a);
}
struct Phase {
  analysis::OperatorNode task;
  analysis::CouplingRelation map;
  std::int64_t stage;
};
struct Endpoint {
  analysis::OperatorNode owner;
  std::vector<Phase> phases;
  analysis::CouplingRelation access;
};
Endpoint AccessEndpoint(mlir::Operation* op,std::string const& tensor,bool write,
                        analysis::ParamBinding const& binding) {
  Endpoint result;
  std::vector<Phase> phases;
  if(auto tile=mlir::dyn_cast_or_null<TileSpaceOp>(op)) {
    auto task=Task(tile);
    phases.push_back({task,Identity(task,binding),std::int64_t(tile.getStage())});
    result.owner=task;
  } else if(auto fused=mlir::dyn_cast_or_null<FusedTileSpaceOp>(op)) {
    // Decode phases without invoking module verification recursively.
    for(auto [text,tiles,map,stage]:llvm::zip(fused.getPhaseSemantics(),
        fused.getPhaseGranularities(),fused.getPhaseMaps(),fused.getPhaseStages())) {
      auto semantic=analysis::DecodeSemanticOp(llvm::cast<mlir::StringAttr>(text).getValue().str());
      phases.push_back({Task(semantic,llvm::cast<mlir::DictionaryAttr>(tiles)),
          llvm::cast<CouplingMapAttr>(map).getMap().BindParams(binding),stage});
    }
    if(phases.empty())throw std::invalid_argument("counted fusion has no phases");
    result.owner=phases.back().task;
    if(!Equal(phases.back().map,Identity(result.owner,binding)) ||
        fused.getTaskCount().getValue().Eval(binding)!=result.owner.Count().Eval(binding,{}))
      throw std::invalid_argument("counted fusion lacks consumer-coordinate ownership");
  } else throw std::invalid_argument("counted scatter references unknown tasks");
  auto domain=Identity(result.owner,binding).Image();
  for(auto const& phase:phases) {
    if(!phase.map.IsSingleValued() || !Equal(phase.map.Reverse().Image(),domain) ||
       !phase.map.Image().IsSubset(Identity(phase.task,binding).Image()))
      throw std::invalid_argument("counted fusion phase exceeds its task domain");
    auto const& access=*phase.task.element_access;
    analysis::CouplingRelation relation;
    auto add=[&](analysis::TensorSpace const& target,analysis::IndexingMap const& map,
                 std::vector<analysis::IndexResult> const& predicates) {
      if(target.name!=tensor)return;
      auto projected=write?analysis::ProjectTaskWrite(access.semantic,phase.task,access.partition,
          target,map,predicates,binding):analysis::ProjectTaskRead(access.semantic,phase.task,
          access.partition,target,map,predicates,binding);
      relation=relation.Union(projected);
    };
    if(write) {
      add(access.semantic.result,access.semantic.result_map,{});
      for(auto const& side:access.semantic.additional_writes)add(side.tensor,side.map,side.nonnegative);
    } else if(!access.semantic.element_reads.empty()) {
      for(auto const& read:access.semantic.element_reads)add(read.tensor,read.map,read.nonnegative);
    } else for(auto const& read:access.semantic.operands)add(read.tensor,read.map,{});
    if(!relation.empty()) {
      result.phases.push_back(phase);
      result.access=result.access.Union(phase.map.ApplyRange(relation));
    }
  }
  if(result.phases.empty())throw std::invalid_argument("counted scatter has no endpoint access");
  if(auto fused=mlir::dyn_cast<FusedTileSpaceOp>(op)) {
    auto stored=(write?fused.getWrites():fused.getReads()).getAs<CouplingMapAttr>(tensor);
    if(!stored || !Equal(stored.getMap().BindParams(binding),result.access))
      throw std::invalid_argument("counted fusion physical access differs from its phases");
  }
  return result;
}
analysis::CountedDependencyForm Contract(TileSpaceOp producer,TileSpaceOp consumer,
    std::string const& tensor,std::vector<unsigned> const& axes,std::string const& source,
    analysis::ParamBinding const& binding) {
  if(producer.getStage()>=consumer.getStage())
    throw std::invalid_argument("counted scatter producer must precede consumer");
  return analysis::BindAlignedCountedScatterDependency(Task(producer),Task(consumer),
      tensor,axes,source,binding);
}
analysis::CountedDependencyForm Contract(mlir::Operation* producer,mlir::Operation* consumer,
    std::string const& tensor,std::vector<unsigned> const& axes,std::string const& source,
    analysis::ParamBinding const& binding,analysis::CouplingRelation* coupling=nullptr) {
  if(auto p=mlir::dyn_cast_or_null<TileSpaceOp>(producer))
    if(auto c=mlir::dyn_cast_or_null<TileSpaceOp>(consumer)) {
      auto result=Contract(p,c,tensor,axes,source,binding);
      if(coupling) {
        auto edges=analysis::CouplingDerivation{}.Derive({{Task(p),Task(c)}},binding);
        for(auto const& edge:edges)*coupling=coupling->Union(edge.C);
      }
      return result;
    }
  auto p=AccessEndpoint(producer,tensor,true,binding);
  auto c=AccessEndpoint(consumer,tensor,false,binding);
  if(p.phases.size()!=1)
    throw std::invalid_argument("counted fusion needs one external writer phase");
  auto const& writer=p.phases.front();
  // Re-executing an external scatter writer would publish a unit twice.
  if(!writer.map.Reverse().IsSingleValued() ||
     !Equal(writer.map.Image(),Identity(writer.task,binding).Image()))
    throw std::invalid_argument("counted fusion scatter writer must execute exactly once");
  for(auto const& reader:c.phases) {
    if(writer.stage>=reader.stage)
      throw std::invalid_argument("counted scatter producer must precede consumer");
    (void)analysis::BindAlignedCountedScatterDependency(writer.task,reader.task,
        tensor,axes,source,binding);
  }
  if(coupling)*coupling=c.access.ApplyRange(p.access.Reverse());
  return analysis::BindCountedTaskDependency(c.owner,c.access,axes,source,binding);
}

}
static mlir::DictionaryAttr EncodeContract(mlir::OpBuilder& builder,
    analysis::CountedDependencyForm const& contract,std::string const& tensor,
    std::vector<unsigned> const& unit_axes,std::string const& binding_source,
    analysis::ParamBinding const& binding);
mlir::DictionaryAttr EncodeBoundCountedScatter(mlir::OpBuilder& builder,
    TileSpaceOp producer,TileSpaceOp consumer,std::string const& tensor,
    std::vector<unsigned> const& unit_axes,std::string const& binding_source,
    analysis::ParamBinding const& binding) {
  analysis::IslReferenceAudit audit(__func__);
  auto contract=Contract(producer,consumer,tensor,unit_axes,binding_source,binding);
  return EncodeContract(builder,contract,tensor,unit_axes,binding_source,binding);
}
static mlir::DictionaryAttr EncodeContract(mlir::OpBuilder& builder,
    analysis::CountedDependencyForm const& contract,std::string const& tensor,
    std::vector<unsigned> const& unit_axes,std::string const& binding_source,
    analysis::ParamBinding const& binding) {
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
mlir::DictionaryAttr RebindBoundCountedScatter(mlir::OpBuilder& builder,
    mlir::Operation* op,analysis::ParamBinding const& binding) {
  analysis::IslReferenceAudit audit(__func__);
  auto edge=mlir::dyn_cast<CouplingOp>(op);
  auto attr=op->getAttrOfType<mlir::DictionaryAttr>("dependency_counted");
  if(!edge || !attr)throw std::invalid_argument("counted rebind lacks its source contract");
  auto tensor=attr.getAs<mlir::StringAttr>("tensor");
  auto source=attr.getAs<mlir::StringAttr>("binding_source");
  auto units=attr.getAs<mlir::DenseI64ArrayAttr>("unit_axes");
  auto contract=attr.getAs<mlir::StringAttr>("binding_contract");
  if(!tensor || !source || !units || contract!="unit_permutation")
    throw std::invalid_argument("counted rebind has an incomplete source contract");
  std::vector<unsigned> axes;
  for(auto axis:units.asArrayRef()) {
    if(axis<0 || std::uint64_t(axis)>std::numeric_limits<unsigned>::max())
      throw std::invalid_argument("counted scatter unit axis overflows");
    axes.push_back(unsigned(axis));
  }
  auto p=mlir::SymbolTable::lookupNearestSymbolFrom(edge,edge.getSrcAttr());
  auto c=mlir::SymbolTable::lookupNearestSymbolFrom(edge,edge.getDstAttr());
  auto rebound=Contract(p,c,tensor.getValue().str(),axes,source.getValue().str(),binding);
  return EncodeContract(builder,rebound,tensor.getValue().str(),axes,source.getValue().str(),binding);
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
  auto p=mlir::SymbolTable::lookupNearestSymbolFrom(coupling,coupling.getSrcAttr());
  auto c=mlir::SymbolTable::lookupNearestSymbolFrom(coupling,coupling.getDstAttr());
  if(!p || !c)throw std::invalid_argument("counted scatter references unknown tasks");
  analysis::CouplingRelation required;
  auto result=Contract(p,c,string("tensor"),axes,string("binding_source"),known,&required);
  if(expected.size()!=result.expected.size())
    throw std::invalid_argument("counted scatter thresholds differ from consumer count");
  for(unsigned target=0;target<result.expected.size();++target)
    if(expected[target]!=result.expected[target])
      throw std::invalid_argument("counted scatter threshold differs from L-sem");
  if(!analysis::Contains(relation.getMap(),result.target_units) ||
     !analysis::Contains(result.target_units,relation.getMap()))
    throw std::invalid_argument("counted scatter contribution image differs from L-sem");
  if(required.empty())throw std::invalid_argument("counted scatter has no producer operand coupling");
  if(!analysis::Contains(coupling.getRelation().getMap().BindParams(known),required))
    throw std::invalid_argument("counted scatter coupling violates I2");
  return result;
}
} // namespace tilemega::dialect
