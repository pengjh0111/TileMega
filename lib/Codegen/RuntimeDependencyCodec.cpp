// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeDependencyCodec.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tilemega::codegen {
namespace {
std::uint32_t Integer(mlir::DictionaryAttr attr,char const* key,bool positive=false) {
  auto value=attr.getAs<mlir::IntegerAttr>(key);
  if(!value || value.getInt()<(positive?1:0) ||
      std::uint64_t(value.getInt())>std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("invalid retained dependency integer");
  return std::uint32_t(value.getInt());
}
std::string String(mlir::DictionaryAttr attr,char const* key) {
  auto value=attr.getAs<mlir::StringAttr>(key);
  if(!value || value.getValue().empty())
    throw std::invalid_argument("missing retained dependency string");
  return value.getValue().str();
}
analysis::CouplingRelation Relation(mlir::DictionaryAttr attr,char const* key) {
  auto value=attr.getAs<dialect::CouplingMapAttr>(key);
  if(!value)throw std::invalid_argument("missing retained dependency relation");
  return value.getMap();
}
std::vector<std::uint32_t> Array(mlir::DictionaryAttr attr,char const* key,bool positive=false) {
  auto value=attr.getAs<mlir::DenseI64ArrayAttr>(key);
  if(!value)throw std::invalid_argument("missing retained dependency array");
  std::vector<std::uint32_t> result;
  for(auto item:value.asArrayRef()) {
    if(item<(positive?1:0) || std::uint64_t(item)>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid retained dependency array element");
    result.push_back(std::uint32_t(item));
  }
  return result;
}
mlir::DenseI64ArrayAttr Array(mlir::OpBuilder& b,std::vector<std::uint32_t> const& values) {
  llvm::SmallVector<std::int64_t> integers(values.begin(),values.end());
  return b.getDenseI64ArrayAttr(integers);
}
}
mlir::DictionaryAttr EncodeRuntimeDependency(mlir::OpBuilder& b,DependencyRecord const& edge) {
  mlir::NamedAttrList result;
  result.set("producer",b.getI64IntegerAttr(edge.producer));
  result.set("consumer",b.getI64IntegerAttr(edge.consumer));
  result.set("window",b.getStringAttr(edge.window.ToString()));
  if(edge.producer_main)result.set("producer_main",b.getBoolAttr(true));
  if(edge.consumer_done)result.set("consumer_done",b.getBoolAttr(true));
  if(edge.phase_window) {
    result.set("phase_window",b.getStringAttr(edge.phase_window->ToString()));
    result.set("phase_tiles",b.getI64IntegerAttr(edge.phase_tiles));
  }
  auto map=[&](auto const& relation){return dialect::CouplingMapAttr::get(b.getContext(),relation);};
  if(edge.table) {
    auto const& table=*edge.table;
    std::vector<std::uint32_t> intervals;
    for(auto const& item:table.intervals){intervals.push_back(item.first);intervals.push_back(item.count);}
    result.set("table",b.getDictionaryAttr({
      b.getNamedAttr("producers",b.getI64IntegerAttr(table.producers)),
      b.getNamedAttr("consumers",b.getI64IntegerAttr(table.consumers)),
      b.getNamedAttr("stride",b.getI64IntegerAttr(table.stride)),
      b.getNamedAttr("intervals",Array(b,intervals)),
      b.getNamedAttr("linear_relation",map(table.linear_relation)),
      b.getNamedAttr("encoded_relation",map(table.encoded_relation))}));
  }
  if(edge.counted) {
    auto const& counted=*edge.counted;
    std::vector<std::uint32_t> axes(counted.unit_axes.begin(),counted.unit_axes.end());
    result.set("counted",b.getDictionaryAttr({
      b.getNamedAttr("producers",b.getI64IntegerAttr(counted.producers)),
      b.getNamedAttr("tensor",b.getStringAttr(counted.tensor)),
      b.getNamedAttr("unit_axes",Array(b,axes)),
      b.getNamedAttr("binding_source",b.getStringAttr(counted.contributions.binding_source)),
      b.getNamedAttr("expected",Array(b,counted.contributions.expected)),
      b.getNamedAttr("target_units",map(counted.contributions.target_units)),
      b.getNamedAttr("conservative_relation",map(counted.conservative_relation))}));
  }
  return result.getDictionary(b.getContext());
}
DependencyRecord DecodeRuntimeDependency(mlir::DictionaryAttr attr) {
  analysis::IslReferenceAudit audit(__func__);
  if(!attr)throw std::invalid_argument("retained dependency is not a dictionary");
  DependencyRecord result{Integer(attr,"producer"),Integer(attr,"consumer"),
      analysis::ParseWaitWindow(String(attr,"window"))};
  for(auto const& field:std::vector<std::pair<char const*,bool*>>{
      {"producer_main",&result.producer_main},{"consumer_done",&result.consumer_done}})
    if(auto value=attr.get(field.first)) {
      auto flag=mlir::dyn_cast<mlir::BoolAttr>(value);
      if(!flag)throw std::invalid_argument("invalid retained dependency endpoint");
      *field.second=flag.getValue();
    }
  if(result.consumer<=result.producer)
    throw std::invalid_argument("retained dependency is not producer-before-consumer");
  if(attr.get("phase_window")) {
    result.phase_window=analysis::ParseWaitWindow(String(attr,"phase_window"));
    auto tiles=Integer(attr,"phase_tiles",true);
    if(tiles>std::uint32_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("retained phase geometry overflows");
    result.phase_tiles=int(tiles);
  }
  if(attr.get("table")) {
    auto table=attr.getAs<mlir::DictionaryAttr>("table");
    if(!table)throw std::invalid_argument("retained table is not a dictionary");
    auto p=Integer(table,"producers",true),c=Integer(table,"consumers",true);
    analysis::DependencyTable derived;
    derived.producers=p;derived.consumers=c;derived.stride=Integer(table,"stride");
    derived.linear_relation=Relation(table,"linear_relation");
    auto intervals=Array(table,"intervals");
    if(intervals.size()%2)throw std::invalid_argument("retained dependency interval array is odd");
    for(std::size_t i=0;i<intervals.size();i+=2)derived.intervals.push_back({intervals[i],intervals[i+1]});
    analysis::ValidateDependencyTableLinear(derived);
    derived.encoded_relation=derived.linear_relation;
    auto encoded=Relation(table,"encoded_relation");
    if(!encoded.IsSubset(derived.encoded_relation) || !derived.encoded_relation.IsSubset(encoded))
      throw std::invalid_argument("retained dependency table changes its coupling");
    result.table=std::move(derived);
  }
  if(attr.get("counted")) {
    auto counted=attr.getAs<mlir::DictionaryAttr>("counted");
    if(!counted || result.table || result.phase_window)
      throw std::invalid_argument("retained counted dependency has competing wait policies");
    CountedWaitRecord record;
    record.producers=Integer(counted,"producers",true);
    record.tensor=String(counted,"tensor");
    auto axes=Array(counted,"unit_axes");record.unit_axes.assign(axes.begin(),axes.end());
    if(axes.empty() || !std::is_sorted(axes.begin(),axes.end()) ||
        std::adjacent_find(axes.begin(),axes.end())!=axes.end())
      throw std::invalid_argument("retained counted unit axes are not distinct and sorted");
    record.contributions.binding_source=String(counted,"binding_source");
    record.contributions.expected=Array(counted,"expected",true);
    if(record.contributions.expected.empty())throw std::invalid_argument("retained counted thresholds are empty");
    record.contributions.target_units=Relation(counted,"target_units");
    auto names=record.contributions.target_units.DomainDimNames();
    if(names.size()!=1 || record.contributions.target_units.RangeDimNames().size()!=axes.size())
      throw std::invalid_argument("retained counted unit rank differs from its ownership");
    auto const consumers=record.contributions.expected.size();
    auto domain=analysis::CouplingRelation::FromIslText("{ [c] -> [c] : 0<=c<"+std::to_string(consumers)+" }");
    auto owned=record.contributions.target_units.Reverse().Image().ImageIdentity();
    if(!domain.IsSubset(owned) || !owned.IsSubset(domain))
      throw std::invalid_argument("retained counted units omit consumer tasks");
    std::vector<analysis::ParamBinding> tasks(consumers);
    for(unsigned i=0;i<consumers;++i)tasks[i].Bind(names[0],i);
    auto counts=record.contributions.target_units.BoundTaskCard().EvalPoints({},tasks);
    for(unsigned i=0;i<consumers;++i)if(counts[i]!=record.contributions.expected[i])
      throw std::invalid_argument("retained counted threshold differs from contribution image");
    record.conservative_relation=Relation(counted,"conservative_relation");
    // Counted waits encode target counts, not producer intervals. Validate the
    // I2 task domains without constructing an unused (potentially huge) table.
    analysis::ValidateLinearTaskBounds(record.conservative_relation,record.producers,consumers);
    result.counted=std::move(record);
  }
  return result;
}
} // namespace tilemega::codegen
