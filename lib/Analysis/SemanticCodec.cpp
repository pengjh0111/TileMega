// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Support/Json.h>
#include <set>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
using json::Value;
using json::Object;
using json::Array;
std::string String(Value const& v, char const* key) { return v.At(key).AsString(key); }
bool Boolean(Value const& v, char const* key) { return v.At(key).AsBool(key); }
ClosedForm Form(Value const& v, char const* key) { return ClosedForm::Parse(String(v,key)); }
template<class E> E Enum(Value const& v, char const* key, E last) {
  double value=v.At(key).AsNumber(key);
  if (value<0 || value>int(last) || value!=int(value))
    throw std::invalid_argument(std::string("invalid semantic enum: ")+key);
  return static_cast<E>(int(value));
}
template<class T,class F> Value EncodeArray(std::vector<T> const& values,F encode) {
  Array array;
  for (auto const& value:values) array.push_back(encode(value));
  return array;
}
Value EncodeIndex(IndexResult const& index) {
  return Object{{"kind",int(index.kind)}, {"offset",index.offset.ToString()},
    {"span",index.span.ToString()}, {"terms",EncodeArray(index.terms,[](auto const& term) {
      return Value(Object{{"dim",term.dim},{"coefficient",term.coefficient.ToString()},
                          {"group",term.group.ToString()}});
    })}};
}
IndexResult DecodeIndex(Value const& value) {
  IndexResult result;
  result.kind=Enum(value,"kind",IndexResult::Kind::kDataDependent);
  result.offset=Form(value,"offset"); result.span=Form(value,"span");
  for (auto const& term:value.At("terms").AsArray("terms"))
    result.terms.push_back({String(term,"dim"),Form(term,"coefficient"),Form(term,"group")});
  return result;
}
Value EncodeMap(IndexingMap const& map) { return EncodeArray(map.results,EncodeIndex); }
IndexingMap DecodeMap(Value const& value) {
  IndexingMap map;
  for (auto const& index:value.AsArray("indexing map")) map.results.push_back(DecodeIndex(index));
  return map;
}
Value EncodeTensor(TensorSpace const& tensor) {
  return Object{{"name",tensor.name},{"layout",tensor.layout_id},
    {"axes",EncodeArray(tensor.axes,[](auto const& axis) {
      return Value(Object{{"name",axis.name},{"extent",axis.extent.ToString()},
                          {"origin",axis.origin.ToString()},{"runtime",axis.runtime}});
    })}};
}
TensorSpace DecodeTensor(Value const& value) {
  TensorSpace tensor; tensor.name=String(value,"name"); tensor.layout_id=String(value,"layout");
  std::set<std::string> names;
  for (auto const& axis:value.At("axes").AsArray("tensor axes")) {
    TensorAxis dim{String(axis,"name"),Form(axis,"extent"),Form(axis,"origin"),Boolean(axis,"runtime")};
    if (dim.name.empty() || !names.insert(dim.name).second)
      throw std::invalid_argument("duplicate or empty semantic tensor axis");
    tensor.axes.push_back(std::move(dim));
  }
  return tensor;
}
Value EncodeEffect(MemoryEffect const& effect) {
  return Object{{"kind",int(effect.kind)},{"alias",effect.alias_set},{"state",effect.state_object}};
}
MemoryEffect DecodeEffect(Value const& value) {
  return {Enum(value,"kind",EffectKind::kReadWrite),String(value,"alias"),String(value,"state")};
}
Value Encode(SemanticOp const& op) {
  return Object{{"version",1},{"name",op.name},{"kind",int(op.kind)},{"dtype",int(op.dtype)},
    {"arithmetic",op.arithmetic},{"generic",op.generic},
    {"domain",EncodeArray(op.domain,[](auto const& dim) {
      return Value(Object{{"name",dim.name},{"extent",dim.extent.ToString()},
        {"origin",dim.origin.ToString()},{"type",int(dim.type)},{"runtime",dim.runtime}});
    })},{"result",EncodeTensor(op.result)},{"result_map",EncodeMap(op.result_map)},
    {"result_effect",EncodeEffect(op.result_effect)},
    {"operands",EncodeArray(op.operands,[](auto const& operand) {
      return Value(Object{{"producer",operand.producer},{"tensor",EncodeTensor(operand.tensor)},
        {"map",EncodeMap(operand.map)},{"effect",EncodeEffect(operand.effect)}});
    })},{"element_reads",EncodeArray(op.element_reads,[](auto const& read) {
      return Value(Object{{"tensor",EncodeTensor(read.tensor)},{"map",EncodeMap(read.map)},
        {"nonnegative",EncodeArray(read.nonnegative,EncodeIndex)}});
    })},{"reduction",Object{{"dim",op.reduction.dim},{"operator",op.reduction.reduction_operator},
        {"partial",op.reduction.partial_tensor},{"combiner",op.reduction.combiner},
        {"splittable",op.reduction.splittable},
        {"ownership",EncodeArray(op.reduction.ownership,[](auto const& name){return Value(name);})}}}};
}
}  // namespace

std::string EncodeSemanticOp(SemanticOp const& op) { return Encode(op).Dump(0); }

SemanticOp DecodeSemanticOp(std::string const& payload) {
  auto value=json::Parse(payload);
  if (value.At("version").AsNumber("version")!=1)
    throw std::invalid_argument("unsupported semantic payload version");
  SemanticOp op;
  op.name=String(value,"name"); op.kind=Enum(value,"kind",OperatorKind::kGather);
  op.dtype=Enum(value,"dtype",ScalarType::kBF16);
  op.arithmetic=String(value,"arithmetic"); op.generic=Boolean(value,"generic");
  std::set<std::string> names;
  for (auto const& dim:value.At("domain").AsArray("domain")) {
    IterationDim axis{String(dim,"name"),Form(dim,"extent"),Form(dim,"origin"),
                     Enum(dim,"type",IteratorType::kReduction),Boolean(dim,"runtime")};
    if (axis.name.empty() || !names.insert(axis.name).second)
      throw std::invalid_argument("duplicate or empty semantic iteration axis");
    op.domain.push_back(std::move(axis));
  }
  op.result=DecodeTensor(value.At("result")); op.result_map=DecodeMap(value.At("result_map"));
  op.result_effect=DecodeEffect(value.At("result_effect"));
  for (auto const& operand:value.At("operands").AsArray("operands"))
    op.operands.push_back({String(operand,"producer"),DecodeTensor(operand.At("tensor")),
                          DecodeMap(operand.At("map")),DecodeEffect(operand.At("effect"))});
  for (auto const& read:value.At("element_reads").AsArray("element_reads")) {
    ElementRead element{DecodeTensor(read.At("tensor")),DecodeMap(read.At("map")),{}};
    for (auto const& predicate:read.At("nonnegative").AsArray("nonnegative"))
      element.nonnegative.push_back(DecodeIndex(predicate));
    op.element_reads.push_back(std::move(element));
  }
  auto const& reduction=value.At("reduction");
  op.reduction={String(reduction,"dim"),String(reduction,"operator"),String(reduction,"partial"),
                String(reduction,"combiner"),Boolean(reduction,"splittable"),{}};
  for (auto const& name:reduction.At("ownership").AsArray("ownership"))
    op.reduction.ownership.push_back(name.AsString("ownership axis"));
  auto check=[&](IndexingMap const& map,TensorSpace const& tensor) {
    if (map.results.size()!=tensor.axes.size()) throw std::invalid_argument("semantic indexing rank mismatch");
    for (auto const& index:map.results) for (auto const& term:index.terms)
      if (!names.count(term.dim)) throw std::invalid_argument("semantic indexing names an unknown axis");
  };
  check(op.result_map,op.result);
  for (auto const& operand:op.operands) check(operand.map,operand.tensor);
  for (auto const& read:op.element_reads) {
    check(read.map,read.tensor);
    for (auto const& predicate:read.nonnegative) for (auto const& term:predicate.terms)
      if (!names.count(term.dim)) throw std::invalid_argument("semantic predicate names an unknown axis");
  }
  if (op.reduction.splittable && !names.count(op.reduction.dim))
    throw std::invalid_argument("semantic reduction names an unknown axis");
  return op;
}
}  // namespace tilemega::analysis
