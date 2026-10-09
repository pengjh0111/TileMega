// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/VirtualTaskBinding.h>
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
  Object encoded{{"kind",int(index.kind)}, {"offset",index.offset.ToString()},
    {"span",index.span.ToString()}, {"terms",EncodeArray(index.terms,[](auto const& term) {
      Object encoded{{"dim",term.dim},{"coefficient",term.coefficient.ToString()},
                     {"group",term.group.ToString()}};
      if (!term.shift.IsLiteral(0)) encoded.emplace_back("shift", term.shift.ToString());
      return Value(std::move(encoded));
    })}};
  if (!index.binding_source.empty()) encoded.emplace_back("binding_source", index.binding_source);
  if(!index.request_dims.empty())encoded.emplace_back("request_dims",
      EncodeArray(index.request_dims,[](auto const& dim){return Value(dim);}));
  if(!index.outer_divisor.IsLiteral(1))
    encoded.emplace_back("outer_divisor",index.outer_divisor.ToString());
  return encoded;
}
IndexResult DecodeIndex(Value const& value) {
  IndexResult result;
  result.kind=Enum(value,"kind",IndexResult::Kind::kDataDependent);
  result.offset=Form(value,"offset"); result.span=Form(value,"span");
  if(auto const* divisor=value.Find("outer_divisor")) {
    result.outer_divisor=ClosedForm::Parse(divisor->AsString("outer_divisor"));
    if(result.kind!=IndexResult::Kind::kAffine ||
       (result.outer_divisor.IsConstant() && result.outer_divisor.Eval({},{})<=0))
      throw std::invalid_argument("outer floor requires an affine index and positive divisor");
  }
  if (auto const* source = value.Find("binding_source")) {
    result.binding_source = source->AsString("binding_source");
    if (result.kind != IndexResult::Kind::kDataDependent)
      throw std::invalid_argument("binding source requires a data-dependent index");
  }
  if(auto const* requests=value.Find("request_dims")) {
    if(result.kind!=IndexResult::Kind::kDataDependent || result.binding_source.empty())
      throw std::invalid_argument("logical requests require a bound data-dependent index");
    std::set<std::string> seen;
    for(auto const& dim:requests->AsArray("request_dims")) {
      auto name=dim.AsString("request dimension");
      if(name.empty() || !seen.insert(name).second)
        throw std::invalid_argument("logical request dimensions must be distinct");
      result.request_dims.push_back(std::move(name));
    }
    if(result.request_dims.empty())throw std::invalid_argument("logical request dimensions are empty");
  }
  for (auto const& term:value.At("terms").AsArray("terms")) {
    IndexResult::Term decoded{String(term,"dim"),Form(term,"coefficient"),Form(term,"group")};
    if (auto const* shift = term.Find("shift")) decoded.shift = ClosedForm::Parse(shift->AsString("shift"));
    result.terms.push_back(std::move(decoded));
  }
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
  (void)VirtualBindings(op);
  Object encoded{{"version",1},{"name",op.name},{"kind",int(op.kind)},{"dtype",int(op.dtype)},
    {"arithmetic",op.arithmetic},{"generic",op.generic},
    {"domain",EncodeArray(op.domain,[](auto const& dim) {
      Object encoded{{"name",dim.name},{"extent",dim.extent.ToString()},
        {"origin",dim.origin.ToString()},{"type",int(dim.type)},{"runtime",dim.runtime}};
      if (dim.capacity) {
        encoded.emplace_back("capacity", dim.capacity->ToString());
        encoded.emplace_back("binding_source", dim.binding_source);
        encoded.emplace_back("binding_requirement", dim.binding_requirement);
      }
      return Value(std::move(encoded));
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
  if (!op.additional_writes.empty())
    encoded.emplace_back("additional_writes",
        EncodeArray(op.additional_writes,[](auto const& write) {
          return Value(Object{{"tensor",EncodeTensor(write.tensor)},
              {"map",EncodeMap(write.map)},
              {"nonnegative",EncodeArray(write.nonnegative,EncodeIndex)},
              {"effect",EncodeEffect(write.effect)}});
        }));
  if (op.exact_task_access) {
    encoded.emplace_back("exact_task_access", true);
    encoded.emplace_back("task_space", EncodeTensor(op.task_space));
    encoded.emplace_back("task_map", EncodeMap(op.task_map));
  }
  if(!op.domain_nonnegative.empty())
    encoded.emplace_back("domain_nonnegative",EncodeArray(op.domain_nonnegative,EncodeIndex));
  return encoded;
}
}  // namespace

std::string EncodeSemanticOp(SemanticOp const& op) { return Encode(op).Dump(0); }

std::string EncodeTaskReductionIndex(TaskReductionIndex const& index) {
  return Value(Object{{"version",1},{"index",EncodeIndex(index.index)},
      {"capacity",index.capacity.ToString()},{"issued_width",index.issued_width.ToString()}}).Dump(0);
}
TaskReductionIndex DecodeTaskReductionIndex(std::string const& payload) {
  auto value=json::Parse(payload);
  if(value.At("version").AsNumber("version")!=1)
    throw std::invalid_argument("unsupported task reduction index version");
  TaskReductionIndex index{DecodeIndex(value.At("index")),Form(value,"capacity"),Form(value,"issued_width")};
  if(index.index.kind!=IndexResult::Kind::kAffine ||
     (index.capacity.IsConstant() && index.capacity.Eval({},{})<=0) ||
     (index.issued_width.IsConstant() && index.issued_width.Eval({},{})<=0))
    throw std::invalid_argument("invalid task reduction index payload");
  return index;
}

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
    if (auto const* capacity = dim.Find("capacity")) axis.capacity = ClosedForm::Parse(capacity->AsString("capacity"));
    if (auto const* source = dim.Find("binding_source")) axis.binding_source = source->AsString("binding_source");
    if (auto const* requirement = dim.Find("binding_requirement")) axis.binding_requirement = requirement->AsString("binding_requirement");
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
  if (auto const* writes=value.Find("additional_writes"))
    for (auto const& write:writes->AsArray("additional_writes")) {
      ElementWrite element{DecodeTensor(write.At("tensor")),
          DecodeMap(write.At("map")),{},DecodeEffect(write.At("effect"))};
      for (auto const& predicate:write.At("nonnegative").AsArray("nonnegative"))
        element.nonnegative.push_back(DecodeIndex(predicate));
      op.additional_writes.push_back(std::move(element));
    }
  auto const& reduction=value.At("reduction");
  op.reduction={String(reduction,"dim"),String(reduction,"operator"),String(reduction,"partial"),
                String(reduction,"combiner"),Boolean(reduction,"splittable"),{}};
  for (auto const& name:reduction.At("ownership").AsArray("ownership"))
    op.reduction.ownership.push_back(name.AsString("ownership axis"));
  if (auto const* exact = value.Find("exact_task_access")) {
    op.exact_task_access = exact->AsBool("exact_task_access");
    if (op.exact_task_access) {
      op.task_space = DecodeTensor(value.At("task_space"));
      op.task_map = DecodeMap(value.At("task_map"));
    }
  }
  if(auto const* predicates=value.Find("domain_nonnegative"))
    for(auto const& predicate:predicates->AsArray("domain_nonnegative"))
      op.domain_nonnegative.push_back(DecodeIndex(predicate));
  if(!op.exact_task_access && !op.domain_nonnegative.empty())
    throw std::invalid_argument("iteration predicates require exact task access");
  for(auto const& predicate:op.domain_nonnegative) {
    if(predicate.kind!=IndexResult::Kind::kAffine)
      throw std::invalid_argument("iteration predicate must be quasi-affine");
    for(auto const& term:predicate.terms)
      if(!names.count(term.dim))throw std::invalid_argument("iteration predicate names an unknown axis");
  }
  auto check=[&](IndexingMap const& map,TensorSpace const& tensor) {
    if (map.results.size()!=tensor.axes.size()) throw std::invalid_argument("semantic indexing rank mismatch");
    for (auto const& index:map.results) for (auto const& term:index.terms)
      if (!names.count(term.dim)) throw std::invalid_argument("semantic indexing names an unknown axis");
    for(auto const& index:map.results)for(auto const& dim:index.request_dims)
      if(!names.count(dim))throw std::invalid_argument("logical request names an unknown iteration axis");
  };
  check(op.result_map,op.result);
  if (op.exact_task_access) check(op.task_map, op.task_space);
  for (auto const& operand:op.operands) check(operand.map,operand.tensor);
  for (auto const& read:op.element_reads) {
    check(read.map,read.tensor);
    for (auto const& predicate:read.nonnegative) for (auto const& term:predicate.terms)
      if (!names.count(term.dim)) throw std::invalid_argument("semantic predicate names an unknown axis");
  }
  for (auto const& write:op.additional_writes) {
    check(write.map,write.tensor);
    for (auto const& predicate:write.nonnegative)
      for (auto const& term:predicate.terms)
        if (!names.count(term.dim))
          throw std::invalid_argument("semantic write predicate names an unknown axis");
  }
  if (op.reduction.splittable && !names.count(op.reduction.dim))
    throw std::invalid_argument("semantic reduction names an unknown axis");
  (void)VirtualBindings(op);
  return op;
}
}  // namespace tilemega::analysis
