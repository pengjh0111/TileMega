// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/DnnStorage.h>
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Analysis/BoundDependencyForm.h>
#include <tilemega/Dialect/CouplingGraph/BoundDependency.h>
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/VirtualTaskBinding.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Solver/MemoryPlan.h>
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tilemega::frontend {
namespace {

std::pair<long, long> parseRange(std::string const& text) {
  static std::regex const pattern(R"(VR\[(-?[0-9]+),\s*(-?[0-9]+)\])");
  std::smatch match;
  if (!std::regex_match(text, match, pattern))
    throw std::invalid_argument("unsupported ShapeEnv range: " + text);
  return {std::stol(match[1].str()), std::stol(match[2].str())};
}

std::vector<std::string> extractGuardTerms(
    std::string const& side, std::vector<std::vector<std::string>> const& shapes) {
  static std::regex const reference(
      R"(L\['flat_args'\]\[([0-9]+)\]\.size\(\)\[([0-9]+)\])");
  std::vector<std::string> terms;
  for (auto it = std::sregex_iterator(side.begin(), side.end(), reference);
       it != std::sregex_iterator(); ++it) {
    std::size_t input = std::stoul((*it)[1].str());
    std::size_t dimension = std::stoul((*it)[2].str());
    if (input >= shapes.size() || dimension >= shapes[input].size())
      throw std::invalid_argument("guard references an unknown input dimension");
    terms.push_back(shapes[input][dimension]);
  }
  std::sort(terms.begin(), terms.end());
  static std::regex const scalar(R"(^\s*\(?\s*(-?[0-9]+)\s*\)?\s*$)");
  std::smatch literal;
  if (std::regex_match(side, literal, scalar)) terms.push_back(literal[1].str());
  return terms;
}

analysis::ClosedForm sumTerms(std::vector<std::string> const& terms) {
  auto result = analysis::ClosedForm::Constant(0);
  for (auto const& term : terms) result = result + analysis::ClosedForm::Parse(term);
  return result;
}

std::string findRoot(std::unordered_map<std::string, std::string>& parent,
                     std::string value) {
  auto found = parent.find(value);
  if (found == parent.end()) return parent.emplace(value, value).first->second;
  if (found->second != value) found->second = findRoot(parent, found->second);
  return found->second;
}

void unite(std::unordered_map<std::string, std::string>& parent,
           std::string a, std::string b) {
  a = findRoot(parent, a); b = findRoot(parent, b);
  if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

std::vector<std::string> readStrings(llvm::json::Array const* array) {
  std::vector<std::string> result;
  if (!array) return result;
  for (auto const& item : *array)
    if (auto value = item.getAsString()) result.push_back(value->str());
  return result;
}

llvm::StringSet<> const& known();

FxArgument ReadFxArgument(llvm::json::Value const& value, unsigned depth=0) {
  if(depth>64)throw std::invalid_argument("FX argument nesting exceeds 64");
  auto* object=value.getAsObject();
  if(!object || !object->getString("t") || !object->get("v"))
    throw std::invalid_argument("FX argument requires typed t/v fields");
  auto tag=*object->getString("t");auto const& data=*object->get("v");
  FxArgument result;using Kind=FxArgument::Kind;
  if(tag=="none") {
    if(!data.getAsNull())throw std::invalid_argument("FX none is not null");
  }else if(tag=="int") {
    auto integer=data.getAsInteger();
    if(!integer)throw std::invalid_argument("FX int is not an integer");
    result.kind=Kind::kInt;result.integer=*integer;
  }else if(tag=="bool") {
    auto boolean=data.getAsBoolean();
    if(!boolean)throw std::invalid_argument("FX bool is not a boolean");
    result.kind=Kind::kBool;result.boolean=*boolean;
  }else if(tag=="float") {
    result.kind=Kind::kFloat;
    if(auto number=data.getAsNumber())result.real=*number;
    else if(auto special=data.getAsString()) {
      if(*special=="+inf")result.real=std::numeric_limits<double>::infinity();
      else if(*special=="-inf")result.real=-std::numeric_limits<double>::infinity();
      else if(*special=="nan")result.real=std::numeric_limits<double>::quiet_NaN();
      else throw std::invalid_argument("unknown nonfinite FX float");
    }else throw std::invalid_argument("FX float is not numeric");
  }else if(tag=="list") {
    auto* array=data.getAsArray();
    if(!array)throw std::invalid_argument("FX list is not an array");
    result.kind=Kind::kList;
    for(auto const& item:*array)result.items.push_back(ReadFxArgument(item,depth+1));
  }else {
    if(tag=="node")result.kind=Kind::kNode;
    else if(tag=="str")result.kind=Kind::kString;
    else if(tag=="dtype")result.kind=Kind::kDtype;
    else if(tag=="device")result.kind=Kind::kDevice;
    else if(tag=="layout")result.kind=Kind::kLayout;
    else if(tag=="memory_format")result.kind=Kind::kMemoryFormat;
    else if(tag=="symbol")result.kind=Kind::kSymbol;
    else throw std::invalid_argument("unknown FX argument tag: "+tag.str());
    auto text=data.getAsString();
    if(!text)throw std::invalid_argument("FX textual argument is not a string");
    result.text=text->str();
  }
  return result;
}

FxConstant ReadFxConstant(llvm::json::Object const& object) {
  FxConstant result;result.present=true;
  auto dtype=object.getString("dtype");auto* shape=object.getArray("shape");
  if(!dtype || !shape)throw std::invalid_argument("FX constant requires dtype and shape");
  result.dtype=dtype->str();
  for(auto const& dimension:*shape) {
    auto integer=dimension.getAsInteger();
    if(!integer || *integer<0)throw std::invalid_argument("invalid constant dimension");
    result.shape.push_back(*integer);
  }
  std::uint64_t elements=1;
  if(std::find(result.shape.begin(),result.shape.end(),0)==result.shape.end())
    for(auto dimension:result.shape) {
      if(elements>(1u<<20)/std::uint64_t(dimension))
        throw std::invalid_argument("FX constant exceeds element limit");
      elements*=std::uint64_t(dimension);
    }
  if(auto* scalar=object.get("scalar")) {
    result.has_scalar=true;result.scalar=ReadFxArgument(*scalar);
  }
  if(auto all_true=object.getBoolean("all_true")) {
    result.has_all_true=true;result.all_true=*all_true;
  }
  if(auto* summary=object.getObject("all_equal")) {
    auto equal=summary->getBoolean("equal");auto* value=summary->get("value");
    if(!equal || !value)throw std::invalid_argument("invalid constant equality summary");
    result.has_all_equal=true;result.all_equal=*equal;
    if(auto boolean=value->getAsBoolean()) {
      result.equal_value.kind=FxArgument::Kind::kBool;result.equal_value.boolean=*boolean;
    }else if(auto integer=value->getAsInteger()) {
      result.equal_value.kind=FxArgument::Kind::kInt;result.equal_value.integer=*integer;
    }else throw std::invalid_argument("constant equality value is not bool/int");
  }
  if(auto data=object.getString("data_base64"))result.data_base64=data->str();
  if(auto order=object.getString("byte_order"))result.byte_order=order->str();
  return result;
}

/// The composite spelling and its Core ATen decomposition classify the same
/// way, so normalization does not move an operator between kinds.
std::string classify(llvm::StringRef target) {
  // An operator no rule covers is not guessed at: it becomes its own task
  // space with conservative semantics (§0.1 prefers degradation to refusal).
  if (!known().contains(target)) return "generic";
  if (target == "aten.linear.default" || target == "aten.matmul.default" ||
      target == "aten.outer.default" || target == "aten.mm.default" ||
      target == "aten.bmm.default") return "gemm";
  if (target.starts_with("aten.mean.") || target.starts_with("aten.softmax.") ||
      target.starts_with("aten._softmax.") || target.starts_with("aten.sum."))
    return "reduction";
  if (target.starts_with("aten.transpose.") || target == "aten.contiguous.default" ||
      target.starts_with("aten.permute.")) return "transpose";
  if (target == "aten.view.default" || target == "aten.clone.default" ||
      target == "aten.alias.default" || target == "aten._to_copy.default")
    return "view";
  if (target == "<built-in function getitem>" || target.starts_with("aten.chunk.") ||
      target.starts_with("aten.split_with_sizes.") ||
      target.starts_with("aten.split.") || target.starts_with("aten.slice."))
    return "slice";
  if (target.starts_with("aten.cat.")) return "concat";
  if (target.starts_with("aten.unsqueeze.") ||
      target.starts_with("aten.repeat_interleave.") ||
      target.starts_with("aten.expand.") || target.starts_with("aten.squeeze."))
    return "broadcast";
  if (target.starts_with("aten._assert_tensor_metadata.") ||
      target.starts_with("aten.sym_size.") || target.starts_with("aten.to."))
    return "frontend";
  return "elementwise";
}

/// The operators a classification rule covers. It is not an admission gate:
/// anything outside it still imports, as one generic task space.
llvm::StringSet<> const& known() {
  static llvm::StringSet<> values = [] {
    llvm::StringSet<> set;
    for (char const* value : {
      "<built-in function add>", "<built-in function getitem>",
      "aten._assert_tensor_metadata.default", "aten.add.Tensor",
      "aten.arange.default", "aten.arange.start", "aten.cat.default",
      "aten.chunk.default", "aten.contiguous.default", "aten.cos.default",
      "aten.div.Tensor", "aten.gt.Tensor", "aten.linear.default",
      "aten.masked_fill.Scalar", "aten.matmul.default", "aten.mean.dim",
      "aten.mul.Tensor", "aten.neg.default", "aten.outer.default",
      "aten.pow.Tensor_Scalar", "aten.repeat_interleave.self_int",
      "aten.rsqrt.default", "aten.silu.default", "aten.sin.default",
      "aten.softmax.int", "aten.sym_size.int", "aten.to.dtype",
      "aten.transpose.int", "aten.unsqueeze.default", "aten.view.default",
      // Core ATen spellings, so a graph normalized by
      // ExportedProgram.run_decompositions() classifies identically.
      "aten._softmax.default", "aten._to_copy.default", "aten.alias.default",
      "aten.arange.start_step", "aten.bmm.default", "aten.clone.default",
      "aten.expand.default", "aten.mm.default", "aten.permute.default",
      "aten.scalar_tensor.default", "aten.sigmoid.default",
      "aten.slice.Tensor", "aten.split_with_sizes.default",
      "aten.squeeze.dim", "aten.sum.dim_IntList", "aten.where.self"})
      set.insert(value);
    return set;
  }();
  return values;
}

mlir::DictionaryAttr dict(mlir::Builder& builder,
                          std::initializer_list<mlir::NamedAttribute> fields) {
  return builder.getDictionaryAttr(fields);
}

/// The CG task kind for a lifted operator.  It follows the role the semantic
/// lifting recognised, never the FX target string: an operator-granularity
/// task space is a fusion of several call_functions and has no single target.
llvm::StringRef taskKindOf(OpRole role) {
  switch (role) {
    case OpRole::kNorm: return "rmsnorm";
    // The per-head normalization is the same task kind; what distinguishes it
    // is the granularity, which the task space carries beside this.
    case OpRole::kQKNorm: return "rmsnorm";
    case OpRole::kEmbedding: return "embedding";
    case OpRole::kQkvProjection:
    case OpRole::kProjection: return "gemm";
    case OpRole::kRoPE: return "rope";
    case OpRole::kKVAppend: return "kvappend";
    case OpRole::kAttention: return "attention";
    case OpRole::kServingQkv:
    case OpRole::kServingSwiGlu:
    case OpRole::kServingArgmaxPartial: return "gemm";
    case OpRole::kServingFusedAttention: return "fused_attention";
    case OpRole::kServingMerge: return "attention_merge";
    case OpRole::kServingArgmax: return "argmax_reduce";
    case OpRole::kActivation:
    case OpRole::kResidualAdd: return "elementwise";
    case OpRole::kGeneric: return "generic";
    case OpRole::kLayerNorm: return "layernorm";
    case OpRole::kEmbeddingSum: return "embedding_sum";
    case OpRole::kLayoutConvert: return "layout_convert";
    case OpRole::kPool: return "pool";
    case OpRole::kGlobalPoolReduce: return "global_pool_reduce";
    case OpRole::kEncoderAttention: return "encoder_attention";
    case OpRole::kDepthwiseConv: return "depthwise_conv";
    case OpRole::kMoERouting: return "moe_topk";
    case OpRole::kMoECombine: return "moe_combine";
  }
  return "generic";
}

/// A derived extent as a metric.  The granularity is substituted first
/// because isl needs a literal floor/ceildiv divisor; theta is deliberately
/// left free so a workload dimension survives as an isl parameter (I1).
analysis::QuasiPolynomial metricOf(analysis::ClosedForm const& value,
                                   analysis::ParamBinding const& granularity) {
  analysis::ClosedForm reduced = value.Substitute(granularity);
  if (reduced.HasPiecewise()) return analysis::QuasiPolynomial::FromClosedForm(reduced);
  std::vector<std::string> free = reduced.FreeSymbols();
  std::sort(free.begin(), free.end());
  free.erase(std::unique(free.begin(), free.end()), free.end());
  std::string prefix;
  if (!free.empty()) prefix = "[" + llvm::join(free, ", ") + "] -> ";
  return analysis::QuasiPolynomial::FromIslText(
      prefix + "{ (" + reduced.ToIslText() + ") }");
}

llvm::StringRef taskKindName(PlanTaskKind kind) {
  switch (kind) {
    case PlanTaskKind::kGemm: return "kGemm";
    case PlanTaskKind::kRMSNorm: return "kRMSNorm";
    case PlanTaskKind::kRoPE: return "kRoPE";
    case PlanTaskKind::kKVAppend: return "kKVAppend";
    case PlanTaskKind::kElementwise: return "kElementwise";
    case PlanTaskKind::kAttention: return "kAttention";
    case PlanTaskKind::kAdd: return "kAdd";
    case PlanTaskKind::kEmbedding: return "kEmbedding";
    case PlanTaskKind::kQKNorm: return "kQKNorm";
    case PlanTaskKind::kFusedAttention: return "kFusedAttention";
    case PlanTaskKind::kAttentionMerge: return "kAttentionMerge";
    case PlanTaskKind::kArgmaxReduce: return "kArgmaxReduce";
    case PlanTaskKind::kDepthwiseConv: return "kDepthwiseConv";
    case PlanTaskKind::kPool: return "kPool";
    case PlanTaskKind::kGlobalPoolReduce: return "kGlobalPoolReduce";
    case PlanTaskKind::kLayerNorm: return "kLayerNorm";
    case PlanTaskKind::kEncoderAttention: return "kEncoderAttention";
    case PlanTaskKind::kEmbeddingSum: return "kEmbeddingSum";
    case PlanTaskKind::kDwPwFused: return "kDwPwFused";
    case PlanTaskKind::kMoETopK: return "kMoETopK";
    case PlanTaskKind::kMoECombine: return "kMoECombine";
    case PlanTaskKind::kLayoutConvert: return "kLayoutConvert";
  }
  llvm_unreachable("unknown plan task kind");
}

mlir::DictionaryAttr modelPlanAttr(mlir::Builder& builder,
                                   ModelPlan const& plan,
                                   std::vector<std::uint8_t> const& written) {
  ValidateDmModelPlan(plan);
  llvm::SmallVector<mlir::Attribute> buffers, gemms, stages, outputs;
  for (auto const& buffer : plan.buffers) {
    std::size_t index = buffers.size();
    llvm::StringRef source = "zero";
    if (buffer.source == PlanBuffer::Source::kFixture) source = "fixture";
    if (buffer.source == PlanBuffer::Source::kWeight) source = "weight";
    llvm::SmallVector<mlir::NamedAttribute> fields = {
        builder.getNamedAttr("name", builder.getStringAttr(buffer.name)),
        builder.getNamedAttr("constant", builder.getI64IntegerAttr(buffer.constant)),
        builder.getNamedAttr("per_seq", builder.getI64IntegerAttr(buffer.per_seq)),
        builder.getNamedAttr("per_past", builder.getI64IntegerAttr(buffer.per_past)),
        builder.getNamedAttr("per_total", builder.getI64IntegerAttr(buffer.per_total)),
        builder.getNamedAttr("source", builder.getStringAttr(source)),
        builder.getNamedAttr("file", builder.getStringAttr(buffer.file)),
        builder.getNamedAttr("no_producer", builder.getBoolAttr(
            index >= written.size() || !written[index]))};
    if (plan.dm || buffer.per_batch || buffer.role != "internal" ||
        !buffer.external_name.empty() || !buffer.pack_json.empty()) {
      fields.push_back(builder.getNamedAttr(
          "per_batch", builder.getI64IntegerAttr(buffer.per_batch)));
      fields.push_back(builder.getNamedAttr(
          "dtype", builder.getStringAttr(buffer.dtype)));
      fields.push_back(builder.getNamedAttr(
          "role", builder.getStringAttr(buffer.role)));
      fields.push_back(builder.getNamedAttr(
          "external_name", builder.getStringAttr(buffer.external_name)));
      fields.push_back(builder.getNamedAttr(
          "pack_json", builder.getStringAttr(buffer.pack_json)));
    }
    if(plan.dm)
      fields.push_back(builder.getNamedAttr("dm_layout",EncodeDm(builder,buffer.layout)));
    if(plan.dm && buffer.arena_offset!=~std::uint64_t(0))
      fields.push_back(builder.getNamedAttr("dm_arena_offset",builder.getI64IntegerAttr(buffer.arena_offset)));
    buffers.push_back(builder.getDictionaryAttr(fields));
  }
  for (auto const& gemm : plan.gemms) {
    llvm::SmallVector<mlir::NamedAttribute> fields = {
        builder.getNamedAttr("n", builder.getI64IntegerAttr(gemm.n)),
        builder.getNamedAttr("k", builder.getI64IntegerAttr(gemm.k)),
        builder.getNamedAttr("a", builder.getI64IntegerAttr(gemm.a)),
        builder.getNamedAttr("b", builder.getI64IntegerAttr(gemm.b)),
        builder.getNamedAttr("c", builder.getI64IntegerAttr(gemm.c)),
        builder.getNamedAttr("d", builder.getI64IntegerAttr(gemm.d)),
        builder.getNamedAttr("beta", builder.getF32FloatAttr(gemm.beta))};
    if(gemm.norm_ss!=0xffffffffu)
      fields.push_back(builder.getNamedAttr("norm_ss",builder.getI64IntegerAttr(gemm.norm_ss)));
    if(gemm.ss_out!=0xffffffffu)
      fields.push_back(builder.getNamedAttr("ss_out",builder.getI64IntegerAttr(gemm.ss_out)));
    if (gemm.epilogue != PlanGemm::Epilogue::kStore) {
      llvm::StringRef epilogue = "store";
      if (gemm.epilogue == PlanGemm::Epilogue::kResidual) epilogue = "residual";
      if (gemm.epilogue == PlanGemm::Epilogue::kSwiGLU) epilogue = "swiglu";
      if (gemm.epilogue == PlanGemm::Epilogue::kArgmaxPartial)
        epilogue = "argmax_partial";
      fields.push_back(builder.getNamedAttr(
          "epilogue", builder.getStringAttr(epilogue)));
      fields.push_back(builder.getNamedAttr(
          "interleave_u", builder.getI64IntegerAttr(gemm.interleave_u)));
      if (gemm.partial_tile_n)
        fields.push_back(builder.getNamedAttr(
            "partial_tile_n", builder.getI64IntegerAttr(gemm.partial_tile_n)));
    }
    if(plan.dm) {
      fields.push_back(builder.getNamedAttr("dm_access",EncodeDm(builder,gemm.access)));
      fields.push_back(builder.getNamedAttr("dm_chain",EncodeDm(builder,gemm.chain)));
    }
    gemms.push_back(builder.getDictionaryAttr(fields));
  }
  for (auto const& stage : plan.stages) {
    llvm::SmallVector<std::int64_t> operands;
    for (auto operand : stage.operands) operands.push_back(operand);
    llvm::SmallVector<mlir::NamedAttribute> fields = {
        builder.getNamedAttr("kind", builder.getStringAttr(taskKindName(stage.kind))),
        builder.getNamedAttr("gemm", builder.getI64IntegerAttr(stage.gemm)),
        builder.getNamedAttr("extent", builder.getI64IntegerAttr(stage.extent)),
        builder.getNamedAttr("width", builder.getI64IntegerAttr(stage.width)),
        builder.getNamedAttr("group", builder.getI64IntegerAttr(stage.group)),
        builder.getNamedAttr("operands", builder.getDenseI64ArrayAttr(operands)),
        builder.getNamedAttr("representative", builder.getStringAttr(stage.representative)),
        builder.getNamedAttr("representative_index",
                             builder.getI64IntegerAttr(stage.representative_index))};
    if (plan.serving || plan.forward) {
      fields.push_back(builder.getNamedAttr(
          "batch_rows", builder.getBoolAttr(stage.batch_rows)));
      fields.push_back(builder.getNamedAttr(
          "row_stride", builder.getI64IntegerAttr(stage.row_stride)));
      fields.push_back(builder.getNamedAttr(
          "row_offset", builder.getI64IntegerAttr(stage.row_offset)));
      fields.push_back(builder.getNamedAttr(
          "attention_kv_block", builder.getI64IntegerAttr(
              stage.attention_kv_block)));
      fields.push_back(builder.getNamedAttr(
          "attention_query_rows", builder.getI64IntegerAttr(
              stage.attention_query_rows)));
    }
    if(plan.dm) {
      fields.push_back(builder.getNamedAttr("dm_conv",builder.getI64IntegerAttr(stage.conv)));
      fields.push_back(builder.getNamedAttr("dm_rows_per_batch",builder.getI64IntegerAttr(stage.rows_per_batch)));
      if(stage.binding_producer!=codegen::kDmNoIndex)
        fields.push_back(builder.getNamedAttr("dm_binding_producer",builder.getI64IntegerAttr(stage.binding_producer)));
      if(stage.kind==PlanTaskKind::kEmbeddingSum)
        fields.push_back(builder.getNamedAttr("dm_workspace_bytes",builder.getI64IntegerAttr(32)));
      if(stage.kind==PlanTaskKind::kEncoderAttention)
        fields.push_back(builder.getNamedAttr("dm_workspace_bytes",builder.getI64IntegerAttr(codegen::EncoderAttentionSharedBytes())));
      if(stage.kind==PlanTaskKind::kMoETopK)
        fields.push_back(builder.getNamedAttr("dm_workspace_bytes",builder.getI64IntegerAttr(codegen::MoeDispatchSharedBytes())));
      if(stage.kind==PlanTaskKind::kMoECombine)
        fields.push_back(builder.getNamedAttr("dm_workspace_bytes",builder.getI64IntegerAttr(codegen::MoeCombineSharedBytes(stage.group,stage.width))));
      if(stage.norm_epsilon!=0.0f)
        fields.push_back(builder.getNamedAttr("dm_norm_epsilon",builder.getF32FloatAttr(stage.norm_epsilon)));
      if(stage.partial_rows_per_image)
        fields.push_back(builder.getNamedAttr("dm_partial_rows_per_image",builder.getI64IntegerAttr(stage.partial_rows_per_image)));
      if(stage.moe.step!=codegen::DmMoeStep::kNone)
        fields.push_back(builder.getNamedAttr("dm_moe",EncodeDm(builder,stage.moe)));
      if(stage.kind==PlanTaskKind::kDepthwiseConv || stage.kind==PlanTaskKind::kDwPwFused) {
        fields.push_back(builder.getNamedAttr("dm_chain",EncodeDm(builder,stage.chain)));
        auto const& c=plan.convolutions.at(stage.conv);
        unsigned gates=0;
        for(unsigned i=0;i<stage.chain.count;++i)
          gates+=stage.chain.operations[i].kind==codegen::DmEpilogueKind::kGatePair;
        auto bytes=std::uint64_t((stage.group-1)*c.stride_h+(c.r-1)*c.dilation_h+1)*
            plan.buffers.at(c.input_layout).layout.physical[2]*stage.width*(gates?2:1)*2;
        if(stage.kind==PlanTaskKind::kDepthwiseConv)
          fields.push_back(builder.getNamedAttr("dm_workspace_bytes",builder.getI64IntegerAttr(std::max<std::uint64_t>(bytes,4096))));
      }
    }
    stages.push_back(builder.getDictionaryAttr(fields));
  }
  for (auto const& output : plan.outputs)
    outputs.push_back(dict(builder, {
        builder.getNamedAttr("buffer", builder.getI64IntegerAttr(output.buffer)),
        builder.getNamedAttr("file", builder.getStringAttr(output.file))}));
  llvm::SmallVector<mlir::NamedAttribute> fields = {
      builder.getNamedAttr("dtype", builder.getStringAttr(plan.dtype)),
      builder.getNamedAttr("norm_epsilon",
                           builder.getF64FloatAttr(plan.norm_epsilon)),
      builder.getNamedAttr("rope_fp32_phase",
                           builder.getBoolAttr(plan.rope_fp32_phase)),
      builder.getNamedAttr("token_id_bits",
                           builder.getI64IntegerAttr(plan.token_id_bits)),
      builder.getNamedAttr("buffers", builder.getArrayAttr(buffers)),
      builder.getNamedAttr("gemms", builder.getArrayAttr(gemms)),
      builder.getNamedAttr("stages", builder.getArrayAttr(stages)),
      builder.getNamedAttr("outputs", builder.getArrayAttr(outputs))};
  if (plan.forward) {
    fields.push_back(builder.getNamedAttr("forward", builder.getBoolAttr(true)));
    if (plan.forward_token_axis)
      fields.push_back(builder.getNamedAttr("forward_token_axis", builder.getBoolAttr(true)));
  }
  if(plan.dm) {
    llvm::SmallVector<mlir::Attribute> convs;
    for(auto const& conv:plan.convolutions)convs.push_back(EncodeDm(builder,conv));
    fields.push_back(builder.getNamedAttr("dm",builder.getBoolAttr(true)));
    if(plan.dm_reduction_mask>=0)fields.push_back(builder.getNamedAttr("dm_reduction_mask",
        builder.getI64IntegerAttr(plan.dm_reduction_mask)));
    if(plan.moe_gemv)fields.push_back(builder.getNamedAttr("moe_gemv",builder.getBoolAttr(true)));
    fields.push_back(builder.getNamedAttr("dm_convolutions",builder.getArrayAttr(convs)));
    if(!plan.deferred_layernorm_edges.empty()) {
      llvm::SmallVector<mlir::Attribute> edges;
      for(auto const& [norm,gemm]:plan.deferred_layernorm_edges)
        edges.push_back(dict(builder,{
            builder.getNamedAttr("norm",builder.getStringAttr(norm)),
            builder.getNamedAttr("gemm",builder.getI64IntegerAttr(gemm))}));
      fields.push_back(builder.getNamedAttr("dm_deferred_layernorm_edges",builder.getArrayAttr(edges)));
    }
    if(plan.memory_reuse!="none") {
      fields.push_back(builder.getNamedAttr("dm_memory_reuse",builder.getStringAttr(plan.memory_reuse)));
      fields.push_back(builder.getNamedAttr("dm_memory_arena_bytes",builder.getI64IntegerAttr(plan.memory_arena_bytes)));
    }
  }
  return builder.getDictionaryAttr(fields);
}

}  // namespace

mlir::DictionaryAttr EncodeModelPlan(mlir::Builder& builder, ModelPlan const& plan,
                                   std::vector<std::uint8_t> const& written) {
  return modelPlanAttr(builder,plan,written);
}

SymbolicShape SymbolicShapeBridge::Parse(
    std::unordered_map<std::string, std::string> const& ranges,
    std::vector<std::string> const& guards,
    std::vector<std::vector<std::string>> const& inputShapes) const {
  SymbolicShape result;
  std::unordered_map<std::string, std::string> parent;
  for (auto const& [symbol, text] : ranges) {
    auto [minimum, maximum] = parseRange(text);
    result.dimensions.push_back(symbol);
    result.ranges.emplace(symbol, ParameterRange{minimum, maximum});
    parent.emplace(symbol, symbol);
  }
  std::sort(result.dimensions.begin(), result.dimensions.end());
  for (auto const& guard : guards) {
    std::size_t split = guard.find("==");
    ShapeConstraint::Predicate predicate = ShapeConstraint::Predicate::kEqual;
    std::size_t width = 2;
    if (split == std::string::npos) {
      split = guard.find("<=");
      predicate = ShapeConstraint::Predicate::kLessEqual;
    }
    if (split == std::string::npos) {
      split = guard.find(">=");
      predicate = ShapeConstraint::Predicate::kLessEqual;
    }
    if (split == std::string::npos)
      throw std::invalid_argument("unsupported ShapeEnv guard predicate: " + guard);
    auto lhs = extractGuardTerms(guard.substr(0, split), inputShapes);
    auto rhs = extractGuardTerms(guard.substr(split + width), inputShapes);
    bool reverseInequality = guard.compare(split, 2, ">=") == 0;
    if (reverseInequality) std::swap(lhs, rhs);
    if (predicate == ShapeConstraint::Predicate::kEqual &&
        guard.substr(0, split).find('%') != std::string::npos) {
      static std::regex const modulo(R"(%\s*([0-9]+))");
      std::smatch divisor;
      std::string left = guard.substr(0, split);
      if (!std::regex_search(left, divisor, modulo))
        throw std::invalid_argument("cannot parse modulo guard: " + guard);
      rhs = {divisor[1].str()};
      predicate = ShapeConstraint::Predicate::kDivisible;
    }
    auto lhsReduced = lhs, rhsReduced = rhs;
    for (auto it = lhsReduced.begin(); it != lhsReduced.end();) {
      auto match = std::find(rhsReduced.begin(), rhsReduced.end(), *it);
      if (match == rhsReduced.end()) { ++it; continue; }
      rhsReduced.erase(match); it = lhsReduced.erase(it);
    }
    bool redundant = predicate == ShapeConstraint::Predicate::kEqual &&
                     lhsReduced.empty() && rhsReduced.empty();
    if (predicate == ShapeConstraint::Predicate::kEqual &&
        lhsReduced.size() == 1 && rhsReduced.size() == 1)
      unite(parent, lhsReduced.front(), rhsReduced.front());
    result.constraints.push_back({sumTerms(lhs), sumTerms(rhs),
                                  predicate,
                                  guard, redundant});
  }
  for (auto const& symbol : result.dimensions)
    result.canonical_symbol[symbol] = findRoot(parent, symbol);
  return result;
}

ExportBridge ReadExportBridge(std::string const& path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file) throw std::runtime_error("cannot read export bridge JSON: " + path);
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) throw std::runtime_error("invalid export bridge JSON: " + path);
  auto* root = parsed->getAsObject();
  if (!root || root->getString("schema") != "tilemega.exported_program.v1")
    throw std::runtime_error("unsupported export bridge schema");

  ExportBridge bridge;
  std::set<std::string> unsupported;
  auto* nodes = root->getArray("nodes");
  if (!nodes) throw std::runtime_error("export JSON has no nodes array");
  for (auto const& item : *nodes) {
    auto* object = item.getAsObject();
    if (!object) throw std::runtime_error("node is not an object");
    FxNodeRecord node;
    node.index = static_cast<int>(*object->getInteger("index"));
    node.name = object->getString("name")->str();
    node.op = object->getString("op")->str();
    node.target = object->getString("target")->str();
    node.inputs = readStrings(object->getArray("inputs"));
    if (auto* scalars = object->getArray("scalar_args")) {
      node.has_scalars = true;
      for (auto const& scalar : *scalars)
        if (auto number = scalar.getAsNumber()) node.scalars.push_back(*number);
    }
    if(auto* arguments=object->get("args")) {
      auto parsed_args=ReadFxArgument(*arguments);
      if(parsed_args.kind!=FxArgument::Kind::kList)
        throw std::invalid_argument("node args must be a typed list");
      node.has_arguments=true;node.args=std::move(parsed_args.items);
    }
    if(auto* value=object->get("kwargs")) {
      auto* kwargs=value->getAsObject();
      if(!kwargs)throw std::invalid_argument("node kwargs must be an object");
      for(auto const& item:*kwargs)node.kwargs.emplace(item.first.str(),ReadFxArgument(item.second));
    }
    if(auto* value=object->get("constant")) {
      auto* constant=value->getAsObject();
      if(!constant)throw std::invalid_argument("node constant must be an object");
      node.constant=ReadFxConstant(*constant);
    }
    if(auto* value=object->get("immutable_buffer_value")) {
      auto* constant=value->getAsObject();
      if(!constant || node.op!="placeholder")
        throw std::invalid_argument("immutable buffer value must describe a placeholder");
      node.immutable_buffer_value=ReadFxConstant(*constant);
    }
    if(auto* shape_constant=object->getObject("shape_constant")) {
      node.shape_constant_symbols=readStrings(shape_constant->getArray("symbols"));
      if(auto* fragment=shape_constant->get("fragment"))
        node.shape_constant_fragment_json=llvm::formatv("{0}",*fragment).str();
      if(auto* bindings=shape_constant->getObject("bindings"))
        for(auto const& item:*bindings) {
          auto integer=item.second.getAsInteger();
          if(!integer)throw std::invalid_argument("shape binding is not an integer");
          node.shape_constant_bindings.emplace(item.first.str(),*integer);
        }
    }
    node.shape = readStrings(object->getArray("shape"));
    if (auto dtype = object->getString("dtype")) node.dtype = dtype->str();
    if (node.op == "call_function") {
      if (!known().contains(node.target)) unsupported.insert(node.target);
      bridge.tasks.push_back(node);
    }
    bridge.nodes.push_back(std::move(node));
  }
  bridge.unsupported.assign(unsupported.begin(), unsupported.end());
  auto* signature = root->getObject("signature");
  if (!signature)
    throw std::runtime_error(
        "export bridge has no structured signature; rerun export_bridge.py");
  if (auto* inputs = signature->getArray("inputs")) {
    for (auto const& item : *inputs) {
      auto* object = item.getAsObject();
      if (!object) throw std::runtime_error("signature input is not an object");
      SignatureInput input;
      input.name = object->getString("name")->str();
      input.kind = object->getString("kind")->str();
      if (auto target = object->getString("target")) input.target = target->str();
      if (auto persistent = object->getBoolean("persistent"))
        input.persistent = *persistent;
      bridge.inputs.push_back(std::move(input));
    }
  }
  if (auto* outputs = signature->getArray("outputs"))
    for (auto const& item : *outputs) {
      auto* object = item.getAsObject();
      if (!object) throw std::runtime_error("signature output is not an object");
      bridge.outputs.push_back(object->getString("name")->str());
    }
  if (bridge.inputs.empty() || bridge.outputs.empty())
    throw std::runtime_error("structured signature has no inputs or outputs");
  if (auto* ranges = root->getObject("range_constraints"))
    for (auto const& item : *ranges)
      bridge.range_texts[item.first.str()] = item.second.getAsString()->str();
  bridge.guards = readStrings(root->getArray("guards"));
  return bridge;
}

static LiftOptions ForwardDimensionRoles(ExportBridge const& bridge,
                                         ModelPlan const& plan,
                                         SymbolicShape const& symbolic) {
  if (!plan.dm || plan.serving || plan.serving_seq <= 0 || plan.serving_capacity)
    throw std::invalid_argument("invalid forward plan phase or dimensions");
  LiftOptions roles;
  roles.forward = true;
  roles.serving = !plan.forward_token_axis;
  roles.static_seq = plan.serving_seq;
  roles.seq_symbol = std::to_string(plan.serving_seq);
  roles.past_symbol.clear();
  std::string row_axis;
  for (auto const& input : bridge.inputs) {
    if (input.kind != "USER_INPUT") continue;
    auto found = std::find_if(bridge.nodes.begin(), bridge.nodes.end(),
        [&](FxNodeRecord const& node) { return node.name == input.name; });
    if (found == bridge.nodes.end() || found->shape.empty())
      throw std::invalid_argument("forward input has no row axis");
    if (!row_axis.empty() && row_axis != found->shape[0])
      throw std::invalid_argument("forward input row axes disagree");
    row_axis = found->shape[0];
  }
  if (!symbolic.ranges.count(row_axis))
    throw std::invalid_argument("forward input row axis must be symbolic");
  if (plan.forward_token_axis) roles.seq_symbol = row_axis;
  else roles.batch_symbol = row_axis;
  return roles;
}

static LiftOptions ServingDimensionRoles(ExportBridge const& bridge,
                                         ModelPlan const& plan,
                                         SymbolicShape const& symbolic) {
  LiftOptions roles;
  roles.serving = true;
  roles.static_seq = plan.serving_seq;
  roles.seq_symbol = std::to_string(plan.serving_seq);
  roles.past_symbol.clear();
  for (auto const& input : bridge.inputs) {
    if (input.kind != "USER_INPUT") continue;
    auto found = std::find_if(bridge.nodes.begin(), bridge.nodes.end(),
                              [&](FxNodeRecord const& node) {
                                return node.name == input.name;
                              });
    if (found == bridge.nodes.end())
      throw std::invalid_argument("serving signature names an unknown input");
    // The token-id tensor structurally identifies batch and fixed seq.  Its
    // symbol role does not depend on order in the export's range dictionary.
    if (found->shape.size() == 2 &&
        (found->dtype == "torch.int64" || found->dtype == "torch.int32")) {
      if (!roles.batch_symbol.empty())
        throw std::invalid_argument("serving export has multiple token-id inputs");
      roles.batch_symbol = found->shape[0];
      if (found->shape[1] != std::to_string(plan.serving_seq))
        throw std::invalid_argument("serving seq is not the plan constant");
    }
  }
  if (roles.batch_symbol.empty() || !symbolic.ranges.count(roles.batch_symbol))
    throw std::invalid_argument("serving batch is not a symbolic token axis");
  for (auto const& input : bridge.inputs) {
    if (input.kind != "USER_INPUT") continue;
    auto found = std::find_if(bridge.nodes.begin(), bridge.nodes.end(),
                              [&](FxNodeRecord const& node) {
                                return node.name == input.name;
                              });
    if (found->shape.size() != 4 ||
        found->shape[0] != roles.batch_symbol) continue;
    // The cache layout is [B,Hkv,past,D].  All caches must agree on axis 2.
    if (symbolic.ranges.count(found->shape[2])) {
      if (!roles.past_symbol.empty() && roles.past_symbol != found->shape[2])
        throw std::invalid_argument("serving caches disagree on past axis");
      roles.past_symbol = found->shape[2];
    } else if (found->shape[2] != "0") {
      throw std::invalid_argument("serving cache past axis is not symbolic or zero");
    }
  }
  if (roles.past_symbol.empty() && plan.serving_seq == 1)
    throw std::invalid_argument("decode requires a symbolic past axis");
  return roles;
}

static mlir::OwningOpRef<mlir::ModuleOp> ImportBridgePlan(
    ExportBridge bridge, ModelPlan const* selected_plan, mlir::MLIRContext& context,
    ImportSummary* summary, ImportOptions const& options,
    ImportedSemantics const* prepared=nullptr,analysis::CouplingCache* cache=nullptr,
    solver::SolverTiming* timing=nullptr) {
  context.getOrLoadDialect<dialect::CGDialect>();
  context.getOrLoadDialect<dialect::ExecDialect>();
  std::vector<FxNodeRecord>& allNodes = bridge.nodes;
  std::vector<FxNodeRecord>& tasks = bridge.tasks;
  std::vector<SignatureInput>& signatureInputs = bridge.inputs;
  std::vector<std::string>& signatureOutputs = bridge.outputs;
  // Degradation, not refusal: the operators no rule covers are reported and
  // each becomes one conservative task space.
  bool selected_dm=selected_plan && selected_plan->dm && !selected_plan->stages.empty();
  if (!selected_dm && !bridge.unsupported.empty())
    llvm::errs() << "IMPORT_DEGRADED " << llvm::join(bridge.unsupported, ", ")
                 << "\n";

  std::unordered_map<std::string, FxNodeRecord const*> nodeByName;
  for (auto const& node : allNodes) nodeByName.emplace(node.name, &node);
  std::vector<std::vector<std::string>> userShapes;
  for (auto const& input : signatureInputs)
    if (input.kind == "USER_INPUT") {
      auto found = nodeByName.find(input.name);
      if (found == nodeByName.end())
        throw std::runtime_error("signature names unknown input " + input.name);
      userShapes.push_back(found->second->shape);
    }

  std::unordered_map<std::string, std::string> const& rangeTexts =
      bridge.range_texts;
  std::vector<std::string> const& guards = bridge.guards;
  SymbolicShape symbolic = prepared ? prepared->symbolic : SymbolicShapeBridge{}.Parse(rangeTexts, guards, userShapes);
  ModelPlan plan = selected_plan ? *selected_plan : BuildModelPlan(allNodes, signatureInputs, signatureOutputs);
  std::optional<analysis::ScopedExactAnalysisMemo> dmMemo;
  if(plan.dm && !analysis::active_exact_memo)dmMemo.emplace();
  if (options.separate_residual_tasks) {
    if (!options.attention.empty())
      throw std::invalid_argument("separate residual stages require attention indices from the expanded plan");
    SeparateResidualTasks(plan);
  }
  if (!options.gemms.empty() && options.gemms.size() != plan.gemms.size())
    throw std::invalid_argument(
        "runtime variant must provide exactly one entry per model GEMM");
  std::vector<GemmGranularity> runtimeGemms = options.gemms;
  if (runtimeGemms.empty()) runtimeGemms.resize(plan.gemms.size());
  std::vector<int> stages = prepared ? std::vector<int>{} : FormSemanticStages(tasks, plan);

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  llvm::SmallVector<mlir::NamedAttribute> theta, domains, aliases;
  for (auto const& symbol : symbolic.dimensions) {
    theta.push_back(builder.getNamedAttr(symbol,
        builder.getI64IntegerAttr(symbolic.ranges.at(symbol).minimum)));
    domains.push_back(builder.getNamedAttr(symbol, builder.getStringAttr(rangeTexts.at(symbol))));
    aliases.push_back(builder.getNamedAttr(symbol,
        builder.getStringAttr(symbolic.canonical_symbol.at(symbol))));
  }
  module->setAttr("tilemega.theta", builder.getDictionaryAttr(theta));
  module->setAttr("tilemega.param_domain", builder.getDictionaryAttr(domains));
  module->setAttr("tilemega.symbol_aliases", builder.getDictionaryAttr(aliases));
  // §2.3's g, as the megakernel actually launches: the GEMM TaskBody owns one
  // 128x128 output tile per CTA, and the split-K chunking is a per-operator
  // decision the solver makes later, not a model-wide tile.
  GemmGranularity representative = runtimeGemms.empty()
      ? GemmGranularity{} : runtimeGemms.front();
  module->setAttr("tilemega.g", dict(builder, {
      builder.getNamedAttr("Tm", builder.getI64IntegerAttr(representative.tile_m)),
      builder.getNamedAttr("Tn", builder.getI64IntegerAttr(representative.tile_n)),
      builder.getNamedAttr("Tkv", builder.getI64IntegerAttr(128))}));
  llvm::SmallVector<mlir::Attribute> runtimePlan;
  for (auto const& impl : runtimeGemms)
    runtimePlan.push_back(dict(builder, {
        builder.getNamedAttr("tile_m", builder.getI64IntegerAttr(impl.tile_m)),
        builder.getNamedAttr("tile_n", builder.getI64IntegerAttr(impl.tile_n)),
        builder.getNamedAttr("tile_k", builder.getI64IntegerAttr(impl.tile_k)),
        builder.getNamedAttr("stages", builder.getI64IntegerAttr(impl.stages)),
        builder.getNamedAttr("split_k", builder.getI64IntegerAttr(impl.split_k))}));
  module->setAttr("tilemega.gemm_runtime", builder.getArrayAttr(runtimePlan));
  if (!options.attention.empty()) {
    llvm::SmallVector<mlir::Attribute> choices;
    std::set<int> selected;
    for (auto const& choice : options.attention) {
      if (choice.stage < 0 || static_cast<std::size_t>(choice.stage) >= plan.stages.size() ||
          plan.stages[choice.stage].kind != PlanTaskKind::kAttention ||
          !selected.insert(choice.stage).second || !choice.runtime.chunks || !choice.runtime.chunk_extent)
        throw std::invalid_argument("attention plan requires unique attention stages and positive geometry");
      choices.push_back(dict(builder, {
          builder.getNamedAttr("stage",builder.getI64IntegerAttr(choice.stage)),
          builder.getNamedAttr("chunks",builder.getI64IntegerAttr(choice.runtime.chunks)),
          builder.getNamedAttr("chunk_extent",builder.getI64IntegerAttr(choice.runtime.chunk_extent))}));
    }
    module->setAttr("tilemega.attention_runtime",builder.getArrayAttr(choices));
  }
  module->setAttr("tilemega.rope_tile_per_block",
                  builder.getBoolAttr(options.rope_tile_per_block));
  module->setAttr("tilemega.kv_tile_per_block",
                  builder.getBoolAttr(options.kv_tile_per_block));
  module->setAttr("tilemega.activation_tile_per_block",
                  builder.getBoolAttr(options.activation_tile_per_block));
  module->setAttr("tilemega.combiner_tile_per_block",
                  builder.getBoolAttr(options.combiner_tile_per_block || plan.dm));
  module->setAttr("tilemega.guard_count", builder.getI64IntegerAttr(guards.size()));
  if (plan.stages.empty())
    llvm::errs() << "IMPORT_DEGRADED no decoder layer; one task space per operator\n";
  builder.setInsertionPointToStart(module.getBody());

  LiftOptions liftOptions;
  if (plan.forward) {
    liftOptions = ForwardDimensionRoles(bridge, plan, symbolic);
  } else if (plan.serving) {
    liftOptions = ServingDimensionRoles(bridge, plan, symbolic);
  } else {
    if (!symbolic.dimensions.empty())
      liftOptions.seq_symbol = symbolic.dimensions.front();
    for (auto const& symbol : symbolic.dimensions)
      if (symbol != liftOptions.seq_symbol) {
        liftOptions.past_symbol = symbol;
        break;
      }
  }
  // Preserve semantic dimension roles for consumers of symbolic CG metrics.
  // Import witness values are not runtime dimensions and must not be reused.
  llvm::SmallVector<mlir::NamedAttribute> dimension_roles = {
      builder.getNamedAttr("seq", builder.getStringAttr(liftOptions.seq_symbol)),
      builder.getNamedAttr("past", builder.getStringAttr(liftOptions.past_symbol))};
  if (plan.serving || (plan.forward && !plan.forward_token_axis))
    dimension_roles.push_back(builder.getNamedAttr(
        "batch", builder.getStringAttr(liftOptions.batch_symbol)));
  module->setAttr("tilemega.dimension_roles",
                  builder.getDictionaryAttr(dimension_roles));
  LiftedModel lifted = prepared ? prepared->lifted : plan.stages.empty()
                           ? LiftGenericSemantics(tasks, stages, liftOptions)
                           : LiftSemantics(plan, liftOptions);
  // The plan attribute is written after lifting because the read-only
  // frontier it carries is the lifting replay's own write relation.
  MaterializeDnnStorage(plan,runtimeGemms,options.phase_batch);
  if(plan.dm) {
    for(unsigned index=0;index<plan.gemms.size();++index) {
      auto const& gemm=plan.gemms[index];
      if(gemm.access.write.kind!=codegen::DmWriteKind::kRowScatter)continue;
      auto stage=std::find_if(plan.stages.begin(),plan.stages.end(),[&](auto const& s) {
        return s.kind==PlanTaskKind::kGemm && s.gemm==index;
      });
      if(stage==plan.stages.end() || stage->binding_producer==codegen::kDmNoIndex)
        throw std::invalid_argument("MoE scatter has no dispatch stage");
      unsigned router=plan.stages.at(stage->binding_producer).moe.router_gemm;
      MaterializeMoeRegionStorage(plan,runtimeGemms.at(router).tile_n,
                                 runtimeGemms.at(index).tile_n,router);
    }
  }
  if (!plan.stages.empty())
    module->setAttr("tilemega.model_plan",
                    modelPlanAttr(builder, plan, lifted.written));
  if (plan.serving || plan.forward) {
    llvm::SmallVector<mlir::NamedAttribute> runtime = {
        builder.getNamedAttr("seq", builder.getI64IntegerAttr(plan.serving_seq)),
        builder.getNamedAttr("capacity", builder.getI64IntegerAttr(plan.serving_capacity))};
    if (plan.forward) runtime.push_back(builder.getNamedAttr("phase", builder.getI64IntegerAttr(2)));
    module->setAttr("tilemega.serving", builder.getDictionaryAttr(runtime));
  }
  if (options.rope_tile_per_block)
    for (auto& op : lifted.ops)
      if (op.role == OpRole::kRoPE)
        op.ownership = OwnershipKind::kTilePerBlock;
  if (options.kv_tile_per_block)
    for (auto& op : lifted.ops)
      if (op.role == OpRole::kKVAppend)
        op.ownership = OwnershipKind::kTilePerBlock;
  if (options.activation_tile_per_block)
    for (auto& op : lifted.ops)
      if (op.role == OpRole::kActivation)
        op.ownership = OwnershipKind::kTilePerBlock;
  analysis::Granularity g = plan.stages.empty()
      ? LaunchGranularity(lifted)
      : LaunchGranularity(lifted, plan, runtimeGemms);
  analysis::OperatorGraph graph = [&] { solver::SolverPhase phase(timing,"instantiate");
    return analysis::Instantiate(lifted.sem,g); }();

  analysis::ParamBinding granularityBinding;
  for (auto const& item :
       module->getAttrOfType<mlir::DictionaryAttr>("tilemega.g"))
    granularityBinding.Bind(
        item.getName().str(),
        llvm::cast<mlir::IntegerAttr>(item.getValue()).getInt());
  // Only what isl *requires* to be literal is bound: `isl_aff_div` rejects a
  // parametric divisor, so tile sizes and the GQA group factor must be numbers
  // before a map is built.  The workload dimensions (S, past, L_s) stay real
  // isl parameters, exactly as §2.7 presents them, so `wait`/`fanout`/
  // `volume`/`count` come out as quasi-polynomials in those parameters rather
  // than as one integer evaluated at the bottom of the parameter range.  P5.1
  // needs functions of theta; an integer measured at S_min is not one.
  analysis::ParamBinding known;
  for (auto const& [name, value] : granularityBinding.values) known.Bind(name, value);
  // The one place a workload minimum is still needed: an event tensor is a
  // real allocation, so its shape is resolved at the smallest instantiation
  // and any axis that is not constant there is emitted dynamic.
  analysis::ParamBinding taskBinding = known;
  for (auto const& [name, value] : options.task_binding.values) taskBinding.Bind(name, value);
  bool const exactTasks = std::any_of(lifted.sem.ops.begin(), lifted.sem.ops.end(),
      [](auto const& op) { return op.exact_task_access; });
  if (exactTasks && (plan.forward || plan.dm)) {
    if (plan.forward_token_axis && symbolic.ranges.count(liftOptions.seq_symbol))
      taskBinding.Bind(liftOptions.seq_symbol, plan.serving_seq);
    if (options.phase_batch > 0 && !liftOptions.batch_symbol.empty())
      taskBinding.Bind(liftOptions.batch_symbol, options.phase_batch);
  }
  if (exactTasks) {
    auto values = llvm::SmallVector<mlir::NamedAttribute>(theta);
    for (auto const& [name, value] : taskBinding.values) {
      if (auto range = symbolic.ranges.find(name); range != symbolic.ranges.end()) {
        if (value < range->second.minimum || value > range->second.maximum)
          throw std::invalid_argument("DM task shape binding is outside the export domain");
        for (auto& attr : values) if (attr.getName().strref() == name)
          attr = builder.getNamedAttr(name, builder.getI64IntegerAttr(value));
      }
    }
    module->setAttr("tilemega.theta", builder.getDictionaryAttr(values));
  }
  analysis::ParamBinding floorBinding = known;
  for (auto const& symbol : symbolic.dimensions)
    floorBinding.Bind(symbol, symbolic.ranges.at(symbol).minimum);

  // A node Instantiate added (a split reduction's combiner) is named after the
  // operator it combines, so the stage and role follow that operator.
  std::unordered_map<std::string, std::size_t> liftedIndex;
  for (std::size_t i = 0; i < lifted.ops.size(); ++i)
    liftedIndex[lifted.ops[i].name] = i;
  auto liftedOf = [&](std::string const& node) -> LiftedOp const& {
    auto found = liftedIndex.find(node);
    if (found != liftedIndex.end()) return lifted.ops[found->second];
    std::size_t dot = node.rfind('.');
    if (dot != std::string::npos) {
      found = liftedIndex.find(node.substr(0, dot));
      if (found != liftedIndex.end()) return lifted.ops[found->second];
    }
    throw std::runtime_error("instantiated node names no lifted operator: " + node);
  };

  std::unordered_map<std::string, std::string> symbols;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    auto const& node = graph.nodes[i];
    LiftedOp const& origin = liftedOf(node.name);
    std::string symbol = "t" + std::to_string(i);
    symbols[node.name] = symbol;
    llvm::SmallVector<mlir::NamedAttribute> tiles;
    for (std::size_t axis = 0; axis < node.output.axes.size(); ++axis)
      tiles.push_back(builder.getNamedAttr(
          node.output.axes[axis].name,
          builder.getStringAttr(node.tile[axis].ToString())));
    // A TaskBody that grid-strides over a linearized element range owns a set
    // `Granularity` (axis tiles only) cannot name.  The tiles above then model
    // it at element granularity -- exact, and finer than what one CTA runs --
    // and this field is what tells Codegen the composition is still owed.
    OwnershipKind ownership = llvm::StringRef(node.name).ends_with(".combine")
        ? ((options.combiner_tile_per_block || (plan.dm && node.element_access)) ? OwnershipKind::kTilePerBlock
                                           : OwnershipKind::kElementChunk)
        : origin.ownership;
    tiles.push_back(builder.getNamedAttr(
        "ownership", builder.getStringAttr(ToString(ownership))));
    mlir::OperationState state(builder.getUnknownLoc(), "tmcg.tile_space");
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(symbol));
    state.addAttribute("kind", dialect::TaskKindAttr::get(
        &context, builder.getStringAttr(taskKindOf(origin.role))));
    if(node.element_access && node.element_access->partition.reduction_index) {
      auto const& partition=node.element_access->partition;
      tiles.push_back(builder.getNamedAttr("reduction_index",builder.getStringAttr(
          analysis::EncodeTaskReductionIndex(*partition.reduction_index))));
      if(!partition.reduction_chunk.IsLiteral(0))
        tiles.push_back(builder.getNamedAttr("reduction_chunk",builder.getStringAttr(
            partition.reduction_chunk.ToString())));
    }
    state.addAttribute("granularity", builder.getDictionaryAttr(tiles));
    llvm::SmallVector<std::string> extents;
    for (auto const& axis : node.output.axes) extents.push_back(axis.extent.ToString());
    state.addAttribute("write_map", dialect::AccessMapAttr::get(&context, dict(builder, {
        builder.getNamedAttr("kind", builder.getStringAttr(taskKindOf(origin.role))),
        builder.getNamedAttr("shape", builder.getStringAttr(
            extents.empty() ? "scalar" : llvm::join(extents, "x")))})));
    state.addAttribute("stage", builder.getI64IntegerAttr(origin.stage));
    state.addAttribute("operator_name", builder.getStringAttr(node.name));
    state.addAttribute("fx_name", builder.getStringAttr(origin.fx_name));
    if (auto const* semantic = lifted.sem.Find(origin.name)) {
#if TILEMEGA_SEMANTIC_COST_INPUT
      if (node.name==origin.name)
        state.addAttribute("semantic",builder.getStringAttr(analysis::EncodeSemanticOp(*semantic)));
#endif
      std::string arithmetic = semantic->arithmetic;
      if (llvm::StringRef(node.name).ends_with(".combine"))
        arithmetic = plan.dm && node.element_access?node.element_access->semantic.arithmetic:
            semantic->reduction.reduction_operator == "add" ? "sum" : "";
      if (!arithmetic.empty())
        state.addAttribute("arithmetic", builder.getStringAttr(arithmetic));
    }
    if (node.element_access) {
      llvm::SmallVector<mlir::Attribute> bindings;
      for (auto const& binding : analysis::VirtualBindings(node.element_access->semantic))
        bindings.push_back(dict(builder, {
            builder.getNamedAttr("dimension", builder.getStringAttr(binding.dimension)),
            builder.getNamedAttr("capacity", builder.getStringAttr(binding.capacity.ToString())),
            builder.getNamedAttr("binding_source", builder.getStringAttr(binding.source)),
            builder.getNamedAttr("extent_kind", builder.getStringAttr("runtime_dynamic")),
            builder.getNamedAttr("runtime_requirement", builder.getStringAttr(binding.requirement))}));
      if (!bindings.empty()) state.addAttribute("virtual_bindings", builder.getArrayAttr(bindings));
      if(plan.dm && node.name!=origin.name)
        state.addAttribute("split_access_semantic",builder.getStringAttr(
            analysis::EncodeSemanticOp(node.element_access->semantic)));
    }
    // A split introduces two distinct task spaces. Keep their exact access
    // witness separate from the g-independent source semantic used by pricing.
    // Serving handoff selection alone promotes it to a phase semantic.
    if(plan.serving && !plan.dm && (node.name==origin.name+".combine" ||
        (node.name==origin.name && lifted.sem.Find(origin.name) &&
         node.output.name!=lifted.sem.Find(origin.name)->result.name))) {
      analysis::SemanticOp witness;
      witness.name=node.name;witness.kind=node.kind;
      witness.dtype=analysis::ScalarType::kF32;
      witness.result=node.output;witness.result_effect.kind=analysis::EffectKind::kWrite;
      // An access witness refines coordinates, not the task's arithmetic
      // declaration (residual/SwiGLU tasks need their original signature).
      if(auto arithmetic=mlir::dyn_cast_or_null<mlir::StringAttr>(
          state.attributes.get("arithmetic")))
        witness.arithmetic=arithmetic.getValue().str();
      for(auto const& axis:node.output.axes) {
        witness.domain.push_back({axis.name,axis.extent,axis.origin,
            analysis::IteratorType::kParallel,axis.runtime});
        witness.result_map.results.push_back(analysis::IndexResult::Dim(axis.name));
      }
      for(auto const& input:node.operands) {
        analysis::SemanticOperand operand;
        operand.producer=input.producer;operand.tensor=input.tensor;
        for(auto const& index:input.axes) {
          analysis::IndexResult mapped;
          if(index.kind==analysis::OperandAxisMap::Kind::kFullRange)
            mapped=analysis::IndexResult::FullRange(index.offset);
          else if(index.kind==analysis::OperandAxisMap::Kind::kBroadcast)
            mapped=analysis::IndexResult::Broadcast(index.span);
          else if(index.kind==analysis::OperandAxisMap::Kind::kDataDependent)
            mapped=analysis::IndexResult::DataDependent();
          else {
            std::vector<analysis::IndexResult::Term> terms;
            for(auto const& term:index.terms)
              terms.push_back({node.output.axes.at(term.output_axis).name,term.scale,term.group});
            mapped=analysis::IndexResult::Affine(std::move(terms),index.offset);
          }
          operand.map.results.push_back(std::move(mapped));
        }
        witness.operands.push_back(std::move(operand));
      }
      state.addAttribute("split_access_semantic",
          builder.getStringAttr(analysis::EncodeSemanticOp(witness)));
    }
    builder.create(state);
  }

  // The real C, from the analysis layer, on the production path.  There is no
  // fallback: an edge the derivation cannot produce is an import failure, not
  // a placeholder.
  std::vector<analysis::CouplingEdge> derived = [&] {
    solver::SolverPhase phase(timing,"derive");
    // A forward variant binds its workload before materializing table waits.
    // Keep L-sem symbolic, but count the exact relation at that same binding:
    // symbolic cardinality of convolution's floordiv unions is unnecessary.
    auto const& analysisKnown=plan.dm && plan.forward && exactTasks?taskBinding:known;
    if(!cache)return analysis::CouplingDerivation{}.Derive(graph,analysisKnown);
    auto hits=cache->hits,misses=cache->misses;
    auto result=cache->Derive(lifted.sem,graph,g,analysisKnown);
    if(timing) { timing->Add("cache_hit",0,cache->hits-hits);timing->Add("cache_miss",0,cache->misses-misses); }
    return result;
  }();
  std::set<std::size_t> storageEdges;

  if(plan.memory_reuse!="none") {
    auto memory=solver::PlanBufferReuse(plan,graph,taskBinding,plan.memory_reuse,
                                      plan.memory_l2_budget_bytes);
    plan.memory_arena_bytes=memory.arena_bytes;
    for(auto const& alias:memory.aliases)plan.buffers.at(alias.buffer).arena_offset=alias.offset;
    for(auto const& hazard:memory.hazards) {
      storageEdges.insert(derived.size());derived.push_back(hazard.coupling);
    }
    module->setAttr("tilemega.model_plan",modelPlanAttr(builder,plan,lifted.written));
    module->setAttr("tilemega.memory_live_peak_bytes",builder.getI64IntegerAttr(memory.live_peak_bytes));
    module->setAttr("tilemega.memory_retained_internal_bytes",builder.getI64IntegerAttr(memory.retained_internal_bytes));
    module->setAttr("tilemega.memory_total_internal_bytes",builder.getI64IntegerAttr(memory.total_internal_bytes));
    module->setAttr("tilemega.memory_fits_l2_budget",builder.getBoolAttr(memory.fits_l2_budget));
    module->setAttr("tilemega.memory_hazard_count",builder.getI64IntegerAttr(memory.hazards.size()));
  }

  // A phase analysis changes only the consumer's reduction tile to one K
  // iteration. Its task graph is an analysis witness; the executable graph
  // above retains the selected split. Every emitted window below is proved
  // against the complete symbolic ISL relation, never inferred from a sample.
  std::map<std::size_t,std::pair<analysis::WaitWindow,int>> phaseWindows;
  if(options.phase_analysis && plan.serving && plan.serving_seq==1) {
    analysis::ParamBinding phaseKnown=known;
    if(!liftOptions.batch_symbol.empty() && options.phase_batch>0)
      phaseKnown.Bind(liftOptions.batch_symbol,options.phase_batch);
    auto phased=g;
    for(auto const& op:lifted.ops) {
      if(op.stage<0 || std::size_t(op.stage)>=plan.stages.size())continue;
      auto const& stage=plan.stages[op.stage];
      if(stage.kind!=PlanTaskKind::kGemm || stage.gemm>=runtimeGemms.size())continue;
      auto semantic=lifted.sem.Find(op.name);
      if(semantic && semantic->reduction.splittable)
        phased.Split(op.name,analysis::ClosedForm::Constant(
            runtimeGemms[stage.gemm].tile_k));
    }
    auto phaseGraph=analysis::Instantiate(lifted.sem,phased);
    auto phaseEdges=cache?cache->Derive(lifted.sem,phaseGraph,phased,known)
        :analysis::CouplingDerivation{}.Derive(phaseGraph,known);
    analysis::ParamBinding witness=phaseKnown;
    if(!liftOptions.batch_symbol.empty() && options.phase_batch<=0)
      witness.Bind(liftOptions.batch_symbol,2L);
    if(!liftOptions.past_symbol.empty())witness.Bind(liftOptions.past_symbol,64L);
    int phase_scanned=0,phase_a_edges=0,phase_sources=0,
        phase_candidates=0,phase_fitted=0;
    for(std::size_t i=0;i<derived.size();++i) {
      auto const& edge=derived[i];auto const& consumer=liftedOf(edge.dst.name);
      if(consumer.stage<0 || std::size_t(consumer.stage)>=plan.stages.size())continue;
      auto const& stage=plan.stages[consumer.stage];
      if(stage.kind!=PlanTaskKind::kGemm || stage.gemm>=runtimeGemms.size())continue;
      ++phase_scanned;
      auto const& gemm=plan.gemms[stage.gemm];
      auto* originalSource=graph.Find(edge.src.name);
      if(!originalSource || originalSource->output.name!=plan.buffers.at(gemm.a).name)
        continue; // Only the A operand has producer phases.
      ++phase_a_edges;
      std::string phaseSource=edge.src.name;
      auto sourceSemantic=lifted.sem.Find(liftedOf(phaseSource).name);
      if(sourceSemantic && phased.ChunkOf(sourceSemantic->name,nullptr))
        phaseSource=sourceSemantic->reduction.combiner;
      auto* source=phaseGraph.Find(phaseSource);
      auto* sink=phaseGraph.Find(edge.dst.name);
      if(!source || !sink)continue;
      ++phase_sources;
      auto found=std::find_if(phaseEdges.begin(),phaseEdges.end(),
          [&](auto const& candidate){return candidate.src.name==phaseSource &&
              candidate.dst.name==edge.dst.name;});
      if(found==phaseEdges.end())continue;
      ++phase_candidates;
      // A GEMM reads the same K slice for every output N tile. A row-major
      // window over (m,n,j) cannot express this periodic reset of n; project
      // to j, then prove that rebuilding the full consumer relation gives
      // precisely the original ISL coupling. The plan has one M tile here.
      int const tk=runtimeGemms[stage.gemm].tile_k;
      int const kt=(gemm.k+tk-1)/tk;
      if(options.phase_batch<=0 || options.phase_batch>runtimeGemms[stage.gemm].tile_m || kt<=1)
        continue;
      auto dimensions=found->C.DomainDimNames();
      std::vector<analysis::ClosedForm> extents;
      for(std::size_t axis=0;axis<sink->output.axes.size();++axis)
        if(sink->IsTiled(axis))extents.push_back(sink->CoordinateExtent(axis).Substitute(phaseKnown));
      if(dimensions.size()!=extents.size() || dimensions.empty() ||
         dimensions.back()!="j")continue;
      std::ostringstream projection;
      projection<<"{ [";
      for(std::size_t axis=0;axis<dimensions.size();++axis){
        if(axis)projection<<',';
        projection<<dimensions[axis];
      }
      projection<<"] -> [jp] : jp = j";
      for(std::size_t axis=0;axis<dimensions.size();++axis)
        projection<<" and 0 <= "<<dimensions[axis]<<" and "
                  <<dimensions[axis]<<" < "<<extents[axis].ToIslText();
      projection<<" }";
      std::optional<analysis::WaitWindow> window;
      try {
        auto project=analysis::CouplingRelation::FromIslText(projection.str());
        auto exact=found->C.BindParams(phaseKnown).IntersectRange(
            analysis::ProducerTaskSpaceText(found->C,*source,phaseKnown));
        auto reduced=project.Reverse().ApplyRange(exact);
        auto rebuilt=project.ApplyRange(reduced);
        if(exact.IsSubset(rebuilt) && rebuilt.IsSubset(exact)) {
          analysis::OperatorNode phaseSink;
          phaseSink.name=edge.dst.name+".phase";
          phaseSink.output.name=phaseSink.name;
          analysis::TensorAxis axis;axis.name="jp";
          axis.extent=analysis::ClosedForm::Constant(kt);
          phaseSink.output.axes.push_back(axis);
          phaseSink.tile.push_back(analysis::ClosedForm::Constant(1));
          analysis::CouplingEdge reducedEdge=*found;
          reducedEdge.C=reduced;
          window=analysis::FitWaitWindowSymbolic(
              reducedEdge,*source,phaseSink,phaseKnown,witness);
        }
      } catch(std::exception const&) {
        // A failed projection proof retains the complete task-level wait.
      }
      if(std::getenv("TILEMEGA_PHASE_TRACE") && phase_candidates<=3)
        llvm::errs()<<"KPHASE_CANDIDATE src="<<phaseSource<<" dst="
                    <<edge.dst.name<<" relation="
                    <<found->C.ToString().substr(0,900)<<" window="
                    <<(window?window->ToString():"none")<<"\n";
      if(!window || !window->narrowed)continue; // Task-level wait is safe.
      ++phase_fitted;
      if(kt>1)phaseWindows.emplace(i,std::make_pair(*window,kt));
    }
    if(std::getenv("TILEMEGA_PHASE_TRACE"))
      llvm::errs()<<"KPHASE scanned="<<phase_scanned<<" a_edges="
                  <<phase_a_edges<<" sources="<<phase_sources<<" candidates="
                  <<phase_candidates<<" fitted="<<phase_fitted<<" selected="
                  <<phaseWindows.size()<<"\n";
  }

  // Part 2: the wait window the generated kernel evaluates per CTA.  It is a
  // property of C at *every* sequence length.  Discover a candidate from one
  // concrete witness, then prove its row-major interval relation equal to the
  // symbolic isl C over the complete parameterized task domain.  Only an edge
  // whose linearization leaves Presburger arithmetic uses the old three-point
  // concrete check; coupling derivation itself is never repeated.  A window
  // is additionally admissible only when
  // both TaskBodies declare `kTilePerBlock`: `kElementChunk` makes the
  // CTA->task map a function of gridDim, so the same id names different tasks
  // on the two sides and no per-CTA narrowing is sound.
  std::vector<std::optional<analysis::BoundDependencyForm>> boundDependencies(derived.size());
  std::vector<std::string> waitMaps(derived.size(), "all");
  std::size_t symbolicWindows = 0, fallbackWindows = 0;
  {
    // Decoder layers repeat the same parameterized relation and task-space
    // shapes.  Prove each distinct window once per variant; operator/stage
    // names do not participate in C, so including both task counts and C's
    // canonical isl text is a complete cache key for the linearized form.
    std::unordered_map<std::string, std::string> windowCache;
    std::vector<bool> owned(derived.size(), false);
    auto owns_tile = [&](std::string const& name) {
      if (llvm::StringRef(name).ends_with(".combine"))
        return options.combiner_tile_per_block || (plan.dm && graph.Find(name)->element_access);
      return liftedOf(name).ownership == OwnershipKind::kTilePerBlock;
    };
    for (std::size_t i = 0; i < derived.size(); ++i)
      owned[i] = owns_tile(derived[i].src.name) &&
                 owns_tile(derived[i].dst.name);
    for (std::size_t i = 0; i < derived.size(); ++i) {
      if (!owned[i]) continue;
      analysis::OperatorNode const* source = graph.Find(derived[i].src.name);
      analysis::OperatorNode const* sink = graph.Find(derived[i].dst.name);
      if (!source || !sink) continue;
      if (source->element_access || sink->element_access) {
        auto bound = analysis::BindExactTaskDependency(derived[i], *source, *sink, taskBinding);
        if (bound.encoding == analysis::BoundDependencyForm::Encoding::kWindow)
          waitMaps[i] = bound.window.ToString();
        boundDependencies[i] = std::move(bound);
        continue;
      }
      auto taskShape = [](analysis::OperatorNode const& node) {
        std::string key;
        for (std::size_t axis = 0; axis < node.output.axes.size(); ++axis)
          if (node.IsTiled(axis))
            key += node.CoordinateExtent(axis).ToString() + ";";
        return key;
      };
      std::string const cacheKey = taskShape(*source) + "\n" +
          taskShape(*sink) + "\n" + derived[i].C.ToString();
      if (auto found = windowCache.find(cacheKey);
          found != windowCache.end()) {
        waitMaps[i] = found->second;
        ++symbolicWindows;
        continue;
      }

      analysis::ParamBinding witness = known;
      if (!liftOptions.batch_symbol.empty())
        witness.Bind(liftOptions.batch_symbol, 2L);
      if (std::getenv("TILEMEGA_WINDOW_TRACE"))
        llvm::errs() << "WINDOW_FIT src=" << derived[i].src.name
                     << " dst=" << derived[i].dst.name << "\n";
      if (!liftOptions.seq_symbol.empty())
        witness.Bind(liftOptions.seq_symbol, 1L);
      if (!liftOptions.past_symbol.empty())
        witness.Bind(liftOptions.past_symbol, 0L);
      auto symbolic = analysis::FitWaitWindowSymbolic(
          derived[i], *source, *sink, known, witness);
      // A one-token witness can collapse the leading row axis and therefore
      // suggest an all/constant map where the symbolic relation is actually
      // a floordiv window.  Endpoint discovery is cheap, so retry at one
      // non-degenerate tile span before invoking point enumeration.
      if (!symbolic && !liftOptions.seq_symbol.empty()) {
        witness.Bind(liftOptions.seq_symbol, 256L);
        symbolic = analysis::FitWaitWindowSymbolic(
            derived[i], *source, *sink, known, witness);
      }
      if (symbolic) {
        waitMaps[i] = symbolic->ToString();
        windowCache.emplace(cacheKey, waitMaps[i]);
        ++symbolicWindows;
        continue;
      }

      ++fallbackWindows;
      if (std::getenv("TILEMEGA_WINDOW_TRACE"))
        llvm::errs() << "WINDOW_FALLBACK src=" << derived[i].src.name
                     << " dst=" << derived[i].dst.name << "\n";
      analysis::WaitWindow fitted;
      bool first_round = true;
      for (long sequence : {256L, 384L, 512L}) {
        analysis::ParamBinding probe = known;
        if (!liftOptions.batch_symbol.empty())
          probe.Bind(liftOptions.batch_symbol, 2L);
        if (!liftOptions.seq_symbol.empty())
          probe.Bind(liftOptions.seq_symbol, sequence);
        if (!liftOptions.past_symbol.empty())
          probe.Bind(liftOptions.past_symbol, sequence - 256L);
        analysis::CouplingEdge concrete = derived[i];
        concrete.C = derived[i].C.BindParams(probe);
        analysis::WaitWindow const here =
            analysis::FitWaitWindow(concrete, *source, *sink, probe);
        if (first_round) fitted = here;
        else if (here != fitted) fitted = analysis::WaitWindow{};
        first_round = false;
      }
      waitMaps[i] = fitted.ToString();
      windowCache.emplace(cacheKey, waitMaps[i]);
    }
  }
  llvm::SmallVector<mlir::Attribute> phaseRecords;
  std::size_t edge = 0;
  for (auto const& item : derived) {
    auto source = symbols.find(item.src.name);
    auto target = symbols.find(item.dst.name);
    if (source == symbols.end() || target == symbols.end())
      throw std::runtime_error("derived coupling names an unknown task space");
    std::string eventName = "e" + std::to_string(edge);
    llvm::SmallVector<std::int64_t> shape;
    llvm::SmallVector<mlir::Attribute> dims;
    analysis::ClosedForm product = analysis::ClosedForm::Constant(1);
    for (auto const& axis : item.event_shape) {
      analysis::ClosedForm reduced = axis.Substitute(granularityBinding);
      shape.push_back(reduced.IsConstant()
                          ? reduced.Eval(floorBinding, floorBinding)
                          : mlir::ShapedType::kDynamic);
      dims.push_back(dialect::MetricAttr::get(&context, metricOf(axis, granularityBinding)));
      product = product * axis;
    }
    mlir::OperationState eventState(builder.getUnknownLoc(), "tmcg.event_tensor");
    eventState.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(eventName));
    eventState.addAttribute("event_type", mlir::TypeAttr::get(
        mlir::RankedTensorType::get(shape, builder.getI32Type())));
    eventState.addAttribute("extent", dialect::MetricAttr::get(
        &context, metricOf(product, granularityBinding)));
    eventState.addAttribute("dims", builder.getArrayAttr(dims));
    builder.create(eventState);

    LiftedOp const& consumer = liftedOf(item.dst.name);
    mlir::OperationState state(builder.getUnknownLoc(), "tmcg.coupling");
    if(storageEdges.count(edge)) {
      if(graph.Find(item.src.name+".combine"))
        state.addAttribute("dependency_producer_main",builder.getBoolAttr(true));
      if(llvm::StringRef(item.dst.name).ends_with(".combine"))
        state.addAttribute("dependency_consumer_done",builder.getBoolAttr(true));
    }
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr("c" + std::to_string(edge)));
    state.addAttribute("src", mlir::FlatSymbolRefAttr::get(&context, source->second));
    state.addAttribute("dst", mlir::FlatSymbolRefAttr::get(&context, target->second));
    state.addAttribute("read_map", dialect::AccessMapAttr::get(&context,
        dict(builder, {builder.getNamedAttr("kind",
            builder.getStringAttr(taskKindOf(consumer.role)))})));
    state.addAttribute("relation", dialect::CouplingMapAttr::get(&context, item.C));
    state.addAttribute("wait_map", builder.getStringAttr(waitMaps[edge]));
    if (boundDependencies[edge]) {
      auto p = graph.Find(item.src.name), c = graph.Find(item.dst.name);
      auto pc = analysis::LinearizeTaskCoordinates(*p, item.C.RangeDimNames(), taskBinding, "_tm_p");
      auto cc = analysis::LinearizeTaskCoordinates(*c, item.C.DomainDimNames(), taskBinding, "_tm_c");
      auto geometry_relation=boundDependencies[edge]->table?
          boundDependencies[edge]->table->linear_relation:boundDependencies[edge]->encoded_relation;
      state.addAttribute("dependency_geometry", dialect::EncodeBoundTaskGeometry(builder,
          {geometry_relation, std::uint32_t(p->Count().Eval(taskBinding, {})),
           std::uint32_t(c->Count().Eval(taskBinding, {}))}, pc, cc, taskBinding));
      if (boundDependencies[edge]->table)
        state.addAttribute("dependency_table", dialect::EncodeBoundDependencyTable(builder,
            *boundDependencies[edge]->table, pc, cc, taskBinding));
    }
    if(plan.dm && consumer.stage>=0 &&
       plan.stages.at(consumer.stage).kind==PlanTaskKind::kMoECombine) {
      auto const& producer=liftedOf(item.src.name);
      auto const& stage=plan.stages.at(producer.stage);
      if(stage.kind==PlanTaskKind::kGemm) {
        auto const& gemm=plan.gemms.at(stage.gemm);
        if(gemm.access.write.kind==codegen::DmWriteKind::kRowScatter &&
            plan.stages.at(consumer.stage).moe.grouped) {
          auto p=mlir::cast<dialect::TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(module,source->second));
          auto c=mlir::cast<dialect::TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(module,target->second));
          state.addAttribute("dependency_counted",dialect::EncodeBoundCountedScatter(builder,p,c,
              plan.buffers.at(gemm.d).name,{0,1},plan.buffers.at(gemm.access.rows).name,taskBinding));
          state.attributes.erase("dependency_table");
        }
      }
    }
    if(auto found=phaseWindows.find(edge);found!=phaseWindows.end()) {
      state.addAttribute("phase_map",builder.getStringAttr(found->second.first.ToString()));
      state.addAttribute("phase_tiles",builder.getI64IntegerAttr(found->second.second));
      phaseRecords.push_back(dict(builder,{
          builder.getNamedAttr("producer",builder.getI64IntegerAttr(liftedOf(item.src.name).stage)),
          builder.getNamedAttr("consumer",builder.getI64IntegerAttr(liftedOf(item.dst.name).stage)),
          builder.getNamedAttr("window",builder.getStringAttr(found->second.first.ToString())),
          builder.getNamedAttr("phase_tiles",builder.getI64IntegerAttr(found->second.second))}));
    }
    state.addAttribute("wait", dialect::MetricAttr::get(&context, item.metrics.wait));
    state.addAttribute("fanout", dialect::MetricAttr::get(&context, item.metrics.fanout));
    state.addAttribute("volume", dialect::MetricAttr::get(&context, item.metrics.volume));
    state.addAttribute("count", dialect::MetricAttr::get(&context, item.metrics.count));
    if (item.shared_elements) {
      state.addAttribute("shared_elements", dialect::CouplingMapAttr::get(&context, *item.shared_elements));
      state.addAttribute("coupled_reads", dialect::CouplingMapAttr::get(&context, *item.coupled_reads));
      state.addAttribute("interface_elements", dialect::MetricAttr::get(&context, *item.interface_elements));
      state.addAttribute("consumer_elements", dialect::CouplingMapAttr::get(&context, *item.consumer_elements));
      state.addAttribute("read_box", dialect::CouplingMapAttr::get(&context, *item.read_box));
      state.addAttribute("read_box_exactness", builder.getStringAttr("over"));
    }
    state.addAttribute("tier", dialect::TierAttr::get(
        &context, std::stol(analysis::ToString(item.tier))));
    state.addAttribute("coupling_attrs", dialect::CouplingAttributesAttr::get(
        &context, builder.getStringAttr(analysis::ToString(item.attributes.relation_kind)),
        builder.getStringAttr(analysis::ToString(item.attributes.extent_kind)),
        builder.getStringAttr(analysis::ToString(item.attributes.exactness)),
        builder.getStringAttr(analysis::ToString(item.attributes.runtime_requirement)),
        builder.getStringAttr(analysis::ToString(item.attributes.countability))));
    state.addAttribute("sync_kind", dialect::SyncKindAttr::get(
        &context, builder.getStringAttr("global")));
    state.addAttribute("event", mlir::FlatSymbolRefAttr::get(&context, eventName));
    builder.create(state);
    ++edge;
  }
  if(!phaseRecords.empty())module->setAttr("tmexec.phase_edges",builder.getArrayAttr(phaseRecords));
  for (auto const& node : graph.nodes) {
    mlir::OperationState state(builder.getUnknownLoc(), "tmexec.placement");
    state.addAttribute("task", mlir::FlatSymbolRefAttr::get(&context, symbols.at(node.name)));
    state.addAttribute("map", builder.getDenseI64ArrayAttr({0}));
    state.addAttribute("cluster", builder.getI64IntegerAttr(1));
    if (options.balanced_placement) {
      state.addAttribute("resident_only",builder.getBoolAttr(true));
      state.addAttribute("mapping_mode",builder.getStringAttr("balanced"));
      // The §5.7.1 plan spelling of the same decision.  Absent attributes mean
      // `legacy_grid_stride`, so the default import stays byte-identical.
      state.addAttribute("mode",builder.getStringAttr(
          dialect::PlacementModeName(dialect::PlacementMode::kBalanced)));
      state.addAttribute("params",builder.getDenseI64ArrayAttr({}));
      state.addAttribute("window",builder.getI64IntegerAttr(
          dialect::kPlacementWindowImplemented));
      state.addAttribute("policy",builder.getStringAttr(dialect::kPlacementPolicyAot));
    }
    builder.create(state);
  }
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("C++ importer produced an invalid CG module");
  if (summary)
    *summary = {graph.nodes.size(), edge, plan.stages.size(), guards.size(),
                symbolicWindows, fallbackWindows, selected_dm?lifted.degraded:bridge.unsupported};
  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}


ImportedSemantics TorchExportImporter::ImportSemantics(std::string const& path,
    ModelPlan const& plan,mlir::MLIRContext& context) const {
  context.getOrLoadDialect<dialect::CGDialect>();
  context.getOrLoadDialect<dialect::ExecDialect>();
  ImportedSemantics result;result.bridge=ReadExportBridge(path);result.plan=plan;
  std::vector<std::vector<std::string>> shapes;
  for(auto const& input:result.bridge.inputs)if(input.kind=="USER_INPUT") {
    auto it=std::find_if(result.bridge.nodes.begin(),result.bridge.nodes.end(),[&](auto const& n){return n.name==input.name;});
    if(it==result.bridge.nodes.end())throw std::invalid_argument("unknown semantic input");
    shapes.push_back(it->shape);
  }
  result.symbolic=SymbolicShapeBridge{}.Parse(result.bridge.range_texts,result.bridge.guards,shapes);
  if (plan.forward) {
    result.lift_options = ForwardDimensionRoles(result.bridge, plan, result.symbolic);
  } else if (plan.serving) {
    result.lift_options = ServingDimensionRoles(
        result.bridge, plan, result.symbolic);
  } else {
    // Static exports retain the same dimension-role defaults as ImportPlan.
    if (!result.symbolic.dimensions.empty())
      result.lift_options.seq_symbol=result.symbolic.dimensions.front();
    for (auto const& symbol:result.symbolic.dimensions)
      if(symbol!=result.lift_options.seq_symbol) {
        result.lift_options.past_symbol=symbol;
        break;
      }
  }
  result.lifted=plan.stages.empty()
      ? LiftGenericSemantics(result.bridge.tasks,FormSemanticStages(result.bridge.tasks,plan),result.lift_options)
      : LiftSemantics(plan,result.lift_options);
  return result;
}
mlir::OwningOpRef<mlir::ModuleOp> TorchExportImporter::InstantiateForGranularity(
    ImportedSemantics const& imported,mlir::MLIRContext& context,
    ImportOptions const& options,analysis::CouplingCache* cache,
    ImportSummary* summary,solver::SolverTiming* timing) const {
  if(options.separate_residual_tasks)
    throw std::invalid_argument("separate residual tasks must be applied before importing semantics");
  return ImportBridgePlan(imported.bridge,&imported.plan,context,summary,options,&imported,cache,timing);
}

mlir::OwningOpRef<mlir::ModuleOp> TorchExportImporter::Import(
    std::string const& path,mlir::MLIRContext& context,
    ImportSummary* summary,ImportOptions const& options) const {
  return ImportBridgePlan(ReadExportBridge(path),nullptr,context,summary,options);
}

mlir::OwningOpRef<mlir::ModuleOp> TorchExportImporter::ImportPlan(
    std::string const& path,ModelPlan const& plan,mlir::MLIRContext& context,
    ImportSummary* summary,ImportOptions const& options) const {
  if (plan.stages.empty() || plan.outputs.empty())
    throw std::invalid_argument("explicit plan requires stages and observable outputs");
  return ImportBridgePlan(ReadExportBridge(path),&plan,context,summary,options);
}

}  // namespace tilemega::frontend
