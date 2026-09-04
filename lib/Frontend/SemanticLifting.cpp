// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tilemega::frontend {
namespace {

using analysis::ClosedForm;
using analysis::IndexResult;
using analysis::IterationDim;
using analysis::IteratorType;
using analysis::OperatorKind;
using analysis::SemanticOp;
using analysis::SemanticOperand;
using analysis::TensorAxis;
using analysis::TensorSpace;

constexpr std::uint32_t kNoOperand = std::numeric_limits<std::uint32_t>::max();

TensorAxis Ax(std::string name, ClosedForm extent) {
  TensorAxis axis;
  axis.name = std::move(name);
  axis.extent = std::move(extent);
  return axis;
}

TensorSpace Space(std::string name, std::vector<TensorAxis> axes,
                  std::string layout = {}) {
  TensorSpace space;
  space.name = std::move(name);
  space.axes = std::move(axes);
  space.layout_id = std::move(layout);
  return space;
}

IterationDim Par(std::string name, ClosedForm extent, bool runtime = false) {
  IterationDim dim;
  dim.name = std::move(name);
  dim.extent = std::move(extent);
  dim.runtime = runtime;
  return dim;
}

IterationDim Red(std::string name, ClosedForm extent, bool runtime = false) {
  IterationDim dim = Par(std::move(name), std::move(extent), runtime);
  dim.type = IteratorType::kReduction;
  return dim;
}

SemanticOperand Read(std::string producer, TensorSpace tensor,
                     std::vector<IndexResult> results) {
  SemanticOperand operand;
  operand.producer = std::move(producer);
  operand.tensor = std::move(tensor);
  operand.map.results = std::move(results);
  operand.effect.kind = analysis::EffectKind::kRead;
  return operand;
}

SemanticOp Op(std::string name, OperatorKind kind,
              std::vector<IterationDim> domain, TensorSpace result,
              std::vector<SemanticOperand> operands) {
  SemanticOp op;
  op.name = std::move(name);
  op.kind = kind;
  op.domain = std::move(domain);
  op.result = std::move(result);
  for (auto const& axis : op.result.axes)
    op.result_map.results.push_back(IndexResult::Dim(axis.name));
  op.result_effect.kind = analysis::EffectKind::kWrite;
  op.operands = std::move(operands);
  return op;
}

std::string StageName(int layer, std::size_t stage, char const* suffix) {
  std::string index = std::to_string(stage);
  if (index.size() < 2) index.insert(index.begin(), '0');
  return "l" + std::to_string(layer) + ".s" + index + "." + suffix;
}

ClosedForm Fixed(std::uint32_t value) {
  return ClosedForm::Constant(static_cast<long>(value));
}

/// Every buffer a RoPE or a KV append reads as its source. A projection whose
/// result lands in one of these is a Q/K/V projection -- the only structural
/// statement needed to give those three their one-head column tile.
std::unordered_set<std::uint32_t> RotatedOrCachedInputs(ModelPlan const& plan) {
  std::unordered_set<std::uint32_t> result;
  for (auto const& stage : plan.stages)
    if (stage.kind == PlanTaskKind::kRoPE ||
        stage.kind == PlanTaskKind::kKVAppend)
      result.insert(stage.operands[0]);
  return result;
}

}  // namespace

std::string ToString(OpRole role) {
  switch (role) {
    case OpRole::kNorm: return "norm";
    case OpRole::kQkvProjection: return "qkv_projection";
    case OpRole::kProjection: return "projection";
    case OpRole::kRoPE: return "rope";
    case OpRole::kKVAppend: return "kv_append";
    case OpRole::kAttention: return "attention";
    case OpRole::kActivation: return "activation";
    case OpRole::kResidualAdd: return "residual_add";
    case OpRole::kGeneric: return "generic";
  }
  return "generic";
}

std::string ToString(OwnershipKind kind) {
  return kind == OwnershipKind::kTilePerBlock ? "tile_per_block"
                                              : "element_chunk";
}

LiftedModel LiftSemantics(ModelPlan const& plan, LiftOptions const& options) {
  LiftedModel model;
  if (plan.stages.empty()) return model;
  model.has_plan = true;

  ClosedForm const S = ClosedForm::Symbol(options.seq_symbol);
  ClosedForm const past = ClosedForm::Symbol(options.past_symbol);
  ClosedForm const total = S + past;

  // Layers are uniform, so the stage count per layer is the total over the
  // attention count. Only naming and the reported layer index use it.
  std::size_t attentions = 0;
  for (auto const& stage : plan.stages)
    if (stage.kind == PlanTaskKind::kAttention) ++attentions;
  std::size_t per_layer = plan.stages.size();
  if (attentions && plan.stages.size() % attentions == 0)
    per_layer = plan.stages.size() / attentions;

  std::unordered_set<std::uint32_t> qkv = RotatedOrCachedInputs(plan);
  std::unordered_map<std::uint32_t, std::string> last_writer;
  // A read names the space its writer declared, so the two sides of an edge
  // agree on axis extents without either recomputing them.
  std::unordered_map<std::uint32_t, TensorSpace> written_space;
  auto name_of = [&](std::uint32_t buffer) -> std::string {
    if (buffer == kNoOperand || buffer >= plan.buffers.size())
      throw std::runtime_error("model plan stage names an unknown buffer");
    return plan.buffers[buffer].name;
  };
  auto producer_of = [&](std::uint32_t buffer) -> std::string {
    auto found = last_writer.find(buffer);
    return found == last_writer.end() ? std::string() : found->second;
  };
  auto space_of = [&](std::uint32_t buffer, std::vector<TensorAxis> fallback) {
    auto found = written_space.find(buffer);
    if (found != written_space.end()) return found->second;
    return Space(name_of(buffer), std::move(fallback));
  };
  auto record = [&](SemanticOp op, OpRole role, OwnershipKind ownership,
                    std::size_t stage, int layer, std::uint32_t result) {
    last_writer[result] = op.name;
    written_space[result] = op.result;
    model.ops.push_back({op.name, role, ownership, static_cast<int>(stage),
                         layer, plan.stages[stage].representative});
    model.sem.ops.push_back(std::move(op));
  };

  for (std::size_t i = 0; i < plan.stages.size(); ++i) {
    PlanStage const& stage = plan.stages[i];
    int layer = static_cast<int>(i / per_layer);
    switch (stage.kind) {
      case PlanTaskKind::kRMSNorm: {
        ClosedForm width = Fixed(stage.width);
        std::string name = StageName(layer, i, "norm");
        // `r` never reaches the result, so the read along it is full range:
        // the whole row is inside one task.
        SemanticOp op = Op(
            name, OperatorKind::kReduction,
            {Par("m", S), Par("h", width), Red("r", width)},
            Space(name_of(stage.operands[2]), {Ax("m", S), Ax("h", width)}),
            {Read(producer_of(stage.operands[0]),
                  space_of(stage.operands[0], {Ax("m", S), Ax("h", width)}),
                  {IndexResult::Dim("m"), IndexResult::Dim("r")})});
        record(std::move(op), OpRole::kNorm, OwnershipKind::kTilePerBlock, i,
               layer, stage.operands[2]);
        break;
      }
      case PlanTaskKind::kGemm: {
        PlanGemm const& gemm = plan.gemms[stage.gemm];
        bool residual = gemm.beta != 0.0f;
        ClosedForm n = Fixed(gemm.n), k = Fixed(gemm.k);
        std::string name = StageName(layer, i, "proj");
        // With beta != 0 the matmul result is the un-added product; the
        // epilogue residual below is what writes `d`.
        std::string product = residual ? name : name_of(gemm.d);
        SemanticOp op = Op(
            name, OperatorKind::kMatmul,
            {Par("m", S), Par("n", n), Red("k", k)},
            Space(product, {Ax("m", S), Ax("n", n)}),
            {Read(producer_of(gemm.a),
                  space_of(gemm.a, {Ax("m", S), Ax("k", k)}),
                  {IndexResult::Dim("m"), IndexResult::Dim("k")})});
        op.reduction.splittable = true;
        op.reduction.dim = "k";
        op.reduction.reduction_operator = "add";
        op.reduction.partial_tensor = name + ".partial";
        op.reduction.combiner = name + ".combine";
        op.reduction.ownership = {"m", "n"};
        OpRole role =
            qkv.count(gemm.d) ? OpRole::kQkvProjection : OpRole::kProjection;
        TensorSpace product_space = op.result;
        if (!residual) {
          record(std::move(op), role, OwnershipKind::kTilePerBlock, i, layer,
                 gemm.d);
          break;
        }
        // The matmul's own result is a value no buffer holds, so it is
        // recorded under the op name rather than a plan buffer.
        model.ops.push_back({op.name, role, OwnershipKind::kTilePerBlock,
                             static_cast<int>(i), layer,
                             plan.stages[i].representative});
        model.sem.ops.push_back(std::move(op));
        std::string add = StageName(layer, i, "add");
        SemanticOp resid = Op(
            add, OperatorKind::kPointwise, {Par("m", S), Par("n", n)},
            Space(name_of(gemm.d), {Ax("m", S), Ax("n", n)}),
            {Read(name, product_space,
                  {IndexResult::Dim("m"), IndexResult::Dim("n")}),
             Read(producer_of(gemm.c),
                  space_of(gemm.c, {Ax("m", S), Ax("n", n)}),
                  {IndexResult::Dim("m"), IndexResult::Dim("n")})});
        record(std::move(resid), OpRole::kResidualAdd,
               OwnershipKind::kTilePerBlock, i, layer, gemm.d);
        break;
      }
      case PlanTaskKind::kRoPE: {
        ClosedForm cols = Fixed(stage.extent * stage.width);
        model.head_dim = Fixed(stage.width);
        std::string name = StageName(layer, i, "rope");
        SemanticOp op = Op(
            name, OperatorKind::kPointwise, {Par("m", S), Par("hh", cols)},
            Space(name_of(stage.operands[1]), {Ax("m", S), Ax("hh", cols)}),
            {Read(producer_of(stage.operands[0]),
                  space_of(stage.operands[0], {Ax("m", S), Ax("hh", cols)}),
                  {IndexResult::Dim("m"), IndexResult::Dim("hh")})});
        record(std::move(op), OpRole::kRoPE, OwnershipKind::kElementChunk, i,
               layer, stage.operands[1]);
        break;
      }
      case PlanTaskKind::kKVAppend: {
        ClosedForm cols = Fixed(stage.extent * stage.width);
        std::string name = StageName(layer, i, "append");
        // The append writes rows [past, past + S) of a cache its consumers
        // address by absolute row; the origin is what keeps W from claiming
        // the whole cache.
        TensorAxis row = Ax("row", S);
        row.origin = past;
        SemanticOp op = Op(
            name, OperatorKind::kConcat, {Par("row", S), Par("hh", cols)},
            Space(name_of(stage.operands[2]), {row, Ax("hh", cols)},
                  "kv_cache"),
            {Read(producer_of(stage.operands[0]),
                  space_of(stage.operands[0], {Ax("row", S), Ax("hh", cols)}),
                  {IndexResult::Dim("row"), IndexResult::Dim("hh")})});
        op.result_effect.kind = analysis::EffectKind::kReadWrite;
        op.result_effect.state_object = "kv_cache";
        op.result_effect.alias_set = "kv_cache";
        record(std::move(op), OpRole::kKVAppend, OwnershipKind::kElementChunk,
               i, layer, stage.operands[2]);
        break;
      }
      case PlanTaskKind::kAttention: {
        ClosedForm q_cols = Fixed(stage.extent * stage.width);
        ClosedForm kv_cols = Fixed(stage.extent / stage.group * stage.width);
        ClosedForm group = Fixed(stage.group);
        model.head_dim = Fixed(stage.width);
        std::string name = StageName(layer, i, "attn");
        // The consumer addresses the cache by absolute row, so it reads a
        // space of runtime extent past + S, not the append's own S rows.
        auto cache = [&](std::uint32_t buffer) {
          TensorAxis row = Ax("row", total);
          row.runtime = true;
          return Space(name_of(buffer), {row, Ax("hh", kv_cols)}, "kv_cache");
        };
        // The KV head is floor(query_head / G): affine only with the grouped
        // term, which is why IndexResult::Term carries `group`.
        SemanticOperand full_k =
            Read(producer_of(stage.operands[1]), cache(stage.operands[1]),
                 {IndexResult::Dim("kv"),
                  IndexResult::Dim("h", ClosedForm::Constant(1), group)});
        full_k.effect.state_object = "kv_cache";
        full_k.effect.alias_set = "kv_cache";
        SemanticOperand full_v = full_k;
        full_v.producer = producer_of(stage.operands[2]);
        full_v.tensor = cache(stage.operands[2]);
        SemanticOp op = Op(
            name, OperatorKind::kPointwise,
            {Par("s", S), Par("h", q_cols), Red("kv", total, /*runtime=*/true)},
            Space(name_of(stage.operands[3]), {Ax("s", S), Ax("h", q_cols)}),
            {Read(producer_of(stage.operands[0]),
                  space_of(stage.operands[0], {Ax("s", S), Ax("h", q_cols)}),
                  {IndexResult::Dim("s"), IndexResult::Dim("h")}),
             full_k, full_v});
        op.reduction.splittable = true;
        op.reduction.dim = "kv";
        op.reduction.reduction_operator = "flash_combine";
        op.reduction.partial_tensor = name + ".partial";
        op.reduction.combiner = name + ".combine";
        op.reduction.ownership = {"s", "h"};
        record(std::move(op), OpRole::kAttention, OwnershipKind::kTilePerBlock,
               i, layer, stage.operands[3]);
        break;
      }
      case PlanTaskKind::kElementwise: {
        ClosedForm cols = Fixed(stage.extent);
        std::string name = StageName(layer, i, "act");
        SemanticOp op = Op(
            name, OperatorKind::kPointwise, {Par("m", S), Par("n", cols)},
            Space(name_of(stage.operands[2]), {Ax("m", S), Ax("n", cols)}),
            {Read(producer_of(stage.operands[0]),
                  space_of(stage.operands[0], {Ax("m", S), Ax("n", cols)}),
                  {IndexResult::Dim("m"), IndexResult::Dim("n")}),
             Read(producer_of(stage.operands[1]),
                  space_of(stage.operands[1], {Ax("m", S), Ax("n", cols)}),
                  {IndexResult::Dim("m"), IndexResult::Dim("n")})});
        record(std::move(op), OpRole::kActivation, OwnershipKind::kElementChunk,
               i, layer, stage.operands[2]);
        break;
      }
    }
  }
  return model;
}

LiftedModel LiftGenericSemantics(std::vector<FxNodeRecord> const& tasks,
                                 std::vector<int> const& stage_of_task,
                                 LiftOptions const& options) {
  LiftedModel model;
  std::unordered_map<std::string, std::size_t> index_of;
  auto axes = [&](FxNodeRecord const& node) {
    std::vector<TensorAxis> result;
    for (std::size_t a = 0; a < node.shape.size(); ++a) {
      std::string const& extent = node.shape[a];
      bool numeric = !extent.empty() &&
                     std::all_of(extent.begin(), extent.end(),
                                 [](unsigned char c) { return std::isdigit(c); });
      TensorAxis axis = Ax("d" + std::to_string(a),
                           numeric ? ClosedForm::Constant(std::stol(extent))
                                   : ClosedForm::Symbol(extent));
      axis.runtime = !numeric;
      result.push_back(std::move(axis));
    }
    if (result.empty()) result.push_back(Ax("d0", ClosedForm::Constant(1)));
    return result;
  };
  (void)options;
  for (std::size_t i = 0; i < tasks.size(); ++i)
    index_of.emplace(tasks[i].name, i);
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    FxNodeRecord const& task = tasks[i];
    std::vector<SemanticOperand> operands;
    std::unordered_set<std::string> seen;
    for (auto const& input : task.inputs) {
      auto found = index_of.find(input);
      if (found == index_of.end() || !seen.insert(input).second) continue;
      operands.push_back(Read(input, Space(input, axes(tasks[found->second])),
                              {}));
    }
    analysis::SemanticOp op = analysis::GenericSemantics(
        task.name, Space(task.name, axes(task)), std::move(operands));
    // GenericSemantics reads full range, which is the sound cover but prints
    // as an exact affine edge. No rule recognised this operator, so no index
    // was established at all; the read is declared data-dependent so the
    // derivation reaches Tier 3 and the I2 relaxation, instead of reporting an
    // exactness the frontend never checked.
    for (auto& operand : op.operands)
      for (auto& result : operand.map.results)
        result = IndexResult::DataDependent();
    model.sem.ops.push_back(std::move(op));
    model.ops.push_back({task.name, OpRole::kGeneric,
                         OwnershipKind::kElementChunk,
                         i < stage_of_task.size() ? stage_of_task[i] : 0, 0,
                         task.name});
    model.degraded.push_back(task.name);
  }
  return model;
}

analysis::Granularity LaunchGranularity(LiftedModel const& model) {
  analysis::Granularity g;
  ClosedForm const one = ClosedForm::Constant(1);
  ClosedForm const Tm = ClosedForm::Symbol("Tm");
  ClosedForm const Tn = ClosedForm::Symbol("Tn");
  for (auto const& op : model.ops) {
    switch (op.role) {
      case OpRole::kNorm:
        // RMSNormTaskBody: one token per CTA.
        g.Tile(op.name, "m", one);
        break;
      case OpRole::kQkvProjection:
      case OpRole::kProjection:
      case OpRole::kResidualAdd:
        // GemmStageTaskBody: one CUTLASS output tile per CTA. The residual
        // add is that GEMM's epilogue, so it owns the same tile.
        g.Tile(op.name, "m", Tm).Tile(op.name, "n", Tn);
        break;
      case OpRole::kRoPE:
        g.Tile(op.name, "m", one).Tile(op.name, "hh", one);
        break;
      case OpRole::kKVAppend:
        g.Tile(op.name, "row", one).Tile(op.name, "hh", one);
        break;
      case OpRole::kAttention:
        // AttentionChunkTaskBody: one (token, head) per CTA, no KV split.
        g.Tile(op.name, "s", one).Tile(op.name, "h", model.head_dim);
        break;
      case OpRole::kActivation:
        g.Tile(op.name, "m", one).Tile(op.name, "n", one);
        break;
      case OpRole::kGeneric:
        break;
    }
  }
  return g;
}

analysis::Granularity ReferenceGranularity(LiftedModel const& model) {
  analysis::Granularity g;
  ClosedForm const one = ClosedForm::Constant(1);
  ClosedForm const Tm = ClosedForm::Symbol("Tm");
  ClosedForm const Tn = ClosedForm::Symbol("Tn");
  ClosedForm const Tkv = ClosedForm::Symbol("Tkv");
  for (auto const& op : model.ops) {
    switch (op.role) {
      case OpRole::kNorm:
        g.Tile(op.name, "m", Tm);
        break;
      case OpRole::kQkvProjection:
        g.Tile(op.name, "m", Tm).Tile(op.name, "n", model.head_dim);
        break;
      case OpRole::kProjection:
      case OpRole::kResidualAdd:
      case OpRole::kActivation:
        g.Tile(op.name, "m", Tm).Tile(op.name, "n", Tn);
        break;
      case OpRole::kRoPE:
        g.Tile(op.name, "m", Tm).Tile(op.name, "hh", model.head_dim);
        break;
      case OpRole::kKVAppend:
        g.Tile(op.name, "row", one).Tile(op.name, "hh", model.head_dim);
        break;
      case OpRole::kAttention:
        g.Tile(op.name, "s", one)
            .Tile(op.name, "h", model.head_dim)
            .Split(op.name, Tkv);
        break;
      case OpRole::kGeneric:
        break;
    }
  }
  return g;
}

}  // namespace tilemega::frontend
