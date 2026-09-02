// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ReferenceModels.h>

#include <utility>

namespace tilemega::analysis {
namespace {

TensorAxis Ax(std::string name, ClosedForm extent) {
  TensorAxis axis;
  axis.name = std::move(name);
  axis.extent = std::move(extent);
  return axis;
}

TensorAxis RuntimeAx(std::string name, ClosedForm extent) {
  TensorAxis axis = Ax(std::move(name), std::move(extent));
  axis.runtime = true;
  return axis;
}

TensorSpace Space(std::string name, std::vector<TensorAxis> axes,
                  std::string layout_id = {}) {
  TensorSpace space;
  space.name = std::move(name);
  space.axes = std::move(axes);
  space.layout_id = std::move(layout_id);
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
  operand.effect.kind = EffectKind::kRead;
  return operand;
}

/// Elementwise result map over the leading `rank` domain dims, named after the
/// result axes; every op here writes its result that way.
IndexingMap Elementwise(TensorSpace const& result) {
  IndexingMap map;
  for (auto const& axis : result.axes)
    map.results.push_back(IndexResult::Dim(axis.name));
  return map;
}

SemanticOp Op(std::string name, OperatorKind kind,
              std::vector<IterationDim> domain, TensorSpace result,
              std::vector<SemanticOperand> operands) {
  SemanticOp op;
  op.name = std::move(name);
  op.kind = kind;
  op.domain = std::move(domain);
  op.result = std::move(result);
  op.result_map = Elementwise(op.result);
  op.result_effect.kind = EffectKind::kWrite;
  op.operands = std::move(operands);
  return op;
}

/// out[m, n] = sum_k in[m, k] * weight[k, n]: two parallel dims and one
/// reduction dim, declared splittable so §2.4's split-K is a granularity
/// choice rather than a TaskBody feature.
SemanticOp Matmul(std::string name, DecoderShape const& s, ClosedForm out_cols,
                  std::string input_producer, TensorSpace input) {
  std::vector<IndexResult> results;
  results.push_back(IndexResult::Dim("m"));
  for (std::size_t i = 1; i < input.axes.size(); ++i)
    results.push_back(IndexResult::Dim("k"));
  ClosedForm depth = input.axes.size() > 1 ? input.axes[1].extent
                                           : ClosedForm::Constant(1);
  std::vector<IterationDim> domain = {Par("m", s.S), Par("n", out_cols),
                                      Red("k", depth)};
  TensorSpace result = Space(name, {Ax("m", s.S), Ax("n", out_cols)});
  std::vector<SemanticOperand> operands = {Read(
      std::move(input_producer), std::move(input), std::move(results))};
  SemanticOp op = Op(name, OperatorKind::kMatmul, std::move(domain),
                     std::move(result), std::move(operands));
  op.reduction.splittable = true;
  op.reduction.dim = "k";
  op.reduction.reduction_operator = "add";
  op.reduction.partial_tensor = name + ".partial";
  op.reduction.combiner = name + ".combine";
  op.reduction.ownership = {"m", "n"};
  return op;
}

}  // namespace

ParamBinding DecoderShape::Table27Theta() {
  ParamBinding theta;
  theta.Bind("H", 4096)
      .Bind("n_h", 32)
      .Bind("n_kv", 8)
      .Bind("G", 4)
      .Bind("d", 128)
      .Bind("I", 14336);
  return theta;
}

ParamBinding DecoderShape::Table27G() {
  ParamBinding g;
  g.Bind("Tm", 128).Bind("Tn", 128).Bind("Tkv", 128);
  return g;
}

ReferenceModel LlamaDecoderLayerSem(DecoderShape const& s,
                                    std::string const& prefix) {
  ReferenceModel model;
  auto& graph = model.sem;
  auto& g = model.g;
  auto n = [&](char const* base) { return prefix + base; };

  ClosedForm q_cols = s.n_h * s.d;
  ClosedForm kv_cols = s.n_kv * s.d;

  TensorSpace hidden = Space(n("hidden"), {Ax("m", s.S), Ax("h", s.H)});
  TensorSpace norm1 = Space(n("norm1"), {Ax("m", s.S), Ax("h", s.H)});

  // The reduction dim `r` never reaches the result, so the operand's read
  // along it is a full-range access: the whole hidden row is inside one task.
  graph.ops.push_back(Op(
      n("rmsnorm1"), OperatorKind::kReduction,
      {Par("m", s.S), Par("h", s.H), Red("r", s.H)}, norm1,
      {Read("", hidden, {IndexResult::Dim("m"), IndexResult::Dim("r")})}));
  g.Tile(n("rmsnorm1"), "m", s.Tm);

  // The QKV projections tile their column axis by one head, not by Tn: §2.7
  // rows 2 and 3 are exact only because RoPE's task is one head of one row
  // block. A column tile unrelated to d relaxes instead (see FINDINGS).
  for (char const* name : {"wq", "wk", "wv"}) {
    ClosedForm cols = std::string(name) == "wq" ? q_cols : kv_cols;
    graph.ops.push_back(Matmul(n(name), s, cols, n("rmsnorm1"), norm1));
    g.Tile(n(name), "m", s.Tm).Tile(n(name), "n", s.d);
  }

  TensorSpace q = Space(n("wq"), {Ax("m", s.S), Ax("n", q_cols)});
  TensorSpace k = Space(n("wk"), {Ax("m", s.S), Ax("n", kv_cols)});
  TensorSpace v = Space(n("wv"), {Ax("m", s.S), Ax("n", kv_cols)});

  graph.ops.push_back(Op(
      n("rope_q"), OperatorKind::kPointwise, {Par("m", s.S), Par("hh", q_cols)},
      Space(n("rope_q"), {Ax("m", s.S), Ax("hh", q_cols)}),
      {Read(n("wq"), q, {IndexResult::Dim("m"), IndexResult::Dim("hh")})}));
  g.Tile(n("rope_q"), "m", s.Tm).Tile(n("rope_q"), "hh", s.d);
  graph.ops.push_back(Op(
      n("rope_k"), OperatorKind::kPointwise, {Par("m", s.S), Par("hh", kv_cols)},
      Space(n("rope_k"), {Ax("m", s.S), Ax("hh", kv_cols)}),
      {Read(n("wk"), k, {IndexResult::Dim("m"), IndexResult::Dim("hh")})}));
  g.Tile(n("rope_k"), "m", s.Tm).Tile(n("rope_k"), "hh", s.d);

  TensorSpace q_rot = Space(n("rope_q"), {Ax("m", s.S), Ax("hh", q_cols)});
  TensorSpace k_rot = Space(n("rope_k"), {Ax("m", s.S), Ax("hh", kv_cols)});

  // The append writes rows [past, past + S) of a cache whose consumers address
  // absolute rows; the origin is what keeps W from claiming the whole cache.
  // Its effect is a read-modify-write on a state object, not a fresh result.
  auto cache = [&](char const* base) {
    TensorAxis row = Ax("row", s.S);
    row.origin = s.past;
    return Space(n(base), {row, Ax("hh", kv_cols)}, "kv_cache");
  };
  auto append = [&](char const* name, char const* out, std::string producer,
                    TensorSpace source) {
    SemanticOp op = Op(n(name), OperatorKind::kConcat,
                       {Par("row", s.S), Par("hh", kv_cols)}, cache(out),
                       {Read(std::move(producer), std::move(source),
                             {IndexResult::Dim("row"), IndexResult::Dim("hh")})});
    op.result_effect.kind = EffectKind::kReadWrite;
    op.result_effect.state_object = "kv_cache";
    op.result_effect.alias_set = "kv_cache";
    graph.ops.push_back(std::move(op));
    g.Tile(n(name), "row", ClosedForm::Constant(1)).Tile(n(name), "hh", s.d);
  };
  append("kvappend_k", "full_k", n("rope_k"), k_rot);
  append("kvappend_v", "full_v", n("wv"), v);

  auto full = [&](char const* base) {
    return Space(n(base), {RuntimeAx("row", s.L_s), Ax("hh", kv_cols)},
                 "kv_cache");
  };
  // The KV head is floor(query_head / G): affine only with the grouped term,
  // which is why IndexResult::Term carries `group`.
  SemanticOperand full_k = Read(
      n("kvappend_k"), full("full_k"),
      {IndexResult::Dim("kv"),
       IndexResult::Dim("h", ClosedForm::Constant(1), s.group)});
  full_k.effect.state_object = "kv_cache";
  full_k.effect.alias_set = "kv_cache";
  SemanticOperand full_v = full_k;
  full_v.producer = n("kvappend_v");
  full_v.tensor = full("full_v");

  // Attention is one op over the whole KV range; FlashDecoding's chunk/combine
  // pair is what §2.4's Split produces from it at a given Tkv, not a separate
  // hand-written operator.
  SemanticOp attn = Op(
      n("attn_chunk"), OperatorKind::kPointwise,
      {Par("s", s.S), Par("h", q_cols), Red("kv", s.L_s, /*runtime=*/true)},
      Space(n("context"), {Ax("s", s.S), Ax("h", q_cols)}),
      {Read(n("rope_q"), q_rot,
            {IndexResult::Dim("s"), IndexResult::Dim("h")}),
       full_k, full_v});
  attn.reduction.splittable = true;
  attn.reduction.dim = "kv";
  attn.reduction.reduction_operator = "flash_combine";
  attn.reduction.partial_tensor = n("partial");
  attn.reduction.combiner = n("attn_combine");
  attn.reduction.ownership = {"s", "h"};
  graph.ops.push_back(std::move(attn));
  g.Tile(n("attn_chunk"), "s", ClosedForm::Constant(1))
      .Tile(n("attn_chunk"), "h", s.d)
      .Split(n("attn_chunk"), s.Tkv);

  TensorSpace context = Space(n("context"), {Ax("s", s.S), Ax("h", q_cols)});
  graph.ops.push_back(Matmul(n("wo"), s, s.H, n("attn_combine"), context));
  g.Tile(n("wo"), "m", s.Tm).Tile(n("wo"), "n", s.Tn);

  TensorSpace attn_out = Space(n("wo"), {Ax("m", s.S), Ax("n", s.H)});
  TensorSpace resid1 = Space(n("resid1"), {Ax("m", s.S), Ax("n", s.H)});
  graph.ops.push_back(Op(
      n("add1"), OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.H)},
      resid1,
      {Read(n("wo"), attn_out, {IndexResult::Dim("m"), IndexResult::Dim("n")}),
       Read("", hidden, {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
  g.Tile(n("add1"), "m", s.Tm).Tile(n("add1"), "n", s.Tn);

  TensorSpace norm2 = Space(n("norm2"), {Ax("i", s.S), Ax("h", s.H)});
  graph.ops.push_back(Op(
      n("rmsnorm2"), OperatorKind::kReduction,
      {Par("i", s.S), Par("h", s.H), Red("r", s.H)}, norm2,
      {Read(n("add1"), resid1, {IndexResult::Dim("i"), IndexResult::Dim("r")})}));
  g.Tile(n("rmsnorm2"), "i", s.Tm);

  graph.ops.push_back(Matmul(n("wgate"), s, s.I, n("rmsnorm2"), norm2));
  g.Tile(n("wgate"), "m", s.Tm).Tile(n("wgate"), "n", s.Tn);
  graph.ops.push_back(Matmul(n("wup"), s, s.I, n("rmsnorm2"), norm2));
  g.Tile(n("wup"), "m", s.Tm).Tile(n("wup"), "n", s.Tn);

  TensorSpace gate = Space(n("wgate"), {Ax("m", s.S), Ax("n", s.I)});
  TensorSpace up = Space(n("wup"), {Ax("m", s.S), Ax("n", s.I)});
  graph.ops.push_back(Op(
      n("silu"), OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.I)},
      Space(n("act"), {Ax("m", s.S), Ax("n", s.I)}),
      {Read(n("wgate"), gate, {IndexResult::Dim("m"), IndexResult::Dim("n")}),
       Read(n("wup"), up, {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
  g.Tile(n("silu"), "m", s.Tm).Tile(n("silu"), "n", s.Tn);

  TensorSpace act = Space(n("act"), {Ax("m", s.S), Ax("n", s.I)});
  graph.ops.push_back(Matmul(n("wdown"), s, s.H, n("silu"), act));
  g.Tile(n("wdown"), "m", s.Tm).Tile(n("wdown"), "n", s.Tn);

  TensorSpace mlp_out = Space(n("wdown"), {Ax("m", s.S), Ax("n", s.H)});
  graph.ops.push_back(Op(
      n("add2"), OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.H)},
      Space(n("resid2"), {Ax("m", s.S), Ax("n", s.H)}),
      {Read(n("wdown"), mlp_out,
            {IndexResult::Dim("m"), IndexResult::Dim("n")}),
       Read(n("add1"), resid1,
            {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
  g.Tile(n("add2"), "m", s.Tm).Tile(n("add2"), "n", s.Tn);

  return model;
}

ReferenceModel LlamaStackSem(DecoderShape const& s, int layers) {
  ReferenceModel model;
  for (int layer = 0; layer < layers; ++layer) {
    std::string prefix = "l" + std::to_string(layer) + ".";
    ReferenceModel one = LlamaDecoderLayerSem(s, prefix);
    if (layer > 0) {
      std::string previous = "l" + std::to_string(layer - 1) + ".add2";
      for (auto& op : one.sem.ops)
        for (auto& operand : op.operands)
          if (operand.producer.empty() &&
              operand.tensor.name == prefix + "hidden")
            operand.producer = previous;
    }
    for (auto& op : one.sem.ops) model.sem.ops.push_back(std::move(op));
    for (auto const& [op, dims] : one.g.tiles)
      for (auto const& [dim, tile] : dims) model.g.Tile(op, dim, tile);
    for (auto const& [op, chunk] : one.g.reduction_chunk)
      model.g.Split(op, chunk);
  }
  return model;
}

ReferenceModel MlpStackSem(DecoderShape const& s, int blocks) {
  ReferenceModel model;
  auto& graph = model.sem;
  auto& g = model.g;
  TensorSpace carry = Space("input", {Ax("m", s.S), Ax("n", s.H)});
  std::string producer;
  for (int block = 0; block < blocks; ++block) {
    std::string prefix = "b" + std::to_string(block) + ".";
    TensorSpace norm = Space(prefix + "norm", {Ax("m", s.S), Ax("h", s.H)});
    graph.ops.push_back(Op(
        prefix + "norm", OperatorKind::kReduction,
        {Par("m", s.S), Par("h", s.H), Red("r", s.H)}, norm,
        {Read(producer, carry,
              {IndexResult::Dim("m"), IndexResult::Dim("r")})}));
    g.Tile(prefix + "norm", "m", s.Tm);

    graph.ops.push_back(Matmul(prefix + "fc1", s, s.I, prefix + "norm", norm));
    g.Tile(prefix + "fc1", "m", s.Tm).Tile(prefix + "fc1", "n", s.Tn);
    TensorSpace fc1 = Space(prefix + "fc1", {Ax("m", s.S), Ax("n", s.I)});
    graph.ops.push_back(Op(
        prefix + "gelu", OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.I)},
        Space(prefix + "act", {Ax("m", s.S), Ax("n", s.I)}),
        {Read(prefix + "fc1", fc1,
              {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
    g.Tile(prefix + "gelu", "m", s.Tm).Tile(prefix + "gelu", "n", s.Tn);
    TensorSpace act = Space(prefix + "act", {Ax("m", s.S), Ax("n", s.I)});
    graph.ops.push_back(Matmul(prefix + "fc2", s, s.H, prefix + "gelu", act));
    g.Tile(prefix + "fc2", "m", s.Tm).Tile(prefix + "fc2", "n", s.Tn);
    TensorSpace fc2 = Space(prefix + "fc2", {Ax("m", s.S), Ax("n", s.H)});
    TensorSpace next = Space(prefix + "resid", {Ax("m", s.S), Ax("n", s.H)});
    graph.ops.push_back(Op(
        prefix + "add", OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.H)},
        next,
        {Read(prefix + "fc2", fc2,
              {IndexResult::Dim("m"), IndexResult::Dim("n")}),
         Read(producer, carry,
              {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
    g.Tile(prefix + "add", "m", s.Tm).Tile(prefix + "add", "n", s.Tn);
    producer = prefix + "add";
    carry = next;
  }
  return model;
}

ReferenceModel MhaModelSem(DecoderShape const& shape, int layers) {
  DecoderShape s = shape;
  s.n_kv = s.n_h;                        // no GQA: one KV head per query head
  s.group = ClosedForm::Constant(1);
  ReferenceModel model;
  for (int layer = 0; layer < layers; ++layer) {
    std::string prefix = "mha" + std::to_string(layer) + ".";
    ReferenceModel one = LlamaDecoderLayerSem(s, prefix);
    // Drop the gated MLP: keep a single fc pair, so the stage sequence differs
    // from Llama's twelve.
    for (auto& op : one.sem.ops) {
      if (op.name.find("wgate") != std::string::npos ||
          op.name.find("silu") != std::string::npos)
        continue;
      if (op.name.find("wdown") != std::string::npos) {
        op.operands.front().producer = prefix + "wup";
        op.operands.front().tensor.name = prefix + "wup";
      }
      model.sem.ops.push_back(std::move(op));
    }
    for (auto const& [op, dims] : one.g.tiles)
      for (auto const& [dim, tile] : dims) model.g.Tile(op, dim, tile);
    for (auto const& [op, chunk] : one.g.reduction_chunk)
      model.g.Split(op, chunk);
  }
  return model;
}

ReferenceModel MisalignedTileModelSem(DecoderShape const& s, long producer_tile,
                                      long consumer_tile) {
  ReferenceModel model;
  ClosedForm rows = s.S;
  ClosedForm width = s.H;
  TensorSpace tensor = Space("staged", {Ax("r", rows), Ax("c", width)});

  model.sem.ops.push_back(Op(
      "produce", OperatorKind::kPointwise, {Par("r", rows), Par("c", width)},
      tensor,
      {Read("", Space("src", {Ax("r", rows), Ax("c", width)}),
            {IndexResult::Dim("r"), IndexResult::FullRange()})}));
  model.g.Tile("produce", "r", ClosedForm::Constant(producer_tile));

  model.sem.ops.push_back(Op(
      "consume", OperatorKind::kPointwise, {Par("r", rows), Par("c", width)},
      Space("consumed", {Ax("r", rows), Ax("c", width)}),
      {Read("produce", tensor,
            {IndexResult::Dim("r"), IndexResult::FullRange()})}));
  model.g.Tile("consume", "r", ClosedForm::Constant(consumer_tile));
  return model;
}

ReferenceModel GatherModelSem(DecoderShape const& s, bool data_dependent) {
  ReferenceModel model;
  TensorSpace table = Space("table", {Ax("m", s.S), Ax("n", s.H)});
  model.sem.ops.push_back(Op(
      "produce", OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.H)}, table,
      {Read("", Space("src", {Ax("m", s.S), Ax("n", s.H)}),
            {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
  model.g.Tile("produce", "m", s.Tm).Tile("produce", "n", s.Tn);
  model.sem.ops.push_back(Op(
      "gather", OperatorKind::kGather, {Par("m", s.S), Par("n", s.H)},
      Space("routed", {Ax("m", s.S), Ax("n", s.H)}),
      {Read("produce", table,
            {data_dependent ? IndexResult::DataDependent()
                            : IndexResult::Dim("m"),
             IndexResult::Dim("n")})}));
  model.g.Tile("gather", "m", s.Tm).Tile("gather", "n", s.Tn);
  return model;
}

ReferenceModel UnknownOperatorModelSem(DecoderShape const& s) {
  ReferenceModel model;
  TensorSpace source = Space("src", {Ax("m", s.S), Ax("n", s.H)});
  TensorSpace staged = Space("staged", {Ax("m", s.S), Ax("n", s.H)});
  model.sem.ops.push_back(Op(
      "produce", OperatorKind::kPointwise, {Par("m", s.S), Par("n", s.H)},
      staged,
      {Read("", source, {IndexResult::Dim("m"), IndexResult::Dim("n")})}));
  model.g.Tile("produce", "m", s.Tm).Tile("produce", "n", s.Tn);
  // No pattern recognizes this operator, so it gets the conservative generic
  // semantics: one task space, elementwise result, full-range reads.
  model.sem.ops.push_back(GenericSemantics(
      "mystery", Space("mystery", {Ax("m", s.S), Ax("n", s.H)}),
      {Read("produce", staged, {})}));
  return model;
}

OperatorGraph LlamaDecoderLayer(DecoderShape const& s,
                                std::string const& prefix) {
  return LlamaDecoderLayerSem(s, prefix).Task();
}

OperatorGraph LlamaStack(DecoderShape const& s, int layers) {
  return LlamaStackSem(s, layers).Task();
}

OperatorGraph MlpStack(DecoderShape const& s, int blocks) {
  return MlpStackSem(s, blocks).Task();
}

OperatorGraph MhaModel(DecoderShape const& s, int layers) {
  return MhaModelSem(s, layers).Task();
}

OperatorGraph MisalignedTileModel(DecoderShape const& s, long producer_tile,
                                  long consumer_tile) {
  return MisalignedTileModelSem(s, producer_tile, consumer_tile).Task();
}

OperatorGraph GatherModel(DecoderShape const& s, bool data_dependent) {
  return GatherModelSem(s, data_dependent).Task();
}

OperatorGraph UnknownOperatorModel(DecoderShape const& s) {
  return UnknownOperatorModelSem(s).Task();
}

}  // namespace tilemega::analysis
