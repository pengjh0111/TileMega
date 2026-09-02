// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ReferenceModels.h>

#include <utility>

namespace tilemega::analysis {
namespace {

using Map = OperandAxisMap;

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

Operand In(std::string producer, TensorSpace tensor,
           std::vector<Map> axes) {
  Operand operand;
  operand.producer = std::move(producer);
  operand.tensor = std::move(tensor);
  operand.axes = std::move(axes);
  return operand;
}

OperatorNode Node(std::string name, OperatorKind kind, TensorSpace output,
                  std::vector<ClosedForm> tile, std::vector<Operand> operands) {
  OperatorNode node;
  node.name = std::move(name);
  node.kind = kind;
  node.output = std::move(output);
  node.tile = std::move(tile);
  node.operands = std::move(operands);
  return node;
}

/// A row-tiled matmul: out[m, n] = sum_k in[m, k] * weight[k, n].
/// `col_tile` defaults to Tn.  The QKV projections pass `d` instead: §2.7's
/// rows 2 and 3 only come out exact when the projection's column tile is one
/// head, because RoPE's task is one head of one row block.  With a column tile
/// unrelated to d the derivation correctly relaxes instead (see FINDINGS).
OperatorNode Matmul(std::string name, DecoderShape const& s,
                    ClosedForm out_cols, std::string input_producer,
                    TensorSpace input, ClosedForm col_tile = ClosedForm()) {
  if (col_tile.IsLiteral(0)) col_tile = s.Tn;  // default: the generic Tn
  std::size_t rank = input.axes.size();
  std::vector<Map> axes;
  axes.push_back(Map::Indexed(0));
  for (std::size_t i = 1; i < rank; ++i) axes.push_back(Map::FullRange());
  return Node(name, OperatorKind::kMatmul,
              Space(name, {Ax("m", s.S), Ax("n", std::move(out_cols))}),
              {s.Tm, std::move(col_tile)},
              {In(std::move(input_producer), std::move(input),
                  std::move(axes))});
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

OperatorGraph LlamaDecoderLayer(DecoderShape const& s,
                                std::string const& prefix) {
  OperatorGraph graph;
  auto n = [&](char const* base) { return prefix + base; };

  ClosedForm q_cols = s.n_h * s.d;
  ClosedForm kv_cols = s.n_kv * s.d;
  ClosedForm chunks = s.L_s.CeilDiv(s.Tkv);

  TensorSpace hidden = Space(n("hidden"), {Ax("m", s.S), Ax("h", s.H)});
  TensorSpace norm1 = Space(n("norm1"), {Ax("m", s.S), Ax("h", s.H)});

  // RMSNorm keeps the whole hidden axis inside one task: its tile equals the
  // axis extent, so that axis contributes no task coordinate and T is 1-D.
  graph.nodes.push_back(Node(
      n("rmsnorm1"), OperatorKind::kReduction, norm1, {s.Tm, s.H},
      {In("", hidden, {Map::Indexed(0), Map::FullRange()})}));

  graph.nodes.push_back(Matmul(n("wq"), s, q_cols, n("rmsnorm1"), norm1, s.d));
  graph.nodes.push_back(Matmul(n("wk"), s, kv_cols, n("rmsnorm1"), norm1, s.d));
  graph.nodes.push_back(Matmul(n("wv"), s, kv_cols, n("rmsnorm1"), norm1, s.d));

  TensorSpace q = Space(n("wq"), {Ax("m", s.S), Ax("n", q_cols)});
  TensorSpace k = Space(n("wk"), {Ax("m", s.S), Ax("n", kv_cols)});
  TensorSpace v = Space(n("wv"), {Ax("m", s.S), Ax("n", kv_cols)});

  // RoPE is pointwise per (token, head): its tile is one head of one row block.
  graph.nodes.push_back(Node(
      n("rope_q"), OperatorKind::kPointwise,
      Space(n("rope_q"), {Ax("m", s.S), Ax("hh", q_cols)}), {s.Tm, s.d},
      {In(n("wq"), q, {Map::Indexed(0), Map::Indexed(1)})}));
  graph.nodes.push_back(Node(
      n("rope_k"), OperatorKind::kPointwise,
      Space(n("rope_k"), {Ax("m", s.S), Ax("hh", kv_cols)}), {s.Tm, s.d},
      {In(n("wk"), k, {Map::Indexed(0), Map::Indexed(1)})}));

  TensorSpace q_rot = Space(n("rope_q"), {Ax("m", s.S), Ax("hh", q_cols)});
  TensorSpace k_rot = Space(n("rope_k"), {Ax("m", s.S), Ax("hh", kv_cols)});

  // The append writes rows [past, past + S) of a cache whose consumers address
  // absolute rows; the origin is what keeps W from claiming the whole cache.
  auto cache = [&](char const* base) {
    TensorAxis row = Ax("row", s.S);
    row.origin = s.past;
    return Space(n(base), {row, Ax("hh", kv_cols)}, "kv_cache");
  };
  graph.nodes.push_back(Node(
      n("kvappend_k"), OperatorKind::kConcat, cache("full_k"),
      {ClosedForm::Constant(1), s.d},
      {In(n("rope_k"), k_rot, {Map::Indexed(0), Map::Indexed(1)})}));
  graph.nodes.push_back(Node(
      n("kvappend_v"), OperatorKind::kConcat, cache("full_v"),
      {ClosedForm::Constant(1), s.d},
      {In(n("wv"), v, {Map::Indexed(0), Map::Indexed(1)})}));

  TensorSpace full_k =
      Space(n("full_k"), {RuntimeAx("row", s.L_s), Ax("hh", kv_cols)},
            "kv_cache");
  TensorSpace full_v =
      Space(n("full_v"), {RuntimeAx("row", s.L_s), Ax("hh", kv_cols)},
            "kv_cache");

  // One attention task per (token, query head, KV chunk).  The KV head is
  // floor(query_head / G): affine only with the grouped term, which is why
  // AffineExpr::Term carries `group`.
  Map kv_head = Map::Indexed(1, ClosedForm::Constant(1), s.group);
  graph.nodes.push_back(Node(
      n("attn_chunk"), OperatorKind::kPointwise,
      Space(n("partial"),
            {Ax("s", s.S), Ax("h", q_cols), RuntimeAx("j", chunks)}),
      {ClosedForm::Constant(1), s.d, ClosedForm::Constant(1)},
      {In(n("rope_q"), q_rot, {Map::Indexed(0), Map::Indexed(1)}),
       In(n("kvappend_k"), full_k, {Map::Indexed(2, s.Tkv), kv_head}),
       In(n("kvappend_v"), full_v, {Map::Indexed(2, s.Tkv), kv_head})}));

  TensorSpace partial =
      Space(n("partial"),
            {Ax("s", s.S), Ax("h", q_cols), RuntimeAx("j", chunks)});

  graph.nodes.push_back(Node(
      n("attn_combine"), OperatorKind::kReduction,
      Space(n("context"), {Ax("s", s.S), Ax("h", q_cols)}),
      {ClosedForm::Constant(1), s.d},
      {In(n("attn_chunk"), partial,
          {Map::Indexed(0), Map::Indexed(1), Map::FullRange()})}));

  TensorSpace context = Space(n("context"), {Ax("s", s.S), Ax("h", q_cols)});
  graph.nodes.push_back(Matmul(n("wo"), s, s.H, n("attn_combine"), context));

  TensorSpace attn_out = Space(n("wo"), {Ax("m", s.S), Ax("n", s.H)});
  TensorSpace resid1 = Space(n("resid1"), {Ax("m", s.S), Ax("n", s.H)});
  graph.nodes.push_back(Node(
      n("add1"), OperatorKind::kPointwise, resid1, {s.Tm, s.Tn},
      {In(n("wo"), attn_out, {Map::Indexed(0), Map::Indexed(1)}),
       In("", hidden, {Map::Indexed(0), Map::Indexed(1)})}));

  TensorSpace norm2 = Space(n("norm2"), {Ax("i", s.S), Ax("h", s.H)});
  graph.nodes.push_back(Node(
      n("rmsnorm2"), OperatorKind::kReduction, norm2, {s.Tm, s.H},
      {In(n("add1"), resid1, {Map::Indexed(0), Map::FullRange()})}));

  graph.nodes.push_back(Matmul(n("wgate"), s, s.I, n("rmsnorm2"), norm2));
  graph.nodes.push_back(Matmul(n("wup"), s, s.I, n("rmsnorm2"), norm2));

  TensorSpace gate = Space(n("wgate"), {Ax("m", s.S), Ax("n", s.I)});
  TensorSpace up = Space(n("wup"), {Ax("m", s.S), Ax("n", s.I)});
  graph.nodes.push_back(Node(
      n("silu"), OperatorKind::kPointwise,
      Space(n("act"), {Ax("m", s.S), Ax("n", s.I)}), {s.Tm, s.Tn},
      {In(n("wgate"), gate, {Map::Indexed(0), Map::Indexed(1)}),
       In(n("wup"), up, {Map::Indexed(0), Map::Indexed(1)})}));

  TensorSpace act = Space(n("act"), {Ax("m", s.S), Ax("n", s.I)});
  graph.nodes.push_back(Matmul(n("wdown"), s, s.H, n("silu"), act));

  TensorSpace mlp_out = Space(n("wdown"), {Ax("m", s.S), Ax("n", s.H)});
  graph.nodes.push_back(Node(
      n("add2"), OperatorKind::kPointwise,
      Space(n("resid2"), {Ax("m", s.S), Ax("n", s.H)}), {s.Tm, s.Tn},
      {In(n("wdown"), mlp_out, {Map::Indexed(0), Map::Indexed(1)}),
       In(n("add1"), resid1, {Map::Indexed(0), Map::Indexed(1)})}));

  return graph;
}

OperatorGraph LlamaStack(DecoderShape const& s, int layers) {
  OperatorGraph graph;
  for (int layer = 0; layer < layers; ++layer) {
    std::string prefix = "l" + std::to_string(layer) + ".";
    OperatorGraph one = LlamaDecoderLayer(s, prefix);
    if (layer > 0) {
      std::string previous = "l" + std::to_string(layer - 1) + ".add2";
      for (auto& node : one.nodes)
        for (auto& operand : node.operands)
          if (operand.producer.empty() &&
              operand.tensor.name == prefix + "hidden")
            operand.producer = previous;
    }
    for (auto& node : one.nodes) graph.nodes.push_back(std::move(node));
  }
  return graph;
}

OperatorGraph MlpStack(DecoderShape const& s, int blocks) {
  OperatorGraph graph;
  TensorSpace carry = Space("input", {Ax("m", s.S), Ax("n", s.H)});
  std::string producer;
  for (int block = 0; block < blocks; ++block) {
    std::string prefix = "b" + std::to_string(block) + ".";
    TensorSpace norm = Space(prefix + "norm", {Ax("m", s.S), Ax("h", s.H)});
    graph.nodes.push_back(Node(
        prefix + "norm", OperatorKind::kReduction, norm, {s.Tm, s.H},
        {In(producer, carry, {Map::Indexed(0), Map::FullRange()})}));
    graph.nodes.push_back(
        Matmul(prefix + "fc1", s, s.I, prefix + "norm", norm));
    TensorSpace fc1 = Space(prefix + "fc1", {Ax("m", s.S), Ax("n", s.I)});
    graph.nodes.push_back(Node(
        prefix + "gelu", OperatorKind::kPointwise,
        Space(prefix + "act", {Ax("m", s.S), Ax("n", s.I)}), {s.Tm, s.Tn},
        {In(prefix + "fc1", fc1, {Map::Indexed(0), Map::Indexed(1)})}));
    TensorSpace act = Space(prefix + "act", {Ax("m", s.S), Ax("n", s.I)});
    graph.nodes.push_back(Matmul(prefix + "fc2", s, s.H, prefix + "gelu", act));
    TensorSpace fc2 = Space(prefix + "fc2", {Ax("m", s.S), Ax("n", s.H)});
    TensorSpace next = Space(prefix + "resid", {Ax("m", s.S), Ax("n", s.H)});
    graph.nodes.push_back(Node(
        prefix + "add", OperatorKind::kPointwise, next, {s.Tm, s.Tn},
        {In(prefix + "fc2", fc2, {Map::Indexed(0), Map::Indexed(1)}),
         In(producer, carry, {Map::Indexed(0), Map::Indexed(1)})}));
    producer = prefix + "add";
    carry = next;
  }
  return graph;
}

OperatorGraph MhaModel(DecoderShape const& shape, int layers) {
  DecoderShape s = shape;
  s.n_kv = s.n_h;                        // no GQA: one KV head per query head
  s.group = ClosedForm::Constant(1);
  OperatorGraph graph;
  for (int layer = 0; layer < layers; ++layer) {
    OperatorGraph one =
        LlamaDecoderLayer(s, "mha" + std::to_string(layer) + ".");
    // Drop the gated MLP: keep a single fc pair, so the stage sequence differs
    // from Llama's twelve.
    for (auto& node : one.nodes) {
      if (node.name.find("wgate") != std::string::npos ||
          node.name.find("silu") != std::string::npos)
        continue;
      if (node.name.find("wdown") != std::string::npos) {
        node.operands.front().producer =
            "mha" + std::to_string(layer) + ".wup";
        node.operands.front().tensor.name =
            "mha" + std::to_string(layer) + ".wup";
      }
      graph.nodes.push_back(std::move(node));
    }
  }
  return graph;
}

OperatorGraph GatherModel(DecoderShape const& s) {
  OperatorGraph graph;
  TensorSpace table = Space("table", {Ax("m", s.S), Ax("n", s.H)});
  graph.nodes.push_back(Node(
      "produce", OperatorKind::kPointwise, table, {s.Tm, s.Tn},
      {In("", Space("src", {Ax("m", s.S), Ax("n", s.H)}),
          {Map::Indexed(0), Map::Indexed(1)})}));
  graph.nodes.push_back(Node(
      "gather", OperatorKind::kGather,
      Space("routed", {Ax("m", s.S), Ax("n", s.H)}), {s.Tm, s.Tn},
      {In("produce", table, {Map::DataDependent(), Map::Indexed(1)})}));
  return graph;
}

}  // namespace tilemega::analysis
