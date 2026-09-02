// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/ModelPlan.h>

#include <tilemega/Frontend/GraphPattern.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace tilemega::frontend {
namespace {

constexpr std::uint32_t kNoOperand = std::numeric_limits<std::uint32_t>::max();

std::string FileComponent(std::string value) {
  for (char& c : value)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return value;
}

std::uint32_t NumericElements(FxNodeRecord const& node) {
  std::uint64_t product = 1;
  if (node.shape.empty())
    throw std::runtime_error("placeholder has no tensor shape: " + node.name);
  for (auto const& extent : node.shape) {
    if (extent.empty() || !std::all_of(extent.begin(), extent.end(), ::isdigit))
      throw std::runtime_error("expected a static parameter shape for " +
                               node.name + "; got " + extent);
    product *= std::stoull(extent);
  }
  if (product > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("parameter is too large for Phase-3 table: " +
                             node.name);
  return static_cast<std::uint32_t>(product);
}

std::uint32_t StaticExtent(FxNodeRecord const& node, std::size_t axis) {
  if (axis >= node.shape.size())
    throw std::runtime_error("missing shape axis on " + node.name);
  auto const& value = node.shape[axis];
  if (value.empty() || !std::all_of(value.begin(), value.end(), ::isdigit))
    throw std::runtime_error("expected static shape axis on " + node.name +
                             "; got " + value);
  return static_cast<std::uint32_t>(std::stoul(value));
}

struct PlanBuilder {
  explicit PlanBuilder(std::vector<FxNodeRecord> const& records)
      : nodes(records) {
    for (auto const& node : nodes) by_name.emplace(node.name, &node);
  }

  FxNodeRecord const& Node(std::string const& name) const {
    auto found = by_name.find(name);
    if (found == by_name.end())
      throw std::runtime_error("model-plan reference names unknown FX node: " + name);
    return *found->second;
  }

  bool DependsOn(std::string const& value, std::string const& ancestor) const {
    if (value == ancestor) return true;
    std::vector<std::string> work{value};
    std::unordered_set<std::string> seen;
    while (!work.empty()) {
      std::string current = std::move(work.back());
      work.pop_back();
      if (!seen.insert(current).second) continue;
      auto found = by_name.find(current);
      if (found == by_name.end()) continue;
      for (auto const& input : found->second->inputs) {
        if (input == ancestor) return true;
        work.push_back(input);
      }
    }
    return false;
  }

  std::uint32_t Buffer(PlanBuffer buffer) {
    auto found = buffer_id.find(buffer.name);
    if (found != buffer_id.end()) return found->second;
    std::uint32_t id = static_cast<std::uint32_t>(plan.buffers.size());
    buffer_id.emplace(buffer.name, id);
    plan.buffers.push_back(std::move(buffer));
    return id;
  }

  std::uint32_t Weight(SignatureInput const& input) {
    FxNodeRecord const& node = Node(input.name);
    return Buffer({input.name, NumericElements(node), 0, 0, 0,
                   PlanBuffer::Source::kWeight,
                   "state_" + FileComponent(input.target) + ".bin"});
  }

  std::uint32_t Scratch(std::string name, std::uint32_t per_seq) {
    return Buffer({std::move(name), 0, per_seq, 0, 0,
                   PlanBuffer::Source::kZero, {}});
  }

  std::uint32_t Gemm(std::uint32_t a, std::uint32_t b, std::uint32_t c,
                     std::uint32_t d, std::uint32_t n, std::uint32_t k,
                     float beta) {
    plan.gemms.push_back({n, k, a, b, c, d, beta});
    return static_cast<std::uint32_t>(plan.gemms.size() - 1);
  }

  void Stage(PlanTaskKind kind, std::string const& representative,
             std::uint32_t gemm, std::uint32_t extent,
             std::uint32_t width, std::uint32_t group,
             std::initializer_list<std::uint32_t> operands = {}) {
    PlanStage stage;
    stage.kind = kind;
    stage.gemm = gemm;
    stage.extent = extent;
    stage.width = width;
    stage.group = group;
    stage.operands.fill(kNoOperand);
    std::copy(operands.begin(), operands.end(), stage.operands.begin());
    stage.representative = representative;
    stage.representative_index = Node(representative).index;
    if (!plan.stages.empty() &&
        stage.representative_index <= plan.stages.back().representative_index)
      throw std::runtime_error("semantic stages are not in FX topological order at " +
                               representative);
    plan.stages.push_back(std::move(stage));
  }

  std::vector<FxNodeRecord> const& nodes;
  std::unordered_map<std::string, FxNodeRecord const*> by_name;
  std::unordered_map<std::string, std::uint32_t> buffer_id;
  ModelPlan plan;
};

/// The decoder layer, stated as use-def structure. Nothing here names a
/// module, a parameter or a layer index: the anchor is the KV concat, and
/// every other slot is pinned relative to it. `Value` is exact modulo
/// layout-only operators, `Dep`/`ancestor_of` are transitive, which is what
/// makes one pattern cover both the composite export and its Core ATen form.
GraphPattern const& DecoderLayerPattern() {
  static GraphPattern const pattern = {
      "decoder_layer",
      {
          // The layer starts at its input norm: the three projections that
          // read one scaled tensor are Q, K and V by definition, and pinning
          // them to a shared operand is what keeps a match inside one layer.
          {"norm1", "multiply", {Any(), Param()}},
          {"q", "contraction", {Value("norm1"), Param()}},
          {"k", "contraction", {Value("norm1"), Param()}},
          {"v", "contraction", {Value("norm1"), Param()}},
          // Each cache appends its own projection to a model input; that is
          // what tells K from V without naming either.
          {"cat_k", "concat", {Input(), Near("contraction", "k")}},
          {"cat_v", "concat", {Input(), Near("contraction", "v")}},
          // Scores read the key cache and not the value cache; the context
          // matmul is the other way round.
          {"score", "contraction",
           {Near("contraction", "q"), Near("concat", "cat_k")}},
          {"prob", "softmax", {Near("contraction", "score")}},
          {"ctx", "contraction", {Value("prob"), Dep("cat_v")}},
          {"o", "contraction", {Value("ctx"), Param()}},
          // The residual takes the projection's own result, not merely
          // something downstream of it, so `Value` and not `Dep`.
          {"resid1", "add", {Value("o")}, /*unordered=*/true},
          // SwiGLU: the gate is the projection under the activation, the up
          // projection is the other operand of the product. `resid2` pins the
          // whole group back to this layer's first residual.
          {"gate", "contraction", {Dep("resid1"), Param()}},
          {"act", "activation", {Value("gate")}},
          {"up", "contraction", {Dep("resid1"), Param()}},
          // `Dep` and not `Value` on the activation: SiLU is one node before
          // `run_decompositions()` and `sigmoid` followed by a product after,
          // so the gate arrives at the product through one more multiply.
          {"swiglu", "multiply", {Dep("act"), Value("up")}, true},
          {"down", "contraction", {Value("swiglu"), Param()}},
          {"resid2", "add", {Value("down"), Value("resid1")}, true},
      },
      // Both residuals carry the hidden width; the eps add inside an RMSNorm
      // does not, which is the only other `add` reachable from `o`.
      {{"resid1", -1, "o", -1}, {"resid2", -1, "down", -1}},
  };
  return pattern;
}

}  // namespace

ModelPlan BuildModelPlan(std::vector<FxNodeRecord> const& nodes,
                         std::vector<SignatureInput> const& inputs,
                         std::vector<std::string> const& outputs) {
  PatternMatcher matcher(nodes, inputs);
  PlanBuilder builder(nodes);
  std::unordered_map<std::string, SignatureInput const*> signature_by_name;
  for (auto const& input : inputs) signature_by_name.emplace(input.name, &input);
  std::vector<PatternBinding> layers = matcher.FindAll(DecoderLayerPattern());
  std::sort(layers.begin(), layers.end(),
            [](PatternBinding const& a, PatternBinding const& b) {
              return a.at("resid2")->index < b.at("resid2")->index;
            });
  for (std::size_t i = 1; i < layers.size(); ++i)
    if (layers[i].at("resid2") == layers[i - 1].at("resid2"))
      throw std::runtime_error("the decoder-layer pattern is ambiguous at " +
                               layers[i].at("resid2")->name);
  // Degradation, not refusal (skeleton §0.1): a graph the decoder pattern does
  // not cover still imports, as one task space per operator, with no plan.
  if (layers.empty()) return {};

  // The hidden state is the model input the first layer's query projection
  // reads; the KV inputs never reach it, so no name convention is needed.
  SignatureInput const* hidden_input = nullptr;
  for (auto const& input : inputs)
    if (input.kind == "USER_INPUT" &&
        matcher.DependsOn(layers.front().at("q")->name, input.name)) {
      hidden_input = &input;
      break;
    }
  if (!hidden_input)
    throw std::runtime_error("no model input reaches the first query projection");

  auto weight = [&](std::string const& operand) -> std::uint32_t {
    std::string name = matcher.Value(operand);
    auto found = signature_by_name.find(name);
    if (found == signature_by_name.end())
      throw std::runtime_error("expected a parameter operand; got " + name);
    return builder.Weight(*found->second);
  };

  FxNodeRecord const& hidden_node = builder.Node(hidden_input->name);
  std::uint32_t hidden = StaticExtent(hidden_node, hidden_node.shape.size() - 1);
  std::uint32_t current_hidden = builder.Buffer(
      {hidden_input->name, 0, hidden, 0, 0, PlanBuffer::Source::kFixture,
       "input_" + hidden_input->name + ".bin"});

  std::unordered_map<std::string, std::uint32_t> semantic_output;
  for (std::size_t number = 0; number < layers.size(); ++number) {
    PatternBinding const& match = layers[number];
    FxNodeRecord const& q_node = *match.at("q");
    FxNodeRecord const& k_node = *match.at("k");
    FxNodeRecord const& v_node = *match.at("v");
    FxNodeRecord const& o_node = *match.at("o");
    FxNodeRecord const& gate_node = *match.at("gate");
    FxNodeRecord const& up_node = *match.at("up");
    FxNodeRecord const& down_node = *match.at("down");
    FxNodeRecord const& cat_k = *match.at("cat_k");
    FxNodeRecord const& cat_v = *match.at("cat_v");
    FxNodeRecord const& score = *match.at("score");
    FxNodeRecord const& residual1 = *match.at("resid1");
    FxNodeRecord const& residual2 = *match.at("resid2");
    if (matcher.Value(q_node.inputs[0]) != matcher.Value(k_node.inputs[0]) ||
        matcher.Value(q_node.inputs[0]) != matcher.Value(v_node.inputs[0]))
      throw std::runtime_error("Q/K/V projections do not share one norm output: " +
                               q_node.name + "/" + k_node.name + "/" + v_node.name);
    if (matcher.Value(gate_node.inputs[0]) != matcher.Value(up_node.inputs[0]))
      throw std::runtime_error("gate/up projections do not share one norm output");

    std::uint32_t q_width = StaticExtent(
        builder.Node(matcher.Value(q_node.inputs[1])), 0);
    std::uint32_t kv_width = StaticExtent(
        builder.Node(matcher.Value(k_node.inputs[1])), 0);
    std::uint32_t intermediate = StaticExtent(
        builder.Node(matcher.Value(up_node.inputs[1])), 0);
    // The head dimension is the contracted axis of the score matmul, so it is
    // read off the attention rather than off a RoPE frequency table.
    // The rotated query, not the reshape that feeds the score matmul: after
    // normalization that reshape sits behind the KV concats in FX order, and
    // stage representatives must stay topologically ordered.
    FxNodeRecord const& q_rot = builder.Node(matcher.Value(score.inputs[0]));
    if (q_rot.shape.empty())
      throw std::runtime_error("attention operand has no shape: " + q_rot.name);
    std::uint32_t head_dim = StaticExtent(q_rot, q_rot.shape.size() - 1);
    if (!head_dim || q_width % head_dim || kv_width % head_dim)
      throw std::runtime_error("Q/K/V widths are incompatible with RoPE head dim");
    std::uint32_t heads = q_width / head_dim;
    std::uint32_t kv_heads = kv_width / head_dim;
    if (!kv_heads || heads % kv_heads)
      throw std::runtime_error("query heads are not divisible by KV heads");

    // The RoPE table is the parameter behind the rotation's own cosine: the
    // nearest parameter to the rotated query is the projection weight, and a
    // plain dependence walk would find the previous layer's table as well.
    // The past-KV tensors are the concat operands that are model inputs.
    std::string inv_freq_name = matcher.NearestParameter(
        matcher.FirstOfRole(q_rot.name, "trig", "contraction"));
    auto inv_freq_sig = signature_by_name.find(inv_freq_name);
    std::string past_k_name, past_v_name;
    for (auto const& operand : cat_k.inputs) {
      auto found = signature_by_name.find(matcher.Value(operand));
      if (found != signature_by_name.end() && found->second->kind == "USER_INPUT")
        past_k_name = found->first;
    }
    for (auto const& operand : cat_v.inputs) {
      auto found = signature_by_name.find(matcher.Value(operand));
      if (found != signature_by_name.end() && found->second->kind == "USER_INPUT")
        past_v_name = found->first;
    }
    if (past_k_name.empty() || past_v_name.empty() ||
        inv_freq_sig == signature_by_name.end())
      throw std::runtime_error("layer " + std::to_string(number) +
                               " has no explicit KV inputs or RoPE table");

    std::string prefix = "l" + std::to_string(number) + ".";
    std::uint32_t next_hidden = builder.Scratch(prefix + "hidden", hidden);
    std::uint32_t norm = builder.Scratch(prefix + "norm", hidden);
    std::uint32_t q = builder.Scratch(prefix + "q", q_width);
    std::uint32_t k = builder.Scratch(prefix + "k", kv_width);
    std::uint32_t v = builder.Scratch(prefix + "v", kv_width);
    std::uint32_t q_rot_buffer = builder.Scratch(prefix + "q_rot", q_width);
    std::uint32_t k_rot = builder.Scratch(prefix + "k_rot", kv_width);
    std::uint32_t context = builder.Scratch(prefix + "context", q_width);
    std::uint32_t gate = builder.Scratch(prefix + "gate", intermediate);
    std::uint32_t up = builder.Scratch(prefix + "up", intermediate);
    std::uint32_t wn1 = weight(matcher.NearestParameter(
        matcher.Value(q_node.inputs[0])));
    std::uint32_t wn2 = weight(matcher.NearestParameter(
        matcher.Value(gate_node.inputs[0])));
    std::uint32_t wq = weight(q_node.inputs[1]);
    std::uint32_t wk = weight(k_node.inputs[1]);
    std::uint32_t wv = weight(v_node.inputs[1]);
    std::uint32_t wo = weight(o_node.inputs[1]);
    std::uint32_t wg = weight(gate_node.inputs[1]);
    std::uint32_t wu = weight(up_node.inputs[1]);
    std::uint32_t wd = weight(down_node.inputs[1]);
    std::uint32_t inv = builder.Weight(*inv_freq_sig->second);
    std::uint32_t past_k = builder.Buffer(
        {past_k_name, 0, 0, kv_width, 0, PlanBuffer::Source::kFixture,
         "input_" + past_k_name + ".bin"});
    std::uint32_t past_v = builder.Buffer(
        {past_v_name, 0, 0, kv_width, 0, PlanBuffer::Source::kFixture,
         "input_" + past_v_name + ".bin"});
    std::uint32_t full_k = builder.Buffer(
        {prefix + "full_k", 0, 0, 0, kv_width,
         PlanBuffer::Source::kZero, {}});
    std::uint32_t full_v = builder.Buffer(
        {prefix + "full_v", 0, 0, 0, kv_width,
         PlanBuffer::Source::kZero, {}});

    builder.Stage(PlanTaskKind::kRMSNorm, q_node.inputs[0], 0, 0, hidden, 1,
                  {current_hidden, wn1, norm});
    builder.Stage(PlanTaskKind::kGemm, q_node.name,
                  builder.Gemm(norm, wq, q, q, q_width, hidden, 0.0f),
                  0, 0, 1);
    builder.Stage(PlanTaskKind::kGemm, k_node.name,
                  builder.Gemm(norm, wk, k, k, kv_width, hidden, 0.0f),
                  0, 0, 1);
    builder.Stage(PlanTaskKind::kGemm, v_node.name,
                  builder.Gemm(norm, wv, v, v, kv_width, hidden, 0.0f),
                  0, 0, 1);
    builder.Stage(PlanTaskKind::kRoPE, q_rot.name, 0, heads, head_dim, 1,
                  {q, q_rot_buffer, inv});
    builder.Stage(PlanTaskKind::kRoPE, cat_k.inputs[1], 0, kv_heads,
                  head_dim, 1, {k, k_rot, inv});
    builder.Stage(PlanTaskKind::kKVAppend, cat_k.name, 0, kv_heads, head_dim, 1,
                  {k_rot, past_k, full_k});
    builder.Stage(PlanTaskKind::kKVAppend, cat_v.name, 0, kv_heads, head_dim, 1,
                  {v, past_v, full_v});
    builder.Stage(PlanTaskKind::kAttention, o_node.inputs[0], 0, heads,
                  head_dim, heads / kv_heads,
                  {q_rot_buffer, full_k, full_v, context});
    builder.Stage(PlanTaskKind::kGemm, residual1.name,
                  builder.Gemm(context, wo, current_hidden, next_hidden,
                               hidden, q_width, 1.0f),
                  0, 0, 1);
    builder.Stage(PlanTaskKind::kRMSNorm, gate_node.inputs[0], 0, 0, hidden, 1,
                  {next_hidden, wn2, norm});
    builder.Stage(PlanTaskKind::kGemm, gate_node.name,
                  builder.Gemm(norm, wg, gate, gate, intermediate, hidden, 0.0f),
                  0, 0, 1);
    builder.Stage(PlanTaskKind::kGemm, up_node.name,
                  builder.Gemm(norm, wu, up, up, intermediate, hidden, 0.0f),
                  0, 0, 1);
    builder.Stage(PlanTaskKind::kElementwise, down_node.inputs[0], 0,
                  intermediate, 0, 1, {gate, up, gate});
    builder.Stage(PlanTaskKind::kGemm, residual2.name,
                  builder.Gemm(gate, wd, next_hidden, next_hidden,
                               hidden, intermediate, 1.0f),
                  0, 0, 1);

    semantic_output[residual2.name] = next_hidden;
    semantic_output[cat_k.name] = full_k;
    semantic_output[cat_v.name] = full_v;
    builder.plan.node_buffer[residual2.name] = next_hidden;
    builder.plan.node_buffer[cat_k.name] = full_k;
    builder.plan.node_buffer[cat_v.name] = full_v;
    current_hidden = next_hidden;
  }

  for (std::size_t index = 0; index < outputs.size(); ++index) {
    auto found = semantic_output.find(outputs[index]);
    if (found == semantic_output.end())
      throw std::runtime_error("unsupported semantic model output: " + outputs[index]);
    builder.plan.outputs.push_back(
        {found->second, "reference_" + std::to_string(index) + ".bin"});
  }
  return std::move(builder.plan);
}

std::vector<int> FormSemanticStages(std::vector<FxNodeRecord> const& tasks,
                                    ModelPlan const& plan) {
  std::vector<int> result;
  if (plan.stages.empty()) return std::vector<int>(tasks.size(), 0);

  result.reserve(tasks.size());
  std::size_t stage = 0;
  for (auto const& task : tasks) {
    while (stage + 1 < plan.stages.size() &&
           task.index > plan.stages[stage].representative_index)
      ++stage;
    result.push_back(static_cast<int>(stage));
  }
  return result;
}

}  // namespace tilemega::frontend
