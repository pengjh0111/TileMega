// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/ModelPlan.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <regex>
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

  FxNodeRecord const& Linear(std::string const& weight) const {
    for (auto const& node : nodes)
      if (node.target == "aten.linear.default" && node.inputs.size() >= 2 &&
          node.inputs[1] == weight)
        return node;
    throw std::runtime_error("no aten.linear uses parameter " + weight);
  }

  FxNodeRecord const& Consumer(std::string const& target,
                               std::string const& input) const {
    for (auto const& node : nodes)
      if (node.target == target &&
          std::find(node.inputs.begin(), node.inputs.end(), input) !=
              node.inputs.end())
        return node;
    throw std::runtime_error("no " + target + " consumes " + input);
  }

  FxNodeRecord const& CacheConcat(std::string const& past) const {
    return Consumer("aten.cat.default", past);
  }

  FxNodeRecord const& FirstAttention(std::string const& q,
                                     std::string const& full_k) const {
    for (auto const& node : nodes)
      if (node.target == "aten.matmul.default" && node.inputs.size() == 2 &&
          DependsOn(node.inputs[0], q) && DependsOn(node.inputs[1], full_k))
        return node;
    throw std::runtime_error("cannot identify attention score matmul");
  }

  FxNodeRecord const& SecondAttention(std::string const& score,
                                      std::string const& full_v) const {
    for (auto const& node : nodes)
      if (node.target == "aten.matmul.default" && node.inputs.size() == 2 &&
          DependsOn(node.inputs[0], score) && DependsOn(node.inputs[1], full_v))
        return node;
    throw std::runtime_error("cannot identify attention value matmul");
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

struct LayerSignature {
  int number = 0;
  std::map<std::string, SignatureInput const*> parameter;
};

SignatureInput const& Require(LayerSignature const& layer,
                              std::string const& suffix) {
  auto found = layer.parameter.find(suffix);
  if (found == layer.parameter.end())
    throw std::runtime_error("layer " + std::to_string(layer.number) +
                             " lacks " + suffix);
  return *found->second;
}

}  // namespace

ModelPlan BuildModelPlan(std::vector<FxNodeRecord> const& nodes,
                         std::vector<SignatureInput> const& inputs,
                         std::vector<std::string> const& outputs) {
  PlanBuilder builder(nodes);
  std::map<int, LayerSignature> layers;
  static std::regex const layer_pattern(R"(^layers\.([0-9]+)\.(.+)$)");
  SignatureInput const* hidden_input = nullptr;
  std::unordered_map<std::string, SignatureInput const*> signature_by_name;
  for (auto const& input : inputs) {
    signature_by_name.emplace(input.name, &input);
    std::smatch match;
    if ((input.kind == "PARAMETER" || input.kind == "BUFFER") &&
        std::regex_match(input.target, match, layer_pattern)) {
      int number = std::stoi(match[1].str());
      layers[number].number = number;
      layers[number].parameter[match[2].str()] = &input;
    } else if (input.kind == "USER_INPUT" &&
               input.name.find("past_") != 0 && !hidden_input) {
      hidden_input = &input;
    }
  }
  if (!hidden_input || layers.empty())
    throw std::runtime_error(
        "Phase-3 model-plan builder needs a tensor input and layers.* parameters");

  FxNodeRecord const& hidden_node = builder.Node(hidden_input->name);
  std::uint32_t hidden = StaticExtent(hidden_node, hidden_node.shape.size() - 1);
  std::uint32_t current_hidden = builder.Buffer(
      {hidden_input->name, 0, hidden, 0, 0, PlanBuffer::Source::kFixture,
       "input_" + hidden_input->name + ".bin"});

  std::unordered_map<std::string, std::uint32_t> semantic_output;
  for (auto const& [number, layer] : layers) {
    auto const& input_norm = Require(layer, "input_norm.weight");
    auto const& post_norm = Require(layer, "post_norm.weight");
    auto const& q_weight = Require(layer, "q_proj.weight");
    auto const& k_weight = Require(layer, "k_proj.weight");
    auto const& v_weight = Require(layer, "v_proj.weight");
    auto const& o_weight = Require(layer, "o_proj.weight");
    auto const& gate_weight = Require(layer, "gate_proj.weight");
    auto const& up_weight = Require(layer, "up_proj.weight");
    auto const& down_weight = Require(layer, "down_proj.weight");
    auto const& inv_freq = Require(layer, "inv_freq");

    FxNodeRecord const& q_node = builder.Linear(q_weight.name);
    FxNodeRecord const& k_node = builder.Linear(k_weight.name);
    FxNodeRecord const& v_node = builder.Linear(v_weight.name);
    FxNodeRecord const& o_node = builder.Linear(o_weight.name);
    FxNodeRecord const& gate_node = builder.Linear(gate_weight.name);
    FxNodeRecord const& up_node = builder.Linear(up_weight.name);
    FxNodeRecord const& down_node = builder.Linear(down_weight.name);
    if (q_node.inputs[0] != k_node.inputs[0] ||
        q_node.inputs[0] != v_node.inputs[0])
      throw std::runtime_error("Q/K/V projections do not share one norm output");
    if (gate_node.inputs[0] != up_node.inputs[0])
      throw std::runtime_error("gate/up projections do not share one norm output");

    std::uint32_t q_width = StaticExtent(builder.Node(q_weight.name), 0);
    std::uint32_t kv_width = StaticExtent(builder.Node(k_weight.name), 0);
    std::uint32_t intermediate = StaticExtent(builder.Node(up_weight.name), 0);
    std::uint32_t head_dim = NumericElements(builder.Node(inv_freq.name)) * 2;
    if (!head_dim || q_width % head_dim || kv_width % head_dim)
      throw std::runtime_error("Q/K/V widths are incompatible with RoPE head dim");
    std::uint32_t heads = q_width / head_dim;
    std::uint32_t kv_heads = kv_width / head_dim;
    if (!kv_heads || heads % kv_heads)
      throw std::runtime_error("query heads are not divisible by KV heads");

    std::string past_k_name = "past_k" + std::to_string(number);
    std::string past_v_name = "past_v" + std::to_string(number);
    auto past_k_sig = signature_by_name.find(past_k_name);
    auto past_v_sig = signature_by_name.find(past_v_name);
    if (past_k_sig == signature_by_name.end() ||
        past_v_sig == signature_by_name.end())
      throw std::runtime_error("missing explicit KV inputs for layer " +
                               std::to_string(number));
    FxNodeRecord const& cat_k = builder.CacheConcat(past_k_name);
    FxNodeRecord const& cat_v = builder.CacheConcat(past_v_name);
    FxNodeRecord const& score = builder.FirstAttention(q_node.name, cat_k.name);
    FxNodeRecord const& value = builder.SecondAttention(score.name, cat_v.name);
    if (o_node.inputs.empty() || !builder.DependsOn(o_node.inputs[0], value.name))
      throw std::runtime_error("o_proj does not consume the matched attention");
    FxNodeRecord const& residual1 = builder.Consumer("aten.add.Tensor", o_node.name);
    FxNodeRecord const& residual2 = builder.Consumer("aten.add.Tensor", down_node.name);
    if (down_node.inputs.empty())
      throw std::runtime_error("down projection has no activation input");

    std::string prefix = "l" + std::to_string(number) + ".";
    std::uint32_t next_hidden = builder.Scratch(prefix + "hidden", hidden);
    std::uint32_t norm = builder.Scratch(prefix + "norm", hidden);
    std::uint32_t q = builder.Scratch(prefix + "q", q_width);
    std::uint32_t k = builder.Scratch(prefix + "k", kv_width);
    std::uint32_t v = builder.Scratch(prefix + "v", kv_width);
    std::uint32_t q_rot = builder.Scratch(prefix + "q_rot", q_width);
    std::uint32_t k_rot = builder.Scratch(prefix + "k_rot", kv_width);
    std::uint32_t context = builder.Scratch(prefix + "context", q_width);
    std::uint32_t gate = builder.Scratch(prefix + "gate", intermediate);
    std::uint32_t up = builder.Scratch(prefix + "up", intermediate);
    std::uint32_t wn1 = builder.Weight(input_norm);
    std::uint32_t wn2 = builder.Weight(post_norm);
    std::uint32_t wq = builder.Weight(q_weight);
    std::uint32_t wk = builder.Weight(k_weight);
    std::uint32_t wv = builder.Weight(v_weight);
    std::uint32_t wo = builder.Weight(o_weight);
    std::uint32_t wg = builder.Weight(gate_weight);
    std::uint32_t wu = builder.Weight(up_weight);
    std::uint32_t wd = builder.Weight(down_weight);
    std::uint32_t inv = builder.Weight(inv_freq);
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
    builder.Stage(PlanTaskKind::kRoPE, score.inputs[0], 0, heads, head_dim, 1,
                  {q, q_rot, inv});
    builder.Stage(PlanTaskKind::kRoPE, cat_k.inputs[1], 0, kv_heads, head_dim, 1,
                  {k, k_rot, inv});
    builder.Stage(PlanTaskKind::kKVAppend, cat_k.name, 0, kv_heads, head_dim, 1,
                  {k_rot, past_k, full_k});
    builder.Stage(PlanTaskKind::kKVAppend, cat_v.name, 0, kv_heads, head_dim, 1,
                  {v, past_v, full_v});
    builder.Stage(PlanTaskKind::kAttention, o_node.inputs[0], 0, heads,
                  head_dim, heads / kv_heads,
                  {q_rot, full_k, full_v, context});
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
  if (plan.stages.empty()) throw std::runtime_error("model plan has no stages");
  std::vector<int> result;
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
