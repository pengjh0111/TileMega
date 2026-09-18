// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.1 frontend grouping and §5.2 generated model data.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tilemega::frontend {

/// Stable FX facts serialized by export_bridge.py. No TileMega classification
/// or scheduling decision is made on the Python side.
struct FxNodeRecord {
  int index = 0;
  std::string name;
  std::string op;
  std::string target;
  std::vector<std::string> inputs;
  std::vector<std::string> shape;
  std::string dtype;
  /// Finite numeric literal arguments in call order. The normalization
  /// epsilon is the one piece of model configuration that reaches FX only
  /// this way.
  std::vector<double> scalars;
  /// Whether the bridge document carried the literals at all. A JSON exported
  /// before `scalar_args` existed says nothing about the epsilon, which is a
  /// different fact from a node that genuinely has no literal operand.
  bool has_scalars = false;
};

struct SignatureInput {
  std::string name;
  std::string kind;
  std::string target;
  bool persistent = false;
};

struct PlanBuffer {
  enum class Source { kZero, kFixture, kWeight };
  std::string name;
  std::uint32_t constant = 0;
  std::uint32_t per_seq = 0;
  std::uint32_t per_past = 0;
  std::uint32_t per_total = 0;
  Source source = Source::kZero;
  std::string file;
};

struct PlanGemm {
  std::uint32_t n = 0, k = 0;
  std::uint32_t a = 0, b = 0, c = 0, d = 0;
  float beta = 0.0f;
};

enum class PlanTaskKind {
  kGemm,
  kRMSNorm,
  kRoPE,
  kKVAppend,
  kElementwise,
  kAttention,
  kAdd,
};

struct PlanStage {
  PlanTaskKind kind = PlanTaskKind::kGemm;
  std::uint32_t gemm = 0;
  std::uint32_t extent = 0;
  std::uint32_t width = 0;
  std::uint32_t group = 1;
  std::array<std::uint32_t, 8> operands{};
  std::string representative;
  int representative_index = -1;
};

struct PlanOutput {
  std::uint32_t buffer = 0;
  std::string file;
};

/// Model-specific data derived structurally from FX. It is serialized as the
/// `tilemega.model_plan` MLIR module attribute; Codegen consumes only that
/// verified CG-side attribute, never this in-memory object.
struct ModelPlan {
  std::string dtype = "f32";
  /// The model's own `rms_norm_eps`, read off the `add(variance, eps)` inside
  /// every matched normalization. Zero means no normalization was matched;
  /// it is never defaulted to a dtype- or model-specific constant.
  double norm_epsilon = 0.0;
  /// Whether the exported graph builds the RoPE phase in FP32. It is read off
  /// the frequency table's own dtype: a table exported in FP32 is multiplied
  /// by FP32 positions and produces an FP32 angle, while a table exported in
  /// model storage rounds the angle to that storage. Both exist, so neither
  /// can be assumed.
  bool rope_fp32_phase = false;
  std::vector<PlanBuffer> buffers;
  std::vector<PlanGemm> gemms;
  std::vector<PlanStage> stages;
  std::vector<PlanOutput> outputs;
  std::unordered_map<std::string, std::uint32_t> node_buffer;
};

ModelPlan BuildModelPlan(std::vector<FxNodeRecord> const& nodes,
                         std::vector<SignatureInput> const& inputs,
                         std::vector<std::string> const& outputs);
void SeparateResidualTasks(ModelPlan& plan);

/// Assign every call_function to the first semantic stage whose representative
/// is at or after it. Representatives come from structural FX matches, not a
/// fixed layer/stage count.
std::vector<int> FormSemanticStages(std::vector<FxNodeRecord> const& tasks,
                                    ModelPlan const& plan);

}  // namespace tilemega::frontend
