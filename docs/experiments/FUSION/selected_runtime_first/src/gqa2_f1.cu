// SPDX-License-Identifier: BSD-3-Clause
// Generated from verified fused L-task phase and dependency projections.
#define TILEMEGA_MODEL_BF16 1
#ifndef TILEMEGA_FUSION_RUNTIME
#define TILEMEGA_FUSION_RUNTIME 1
#endif
#ifndef TILEMEGA_FUSION_GEMM_RUNTIME
#define TILEMEGA_FUSION_GEMM_RUNTIME 0
#endif
#ifndef TILEMEGA_FUSION_ROPE_RUNTIME
#define TILEMEGA_FUSION_ROPE_RUNTIME 1
#endif
#ifndef TILEMEGA_FUSION_ROPE_MAX_WIDTH
#define TILEMEGA_FUSION_ROPE_MAX_WIDTH 128
#endif
#define TILEMEGA_GEMM_TILE_M 128
#define TILEMEGA_GEMM_TILE_N 128
#define TILEMEGA_GEMM_TILE_K 16
#define TILEMEGA_GEMM_STAGES 3
#define TILEMEGA_GEMM_VARIANT_COUNT 1
#define TILEMEGA_GENERATED_WAIT_global(ev, need) do { \
  while (::tilemega::codegen::EventPoll((ev)) < (need)) __nanosleep(64); \
} while (0)
#define TILEMEGA_GENERATED_NOTIFY_global(ev, value) atomicExch((ev), (value))
#define TILEMEGA_GENERATED_RESIDENT_GRID(target, function, block_size, dynamic_smem) \
  ((target).res.num_sms * (target).ActiveBlocksPerSM( \
      reinterpret_cast<void const*>(function), (block_size), (dynamic_smem)))
// Model invocation tables are allocated on device and passed as one Params const* by ModelHarness (F-17b).
// Launcher specialization: l1_kernel
#include <tilemega/Codegen/tasks/ModelHarness.cuh>

namespace {
using namespace tilemega::codegen;

constexpr ModelDims kDims = {};

constexpr BufferDesc kBuffers[] = {
  {"hidden", 0u, 512u, 0u, 0u, BufferSource::kFixture, "input_hidden.bin"},
  {"l0.hidden", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.norm", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.q", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.k", 0u, 256u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.v", 0u, 256u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.q_rot", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.k_rot", 0u, 256u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.context", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.gate", 0u, 1024u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l0.up", 0u, 1024u, 0u, 0u, BufferSource::kZero, nullptr},
  {"p_layers_0_input_norm_weight", 512u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_input_norm_weight.bin"},
  {"p_layers_0_post_norm_weight", 512u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_post_norm_weight.bin"},
  {"p_layers_0_q_proj_weight", 262144u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_q_proj_weight.bin"},
  {"p_layers_0_k_proj_weight", 131072u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_k_proj_weight.bin"},
  {"p_layers_0_v_proj_weight", 131072u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_v_proj_weight.bin"},
  {"p_layers_0_o_proj_weight", 262144u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_o_proj_weight.bin"},
  {"p_layers_0_gate_proj_weight", 524288u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_gate_proj_weight.bin"},
  {"p_layers_0_up_proj_weight", 524288u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_up_proj_weight.bin"},
  {"p_layers_0_down_proj_weight", 524288u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_down_proj_weight.bin"},
  {"b_layers_0_inv_freq", 64u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_0_inv_freq.bin"},
  {"past_k0", 0u, 0u, 256u, 0u, BufferSource::kFixture, "input_past_k0.bin"},
  {"past_v0", 0u, 0u, 256u, 0u, BufferSource::kFixture, "input_past_v0.bin"},
  {"l0.full_k", 0u, 0u, 0u, 256u, BufferSource::kZero, nullptr},
  {"l0.full_v", 0u, 0u, 0u, 256u, BufferSource::kZero, nullptr},
  {"l1.hidden", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.norm", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.q", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.k", 0u, 256u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.v", 0u, 256u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.q_rot", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.k_rot", 0u, 256u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.context", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.gate", 0u, 1024u, 0u, 0u, BufferSource::kZero, nullptr},
  {"l1.up", 0u, 1024u, 0u, 0u, BufferSource::kZero, nullptr},
  {"p_layers_1_input_norm_weight", 512u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_input_norm_weight.bin"},
  {"p_layers_1_post_norm_weight", 512u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_post_norm_weight.bin"},
  {"p_layers_1_q_proj_weight", 262144u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_q_proj_weight.bin"},
  {"p_layers_1_k_proj_weight", 131072u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_k_proj_weight.bin"},
  {"p_layers_1_v_proj_weight", 131072u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_v_proj_weight.bin"},
  {"p_layers_1_o_proj_weight", 262144u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_o_proj_weight.bin"},
  {"p_layers_1_gate_proj_weight", 524288u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_gate_proj_weight.bin"},
  {"p_layers_1_up_proj_weight", 524288u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_up_proj_weight.bin"},
  {"p_layers_1_down_proj_weight", 524288u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_down_proj_weight.bin"},
  {"b_layers_1_inv_freq", 64u, 0u, 0u, 0u, BufferSource::kWeight, "state_layers_1_inv_freq.bin"},
  {"past_k1", 0u, 0u, 256u, 0u, BufferSource::kFixture, "input_past_k1.bin"},
  {"past_v1", 0u, 0u, 256u, 0u, BufferSource::kFixture, "input_past_v1.bin"},
  {"l1.full_k", 0u, 0u, 0u, 256u, BufferSource::kZero, nullptr},
  {"l1.full_v", 0u, 0u, 0u, 256u, BufferSource::kZero, nullptr},
};

constexpr GemmDesc kGemms[] = {
  {512, 512, 2u, 13u, 3u, 3u, 0.00000000f},
  {256, 512, 2u, 14u, 4u, 4u, 0.00000000f},
  {256, 512, 2u, 15u, 5u, 5u, 0.00000000f},
  {512, 512, 8u, 16u, 0u, 1u, 1.00000000f},
  {1024, 512, 2u, 17u, 9u, 9u, 0.00000000f},
  {1024, 512, 2u, 18u, 10u, 10u, 0.00000000f},
  {512, 1024, 9u, 19u, 1u, 1u, 1.00000000f},
  {512, 512, 26u, 37u, 27u, 27u, 0.00000000f},
  {256, 512, 26u, 38u, 28u, 28u, 0.00000000f},
  {256, 512, 26u, 39u, 29u, 29u, 0.00000000f},
  {512, 512, 32u, 40u, 1u, 25u, 1.00000000f},
  {1024, 512, 26u, 41u, 33u, 33u, 0.00000000f},
  {1024, 512, 26u, 42u, 34u, 34u, 0.00000000f},
  {512, 1024, 33u, 43u, 25u, 25u, 1.00000000f},
};

constexpr StageDesc kStages[] = {
  {TaskKind::kRMSNorm, 0u, 0u, 512u, 1u, {0u, 11u, 2u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 0u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 1u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 2u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRoPE, 0u, 4u, 128u, 1u, {3u, 6u, 20u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRoPEKVAppend, 0u, 2u, 128u, 1u, {4u, 20u, 21u, 23u, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kKVAppend, 0u, 2u, 128u, 1u, {5u, 22u, 24u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kAttention, 0u, 4u, 128u, 2u, {6u, 23u, 24u, 8u, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 3u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRMSNorm, 0u, 0u, 512u, 1u, {1u, 12u, 2u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 4u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 5u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kElementwise, 0u, 1024u, 0u, 1u, {9u, 10u, 9u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 6u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRMSNorm, 0u, 0u, 512u, 1u, {1u, 35u, 26u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 7u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 8u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 9u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRoPE, 0u, 4u, 128u, 1u, {27u, 30u, 44u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRoPEKVAppend, 0u, 2u, 128u, 1u, {28u, 44u, 45u, 47u, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kKVAppend, 0u, 2u, 128u, 1u, {29u, 46u, 48u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kAttention, 0u, 4u, 128u, 2u, {30u, 47u, 48u, 32u, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 10u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kRMSNorm, 0u, 0u, 512u, 1u, {25u, 36u, 26u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 11u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 12u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kElementwise, 0u, 1024u, 0u, 1u, {33u, 34u, 33u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
  {TaskKind::kGemm, 13u, 0u, 0u, 1u, {kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
};

constexpr OutputDesc kOutputs[] = {
  {25u, "reference_0.bin"},
  {23u, "reference_1.bin"},
  {24u, "reference_2.bin"},
  {47u, "reference_3.bin"},
  {48u, "reference_4.bin"},
};

constexpr GemmRuntimeDesc kRuntimeGemms0[] = {
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
  {0u, 1u, 128u, 128u, 16u, 3u},
};

constexpr StageDependency kDependencies0[] = {
  {0u, 1u, StageDependency::Map::kWindow, 4u, 128, 0, 128u},
  {0u, 2u, StageDependency::Map::kWindow, 2u, 128, 0, 128u},
  {0u, 3u, StageDependency::Map::kWindow, 2u, 128, 0, 128u},
  {1u, 4u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {2u, 5u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {3u, 6u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {4u, 7u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {5u, 7u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {6u, 7u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {7u, 8u, StageDependency::Map::kWindow, 4u, 512, 0, 512u},
  {8u, 9u, StageDependency::Map::kWindow, 128u, 4, 0, 4u},
  {9u, 10u, StageDependency::Map::kWindow, 8u, 128, 0, 128u},
  {9u, 11u, StageDependency::Map::kWindow, 8u, 128, 0, 128u},
  {10u, 12u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {11u, 12u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {8u, 13u, StageDependency::Map::kIdentity, 1u, 1, 0, 1u},
  {12u, 13u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {13u, 14u, StageDependency::Map::kWindow, 128u, 4, 0, 4u},
  {14u, 15u, StageDependency::Map::kWindow, 4u, 128, 0, 128u},
  {14u, 16u, StageDependency::Map::kWindow, 2u, 128, 0, 128u},
  {14u, 17u, StageDependency::Map::kWindow, 2u, 128, 0, 128u},
  {15u, 18u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {16u, 19u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {17u, 20u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {18u, 21u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {19u, 21u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {20u, 21u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {13u, 22u, StageDependency::Map::kIdentity, 1u, 1, 0, 1u},
  {21u, 22u, StageDependency::Map::kWindow, 4u, 512, 0, 512u},
  {22u, 23u, StageDependency::Map::kWindow, 128u, 4, 0, 4u},
  {23u, 24u, StageDependency::Map::kWindow, 8u, 128, 0, 128u},
  {23u, 25u, StageDependency::Map::kWindow, 8u, 128, 0, 128u},
  {24u, 26u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {25u, 26u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
  {22u, 27u, StageDependency::Map::kIdentity, 1u, 1, 0, 1u},
  {26u, 27u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
};

constexpr std::uint32_t kDependencyOffsets0[] = {
  0u, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 9u, 10u, 11u, 12u, 13u, 15u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u, 27u, 29u, 30u, 31u, 32u, 34u, 36u, 
};

constexpr ScheduleStageDesc kSchedule0[] = {
  {0u, 0u, 0u},
  {1u, 0u, 1u},
  {2u, 1u, 1u},
  {3u, 2u, 1u},
  {4u, 3u, 1u},
  {5u, 4u, 1u},
  {6u, 5u, 1u},
  {7u, 6u, 3u},
  {8u, 9u, 1u},
  {9u, 10u, 1u},
  {10u, 11u, 1u},
  {11u, 12u, 1u},
  {12u, 13u, 2u},
  {13u, 15u, 2u},
  {14u, 17u, 1u},
  {15u, 18u, 1u},
  {16u, 19u, 1u},
  {17u, 20u, 1u},
  {18u, 21u, 1u},
  {19u, 22u, 1u},
  {20u, 23u, 1u},
  {21u, 24u, 3u},
  {22u, 27u, 2u},
  {23u, 29u, 1u},
  {24u, 30u, 1u},
  {25u, 31u, 1u},
  {26u, 32u, 2u},
  {27u, 34u, 2u},
};

constexpr RuntimeExactDependencyDesc kExactDependencies0 = {"[s11, s14] -> { [] -> [n = 27, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + o1; [] -> [n = 26, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < 8s11; [] -> [n = 25, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 8*floor((-1 + s11)/128) >= -7 + o1; [] -> [n = 24, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 8*floor((-1 + s11)/128) >= -7 + o1; [] -> [n = 23, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < s11; [] -> [n = 22, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + o1; [] -> [n = 21, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < 4s11; [] -> [n = 20, o1] : s14 <= 512 and o1 >= 0 and ((s11 > 0 and s14 >= s11 and o1 < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and o1 < 2s11)); [] -> [n = 3, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 2*floor((-1 + s11)/128) >= -1 + o1; [] -> [n = 19, o1] : s14 <= 512 and o1 >= 0 and ((s11 > 0 and s14 >= s11 and o1 < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and o1 < 2s11)); [] -> [n = 2, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 2*floor((-1 + s11)/128) >= -1 + o1; [] -> [n = 18, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < 2s11; [] -> [n = 17, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 2*floor((-1 + s11)/128) >= -1 + o1; [] -> [n = 16, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 2*floor((-1 + s11)/128) >= -1 + o1; [] -> [n = 15, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + o1; [] -> [n = 14, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < s11; [] -> [n = 13, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + o1; [] -> [n = 12, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < 8s11; [] -> [n = 11, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 8*floor((-1 + s11)/128) >= -7 + o1; [] -> [n = 10, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 8*floor((-1 + s11)/128) >= -7 + o1; [] -> [n = 9, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < s11; [] -> [n = 8, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + o1; [] -> [n = 7, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < 4s11; [] -> [n = 6, o1] : s14 <= 512 and o1 >= 0 and ((s11 > 0 and s14 >= s11 and o1 < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and o1 < 2s11)); [] -> [n = 1, o1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and o1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + o1; [] -> [n = 5, o1] : s14 <= 512 and o1 >= 0 and ((s11 > 0 and s14 >= s11 and o1 < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and o1 < 2s11)); [] -> [n = 0, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < s11; [] -> [n = 4, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < 2s11 }", "[s11, s14] -> { [n = 27, i1] -> [n' = 26, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < 8s11 and 4*floor((-1 + s11)/128) >= -3 + i1; [n = 26, i1] -> [n' = 25, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 8s11 and i1' >= 0 and 8*floor((-1 + s11)/128) >= -7 + i1'; [n = 26, i1] -> [n' = 24, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 8s11 and i1' >= 0 and 8*floor((-1 + s11)/128) >= -7 + i1'; [n = 25, i1] -> [n' = 23, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 8*floor((-1 + s11)/128) >= -7 + i1 and -127 + i1' <= 128*floor((i1)/8) <= i1'; [n = 24, i1] -> [n' = 23, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 8*floor((-1 + s11)/128) >= -7 + i1 and -127 + i1' <= 128*floor((i1)/8) <= i1'; [n = 23, i1] -> [n' = 22, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < s11 and i1' >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1' and -3 + i1' <= 4*floor((i1)/128) <= i1'; [n = 22, i1] -> [n' = 21, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < 4s11 and 4*floor((-1 + s11)/128) >= -3 + i1 and -511 + i1' <= 512*floor((i1)/4) <= i1'; [n = 21, i1] -> [n' = 20, i1'] : s14 <= 512 and 0 <= i1 < 4s11 and i1' >= 0 and ((s14 >= s11 and i1' < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and i1' < 2s11)); [n = 13, i1] -> [n' = 8, i1' = i1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1; [n = 21, i1] -> [n' = 19, i1'] : s14 <= 512 and 0 <= i1 < 4s11 and i1' >= 0 and ((s14 >= s11 and i1' < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and i1' < 2s11)); [n = 4, i1] -> [n' = 1, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 2s11 and i1' >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1'; [n = 21, i1] -> [n' = 18, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 4s11 and 0 <= i1' < 2s11; [n = 20, i1] -> [n' = 17, i1'] : s14 <= 512 and i1 >= 0 and i1' >= 0 and 2*floor((-1 + s11)/128) >= -1 + i1' and ((s11 > 0 and s14 >= s11 and i1 < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and i1 < 2s11)); [n = 3, i1] -> [n' = 0, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 2*floor((-1 + s11)/128) >= -1 + i1 and -127 + i1' <= 128*floor((i1)/2) <= i1'; [n = 19, i1] -> [n' = 16, i1'] : s14 <= 512 and 0 <= i1 < 2s11 and i1' >= 0 and 2*floor((-1 + s11)/128) >= -1 + i1' and (s14 >= s11 or (s11 <= 2048 and 0 <= s14 < s11)); [n = 2, i1] -> [n' = 0, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 2*floor((-1 + s11)/128) >= -1 + i1 and -127 + i1' <= 128*floor((i1)/2) <= i1'; [n = 18, i1] -> [n' = 15, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 2s11 and i1' >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1'; [n = 17, i1] -> [n' = 14, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 2*floor((-1 + s11)/128) >= -1 + i1 and -127 + i1' <= 128*floor((i1)/2) <= i1'; [n = 16, i1] -> [n' = 14, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 2*floor((-1 + s11)/128) >= -1 + i1 and -127 + i1' <= 128*floor((i1)/2) <= i1'; [n = 15, i1] -> [n' = 14, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 4*floor((-1 + s11)/128) >= -3 + i1 and -127 + i1' <= 128*floor((i1)/4) <= i1'; [n = 14, i1] -> [n' = 13, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < s11 and i1' >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1' and -3 + i1' <= 4*floor((i1)/128) <= i1'; [n = 13, i1] -> [n' = 12, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < 8s11 and 4*floor((-1 + s11)/128) >= -3 + i1; [n = 12, i1] -> [n' = 11, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 8s11 and i1' >= 0 and 8*floor((-1 + s11)/128) >= -7 + i1'; [n = 12, i1] -> [n' = 10, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 8s11 and i1' >= 0 and 8*floor((-1 + s11)/128) >= -7 + i1'; [n = 11, i1] -> [n' = 9, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 8*floor((-1 + s11)/128) >= -7 + i1 and -127 + i1' <= 128*floor((i1)/8) <= i1'; [n = 10, i1] -> [n' = 9, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 8*floor((-1 + s11)/128) >= -7 + i1 and -127 + i1' <= 128*floor((i1)/8) <= i1'; [n = 9, i1] -> [n' = 8, i1'] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < s11 and i1' >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1' and -3 + i1' <= 4*floor((i1)/128) <= i1'; [n = 8, i1] -> [n' = 7, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < 4s11 and 4*floor((-1 + s11)/128) >= -3 + i1 and -511 + i1' <= 512*floor((i1)/4) <= i1'; [n = 7, i1] -> [n' = 6, i1'] : s14 <= 512 and 0 <= i1 < 4s11 and i1' >= 0 and ((s14 >= s11 and i1' < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and i1' < 2s11)); [n = 1, i1] -> [n' = 0, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 0 <= i1' < s11 and 4*floor((-1 + s11)/128) >= -3 + i1 and -127 + i1' <= 128*floor((i1)/4) <= i1'; [n = 7, i1] -> [n' = 5, i1'] : s14 <= 512 and 0 <= i1 < 4s11 and i1' >= 0 and ((s14 >= s11 and i1' < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and i1' < 2s11)); [n = 27, i1] -> [n' = 22, i1' = i1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1; [n = 7, i1] -> [n' = 4, i1'] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= i1 < 4s11 and 0 <= i1' < 2s11; [n = 6, i1] -> [n' = 3, i1'] : s14 <= 512 and i1 >= 0 and i1' >= 0 and 2*floor((-1 + s11)/128) >= -1 + i1' and ((s11 > 0 and s14 >= s11 and i1 < 2s14) or (s11 <= 2048 and 0 <= s14 < s11 and i1 < 2s11)); [n = 22, i1] -> [n' = 13, i1' = i1] : 0 < s11 <= 2048 and 0 <= s14 <= 512 and i1 >= 0 and 4*floor((-1 + s11)/128) >= -3 + i1; [n = 5, i1] -> [n' = 2, i1'] : s14 <= 512 and 0 <= i1 < 2s11 and i1' >= 0 and 2*floor((-1 + s11)/128) >= -1 + i1' and (s14 >= s11 or (s11 <= 2048 and 0 <= s14 < s11)) }", "s11", "s14"};
constexpr RuntimeVariantDesc kRuntimeVariants[] = {
  {kRuntimeGemms0, kDependencies0, 36u, kDependencyOffsets0, kSchedule0, 28u, 9u, 1u, 2048u, 0u, nullptr, true, false, &kExactDependencies0},
};

constexpr auto MakeSeqVariant() {
  std::array<std::uint16_t, 2049> table{};
  for (std::uint32_t s = 1u; s <= 2048u; ++s) table[s] = 0u;
  return table;
}
constexpr auto kSeqVariant = MakeSeqVariant();

constexpr ModelSpec kModel = {kDims, ScalarType::kBF16, kBuffers, 49u, kGemms, 14u, kStages, 28u, kOutputs, 5u, kRuntimeVariants, 1u, kSeqVariant.data(), 2049u};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: e2e FIXTURE_DIR\n"); return 2; }
  return tilemega::codegen::RunModel(kModel, argv[1]);
}
