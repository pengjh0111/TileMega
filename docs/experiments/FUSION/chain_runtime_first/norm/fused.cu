// SPDX-License-Identifier: BSD-3-Clause
// Generated from verified fused L-task phase and dependency projections.
#define TILEMEGA_MODEL_BF16 1
#ifndef TILEMEGA_FUSION_RUNTIME
#define TILEMEGA_FUSION_RUNTIME 1
#endif
#ifndef TILEMEGA_FUSION_GEMM_RUNTIME
#define TILEMEGA_FUSION_GEMM_RUNTIME 1
#endif
#ifndef TILEMEGA_FUSION_ROPE_RUNTIME
#define TILEMEGA_FUSION_ROPE_RUNTIME 0
#endif
#ifndef TILEMEGA_FUSION_ROPE_MAX_WIDTH
#define TILEMEGA_FUSION_ROPE_MAX_WIDTH 0
#endif
#define TILEMEGA_GEMM_TILE_M 32
#define TILEMEGA_GEMM_TILE_N 512
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
  {"input", 0u, 512u, 0u, 0u, BufferSource::kFixture, "input.bin"},
  {"weight", 262144u, 0u, 0u, 0u, BufferSource::kFixture, "weight.bin"},
  {"intermediate", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
  {"consumer_input", 512u, 0u, 0u, 0u, BufferSource::kFixture, "consumer_input.bin"},
  {"output", 0u, 512u, 0u, 0u, BufferSource::kZero, nullptr},
};

constexpr GemmDesc kGemms[] = {
  {512, 512, 0u, 1u, 2u, 2u, 0.00000000f},
};

constexpr StageDesc kStages[] = {
  {TaskKind::kGemmRMSNorm, 0u, 512u, 512u, 1u, {3u, 4u, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand, kNoOperand}},
};

constexpr OutputDesc kOutputs[] = {
  {4u, "reference.bin"},
};

constexpr GemmRuntimeDesc kRuntimeGemms0[] = {
  {0u, 1u, 32u, 512u, 16u, 3u},
};

constexpr StageDependency kDependencies0[] = {
  {0u, 0u, StageDependency::Map::kAll, 1u, 0, 0, 1u},
};

constexpr std::uint32_t kDependencyOffsets0[] = {
  0u, 0u, 
};

constexpr ScheduleStageDesc kSchedule0[] = {
  {0u, 0u, 0u},
};

constexpr RuntimeExactDependencyDesc kExactDependencies0 = {"[s11, s14] -> { [] -> [n = 0, o1] : s11 <= 2048 and 0 <= s14 <= 512 and 0 <= o1 < s11 }", "[s11, s14] -> { [n, i1] -> [n', i1'] : false }", "s11", "s14"};
constexpr RuntimeVariantDesc kRuntimeVariants[] = {
  {kRuntimeGemms0, kDependencies0, 0u, kDependencyOffsets0, kSchedule0, 1u, 0u, 1u, 2048u, 0u, nullptr, true, false, &kExactDependencies0},
};

constexpr auto MakeSeqVariant() {
  std::array<std::uint16_t, 2049> table{};
  for (std::uint32_t s = 1u; s <= 2048u; ++s) table[s] = 0u;
  return table;
}
constexpr auto kSeqVariant = MakeSeqVariant();

constexpr ModelSpec kModel = {kDims, ScalarType::kBF16, kBuffers, 5u, kGemms, 1u, kStages, 1u, kOutputs, 1u, kRuntimeVariants, 1u, kSeqVariant.data(), 2049u};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: e2e FIXTURE_DIR\n"); return 2; }
  return tilemega::codegen::RunModel(kModel, argv[1]);
}
