// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_PHASE 2
#ifndef TILEMEGA_FORWARD_TEST_SEQ
#define TILEMEGA_FORWARD_TEST_SEQ 1
#endif
#define TILEMEGA_SERVING_SEQ TILEMEGA_FORWARD_TEST_SEQ
#define TILEMEGA_SERVING_BATCH_LO 1
#define TILEMEGA_SERVING_BATCH_HI 8
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 0
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 3
#include <tilemega/Codegen/DmDescriptors.h>
namespace tilemega::codegen {
using ForwardChain = DmEpilogueProgram<
    DmEpilogueStep<DmEpilogueKind::kBias, DmActivation::kRelu, DmGatePair::kSwiGLU,
                   16, DmRounding::kFP32, DmRounding::kFP32, DmWriteKind::kDense, 1>,
    DmEpilogueStep<DmEpilogueKind::kActivation, DmActivation::kRelu, DmGatePair::kSwiGLU,
                   16, DmRounding::kFP32, DmRounding::kBF16, DmWriteKind::kDense, 1>>;
using ForwardSpec = DmEpilogueSpec<ForwardChain>;
template <class Runner>
__device__ void DispatchDmEpilogue(std::uint32_t id, Runner const& runner) {
  if (id == 0) runner.template Run<ForwardSpec>(); else asm volatile("trap;");
}
}
#include <tilemega/Codegen/tasks/ModelHarness.cuh>

namespace {
using namespace tilemega::codegen;
constexpr BufferDesc kBuffers[] = {
    {"input", 0, 0, 0, 0, BufferSource::kZero, nullptr, 17 * 128, 0, 1, "input"},
    {"weight", 64 * 128, 0, 0, 0, BufferSource::kZero, nullptr, 0, 0, 1, "weight"},
    {"bias", 64, 0, 0, 0, BufferSource::kZero, nullptr, 0, 1, 1, "bias"},
    {"output", 0, 0, 0, 0, BufferSource::kZero, nullptr, 17 * 64, 0, 1, "output"}};
constexpr auto MakeGemm() {
  GemmDesc g{64, 128, 0, 1, 0, 3, 0};
  g.access.rows_per_batch = 17; g.chain.count = 2;
  g.chain.operations[0].kind = DmEpilogueKind::kBias;
  g.chain.operations[0].output_rounding = DmRounding::kFP32;
  g.chain.operations[0].parameter[0] = 2;
  g.chain.operations[1].kind = DmEpilogueKind::kActivation;
  g.chain.operations[1].output_rounding = DmRounding::kBF16;
  return g;
}
constexpr GemmDesc kGemms[] = {MakeGemm()};
constexpr StageDesc kStages[] = {{TaskKind::kGemm, 0, 0, 64, 0, {}}};
constexpr GemmRuntimeDesc kGeometry[] = {{0, 1, 16, 64, 64, 3}};
constexpr std::uint32_t kOffsets[] = {0, 0};
constexpr ScheduleStageDesc kSchedule[] = {{0, 0, 0}};
constexpr RuntimeVariantDesc kVariants[] = {
    {kGeometry, nullptr, 0, kOffsets, kSchedule, 1, 0, TILEMEGA_SERVING_SEQ, TILEMEGA_SERVING_SEQ, 0}};
constexpr std::uint16_t kSeqVariant[TILEMEGA_SERVING_SEQ + 1] = {};
constexpr OutputDesc kOutputs[] = {{3, nullptr}};
constexpr ModelSpec kModel = {{0, 0, 0, TILEMEGA_SERVING_SEQ, 0}, ScalarType::kBF16, kBuffers, 4,
    kGemms, 1, kStages, 1, kOutputs, 1, kVariants, 1, kSeqVariant, TILEMEGA_SERVING_SEQ + 1};
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
#include <cassert>
#include <cstring>

int main() {
  using E = cutlass::bfloat16_t;
  tm_plan_info info;
  assert(tm_plan_query(&info) == 0 && info.phase == TM_SERVING_FORWARD && info.seq == TILEMEGA_SERVING_SEQ);
  assert(info.past_lo == 0 && info.past_hi == 0 && info.capacity == 0 && info.buffer_count == 4);
  tm_buffer_info input;
  assert(tm_plan_buffer(0, &input) == 0 && std::strcmp(input.name, "input") == 0);
  assert(input.elements_per_batch == 17 * 128 && input.role == TM_BUFFER_EXTERNAL);
  for (int batch : {1, 8}) {
    E *a, *b, *out; float* bias;
    TILEMEGA_CUDA_CHECK(cudaMallocManaged(&a, batch * 17 * 128 * sizeof(E)));
    TILEMEGA_CUDA_CHECK(cudaMallocManaged(&b, 64 * 128 * sizeof(E)));
    TILEMEGA_CUDA_CHECK(cudaMallocManaged(&out, batch * 17 * 64 * sizeof(E)));
    TILEMEGA_CUDA_CHECK(cudaMallocManaged(&bias, 64 * sizeof(float)));
    for (int i = 0; i < batch * 17 * 128; ++i) a[i] = E(float(i * 7 % 31 - 15) / 32);
    for (int i = 0; i < 64 * 128; ++i) b[i] = E(float(i * 11 % 17 - 8) / 64);
    for (int i = 0; i < 64; ++i) bias[i] = float(i % 7 - 3) / 8;
    void* buffers[] = {a, b, bias, out};
    void* plan = tm_plan_create(batch, buffers, 0); assert(plan);
    assert(tm_plan_loop_modes(plan) == 0);
    std::int32_t invalid[] = {0, 0}, step = 0;
    assert(tm_plan_set_steps(plan, invalid, 2) == -2);
    assert(tm_plan_set_steps(plan, &step, 1) == 0);
    assert(tm_plan_launch(plan, 1, TM_SERVING_L1, 0, nullptr) == -1);
    std::vector<E> first;
    for (unsigned mode : {TM_SERVING_L1, TM_SERVING_L2}) {
      assert(tm_plan_launch(plan, 0, mode, 0, nullptr) == 0);
      TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
      for (int row = 0; row < batch * 17; ++row) for (int column = 0; column < 64; ++column) {
        float ref = bias[column];
        for (int k = 0; k < 128; ++k) ref += float(a[row * 128 + k]) * float(b[column * 128 + k]);
        E expected(std::max(ref, 0.f));
        assert(float(out[row * 64 + column]) == float(expected));
      }
      if (mode == TM_SERVING_L1) first.assign(out, out + batch * 17 * 64);
      else assert(std::memcmp(first.data(), out, first.size() * sizeof(E)) == 0);
    }
    tm_plan_destroy(plan);
    for (void* p : buffers) TILEMEGA_CUDA_CHECK(cudaFree(p));
  }
  std::puts("forward ABI: step zero, no KV state, B=1/8, L1/L2 exact outputs");
}
