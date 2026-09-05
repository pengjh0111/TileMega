// SPDX-License-Identifier: BSD-3-Clause
// Compile-only coverage for the production BF16 dispatch. Including the full
// harness forces nvcc to instantiate GEMM, norm, RoPE, KV, elementwise,
// attention, and split-combine bodies under the BF16 model ABI.
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_GEMM_TILE_M 128
#define TILEMEGA_GEMM_TILE_N 128
#define TILEMEGA_GEMM_TILE_K 16
#define TILEMEGA_GEMM_STAGES 3
#define TILEMEGA_GEMM_VARIANT_COUNT 1
#include <tilemega/Codegen/tasks/ModelHarness.cuh>

static_assert(tilemega::codegen::kCompiledScalarType ==
              tilemega::codegen::ScalarType::kBF16);
static_assert(tilemega::codegen::GemmImpl::kThreads == 128);

int main() { return 0; }
