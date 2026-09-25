// SPDX-License-Identifier: BSD-3-Clause
// Instantiates the serving scalar dispatch in the generated persistent kernel.
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_GEMM_TILE_M 128
#define TILEMEGA_GEMM_TILE_N 128
#define TILEMEGA_GEMM_TILE_K 16
#define TILEMEGA_GEMM_STAGES 3
#define TILEMEGA_GEMM_VARIANT_COUNT 1
#define TILEMEGA_EMBEDDING_RUNTIME 1
#include <tilemega/Codegen/tasks/ModelHarness.cuh>

static_assert(tilemega::codegen::kCompiledScalarType ==
              tilemega::codegen::ScalarType::kBF16);

int main() { return 0; }
