// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 8
#define TILEMEGA_GEMM_TILE_K 128
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_GEMM_IMPL_0 1
#define TILEMEGA_GEMM_V1_TILE_M 16
#define TILEMEGA_GEMM_V1_TILE_N 16
#define TILEMEGA_GEMM_V1_TILE_K 128
#define TILEMEGA_GEMM_V1_STAGES 2
#define TILEMEGA_GEMM_IMPL_1 1
#define TILEMEGA_GEMM_V2_TILE_M 16
#define TILEMEGA_GEMM_V2_TILE_N 32
#define TILEMEGA_GEMM_V2_TILE_K 128
#define TILEMEGA_GEMM_V2_STAGES 2
#define TILEMEGA_GEMM_IMPL_2 1
#define TILEMEGA_GEMM_VARIANT_COUNT 3
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
using namespace tilemega::codegen;
using Body=GemmStageTaskBody<tilemega::arch::CurrentArch,GemmVariantSmem,128>;
extern "C" __global__ void gemv_dispatch(GemmInvocation const* invocation) {
  extern __shared__ char bytes[];
  Body::RunTask<0>(*invocation,0,bytes);
  Body::RunTask<1>(*invocation,0,bytes);
  Body::RunTask<2>(*invocation,0,bytes);
}
