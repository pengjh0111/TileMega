#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_MIDPOINT_REFINE 0
#define TILEMEGA_GEMM_TILE_M 32
#define TILEMEGA_GEMM_TILE_N 16
#define TILEMEGA_GEMM_TILE_K 32
#define TILEMEGA_GEMM_STAGES 4
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Target/ArchDispatch.h>
#include <cstdio>
using namespace tilemega::codegen;
extern "C" __global__ __launch_bounds__(128) void probe_gemm(GemmInvocation const* g,int task) { extern __shared__ char bytes[]; GemmStageTaskBody<tilemega::arch::CurrentArch,GemmVariantSmem,128>::RunTask<0>(*g,task,bytes,nullptr); }
int main() { std::printf("%zu 128\n",sizeof(GemmVariantSmem)); }
