#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_MIDPOINT_REFINE 0
#define TILEMEGA_GEMM_TILE_M 32
#define TILEMEGA_GEMM_TILE_N 16
#define TILEMEGA_GEMM_TILE_K 16
#define TILEMEGA_GEMM_STAGES 2
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Target/ArchDispatch.h>
#include <cstdio>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/RoPETaskBody.h>
#include <tilemega/Codegen/tasks/KVAppendTaskBody.h>
#include <tilemega/Codegen/tasks/ElementwiseTaskBody.h>
#include <tilemega/Codegen/tasks/AddTaskBody.h>
#include <tilemega/Codegen/tasks/EmbeddingTaskBody.h>
#include <tilemega/Codegen/tasks/QKNormTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/GemmCombineTaskBody.h>
using namespace tilemega::codegen;
union ProbeSmem { float rms[256]; float attention[TILEMEGA_ATTENTION_SCRATCH_EXTENT]; float pointwise[1]; };
extern "C" __global__ __launch_bounds__(128) void probe_RMSNorm(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; RMSNormTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_RoPE(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; RoPETaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_KVAppend(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; KVAppendTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_Elementwise(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; ElementwiseTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_Add(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; AddTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_Embedding(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; EmbeddingTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_QKNorm(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; QKNormTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_Attention(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; AttentionTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
extern "C" __global__ __launch_bounds__(128) void probe_GemmCombine(Params const* p,StageDesc const* stage,int task) { extern __shared__ char bytes[]; GemmCombineTaskBody<tilemega::arch::CurrentArch,ProbeSmem,128>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }
int main() { std::printf("%zu 128\n",sizeof(ProbeSmem)); }
