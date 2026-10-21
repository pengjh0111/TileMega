// SPDX-License-Identifier: BSD-3-Clause
// Numerical test entry points for the xcore1000 serving GEMM and GEMV bodies.
// test/python/test_mx_task_bodies.py builds operands with torch, calls these
// through ctypes and compares against an FP32 torch reference.
#include <tilemega/Codegen/tasks/ServingGemvTaskBody.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Target/ArchDispatch.h>

#include <cuda_runtime.h>
#include <cstdint>

using namespace tilemega;
using namespace tilemega::codegen;

namespace {

struct GemmArgs {
  Bf16 const* a;
  Bf16 const* b;
  Bf16 const* residual;
  Bf16* output;
  float* partial;
  float* argmax_value;
  int* argmax_index;
  float const* norm_ss;
  float* ss_out;
  int m, n, k_total, k_begin, k_count;
  int output_stride, partial_stride, a_row_stride, b_row_stride;
  int norm_k;
  float norm_eps;
  int epilogue;
};

__device__ ServingGemmOperands Operands(GemmArgs const& g) {
  ServingGemmOperands p;
  p.a = g.a;
  p.b = g.b;
  p.residual = g.residual;
  p.output = g.output;
  p.partial = g.partial;
  p.argmax_value = g.argmax_value;
  p.argmax_index = g.argmax_index;
  p.norm_ss = g.norm_ss;
  p.ss_out = g.ss_out;
  p.m = g.m;
  p.n = g.n;
  p.k_total = g.k_total;
  p.k_begin = g.k_begin;
  p.k_count = g.k_count;
  p.output_stride = g.output_stride;
  p.partial_stride = g.partial_stride;
  p.a_row_stride = g.a_row_stride;
  p.b_row_stride = g.b_row_stride;
  p.norm_k = g.norm_k;
  p.norm_eps = g.norm_eps;
  p.epilogue = static_cast<backend::ServingEpilogueOp>(g.epilogue);
  return p;
}

template <class Body>
__global__ void __launch_bounds__(128) RunGemm(GemmArgs g, int tiles_n) {
  extern __shared__ __align__(16) char shared[];
  auto p = Operands(g);
  Body::Run(p, blockIdx.x / tiles_n, blockIdx.x % tiles_n, shared);
}

template <class Body>
int Launch(GemmArgs const& g, int tile_m, int tile_n, void* stream) {
  int const tiles_m = (g.m + tile_m - 1) / tile_m, tiles_n = (g.n + tile_n - 1) / tile_n;
  int const smem = Body::kSharedBytes;
  if (cudaFuncSetAttribute(RunGemm<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    return 1;
  RunGemm<Body><<<tiles_m * tiles_n, 128, smem, static_cast<cudaStream_t>(stream)>>>(g, tiles_n);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace

using X = arch::Xcore1000;

/// MMA GEMM: tile_m in {16,32,64}, tile_n in {32,64,128}, tile_k fixed 64;
/// row-major B at depth 2 where the registers allow it (64x128 does not).
extern "C" int tm_test_gemm(GemmArgs const* g, int tile_m, int tile_n, void* stream) {
#define TM_CASE(M, N)                                                                                \
  if (tile_m == M && tile_n == N)                                                                    \
    return Launch<ServingGemmTaskBody<X, M, N, 64, tilemega::solver::ServingGemmDepthFits(M, N, 2, 2) ? 2 : 1>>( \
        *g, M, N, stream);
  TM_CASE(16, 32) TM_CASE(16, 64) TM_CASE(16, 128) TM_CASE(32, 64) TM_CASE(64, 64) TM_CASE(64, 128)
#undef TM_CASE
  return 3;
}

/// SIMT GEMV (M <= 4): tile_n in {8,16,32,64}, tile_k 128.
extern "C" int tm_test_gemv(GemmArgs const* g, int tile_n, void* stream) {
#define TM_CASE(N) \
  if (tile_n == N) return Launch<ServingGemvTaskBody<X, N, 128>>(*g, 16, N, stream);
  TM_CASE(8) TM_CASE(16) TM_CASE(32) TM_CASE(64)
#undef TM_CASE
  return 3;
}

extern "C" int tm_test_gemm_args_size() { return sizeof(GemmArgs); }
// Fragment-packed (tiled) MMA mainloop: B from `weight` packed by the loader's
// xcore_mma tile_pages recipe; the FP32 tile of every (tile_m, tile_n) goes to
// `tiles` as [tiles_m*tiles_n][TM*TN].
namespace {
template <int TM, int TN, int TK, int S>
__global__ void __launch_bounds__(128) RunTiledDense(GemmArgs g, Bf16 const* weight, int k_total_full,
                                                     int tiles_n, float* tiles) {
  extern __shared__ __align__(16) char shared[];
  auto p = Operands(g);
  p.weight_base = weight;
  p.k_total_full = k_total_full;
  int const tm = blockIdx.x / tiles_n, tn = blockIdx.x % tiles_n;
  float* tile = backend::ServingGemmMma<TM, TN, TK, S>::template Dense<true>(p, tm, tn, shared);
  for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) tiles[std::size_t(blockIdx.x) * TM * TN + i] = tile[i];
}
}  // namespace

extern "C" int tm_test_gemm_tiled_staged(GemmArgs const* g, Bf16 const* weight, int k_total_full,
                                         int tile_m, int tile_n, int tile_k, int stages,
                                         float* tiles, void* stream) {
#define TM_LAUNCH(M, N, K, S)                                                                       \
  if (tile_m == M && tile_n == N && tile_k == K && stages == S &&                                   \
      tilemega::solver::ServingGemmDepthFits(M, N, S, 2)) {                                                  \
    if constexpr (tilemega::solver::ServingGemmDepthFits(M, N, S, 2)) {                                      \
    int const tm = (g->m + M - 1) / M, tn = (g->n + N - 1) / N, smem = 4 * M * N + 4 * M;           \
    if (cudaFuncSetAttribute(RunTiledDense<M, N, K, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) \
      return 1;                                                                                     \
    RunTiledDense<M, N, K, S><<<tm * tn, 128, smem, static_cast<cudaStream_t>(stream)>>>(*g, weight, k_total_full, tn, tiles); \
    return cudaGetLastError() == cudaSuccess ? 0 : 2;                                               \
    }                                                                                               \
  }
#define TM_CASE(M, N, K) TM_LAUNCH(M, N, K, 1) TM_LAUNCH(M, N, K, 2) TM_LAUNCH(M, N, K, 3) \
                         TM_LAUNCH(M, N, K, 4) TM_LAUNCH(M, N, K, 6)
  TM_CASE(16, 32, 64) TM_CASE(16, 32, 128) TM_CASE(16, 128, 64) TM_CASE(32, 64, 128) TM_CASE(64, 128, 64)
#undef TM_CASE
#undef TM_LAUNCH
  return 3;
}

extern "C" int tm_test_gemm_tiled(GemmArgs const* g, Bf16 const* weight, int k_total_full, int tile_m, int tile_n,
                                  int tile_k, float* tiles, void* stream) {
  return tm_test_gemm_tiled_staged(g, weight, k_total_full, tile_m, tile_n, tile_k, 1, tiles, stream);
}


// ---- attention -------------------------------------------------------------
#include <tilemega/Codegen/tasks/mx/DecodeAttentionSimt.h>
#include <tilemega/Codegen/tasks/mx/PrefillAttentionMma.h>

namespace {
struct AttentionArgs {
  Bf16 const* qkv;
  Bf16* key_cache;
  Bf16* value_cache;
  Bf16 const* cosine;
  Bf16 const* sine;
  Bf16 const* q_norm;
  Bf16 const* k_norm;
  Bf16* context;
  float* partial;
  float* lse;
  int batch, heads_kv, capacity, past, block_extent;
  float epsilon;
};
__device__ ServingAttentionOperands AttentionOperands(AttentionArgs const& a) {
  ServingAttentionOperands p{};
  p.qkv = a.qkv; p.key_cache = a.key_cache; p.value_cache = a.value_cache;
  p.cosine = a.cosine; p.sine = a.sine; p.q_norm = a.q_norm; p.k_norm = a.k_norm;
  p.context = a.context; p.partial = a.partial; p.lse = a.lse;
  p.batch = a.batch; p.heads_kv = a.heads_kv; p.capacity = a.capacity; p.past = a.past;
  p.block_extent = a.block_extent; p.epsilon = a.epsilon;
  return p;
}
template <class Body>
__global__ void __launch_bounds__(128) RunDecodeAttention(AttentionArgs a, int blocks) {
  extern __shared__ __align__(16) char shared[];
  int const c = blockIdx.x % blocks, g = (blockIdx.x / blocks) % a.heads_kv, b = blockIdx.x / blocks / a.heads_kv;
  Body::Run(AttentionOperands(a), *reinterpret_cast<typename Body::SharedStorage*>(shared), b, g, c);
}
template <class Body>
__global__ void __launch_bounds__(128) RunPrefillAttention(AttentionArgs a, int query_blocks) {
  extern __shared__ __align__(16) char shared[];
  int const qb = blockIdx.x % query_blocks, g = (blockIdx.x / query_blocks) % a.heads_kv,
            b = blockIdx.x / query_blocks / a.heads_kv;
  Body::Run(AttentionOperands(a), *reinterpret_cast<typename Body::SharedStorage*>(shared), b, g, qb, 0);
}
/// Decode through the MMA body: one 16-row query block (qb = 0) and one task
/// per KV block, which is the opposite mapping from prefill, where the KV
/// blocks are streamed inside Run and the grid covers query blocks.
template <class Body>
__global__ void __launch_bounds__(128) RunDecodeAttentionMma(AttentionArgs a, int blocks) {
  extern __shared__ __align__(16) char shared[];
  int const c = blockIdx.x % blocks, g = (blockIdx.x / blocks) % a.heads_kv,
            b = blockIdx.x / blocks / a.heads_kv;
  Body::Run(AttentionOperands(a), *reinterpret_cast<typename Body::SharedStorage*>(shared), b, g, 0, c);
}

template <class Body, class Kernel>
int LaunchAttention(Kernel kernel, AttentionArgs const& a, int per_group, void* stream) {
  int const smem = sizeof(typename Body::SharedStorage);
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 1;
  kernel<<<a.batch * a.heads_kv * per_group, 128, smem, static_cast<cudaStream_t>(stream)>>>(a, per_group);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
}  // namespace

/// Decode attention: (D, Q, norm) in {(128,2,1), (64,4,0), (128,4,0), (128,3,0), (128,8,1)}.
/// `impl` 3 is simt, 0 is mma16. mma16 reuses PrefillAttentionMma at
/// Tokens=1, so the same inputs must give the same partials, LSE and context
/// as the SIMT body at the tolerance the SIMT body already meets (MX3-3 item
/// 4). Its query block is a 16-row MMA tile, so `per_group` stays the block
/// count and the query-block argument is 0.
extern "C" int tm_test_decode_attention_impl(AttentionArgs const* a, int d, int q, int norm,
                                             int impl, void* stream) {
  int const blocks = (a->capacity + a->block_extent - 1) / a->block_extent;
#define TM_SIMT(D, Q, N)                                                                   \
  if (d == D && q == Q && norm == N && impl == 3) {                                        \
    using B = mx::DecodeAttentionSimt<D, Q, N != 0>;                                       \
    return LaunchAttention<B>(RunDecodeAttention<B>, *a, blocks, stream);                  \
  }
#define TM_MMA(D, Q, N)                                                                    \
  if (d == D && q == Q && norm == N && impl == 0) {                                        \
    using B = mx::PrefillAttentionMma<D, Q, 1, 16, (Q >= 8 ? 64 : 32), N != 0>;            \
    return LaunchAttention<B>(RunDecodeAttentionMma<B>, *a, blocks, stream);                \
  }
#define TM_CASE(D, Q, N) TM_SIMT(D, Q, N) TM_MMA(D, Q, N)
  TM_CASE(128, 2, 1) TM_CASE(64, 4, 0) TM_CASE(128, 4, 0) TM_CASE(128, 3, 0) TM_CASE(128, 8, 1)
#undef TM_CASE
#undef TM_MMA
#undef TM_SIMT
  return 3;
}

extern "C" int tm_test_decode_attention(AttentionArgs const* a, int d, int q, int norm, void* stream) {
  return tm_test_decode_attention_impl(a, d, q, norm, 3, stream);
}

/// Prefill attention, Tokens = 64: (D, Q, norm, kv_tile, q_rows).
extern "C" int tm_test_prefill_attention(AttentionArgs const* a, int d, int q, int norm, int kv, int rows,
                                         void* stream) {
#define TM_CASE(D, Q, N, KV, R)                                                            \
  if (d == D && q == Q && norm == N && kv == KV && rows == R) {                            \
    using B = mx::PrefillAttentionMma<D, Q, 64, R, KV, N != 0>;                             \
    return LaunchAttention<B>(RunPrefillAttention<B>, *a, (64 * Q + R - 1) / R, stream);   \
  }
  TM_CASE(128, 2, 1, 32, 64) TM_CASE(128, 2, 1, 64, 16) TM_CASE(64, 4, 0, 64, 64) TM_CASE(128, 4, 0, 32, 32)
#undef TM_CASE
  return 3;
}
extern "C" int tm_test_attention_args_size() { return sizeof(AttentionArgs); }
