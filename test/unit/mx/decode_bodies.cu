// SPDX-License-Identifier: BSD-3-Clause
// Numerical test entry points for the MX-4 decode stage bodies
// (Backend/Mx/DecodeGemm.h, DecodeAttention.h). test/python/
// test_mx_decode_bodies.py builds operands with torch, calls these through
// ctypes and compares against FP32 torch, or bit for bit against the
// RotateRow reference below. Every body runs as the program runs it: G = 104
// CTAs of 512 threads, its own prefetch first, dynamic shared memory only.
#include <tilemega/Backend/Mx/DecodeAttention.h>
#include <tilemega/Backend/Mx/DecodeGemm.h>
#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>
#include <tilemega/Codegen/tasks/mx/PrefillAttentionMma.h>

#include <cuda_runtime.h>
#include <cstdint>

using namespace tilemega;
using namespace tilemega::backend::mx;

namespace {

constexpr int kGrid = 104;
/// The program's request at G = 104 (MX4 prompt §3.1).
constexpr int kProgramSmem = 65536 / 2 + 16;

struct DecodeGemmArgs {
  Bf16 const* a;
  Bf16 const* w;
  Bf16* out;
  float* partial;
  unsigned* tickets;
  Bf16 const* q_norm;
  Bf16 const* k_norm;
  Bf16 const* cosine;
  Bf16 const* sine;
  Bf16* k_cache;
  Bf16* vt_cache;
  float* head_value;
  int* head_index;
  unsigned* head_ticket;
  int* tokens;
  int capacity, past;
  float epsilon;
  int n, k, b;
};

__device__ MxGemmOperands Operands(DecodeGemmArgs const& g) {
  return MxGemmOperands{g.a,        g.w,        g.out,        g.partial,     g.tickets,     g.q_norm,
                        g.k_norm,   g.cosine,   g.sine,       g.k_cache,     g.vt_cache,    g.head_value,
                        g.head_index, g.head_ticket, g.tokens, g.capacity, g.past, g.epsilon};
}

template <class Body>
__global__ void __launch_bounds__(512) RunGemm(DecodeGemmArgs g) {
  extern __shared__ __align__(16) char smem[];
  MxGemmOperands const p = Operands(g);
  MxShape const s{g.n, g.k, g.b};
  auto const pf = Body::Prefetch(p, s);
  __syncthreads();
  Body::Run(pf, p, s, smem);
}

template <class Body>
int Launch(DecodeGemmArgs const& g, int tn, void* stream) {
  int const smem = max(MxGemmSmem(g.b, tn).bytes, kProgramSmem);
  if (cudaFuncSetAttribute(RunGemm<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    return 1;
  RunGemm<Body><<<kGrid, 512, smem, static_cast<cudaStream_t>(stream)>>>(g);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <MxDecodeOp Op, bool Norm, MxPart Part, int TN, int D, class Rope = MxNoRope>
int Gemm(DecodeGemmArgs const& g, void* stream) {
  constexpr bool kLegal = MxDepthFits(TN, D) && (Op != MxDecodeOp::kSwiGLU || TN >= 32) &&
                          (Op != MxDecodeOp::kArgmax || Part == MxPart::kTileRange) &&
                          (Op != MxDecodeOp::kQkvRope ||
                           ((TN == Rope::kHeadDim || TN == 2 * Rope::kHeadDim) &&
                            ((Rope::kQ + 2) * Rope::kHeadDim) % TN == 0));
  if constexpr (!kLegal) {
    return 4;
  } else {
    return Launch<MxDecodeGemm<TN, D, Op, Norm, Part, Rope>>(g, TN, stream);
  }
}

template <MxDecodeOp Op, bool Norm, MxPart Part, class Rope = MxNoRope>
int GemmTD(DecodeGemmArgs const& g, int tn, int d, void* stream) {
#define TM_CASE(TN, D) \
  if (tn == TN && d == D) return Gemm<Op, Norm, Part, TN, D, Rope>(g, stream);
#define TM_TN(TN) TM_CASE(TN, 1) TM_CASE(TN, 2) TM_CASE(TN, 3) TM_CASE(TN, 4)
  if constexpr (Op == MxDecodeOp::kQkvRope) {
    if constexpr (Rope::kHeadDim == 64) {
      TM_TN(64) TM_TN(128)
    } else {
      TM_TN(128)
    }
  } else {
    TM_TN(16) TM_TN(32) TM_TN(64) TM_TN(128)
  }
#undef TM_TN
#undef TM_CASE
  return 3;
}

template <MxDecodeOp Op, class Rope = MxNoRope>
int GemmNP(DecodeGemmArgs const& g, int tn, int d, int norm, int part, void* stream) {
  bool const stream_k = part == 0;
  if constexpr (Op == MxDecodeOp::kArgmax) {
    if (stream_k) return 4;
    return norm ? GemmTD<Op, true, MxPart::kTileRange, Rope>(g, tn, d, stream)
                : GemmTD<Op, false, MxPart::kTileRange, Rope>(g, tn, d, stream);
  } else {
    if (norm)
      return stream_k ? GemmTD<Op, true, MxPart::kStreamK, Rope>(g, tn, d, stream)
                      : GemmTD<Op, true, MxPart::kTileRange, Rope>(g, tn, d, stream);
    return stream_k ? GemmTD<Op, false, MxPart::kStreamK, Rope>(g, tn, d, stream)
                    : GemmTD<Op, false, MxPart::kTileRange, Rope>(g, tn, d, stream);
  }
}

}  // namespace

extern "C" int tm_decode_gemm_args_size() { return sizeof(DecodeGemmArgs); }
extern "C" int tm_decode_gemm_smem(int b, int tn) { return MxGemmSmem(b, tn).bytes; }
extern "C" int tm_decode_max_contributors(int tiles, int grid) { return MxMaxContributors(tiles, grid); }
extern "C" int tm_decode_grid() { return kGrid; }

/// op: 1 kResidual (out = x in place), 2 kSwiGLU, 3 kArgmax; part: 0
/// kStreamK, 1 kTileRange. 3 = no such (TN, D), 4 = illegal combination.
extern "C" int tm_decode_gemm(DecodeGemmArgs const* g, int tn, int d, int op, int norm, int part, void* stream) {
  switch (op) {
    case 1: return GemmNP<MxDecodeOp::kResidual>(*g, tn, d, norm, part, stream);
    case 2: return GemmNP<MxDecodeOp::kSwiGLU>(*g, tn, d, norm, part, stream);
    case 3: return GemmNP<MxDecodeOp::kArgmax>(*g, tn, d, norm, part, stream);
    default: return 3;
  }
}

/// kQkvRope with a head configuration: 0 = D128 Q2 hkv8 q/k-norm (Qwen3),
/// 1 = D64 Q4 hkv8 (Llama-3.2-1B), 2 = D128 Q4 hkv8 (Llama-3.1-8B). Norm on.
extern "C" int tm_decode_qkv(DecodeGemmArgs const* g, int tn, int d, int config, int part, void* stream) {
  switch (config) {
    case 0: return GemmNP<MxDecodeOp::kQkvRope, MxRope<128, 2, 8, true>>(*g, tn, d, 1, part, stream);
    case 1: return GemmNP<MxDecodeOp::kQkvRope, MxRope<64, 4, 8, false>>(*g, tn, d, 1, part, stream);
    case 2: return GemmNP<MxDecodeOp::kQkvRope, MxRope<128, 4, 8, false>>(*g, tn, d, 1, part, stream);
    default: return 3;
  }
}

// ---- kQkvRope reference -----------------------------------------------------
// The un-fused path, bit for bit: y = bf16(x W^T * inv) with W row-major and
// an exact FP32 product (the test's operands are small integers, so every
// accumulation order gives the same sum), then RotateRow on the stored row.
namespace {
template <int D, int Q, bool QkNorm>
__global__ void __launch_bounds__(32) QkvReference(Bf16 const* x, Bf16 const* w, int k, int hkv, float epsilon,
                                                   Bf16 const* q_norm, Bf16 const* k_norm, Bf16 const* cosine,
                                                   Bf16 const* sine, int past, int capacity, Bf16* row_scratch,
                                                   Bf16* q_out, Bf16* k_cache, Bf16* vt_cache) {
  int const b = blockIdx.y, head = blockIdx.x, group = head / (Q + 2), j = head % (Q + 2);
  int const n = hkv * (Q + 2) * D;
  float ss = 0;
  for (int i = 0; i < k; ++i) ss += float(x[std::size_t(b) * k + i]) * float(x[std::size_t(b) * k + i]);
  float const inv = maca::Rsqrt(ss / float(k) + epsilon);
  Bf16* const row = row_scratch + std::size_t(b) * n + std::size_t(head) * D;
  for (int d = threadIdx.x; d < D; d += 32) {
    float acc = 0;
    for (int i = 0; i < k; ++i) acc += float(x[std::size_t(b) * k + i]) * float(w[(std::size_t(head) * D + d) * k + i]);
    row[d] = Bf16(acc * inv);
  }
  __syncwarp();
  std::size_t const bg = std::size_t(b) * hkv + group;
  if (j == Q + 1) {
    for (int d = threadIdx.x; d < D; d += 32) vt_cache[(bg * D + d) * capacity + past] = row[d];
    return;
  }
  Bf16 const* norm = j < Q ? q_norm : k_norm;
  codegen::mx::RotateRow<D, QkNorm>(row, norm, cosine + std::size_t(past) * D, sine + std::size_t(past) * D, epsilon,
                                    [&](int d, uint4 v) {
                                      Bf16* const target = j < Q ? q_out + (bg * (Q + 2) + j) * D + d
                                                                 : k_cache + (bg * capacity + past) * D + d;
                                      *reinterpret_cast<uint4*>(target) = v;
                                    });
}
}  // namespace

extern "C" int tm_decode_qkv_reference(DecodeGemmArgs const* g, Bf16 const* w_rows, int config, Bf16* row_scratch,
                                       void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
#define TM_REF(D, Q, N)                                                                                         \
  QkvReference<D, Q, N><<<dim3(8 * (Q + 2), g->b), 32, 0, s>>>(g->a, w_rows, g->k, 8, g->epsilon, g->q_norm,    \
                                                               g->k_norm, g->cosine, g->sine, g->past,          \
                                                               g->capacity, row_scratch, g->out, g->k_cache,    \
                                                               g->vt_cache);
  switch (config) {
    case 0: TM_REF(128, 2, true) break;
    case 1: TM_REF(64, 4, false) break;
    case 2: TM_REF(128, 4, false) break;
    default: return 3;
  }
#undef TM_REF
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// ---- attention ----------------------------------------------------------------
namespace {
struct DecodeAttentionArgs {
  Bf16 const* qkv;
  Bf16 const* k_cache;
  Bf16 const* vt_cache;
  Bf16* ctx;
  float* partial;
  unsigned* tickets;
  int capacity, past, b, hkv;
};

template <class Body>
__global__ void __launch_bounds__(512) RunAttention(DecodeAttentionArgs g) {
  extern __shared__ __align__(16) char smem[];
  MxAttentionOperands const p{g.qkv, g.k_cache, g.vt_cache, g.ctx, g.partial, g.tickets, g.capacity, g.past};
  MxAttentionShape const s{g.b, g.hkv};
  auto const pf = Body::Prefetch(p, s);
  __syncthreads();
  Body::Run(pf, p, s, smem);
}

template <int D, int Q, int Tmin>
int LaunchAttention(DecodeAttentionArgs const& g, void* stream) {
  using Body = MxDecodeAttention<D, Q, Tmin>;
  int const smem = max(Body::kSharedBytes, kProgramSmem);
  if (cudaFuncSetAttribute(RunAttention<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    return 1;
  RunAttention<Body><<<kGrid, 512, smem, static_cast<cudaStream_t>(stream)>>>(g);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
}  // namespace

extern "C" int tm_decode_attention_args_size() { return sizeof(DecodeAttentionArgs); }

extern "C" int tm_decode_attention(DecodeAttentionArgs const* g, int d, int q, int tmin, void* stream) {
#define TM_CASE(D, Q, T) \
  if (d == D && q == Q && tmin == T) return LaunchAttention<D, Q, T>(*g, stream);
#define TM_DQ(D, Q) TM_CASE(D, Q, 32) TM_CASE(D, Q, 64) TM_CASE(D, Q, 128)
  TM_DQ(64, 2) TM_DQ(64, 4) TM_DQ(128, 2) TM_DQ(128, 4)
#undef TM_DQ
#undef TM_CASE
  return 3;
}

// ---- embed ----------------------------------------------------------------------
namespace {
__global__ void __launch_bounds__(512) RunEmbed(Bf16 const* embed, int const* tokens, Bf16* x, int b, int hidden,
                                                int capacity, int past) {
  MxDecodeEmbed(embed, tokens, x, MxShape{0, 0, b}, hidden, capacity, past);
}
}  // namespace

extern "C" int tm_decode_embed(Bf16 const* embed, int const* tokens, Bf16* x, int b, int hidden, int capacity,
                               int past, void* stream) {
  RunEmbed<<<kGrid, 512, 0, static_cast<cudaStream_t>(stream)>>>(embed, tokens, x, b, hidden, capacity, past);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// ---- prefill (V^T) -> decode -------------------------------------------------------
// §8: the prefill attention body writes K rows and V^T columns [0, 64) of the
// `vt` cache layout; MX-4 decode attention then reads them at past = 64.
namespace {
struct PrefillArgs {
  Bf16 const* qkv;  // [B][64][hkv][Q+2][D], before q/k-norm and RoPE
  Bf16* k_cache;
  Bf16* vt_cache;
  Bf16 const* cosine;
  Bf16 const* sine;
  Bf16 const* q_norm;
  Bf16 const* k_norm;
  Bf16* context;    // [B][64][hkv*Q][D]
  int batch, heads_kv, capacity;
  float epsilon;
};
template <class Body>
__global__ void __launch_bounds__(128) RunPrefillVt(PrefillArgs a, int query_blocks) {
  extern __shared__ __align__(16) char shared[];
  codegen::ServingAttentionOperands p{};
  p.qkv = a.qkv; p.key_cache = a.k_cache; p.value_cache = a.vt_cache; p.cosine = a.cosine; p.sine = a.sine;
  p.q_norm = a.q_norm; p.k_norm = a.k_norm; p.context = a.context; p.batch = a.batch; p.heads_kv = a.heads_kv;
  p.capacity = a.capacity; p.past = 0; p.block_extent = a.capacity; p.epsilon = a.epsilon;
  int const qb = blockIdx.x % query_blocks, g = (blockIdx.x / query_blocks) % a.heads_kv,
            b = blockIdx.x / query_blocks / a.heads_kv;
  Body::Run(p, *reinterpret_cast<typename Body::SharedStorage*>(shared), b, g, qb, 0);
}
}  // namespace

extern "C" int tm_decode_prefill_args_size() { return sizeof(PrefillArgs); }

/// (D, Q, q/k-norm) in {(128,2,1), (64,4,0), (128,4,0)}, Tokens = 64.
extern "C" int tm_decode_prefill_vt(PrefillArgs const* a, int d, int q, int norm, void* stream) {
#define TM_CASE(D, Q, N, KV, R)                                                                       \
  if (d == D && q == Q && norm == N) {                                                                \
    using B = codegen::mx::PrefillAttentionMma<D, Q, 64, R, KV, N != 0, true>;                        \
    int const smem = sizeof(typename B::SharedStorage), blocks = (64 * Q + R - 1) / R;                \
    if (cudaFuncSetAttribute(RunPrefillVt<B>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) \
      return 1;                                                                                       \
    RunPrefillVt<B><<<a->batch * a->heads_kv * blocks, 128, smem, static_cast<cudaStream_t>(stream)>>>(*a, blocks); \
    return cudaGetLastError() == cudaSuccess ? 0 : 2;                                                 \
  }
  TM_CASE(128, 2, 1, 32, 64) TM_CASE(64, 4, 0, 64, 64) TM_CASE(128, 4, 0, 32, 32)
#undef TM_CASE
  return 3;
}
