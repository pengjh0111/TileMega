// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/DecodeShapes.h
//
// The MX-4 decode bodies' shape rules (MX4 prompt §3.5, §3.6), shared by the
// bodies (DecodeGemm.h, DecodeAttention.h), the program generator
// (lib/Codegen/MxDecodeProgram.cpp) and the solver: one definition of what is
// legal and how much memory it takes. Plain C++ outside CUDA.
#pragma once

#include <tilemega/Target/Maca/Bf16.h>

// Force-inlined on the device: the plan build (cucc -shared) left these as
// calls inside the program, each with a stack frame (measured).
#if defined(__CUDACC__) || defined(__MACACC__) || defined(__HIP__)
#define TM_MX_HD __host__ __device__ __forceinline__
#else
#define TM_MX_HD inline
#endif

namespace tilemega::backend::mx {

inline constexpr int kDecodeThreads = 512;
inline constexpr int kDecodeWaves = kDecodeThreads / 64;
/// R_max of §3.1: the CTA's 64 KiB less 1 KiB for everything else.
inline constexpr int kDecodeReduceBytes = 65536 - 1024;

enum class MxDecodeOp { kQkvRope, kResidual, kSwiGLU, kArgmax };
enum class MxPart { kStreamK, kTileRange };

/// N, K, B: constexpr in the program (MxStaticShape), runtime in the stage
/// bench (MxShape); the body reads them the same way.
struct MxShape {
  int n, k, b;
};
template <int N, int K, int B>
struct MxStaticShape {
  static constexpr int n = N, k = K, b = B;
};

/// Head constants of the Qkv epilogue (kQkvRope only).
template <int HeadDim, int QPerKv, int HeadsKv, bool QkNorm>
struct MxRope {
  static constexpr int kHeadDim = HeadDim, kQ = QPerKv, kHeadsKv = HeadsKv;
  static constexpr bool kQkNorm = QkNorm;
};
struct MxNoRope {
  static constexpr int kHeadDim = 64, kQ = 1, kHeadsKv = 1;
  static constexpr bool kQkNorm = false;
};

/// Every address a GEMM stage touches. Passed by value into inlined code.
struct MxGemmOperands {
  tilemega::Bf16 const* a;  // [B][K]
  tilemega::Bf16 const* w;  // tile_pages(xcore_mma, TN, 32)
  tilemega::Bf16* out;      // kResidual: x (in place); kSwiGLU: h; kQkvRope: qkv
  float* partial;           // [T][P][B][TN+1], P = MaxContributors
  unsigned* tickets;        // [T]
  // kQkvRope
  tilemega::Bf16 const *q_norm, *k_norm, *cosine, *sine;
  tilemega::Bf16 *k_cache, *vt_cache;
  // kArgmax
  float* head_value;   // [B][G]
  int* head_index;     // [B][G]
  unsigned* head_ticket;
  int* tokens;         // [B][capacity]
  int capacity, past;
  float epsilon;
};

/// Bound on the CTAs sharing a tile: ceil(G/T) + 1.
TM_MX_HD constexpr int MxMaxContributors(int tiles, int grid) { return (grid + tiles - 1) / tiles + 1; }

/// §3.5 reduction legality: rounds 1 or 2, 0 = illegal.
TM_MX_HD constexpr int MxReduceRounds(int b, int tn) {
  return kDecodeWaves * b * (tn + 1) * 4 <= kDecodeReduceBytes         ? 1
         : (kDecodeWaves / 2) * b * (tn + 1) * 4 <= kDecodeReduceBytes ? 2
                                                                        : 0;
}
/// §3.5 rule 6: accumulators plus the ring within 160 VGPRs.
TM_MX_HD constexpr bool MxDepthFits(int tn, int depth) {
  return (tn / 16) * 4 + depth * (tn / 16 + 1) * 4 <= 160;
}

/// Shared-memory layout of one GEMM stage, in floats.
struct MxGemmSmem {
  int buf, ss, res, square, cand_value, cand_index, best_value, best_index, flag, bytes;
  TM_MX_HD constexpr MxGemmSmem(int b, int tn)
      : buf(0),
        ss(buf + (MxReduceRounds(b, tn) == 1 ? kDecodeWaves : kDecodeWaves / 2) * b * tn),
        res(ss + kDecodeWaves * 64),
        square(res + b * (tn + 1)),
        cand_value(square + b * tn / 8),
        cand_index(cand_value + b * tn / 8),
        best_value(cand_index + b * tn / 8),
        best_index(best_value + 16),
        flag(best_index + 16),
        bytes(((flag + 1) * 4 + 15) / 16 * 16) {}
};

/// T_c of §3.6: the smallest multiple of 32, >= Tmin, leaving at most
/// `slots` = G*W items for L tokens; capped at 256.
TM_MX_HD constexpr int MxAttentionChunkTokens(int heads, int length, int tmin, int slots) {
  for (int t = (tmin + 31) / 32 * 32; t < 256; t += 32)
    if (heads * ((length + t - 1) / t) <= slots) return t;
  return 256;
}
/// The most 32-token blocks an item can hold over L in [1, capacity]: the
/// bodies for larger counts are never reached by the plan and are not
/// instantiated (MaxBlocks below).
TM_MX_HD constexpr int MxAttentionMaxBlocks(int heads, int capacity, int tmin, int slots) {
  int most = 1;
  for (int length = 1; length <= capacity; ++length) {
    int const t = MxAttentionChunkTokens(heads, length, tmin, slots);
    int const blocks = ((t < length ? t : length) + 31) / 32;
    most = blocks > most ? blocks : most;
  }
  return most;
}


}  // namespace tilemega::backend::mx
