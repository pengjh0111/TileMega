// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>

namespace tilemega::backend::mx {

template <int D, bool QkNorm>
__device__ __forceinline__ void MxRowTransformV2(uint4 (&x)[D / 32], uint4 const (&weights)[D / 32],
                                               uint4 const (&cosine)[D / 32], uint4 const (&sine)[D / 32],
                                               float epsilon, float* exchange, int lane) {
  static_assert(D == 64 || D == 128);
  static_assert(!QkNorm || D == 128, "the covered q/k-normalized models use D=128");
  constexpr int chunks = D / 32;
  uint4 original[chunks];
#pragma unroll
  for (int dc = 0; dc < chunks; ++dc) original[dc] = x[dc];
  float inverse = 1.f;
  if constexpr (QkNorm) {
    float squares[chunks];
#pragma unroll
    for (int dc = 0; dc < chunks; ++dc) {
      alignas(16) Bf16 values[8];
      *reinterpret_cast<uint4*>(values) = original[dc];
      squares[dc] = codegen::mx::SquareEight(values);
    }
    int const row = lane & 15, group = lane >> 4;
    exchange[group * 16 + row] = (squares[0] + squares[2]) + (squares[1] + squares[3]);
    __syncwarp();
    float const sum = (exchange[row] + exchange[32 + row]) + (exchange[16 + row] + exchange[48 + row]);
    inverse = maca::Rsqrt(sum / float(D) + epsilon);
  }
#pragma unroll
  for (int dc = 0; dc < chunks; ++dc) {
    int const partner = dc ^ (chunks / 2);
    uint4 packed_w = weights[dc], packed_pw = weights[partner];
    uint4 packed_c = cosine[dc], packed_s = sine[dc];
    if constexpr (QkNorm) {
      // Keep BF16 operands packed until the normalization inverse is ready.
      asm volatile("" : "+r"(packed_w.x), "+r"(packed_w.y), "+r"(packed_w.z), "+r"(packed_w.w),
                        "+r"(packed_pw.x), "+r"(packed_pw.y), "+r"(packed_pw.z), "+r"(packed_pw.w),
                        "+r"(packed_c.x), "+r"(packed_c.y), "+r"(packed_c.z), "+r"(packed_c.w),
                        "+r"(packed_s.x), "+r"(packed_s.y), "+r"(packed_s.z), "+r"(packed_s.w)
                      : "r"(inverse));
    }
    alignas(16) Bf16 a[8], b[8], w[8], pw[8], c[8], s[8];
    // RotateEight changes both operands; every pair starts from raw copies.
    *reinterpret_cast<uint4*>(a) = original[dc];
    *reinterpret_cast<uint4*>(b) = original[partner];
    *reinterpret_cast<uint4*>(w) = packed_w;
    *reinterpret_cast<uint4*>(pw) = packed_pw;
    *reinterpret_cast<uint4*>(c) = packed_c;
    *reinterpret_cast<uint4*>(s) = packed_s;
    codegen::mx::RotateEight<QkNorm>(a, b, w, pw, c, s, inverse, dc < chunks / 2);
    x[dc] = *reinterpret_cast<uint4 const*>(a);
  }
}

}  // namespace tilemega::backend::mx
