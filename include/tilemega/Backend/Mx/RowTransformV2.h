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
  float inverse = 1.f;
  if constexpr (QkNorm) {
    float squares[chunks];
#pragma unroll
    for (int dc = 0; dc < chunks; ++dc) {
      alignas(16) Bf16 values[8];
      *reinterpret_cast<uint4*>(values) = x[dc];
      squares[dc] = codegen::mx::SquareEight(values);
    }
    int const row = lane & 15, group = lane >> 4;
    exchange[group * 16 + row] = (squares[0] + squares[2]) + (squares[1] + squares[3]);
    __syncwarp();
    float const sum = (exchange[row] + exchange[32 + row]) + (exchange[16 + row] + exchange[48 + row]);
    inverse = maca::Rsqrt(sum / float(D) + epsilon);
  }
#pragma unroll
  for (int dc = 0; dc < chunks / 2; ++dc) {
    int const partner = dc + chunks / 2;
    // Both rotations start from the same unmodified pair.
    uint4 const original_pair[2] = {x[dc], x[partner]};
    uint4 prepared_pair[2] = {weights[dc], weights[partner]};
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      if constexpr (QkNorm) {
        auto& packed = prepared_pair[side];
        asm volatile("" : "+r"(packed.x), "+r"(packed.y), "+r"(packed.z), "+r"(packed.w) : "r"(inverse));
      }
    }
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      int const at = dc + side * (chunks / 2);
      uint4 packed_w = prepared_pair[side], packed_pw = prepared_pair[side ^ 1];
      uint4 packed_c = cosine[at], packed_s = sine[at];
      if constexpr (QkNorm) {
        // Keep BF16 operands packed until the normalization inverse is ready.
        asm volatile("" : "+r"(packed_c.x), "+r"(packed_c.y), "+r"(packed_c.z), "+r"(packed_c.w),
                          "+r"(packed_s.x), "+r"(packed_s.y), "+r"(packed_s.z), "+r"(packed_s.w)
                        : "r"(inverse));
      }
      alignas(16) Bf16 a[8], b[8], w[8], pw[8], c[8], s[8];
      *reinterpret_cast<uint4*>(a) = original_pair[side];
      *reinterpret_cast<uint4*>(b) = original_pair[side ^ 1];
      *reinterpret_cast<uint4*>(w) = packed_w;
      *reinterpret_cast<uint4*>(pw) = packed_pw;
      *reinterpret_cast<uint4*>(c) = packed_c;
      *reinterpret_cast<uint4*>(s) = packed_s;
      codegen::mx::RotateEight<QkNorm>(a, b, w, pw, c, s, inverse, side == 0);
      x[at] = *reinterpret_cast<uint4 const*>(a);
    }
  }
}

}  // namespace tilemega::backend::mx
