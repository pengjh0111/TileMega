// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Mx/KvFragmentLayoutV2.h>
#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>

namespace tilemega::codegen::mx {

// One query tile owns each cache block. The staging arrays are reused before
// attention starts. MX-5's 64-token prompt produces two complete cache blocks.
template <int D, int Q, int Tokens, int QRows, int KvTile, bool QkNorm>
struct PrefillKvFragmentsV2 {
  static_assert(Tokens == 64);
  using Layout = backend::mx::MxKvFragmentLayoutV2<D>;
  using Element = tilemega::Bf16;

  template <class Storage>
  __device__ static void Run(ServingAttentionOperands const& p, Storage& s, int b, int g, int qb, int c) {
    int const head = b * p.heads_kv + g;
    if (p.past != 0 || p.capacity < Tokens || p.capacity % 32 || p.block_extent <= 0 || p.block_extent % 32)
      maca::Trap();
    int const qbegin = qb * QRows;
    int const warp = ComputeThread() / 32, lane = ComputeThread() % 32;
    for (int block = 0; block < Tokens / 32; ++block) {
      int const first_new = block * 32;
      int const query = first_new * Q;
      if (first_new / p.block_extent != c || query < qbegin || query >= qbegin + QRows) continue;

      for (int row = warp; row < 32; row += kComputeThreads / 32) {
        int const position = block * 32 + row;
        int const token = position;
        auto const* x = p.qkv + ((std::size_t(b) * Tokens + token) * p.heads_kv + g) * ((Q + 2) * D) + Q * D;
        RotateRow<D, QkNorm>(x, p.k_norm, p.cosine + std::size_t(position) * D,
                             p.sine + std::size_t(position) * D, p.epsilon,
                             [&](int d, uint4 value) { *reinterpret_cast<uint4*>(s.key + row * D + d) = value; });
        if (lane * 8 < D) {
          uint4 const value = backend::LoadGlobal16(x + D + lane * 8);
          auto const* elements = reinterpret_cast<Element const*>(&value);
#pragma unroll
          for (int i = 0; i < 8; ++i) s.value_t[(lane * 8 + i) * KvTile + row] = elements[i];
        }
      }
      ComputeSync();
      for (int item = ComputeThread(); item < D / 32 * 2 * 64; item += kComputeThreads) {
        int const l = item % 64, atom = item / 64 % 2, dc = item / 128;
        int const r = l % 16, row = 8 * (r / 4) + 4 * atom + r % 4, d = 32 * dc + 8 * (l / 16);
        backend::StoreGlobal16(p.key_cache + Layout::KWave(head, block, dc, atom, l, p.capacity),
                               *reinterpret_cast<uint4 const*>(s.key + row * D + d));
      }
      for (int item = ComputeThread(); item < D / 16 * 64; item += kComputeThreads) {
        int const l = item % 64, atom = item / 64, d = atom * 16 + l % 16, row = 8 * (l / 16);
        backend::StoreGlobal16(p.value_cache + Layout::VWave(head, block, atom, l, p.capacity),
                               *reinterpret_cast<uint4 const*>(s.value_t + d * KvTile + row));
      }
      ComputeSync();
    }
  }
};

}  // namespace tilemega::codegen::mx
