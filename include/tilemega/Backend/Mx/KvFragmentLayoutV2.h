// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeShapes.h>

namespace tilemega::backend::mx {

template <int D>
struct MxKvFragmentLayoutV2 {
  static_assert(D == 64 || D == 128);
  static constexpr unsigned kBlockElements = 32 * D;

  TM_MX_HD static constexpr unsigned K(unsigned head, unsigned token, unsigned dim, unsigned capacity) {
    unsigned const t = token % 32, dc = dim / 32, a = t % 8 / 4;
    unsigned const r = 4 * (t / 8) + t % 4, g = dim % 32 / 8;
    return ((head * (capacity / 32) + token / 32) * (D / 32) * 2 + dc * 2 + a) * 512
           + (r + 16 * g) * 8 + dim % 8;
  }
  TM_MX_HD static constexpr unsigned V(unsigned head, unsigned token, unsigned dim, unsigned capacity) {
    unsigned const lane = dim % 16 + 16 * (token % 32 / 8);
    return ((head * (capacity / 32) + token / 32) * (D / 16) + dim / 16) * 512
           + lane * 8 + token % 8;
  }
  TM_MX_HD static constexpr unsigned KWave(unsigned head, unsigned block, unsigned dc,
                                          unsigned atom, unsigned lane, unsigned capacity) {
    return ((head * (capacity / 32) + block) * (D / 32) * 2 + dc * 2 + atom) * 512 + lane * 8;
  }
  TM_MX_HD static constexpr unsigned VWave(unsigned head, unsigned block, unsigned atom,
                                          unsigned lane, unsigned capacity) {
    return ((head * (capacity / 32) + block) * (D / 16) + atom) * 512 + lane * 8;
  }
};

}  // namespace tilemega::backend::mx
