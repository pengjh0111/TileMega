// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Mx/DecodeShapes.h>

namespace tilemega::backend::mx {

enum class MxOpV2 { kStore, kResidual, kSwiGLU };

struct MxSegmentV2 {
  int tile, k0, k1;
  bool active;
};

template <int TN, int WN, MxPart Part>
struct MxGeometryV2 {
  int kc, tiles, grid, cta, rounds;
  int units, begin, end;

  TM_MX_HD constexpr MxGeometryV2(int n, int k, int g, int c)
      : kc(k / 32), tiles(n / TN), grid(g), cta(c), rounds(0),
        units(tiles * kc), begin(0), end(0) {
    if constexpr (Part == MxPart::kTileRange) {
      begin = c * tiles / g;
      end = (c + 1) * tiles / g;
      rounds = int((end - begin + WN - 1) / WN);
    } else {
      static_assert(WN == 1, "stream-K uses one wave group");
      begin = c * units / g;
      end = (c + 1) * units / g;
      rounds = begin < end ? int((end - 1) / kc - begin / kc + 1) : 0;
    }
  }

  TM_MX_HD constexpr MxSegmentV2 Segment(int round, int group) const {
    if constexpr (Part == MxPart::kTileRange) {
      int const tile = int(begin) + round * WN + group;
      return {tile, 0, kc, tile < end};
    } else {
      int const tile = int(begin / kc) + round;
      int const first = tile * kc;
      int const k0 = int(begin > first ? begin - first : 0);
      int const k1 = int(end - first < kc ? end - first : kc);
      return {tile, k0, k1, round < rounds && k0 < k1};
    }
  }

  TM_MX_HD constexpr int Owner(int unit) const {
    if constexpr (Part == MxPart::kTileRange) return int(((unit / kc + 1) * grid - 1) / tiles);
    else return int(((unit + 1) * grid - 1) / units);
  }
  TM_MX_HD constexpr int FirstOwner(int tile) const { return Owner(tile * kc); }
  TM_MX_HD constexpr int Producers(int tile) const {
    return Owner((tile + 1) * kc - 1) - FirstOwner(tile) + 1;
  }
};

TM_MX_HD constexpr int MxWaveChunksV2(MxSegmentV2 segment, int wave_k, int wk) {
  int const length = segment.active ? segment.k1 - segment.k0 : 0;
  return length > wave_k ? (length - wave_k + wk - 1) / wk : 0;
}

template <int TN, int WN, MxPart Part>
TM_MX_HD constexpr int MxFirstChunksV2(int n, int k, int grid) {
  int least = k / 32;
  for (int c = 0; c < grid; ++c) {
    MxGeometryV2<TN, WN, Part> const geometry(n, k, grid, c);
    for (int wave = 0; wave < 8; ++wave) {
      auto const segment = geometry.Segment(0, wave / (8 / WN));
      int const count = MxWaveChunksV2(segment, wave % (8 / WN), 8 / WN);
      least = count < least ? count : least;
    }
  }
  return least;
}

template <int TN, int WN, int D, MxPart Part>
TM_MX_HD constexpr int MxPrefetchWidthV2(int n, int k, int grid, int threads, int next_registers) {
  int const cap = threads == 576 ? 220 : 248;
  int const available = cap - next_registers < 64 ? cap - next_registers : 64;
  int const limit = available / (TN / 4);
  int const first = MxFirstChunksV2<TN, WN, Part>(n, k, grid);
  int const bounded = first < limit ? first : limit;
  return D > bounded ? D : bounded;
}

TM_MX_HD constexpr bool MxGemmLegalV2(int n, int k, int b, int tn, int wn, int depth,
                                     MxOpV2 op, MxPart part, int grid = 104) {
  if (!(tn == 16 || tn == 32 || tn == 64 || tn == 128) || n <= 0 || n % tn || k <= 0 || k % 32) return false;
  if (!(wn == 1 || wn == 2 || wn == 4 || wn == 8) || wn > (n / tn + grid - 1) / grid) return false;
  if (b < 1 || b > 16 || depth < 1 || depth > 4 || !MxDepthFits(tn, depth)) return false;
  if ((depth - 1) * (tn / 16 + 1) > 63) return false;
  if (op == MxOpV2::kSwiGLU && tn < 32) return false;
  if (wn < 8 && !MxReduceRounds(b, tn)) return false;
  if (part == MxPart::kStreamK && (wn != 1 || static_cast<long long>(n / tn) * (k / 32) < grid)) return false;
  return true;
}

// V3's grouped work is bounded by captured-independent maximum activity.
// Small unit counts do not restrict the number of wave groups in a CTA.
TM_MX_HD constexpr bool MxGemmLegalV3(int n, int k, int b, int tn, int wn, int depth,
                                     MxOpV2 op, MxPart part, bool grouped, int grid = 104) {
  if (!(tn == 16 || tn == 32 || tn == 64 || tn == 128) || n <= 0 || n % tn || k <= 0 || k % 32) return false;
  if (!(wn == 1 || wn == 2 || wn == 4 || wn == 8)) return false;
  if (!grouped && wn > (n / tn + grid - 1) / grid) return false;
  if (b < 1 || b > 16 || depth < 1 || depth > 4 || !MxDepthFits(tn, depth, 192)) return false;
  if ((depth - 1) * (tn / 16 + 1) > 63) return false;
  if (op == MxOpV2::kSwiGLU && tn < 32) return false;
  if (wn < 8 && !MxReduceRounds(b, tn)) return false;
  if (grouped && part != MxPart::kTileRange) return false;
  if (part == MxPart::kStreamK && (wn != 1 || static_cast<long long>(n / tn) * (k / 32) < grid)) return false;
  return true;
}

}  // namespace tilemega::backend::mx
