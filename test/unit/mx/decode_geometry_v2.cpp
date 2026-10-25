// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <tilemega/Backend/Mx/KvFragmentLayoutV2.h>

#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <vector>

using namespace tilemega::backend::mx;

namespace {

template <int TN, int WN, MxPart Part>
void Check(int n, int k) {
  constexpr int grid = 104, wk = 8 / WN;
  if (!MxGemmLegalV2(n, k, 1, TN, WN, 1, MxOpV2::kStore, Part, grid)) return;
  int const kc = k / 32, tiles = n / TN, units = tiles * kc;
  std::vector<int> owners(units, -1);
  for (int c = 0; c < grid; ++c) {
    MxGeometryV2<TN, WN, Part> const geometry(n, k, grid, c);
    for (int round = 0; round < geometry.rounds; ++round) {
      for (int wave = 0; wave < 8; ++wave) {
        auto const segment = geometry.Segment(round, wave / wk);
        int count = 0;
        if (segment.active) {
          assert(segment.tile >= 0 && segment.tile < tiles);
          for (int chunk = segment.k0 + wave % wk; chunk < segment.k1; chunk += wk) {
            int const index = segment.tile * kc + chunk;
            assert(index >= 0 && index < units && owners[index] == -1);
            owners[index] = c;
            ++count;
          }
        }
        assert(count == MxWaveChunksV2(segment, wave % wk, wk));
      }
    }
  }
  MxGeometryV2<TN, WN, Part> const geometry(n, k, grid, 0);
  for (int u = 0; u < units; ++u) {
    assert(owners[u] >= 0);
    assert(geometry.Owner(u) == owners[u]);
  }
  for (int t = 0; t < tiles; ++t) {
    auto const first = owners.begin() + t * kc;
    auto const last = first + kc;
    int const lo = *std::min_element(first, last), hi = *std::max_element(first, last);
    assert(geometry.FirstOwner(t) == lo && geometry.Producers(t) == hi - lo + 1);
  }
}

template <int TN>
void Cases() {
  for (int n : {1024, 2048, 4096, 65536})
    for (int k : {128, 1024, 8192}) {
      Check<TN, 1, MxPart::kTileRange>(n, k);
      Check<TN, 2, MxPart::kTileRange>(n, k);
      Check<TN, 4, MxPart::kTileRange>(n, k);
      Check<TN, 8, MxPart::kTileRange>(n, k);
      Check<TN, 1, MxPart::kStreamK>(n, k);
    }
}

template <int D>
void CacheLayout() {
  using Layout = MxKvFragmentLayoutV2<D>;
  for (unsigned capacity : {32u, 64u, 1088u}) {
    std::vector<bool> keys(2 * capacity * D), values(keys.size());
    for (unsigned head = 0; head < 2; ++head)
      for (unsigned block = 0; block < capacity / 32; ++block)
        for (unsigned lane = 0; lane < 64; ++lane) {
          unsigned const r = lane & 15, g = lane >> 4;
          for (unsigned dc = 0; dc < D / 32; ++dc)
            for (unsigned atom = 0; atom < 2; ++atom)
              for (unsigned e = 0; e < 8; ++e) {
                unsigned const token = block * 32 + 8 * (r / 4) + atom * 4 + r % 4;
                unsigned const dim = dc * 32 + g * 8 + e;
                unsigned const index = Layout::K(head, token, dim, capacity);
                assert(index < keys.size() && !keys[index]);
                assert(index == Layout::KWave(head, block, dc, atom, lane, capacity) + e);
                keys[index] = true;
              }
          for (unsigned atom = 0; atom < D / 16; ++atom)
            for (unsigned e = 0; e < 8; ++e) {
              unsigned const index = Layout::V(head, block * 32 + g * 8 + e, atom * 16 + r, capacity);
              assert(index < values.size() && !values[index]);
              assert(index == Layout::VWave(head, block, atom, lane, capacity) + e);
              values[index] = true;
            }
        }
    assert(std::all_of(keys.begin(), keys.end(), [](bool v) { return v; }));
    assert(std::all_of(values.begin(), values.end(), [](bool v) { return v; }));
  }
}

// R_next=156 is a permitted resource example, not a measured v2 MTreg value.
constexpr int kOPw = MxPrefetchWidthV2<16, 1, 1, MxPart::kTileRange>(2048, 2048, 104, 576, 156);
constexpr int kDownPw = MxPrefetchWidthV2<16, 1, 1, MxPart::kTileRange>(2048, 8192, 104, 576, 156);
static_assert(kOPw == 8 && kDownPw == 16);
static_assert(MxGemmLegalV2(2048, 2048, 1, 16, 1, 1, MxOpV2::kResidual, MxPart::kTileRange));
static_assert(MxGemmLegalV2(2048, 8192, 1, 16, 1, 1, MxOpV2::kResidual, MxPart::kTileRange));

}  // namespace

int main() {
  Cases<16>(); Cases<32>(); Cases<64>(); Cases<128>();
  CacheLayout<64>(); CacheLayout<128>();
  std::puts("PASS v2 geometry coverage, ownership and contributor counts");
  std::printf("CONTRACT same six-coordinate body: O Pw=%d, down Pw=%d (conditional R_next=156)\n",
               kOPw, kDownPw);
}
