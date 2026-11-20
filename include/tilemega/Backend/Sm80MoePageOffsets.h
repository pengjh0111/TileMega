// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if defined(__CUDACC__)
#define TILEMEGA_MOE_PAGE_HD __host__ __device__
#else
#define TILEMEGA_MOE_PAGE_HD
#endif

namespace tilemega::backend {

// A whole page contains 64 output rows and one 64-column K panel.
template<int TileN,int Hidden,int TileK>
struct Sm80MoePageOffsets {
  static_assert(TileN%64==0 && Hidden%64==0 && TileK%64==0);
  static constexpr int kElements=64*64,kPages=TileN/64,kPanels=TileK/64;

  TILEMEGA_MOE_PAGE_HD static constexpr int Task(int output_tile) {
    return output_tile*kPages*(Hidden/64)*kElements;
  }

  TILEMEGA_MOE_PAGE_HD static constexpr int Source(int k_tile,int panel,int page) {
    return (page*(Hidden/64)+k_tile*kPanels+panel)*kElements;
  }

  TILEMEGA_MOE_PAGE_HD static constexpr int Destination(int panel,int page) {
    return (panel*kPages+page)*kElements;
  }
};

}  // namespace tilemega::backend
#undef TILEMEGA_MOE_PAGE_HD
