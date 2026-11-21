// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>

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

template<int TileN,int Hidden,int TileK,int Phase>
struct Sm80MoePageSegments {
  static_assert(TileN>0 && TileN%32==0 && Hidden%64==0 && TileK%64==0);
  static_assert(Phase==0 || (Phase==32 && TileN%64==32));
  static constexpr int kSegments=(Phase+TileN+63)/64;
  static constexpr int kPanels=TileK/64;

  TILEMEGA_MOE_PAGE_HD static constexpr std::size_t Task(int output_tile) {
    return (std::size_t(output_tile)*TileN/64)*(Hidden/64)*4096;
  }

  template<int Segment>
  struct Span {
    static_assert(Segment>=0 && Segment<kSegments);
    static constexpr int kLocal=Segment==0?0:Segment*64-Phase;
    static constexpr int kInside=Segment==0?Phase:0;
    static constexpr int kRemaining=TileN-kLocal;
    static constexpr int kRows=kRemaining<64-kInside?kRemaining:64-kInside;
    static constexpr int kElements=kRows*64;

    TILEMEGA_MOE_PAGE_HD static constexpr int Source(int k_tile,int panel) {
      return (Segment*(Hidden/64)+k_tile*kPanels+panel)*4096+kInside*64;
    }

    TILEMEGA_MOE_PAGE_HD static constexpr int Destination(int panel) {
      return panel*TileN*64+kLocal*64;
    }
  };
};

}  // namespace tilemega::backend
#undef TILEMEGA_MOE_PAGE_HD
