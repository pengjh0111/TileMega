// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if defined(__CUDACC__)
#define TILEMEGA_ATTENTION_HD __host__ __device__
#else
#define TILEMEGA_ATTENTION_HD
#endif

namespace tilemega::codegen {
enum class AttentionPagePolicy { kWarpPrivate, kPacked };

// Positions are relative to the cache chunk. Loader and consumer share this
// mapping; padding rows have no cache owner and must never be read as keys.
template<int D,int PageBytes,AttentionPagePolicy Policy>
struct AttentionPageLayout {
  static_assert(D==64 || D==128);
  static_assert(PageBytes==8192 || PageBytes==16384);
  static constexpr int page_rows=PageBytes/(4*D);
  int count;
  TILEMEGA_ATTENTION_HD constexpr int extent() const {return ((count+63)/64)*16;}
  TILEMEGA_ATTENTION_HD constexpr int owners_per_page() const {
    if(Policy==AttentionPagePolicy::kWarpPrivate || !count || extent()>=page_rows)return 1;
    int owners=page_rows/extent();return owners<4?owners:4;
  }
  TILEMEGA_ATTENTION_HD constexpr int waves() const {return (extent()+page_rows-1)/page_rows;}
  TILEMEGA_ATTENTION_HD constexpr int pages_per_wave() const {return 4/owners_per_page();}
  TILEMEGA_ATTENTION_HD constexpr int pages_per_task() const {return waves()*pages_per_wave();}
  TILEMEGA_ATTENTION_HD constexpr int PageOf(int owner,int wave) const {
    return wave*pages_per_wave()+owner/owners_per_page();
  }
  TILEMEGA_ATTENTION_HD constexpr int RowOffset(int owner) const {
    return (owner%owners_per_page())*extent();
  }
  TILEMEGA_ATTENTION_HD constexpr int Position(int owner,int wave,int row) const {
    return owner*extent()+wave*page_rows+row;
  }
  TILEMEGA_ATTENTION_HD constexpr bool Valid(int owner,int wave,int row) const {
    return wave*page_rows+row<extent() && Position(owner,wave,row)<count;
  }
  TILEMEGA_ATTENTION_HD constexpr int OwnerOf(int page,int row) const {
    return (page%pages_per_wave())*owners_per_page()+
        (owners_per_page()>1?row/extent():0);
  }
  TILEMEGA_ATTENTION_HD constexpr int LocalRow(int row) const {
    return owners_per_page()>1?row%extent():row;
  }
  TILEMEGA_ATTENTION_HD constexpr int RowsPerOwner() const {
    return owners_per_page()>1?extent():page_rows;
  }
  TILEMEGA_ATTENTION_HD constexpr int ReleaseArrivals(int) const {return 32*owners_per_page();}
  // PageRing's empty barrier has 128 arrivals. Each participating lane
  // supplies this many arrivals, derived from the actual set of consumers.
  TILEMEGA_ATTENTION_HD constexpr int ReleaseIterations(int page) const {
    return 128/ReleaseArrivals(page);
  }
  TILEMEGA_ATTENTION_HD constexpr bool needs_cta_sync() const {return owners_per_page()>1;}
  TILEMEGA_ATTENTION_HD constexpr int pages_in_flight() const {return 2*pages_per_wave();}
};
} // namespace tilemega::codegen
#undef TILEMEGA_ATTENTION_HD
