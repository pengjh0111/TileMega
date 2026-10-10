#pragma once
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Codegen/tasks/DmAttentionPageLayout.h>
#else
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if defined(__CUDACC__)
#define TILEMEGA_LAYOUT_HD __host__ __device__
#else
#define TILEMEGA_LAYOUT_HD
#endif
namespace tilemega::codegen {
enum class AttentionPagePolicy { WarpPrivate, Packed };

// One description owns both transport and consumer coordinates. In particular,
// a warp-private producer must never be paired with a packed consumer.
struct AttentionPageLayout {
  using Policy = AttentionPagePolicy;
  int page_rows, extent, owners_per_page, pages_per_wave, waves;
  int pages_per_task, pages_in_flight, begin;
  Policy policy;
  bool needs_cta_sync;

  TILEMEGA_LAYOUT_HD constexpr AttentionPageLayout(int rows, int count,
      Policy selected, int first=0)
      : page_rows(rows), extent(((count+63)/64)*16), owners_per_page(1),
        pages_per_wave(4), waves(0), pages_per_task(0), pages_in_flight(0),
        begin(first), policy(selected), needs_cta_sync(false) {
    if(count<=0 || rows<=0)return;
    if(selected==Policy::Packed && extent<rows) {
      while(owners_per_page<4 && 2*owners_per_page*extent<=rows)
        owners_per_page*=2;
    }
    pages_per_wave=4/owners_per_page;
    waves=(extent+rows-1)/rows;
    pages_per_task=waves*pages_per_wave;
    // A wave is the live frontier, not the sum over sequential waves.
    pages_in_flight=pages_per_wave;
    needs_cta_sync=owners_per_page>1;
  }
  TILEMEGA_LAYOUT_HD constexpr int PageOf(int owner,int wave) const {
    return wave*pages_per_wave+owner/owners_per_page;
  }
  TILEMEGA_LAYOUT_HD constexpr int RowOffset(int owner) const {
    return (owner%owners_per_page)*extent;
  }
  TILEMEGA_LAYOUT_HD constexpr int Position(int owner,int wave,int row) const {
    return begin+owner*extent+wave*page_rows+row;
  }
  TILEMEGA_LAYOUT_HD constexpr int RowsPerOwner() const {
    return owners_per_page>1?extent:page_rows;
  }
  TILEMEGA_LAYOUT_HD constexpr bool Valid(int owner,int wave,int row,
      int end,int past) const {
    return owner>=0 && owner<4 && row>=0 && row<RowsPerOwner() &&
        wave*page_rows+row<extent && Position(owner,wave,row)<end &&
        Position(owner,wave,row)<past;
  }
  TILEMEGA_LAYOUT_HD constexpr int OwnerOf(int page,int row) const {
    return (page%pages_per_wave)*owners_per_page+
        (owners_per_page>1?row/extent:0);
  }
  TILEMEGA_LAYOUT_HD constexpr int LocalRow(int row) const {
    return owners_per_page>1?row%extent:row;
  }
  // Each of 32 lanes in every owning warp contributes this many arrivals;
  // exactly 128 arrivals release a page, independent of packing.
  TILEMEGA_LAYOUT_HD constexpr int ReleaseArrivals(int page) const {
    return page>=0 && page<pages_per_task?4/owners_per_page:0;
  }
  TILEMEGA_LAYOUT_HD constexpr bool Full(int page,int end,int past) const {
    int wave=page/pages_per_wave;
    if(owners_per_page>1 && owners_per_page*extent!=page_rows)return false;
    // Positions are contiguous within a full page; checking the two ends
    // avoids a per-row predicate loop in the loader.
    return Valid(OwnerOf(page,0),wave,0,end,past) &&
        Valid(OwnerOf(page,page_rows-1),wave,LocalRow(page_rows-1),end,past);
  }
};
} // namespace tilemega::codegen
#undef TILEMEGA_LAYOUT_HD

#endif
