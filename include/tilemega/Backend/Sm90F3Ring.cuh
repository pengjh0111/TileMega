// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <tilemega/Backend/Sm90F3Async.cuh>
#endif

// The F3 forward skeleton's shared-memory budget and its per-arena ring
// (DNN3 prompt §5.2 and §5.3). The byte arithmetic is plain constexpr so the
// solver prices a stage with the very function the body lays itself out with
// (§5.2 item 4); only the barrier protocol below needs the device compiler.
namespace tilemega::backend {

// H800: 232448 B of dynamic shared memory at residency 1. The first 2 KB is
// the global header (both rings and the gate flags), and the rest splits into
// two equal 1 KB-aligned arenas, one per stage parity.
struct Sm90F3Budget {
  static constexpr int kBudget=232448;
  static constexpr int kHeader=2048;
  static constexpr int kArena=(kBudget-kHeader)/2048*1024;
  static_assert(kArena==114688,"the H800 arena is 112 KiB");
  static constexpr int kArena0=kHeader;
  static constexpr int kArena1=kHeader+kArena;
  // A stage whose storage exceeds one arena spans both (§5.2 item 3).
  static constexpr int kSpanBudget=kBudget-kHeader;
  static constexpr int Offset(unsigned stage,int bytes) {
    return bytes>kArena?kHeader:(stage%2?kArena1:kArena0);
  }
  static constexpr bool Spanning(int bytes) {return bytes>kArena;}
  static constexpr bool Fits(int bytes) {return bytes>0 && bytes<=kSpanBudget;}
};

// One slot of a normal or gather GEMM stage: the A panel and the B page, both
// SW128 K-major over a 64-wide k block (§6.2 item 2).
constexpr int Sm90F3SlotBytes(int tile_m,int tile_n) {
  return 128*(tile_m+tile_n);
}
// swap-AB: the weight page is 128 rows and the activation panel is Mp rows
// (§6.7 item 3).
constexpr int Sm90F3SwapSlotBytes(int rows) {
  return 2*64*(128+rows);
}
// The residual tile the producer prefetches: ceil(TN/64) boxes of TM rows,
// each row a 128 B SW128 line (§6.4 step 1).
constexpr int Sm90F3ResidualBytes(int tile_m,int tile_n) {
  return (tile_n+63)/64*tile_m*128;
}
// Per-column vectors are FP32 over the tile's N, and the row statistics are
// two FP32 per live row.
constexpr int Sm90F3VectorBytes(int tile_n,int vectors) {return 4*tile_n*vectors;}
constexpr int Sm90F3RowStatBytes(int tile_m) {return 8*tile_m;}

// A GAP producer's reduction area: four warps per warpgroup, each holding its
// own partial for every column of its half tile and every 64-row segment
// (§6.4 step 4 item 3).
constexpr int Sm90F3PoolScratchBytes(int columns,int segments) {
  return 4*columns*segments*4;
}

// Everything a GEMM stage holds in its arena. `vectors` counts the per-column
// FP32 operands the chain names, `row_stats` whether it reads them, and
// `pool_segments` is nonzero only for a GAP producer.
struct Sm90F3GemmLayout {
  int slots=0,slot_bytes=0,side_bytes=0,pool_bytes=0,end=0;
  int slot_base=0,side_base=0,pool_base=0;
  constexpr bool Spanning() const {return Sm90F3Budget::Spanning(end);}
};
constexpr Sm90F3GemmLayout Sm90F3MakeGemmLayout(int tile_m,int tile_n,int slots,
    bool swap_ab,int swap_rows,bool residual,int vectors,bool row_stats,
    int pool_columns,int pool_segments) {
  Sm90F3GemmLayout l;
  l.slots=slots;
  l.slot_bytes=swap_ab?Sm90F3SwapSlotBytes(swap_rows):Sm90F3SlotBytes(tile_m,tile_n);
  // The side area holds, in order, the residual tile, the per-column vectors
  // and the row statistics; the producer fills it under one barrier.
  l.side_bytes=(residual?Sm90F3ResidualBytes(tile_m,tile_n):0)+
      Sm90F3VectorBytes(tile_n,vectors)+(row_stats?Sm90F3RowStatBytes(tile_m):0);
  l.pool_bytes=pool_segments?Sm90F3PoolScratchBytes(pool_columns,pool_segments):0;
  l.slot_base=0;
  l.side_base=l.slot_base+slots*l.slot_bytes;
  // Both the side area and the pool scratch start 1 KB aligned, so an SW128
  // descriptor can address the residual tile from its own base.
  l.side_base=(l.side_base+1023)/1024*1024;
  l.pool_base=(l.side_base+l.side_bytes+1023)/1024*1024;
  l.end=(l.pool_base+l.pool_bytes+1023)/1024*1024;
  return l;
}

#if defined(__CUDACC__)

// One arena's ring (§5.3). The counts are the protocol: `full` is completed by
// the producer leader's own `arrive.expect_tx`; `gfull` adds one arrival per
// gathering lane; `empty` and `side_empty` take one arrival from each of the
// eight consumer warps.
struct alignas(16) Sm90F3Ring {
  static constexpr int kSlots=8;
  static constexpr int kConsumerWarps=8;
  static constexpr int kGatherLanes=32;
  std::uint64_t full[kSlots],gfull[kSlots],empty[kSlots];
  std::uint64_t side_full,side_empty;

  using Async=Sm90F3Async::Base;
  // Thread zero alone initializes both arenas, then one fence publishes every
  // barrier before any arrival (§5.3 item 1).
  __device__ void Initialize() {
    #pragma unroll
    for(int i=0;i<kSlots;++i) {
      Async::Init(full+i,1);
      Async::Init(gfull+i,kGatherLanes+1);
      Async::Init(empty+i,kConsumerWarps);
    }
    Async::Init(&side_full,1);
    Async::Init(&side_empty,kConsumerWarps);
  }
  __device__ void Invalidate() {
    #pragma unroll
    for(int i=0;i<kSlots;++i) {
      Async::Invalidate(full+i);Async::Invalidate(gfull+i);Async::Invalidate(empty+i);
    }
    Async::Invalidate(&side_full);Async::Invalidate(&side_empty);
  }
  // Every consumer warp's lane 0 arrives once on each empty barrier, so the
  // producer's first wait on phase 0 passes.
  __device__ void Prime() {
    #pragma unroll
    for(int i=0;i<kSlots;++i)Async::Arrive(empty+i);
    Async::Arrive(&side_empty);
  }
};

// The phase bits each role carries in registers, one per barrier, continued
// across tasks and across the stages that share an arena (§5.3 item 2).
struct Sm90F3Phase {
  unsigned full=0,gather=0,empty=0;
  // The side barriers are single, so one bit each.
  unsigned side_full=0,side_empty=0;
  __device__ __forceinline__ unsigned Bit(unsigned mask,int slot) const {
    return (mask>>slot)&1u;
  }
};

// The global header: both rings, then the gate's broadcast flags.
struct alignas(16) Sm90F3Header {
  Sm90F3Ring rings[2];
  // The split-K last-arriver decision and the GAP segment count, broadcast
  // from thread 0 of the arriving warpgroup to all 256 consumers.
  int last_arriver;
  int reserved[3];
};
static_assert(sizeof(Sm90F3Header)<=Sm90F3Budget::kHeader,
    "both F3 rings and the gate flags live in the 2 KB header");

#endif  // __CUDACC__

}  // namespace tilemega::backend
