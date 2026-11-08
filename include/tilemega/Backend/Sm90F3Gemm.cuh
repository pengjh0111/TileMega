// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Epilogue.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <tilemega/Backend/Sm90Gmma.h>
#include <cute/tensor.hpp>
#include <type_traits>

// The F3 forward GEMM/convolution body (DNN3 prompt §6). Two consumer
// warpgroups share one CTA tile, a single producer warp feeds an eight-deep
// ring, and the epilogue runs in registers over the producer's prefetched
// side data. The main loop keeps one WGMMA batch in flight (`wait<1>`), which
// is CUTLASS's K_PIPE_MMAS=1 (§6.3, appendix B).
namespace tilemega::backend {

enum class Sm90F3Mode {kNormal, kGather, kSwapAB, kTmaStem};

struct Sm90F3GemmOperands {
  CUtensorMap const* activation;     // 2D {K, M} dense, or 4D im2col NHWC
  CUtensorMap const* residual;       // 2D {N, M} of the residual buffer
  cutlass::bfloat16_t const* weight;
  cutlass::bfloat16_t const* input;  // gather mode reads the image directly
  Sm90F3EpilogueOperands epilogue;
  CUtensorMap const* output_map=nullptr;
};

// A stage's convolution geometry, as compile-time constants exactly as the
// legacy body takes them.
struct Sm90F3NoConv {
  static constexpr int kQ=1,kP=1,kC=64,kS=1,kR=1,kStrideW=1,kStrideH=1,
      kPadW=0,kPadH=0,kDilationW=1,kDilationH=1,kW=1,kH=1;
};

// How the two consumer warpgroups divide one CTA tile (§6.1). A 64-row tile
// splits by N so both warpgroups keep a full-height accumulator; taller tiles
// split by M, and a 256-row tile gives each warpgroup two m64 operations.
template<int TM,int TN>
struct Sm90F3Split {
  static constexpr int kSubTiles=TM==256?2:1;
  static constexpr bool kSplitN=TM==64;
  static constexpr int kWgRows=kSplitN?64:64*kSubTiles;
  static constexpr int kWgCols=kSplitN?TN/2:TN;
  static_assert(TM==64 || TM==128 || TM==256,"an F3 CTA tile is 64, 128 or 256 rows");
  static_assert(TN==32 || TN==64 || TN==96 || TN==128 || TN==192 || TN==256);
  static_assert(!kSplitN || TN>=32,"a 64-row tile splits its columns between the warpgroups");
  // Each warpgroup's accumulator stays inside 128 FP32 registers per thread.
  static_assert(kWgRows*kWgCols/128<=128,"an F3 warpgroup accumulator exceeds 128 registers");
  // Byte offset of this warpgroup's own operand base inside the slot.
  __device__ static __forceinline__ int ARows(int wg) {return kSplitN?0:wg*kWgRows;}
  __device__ static __forceinline__ int BRows(int wg) {return kSplitN?wg*kWgCols:0;}
};

// A GAP producer's extra epilogue duty (DNN3 prompt §4 F-7, §6.4 step 4): the
// per-channel partial sums the later global-average stage reduces. The slot is
// (image, 64-row block), so a 64-row sub-tile contributes to exactly one block
// and each (image, block, column) has one writer — the sub-tile that owns
// those rows. `Area` is the producer's pixels per image.
template<int Area,int Parts,int Channels>
struct Sm90F3Pool {
  static constexpr int kArea=Area,kParts=Parts,kChannels=Channels;
  static_assert(Area>0 && Parts>0 && Channels>0);
  // How many images a 64-row block can touch, including its leading overlap.
  static constexpr int kImages=Area>=64?2:(64+Area-1)/Area+1;
  static_assert(Parts>=(Area+63)/64+1,"F-7 reserves ceil(area/64) + 1 slots per image");
};
struct Sm90F3NoPool {
  static constexpr int kArea=0,kParts=0,kChannels=0,kImages=0;
};

template<int TM,int TN,int Slots,int SplitK,Sm90F3Mode Mode,int M,int N,int K,
         int NStore,class Chain,class Conv=Sm90F3NoConv,class Pool=Sm90F3NoPool,int TK=64,
         bool TmaStore=false>
struct Sm90F3Gemm {
  using Element=cutlass::bfloat16_t;
  using Async=Sm90F3Async;
  using Base=Sm90F3Async::Base;
  using Split=Sm90F3Split<TM,TN>;
  using Side=typename Chain::Layout;

  static constexpr int kThreads=384;
  static constexpr bool kCanPreissue=Mode==Sm90F3Mode::kNormal;
  static constexpr int kSlots=Slots;
  static constexpr int kTK=TK,kPanels=TK/64;
  static constexpr bool kTmaStore=TmaStore;
  static_assert(!TmaStore || (TM>=128 && TN%64==0 && Mode!=Sm90F3Mode::kSwapAB));
  static constexpr int kIterations=K/(SplitK*TK);
  static_assert(TK==64 || TK==128);
  static_assert(Slots>=2 && Slots<=8);
  static_assert(K%(SplitK*TK)==0,"split-K divides the contraction into whole slots");
  static_assert(N%TN==0,"N is padded to a whole number of TN tiles when the plan is built");
  static_assert(NStore>=1 && NStore<=N);
  static constexpr int kTiles=N/TN,kMTiles=(M+TM-1)/TM;
  static constexpr int kOutputTiles=kMTiles*kTiles,kTasks=kOutputTiles*SplitK;
  static constexpr int kSlotBytes=Sm90F3SlotBytes(TM,TN,TK);
  static constexpr std::size_t kWeightBytes=std::size_t(N)*K*2;
  // Gather reads R*S*8 taps as whole 64-wide k blocks (§6.6 item 2).
  static constexpr int kGatherTaps=Conv::kR*Conv::kS;

  // The WGMMA one warpgroup issues, over its own slice of the tile.
  using Gmma=Sm90Gmma<arch::Sm90>;
  using Mma=decltype(cute::make_tiled_mma(typename Gmma::template SS<Split::kWgCols>{}));
  // Each operation is m64, so a 256-row tile gives a warpgroup two separate
  // 64-row accumulators rather than one 128-row one (§6.1, third row).
  using Fragment=decltype(cute::partition_fragment_C(Mma{},
      cute::Shape<cute::_64,cute::Int<Split::kWgCols>>{}));
  struct Accumulators {Fragment tile[Split::kSubTiles];};
  struct RowAccumulator {
    float sum[Split::kSubTiles][2]{},square[Split::kSubTiles][2]{};
  };

  __device__ static __forceinline__ int NTile(int task) {return task%kTiles;}
  __device__ static __forceinline__ int MTile(int task) {return (task/kTiles)%kMTiles;}
  __device__ static __forceinline__ int Slice(int task) {return task/kOutputTiles;}

  // A slot's A panel and B page.
  __device__ static __forceinline__ Element* SlotA(Element* arena,int slot) {
    return arena+std::size_t(slot)*(kSlotBytes/2);
  }
  __device__ static __forceinline__ Element* SlotB(Element* arena,int slot) {
    return SlotA(arena,slot)+std::size_t(TM)*TK;
  }

  // ---- producer ------------------------------------------------------------

  __device__ static __forceinline__ void Preissue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,Sm90F3Phase& phase) {
    if(int(threadIdx.x)==256 && int(blockIdx.x)<kTasks) {
      constexpr int prefix=Slots<kIterations?Slots:kIterations;
      int first=int(blockIdx.x),n_tile=NTile(first),slice=Slice(first);
      #pragma unroll
      for(int slot=0;slot<prefix;++slot) {
        Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
        phase.empty^=1u<<slot;
        Base::ExpectTx(ring.full+slot,kSlotBytes);
        Base::Bulk(SlotB(arena,slot),p.weight+
            (std::size_t(n_tile)*(K/64)+(slice*kIterations+slot)*kPanels)*TN*64,
            2*TN*TK,ring.full+slot);
      }
      Base::PrefetchTensorMap(p.activation);
      if(p.residual)Base::PrefetchTensorMap(p.residual);
    }
    __syncwarp();
  }

  __device__ static __forceinline__ void IssueAK64(Element* a,Sm90F3GemmOperands const& p,
      int m_tile,int k,std::uint64_t* barrier) {
    if constexpr(Mode==Sm90F3Mode::kTmaStem) {
      static_assert(Conv::kC==8);
      #pragma unroll
      for(int j=0;j<8;++j) {
        int tap=k/8+j;
        bool live=tap<kGatherTaps;
        #pragma unroll
        for(int base=0;base<TM;base+=64) {
          int row=m_tile*TM+base,image=row/(Conv::kP*Conv::kQ),pixel=row%(Conv::kP*Conv::kQ);
          int w=(pixel%Conv::kQ)*Conv::kStrideW-Conv::kPadW;
          int h=(pixel/Conv::kQ)*Conv::kStrideH-Conv::kPadH;
          // The padded final taps use an OOB C8 coordinate, yielding zeros.
          Base::TensorIm2col4D(a+j*TM*8+base*8,p.activation,live?0:8,w,h,image,
              std::uint16_t(live?(tap%Conv::kS)*Conv::kDilationW:0),
              std::uint16_t(live?(tap/Conv::kS)*Conv::kDilationH:0),barrier);
        }
      }
    }else if constexpr(Mode==Sm90F3Mode::kNormal) {
      if constexpr(std::is_same_v<Conv,Sm90F3NoConv>)
        Base::Tensor2D(a,p.activation,k,m_tile*TM,barrier);
      else {
        // One 64-pixel im2col request per 64 rows of the tile, all completing
        // the same mbarrier, exactly as the legacy body issues them.
        constexpr int kBlocks=Conv::kC/64;
        int panel=k/64,tap=panel/kBlocks,c=(panel%kBlocks)*64;
        #pragma unroll
        for(int base=0;base<TM;base+=64) {
          int row=m_tile*TM+base,image=row/(Conv::kP*Conv::kQ),pixel=row%(Conv::kP*Conv::kQ);
          int w=(pixel%Conv::kQ)*Conv::kStrideW-Conv::kPadW;
          int h=(pixel/Conv::kQ)*Conv::kStrideH-Conv::kPadH;
          Base::TensorIm2col4D(a+std::size_t(base)*64,p.activation,c,w,h,image,
              std::uint16_t((tap%Conv::kS)*Conv::kDilationW),
              std::uint16_t((tap/Conv::kS)*Conv::kDilationH),barrier);
        }
      }
    }
  }

  __device__ static __forceinline__ void IssueA(Element* a,Sm90F3GemmOperands const& p,
      int m_tile,int k,std::uint64_t* barrier) {
    #pragma unroll
    for(int panel=0;panel<kPanels;++panel)
      IssueAK64(a+std::size_t(panel)*TM*64,p,m_tile,k+panel*64,barrier);
  }

  // Dense A is one 2D box; an im2col A is TM/64 boxes. Either way the slot's
  // transaction count is the whole slot.
  __device__ static __forceinline__ void ProduceTaskK(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,Sm90F3Phase& phase,int task,int k_index,int slot,
      bool skip_b) {
    int n_tile=NTile(task),m_tile=MTile(task),slice=Slice(task);
    int k=(slice*kIterations+k_index)*TK;
    if(!skip_b) {
      Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
      phase.empty^=1u<<slot;
      Base::ExpectTx(ring.full+slot,kSlotBytes);
      Base::Bulk(SlotB(arena,slot),
          p.weight+(std::size_t(n_tile)*(K/64)+k/64)*TN*64,2*TN*TK,ring.full+slot);
    }
    IssueA(SlotA(arena,slot),p,m_tile,k,ring.full+slot);
  }

  // ---- consumer main loop (§6.3) ------------------------------------------

  template<class Accum>
  __device__ static __forceinline__ void MainLoop(Sm90F3Ring& ring,Element* arena,
      Sm90F3Phase& phase,Accum& accumulators,int wg,int first_slot,
      int iterations) {
    using namespace cute;
    Mma mma;
    auto thread=mma.get_slice(int(threadIdx.x)%128);
    auto& accum=accumulators.tile;
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s)cute::clear(accum[s]);
    int previous=-1;
    for(int k=0;k<iterations;++k) {
      int slot=(first_slot+k)%Slots;
      TM_F3_TRACE_BEGIN(kSlotWait);
      if constexpr(Mode==Sm90F3Mode::kGather) {
        Base::Wait(ring.gfull+slot,phase.Bit(phase.gather,slot));
        phase.gather^=1u<<slot;
        // cp.async writes through the generic proxy and WGMMA reads the async
        // proxy: this fence cannot be dropped (§6.6 item 4).
        Base::FenceProxyAsyncShared();
      }else {
        Base::Wait(ring.full+slot,phase.Bit(phase.full,slot));
        phase.full^=1u<<slot;
      }
      auto* a=SlotA(arena,slot);
      TM_F3_TRACE_END(kSlotWait);
      if(k==0)TM_F3_TRACE_FIRST_READY();
      auto* b=SlotB(arena,slot);
      #pragma unroll
      for(int s=0;s<Split::kSubTiles;++s) {
        cute::warpgroup_fence_operand(accum[s]);
        cute::warpgroup_arrive();
        #pragma unroll
        for(int kb=0;kb<kPanels;++kb) {
        using StemLayout=decltype(tile_to_shape(gmma::Layout_K_INTER_Atom<Element>{},
            Shape<Int<TM>,_64>{}));
        using ALayout=std::conditional_t<Mode==Sm90F3Mode::kTmaStem,StemLayout,Sm90KLayout<TM,64>>;
        auto full_a=make_tensor(make_smem_ptr(a+std::size_t(kb)*TM*64),ALayout{});
        auto panel=local_tile(full_a,Shape<_64,_64>{},
            make_coord((Split::ARows(wg)+s*64)/64,0));
        auto page=make_tensor(make_smem_ptr(
            b+std::size_t(kb)*TN*64+std::size_t(Split::BRows(wg))*64),Sm90KLayout<Split::kWgCols,64>{});
        auto ra=thread.partition_fragment_A(panel);
        auto rb=thread.partition_fragment_B(page);
        CUTE_UNROLL
        for(int step=0;step<size<2>(ra);++step)
          cute::gemm(mma,ra(cute::_,cute::_,step),rb(cute::_,cute::_,step),accum[s]);
        }
        cute::warpgroup_commit_batch();
      }
      // The batch just issued may stay in flight; the previous one is done,
      // so its slot goes back to the producer.
      cute::warpgroup_wait<Split::kSubTiles>();
      #pragma unroll
      for(int s=0;s<Split::kSubTiles;++s)cute::warpgroup_fence_operand(accum[s]);
      if(previous>=0)Release(ring,previous);
      previous=slot;
    }
    cute::warpgroup_wait<0>();
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s)cute::warpgroup_fence_operand(accum[s]);
    if(previous>=0)Release(ring,previous);
  }

  // One arrival per consumer warp, after the warp's own WGMMA reads retired.
  __device__ static __forceinline__ void Release(Sm90F3Ring& ring,int slot) {
    __syncwarp();
    if((tilemega::backend::Sm90F3Async::ThreadIndex()&31)==0)Base::Arrive(ring.empty+slot);
  }

  // ---- producer side prefetch (§6.4 step 1) -------------------------------

  static constexpr int kVectors=Side::kVectorBytes/(4*TN);
  static constexpr int kResidualBoxes=(TN+63)/64;
  // The stage's own arena layout, from the same function the solver prices it
  // with (§5.2 item 4). A GAP producer's reduction area is added by codegen
  // through the chain's pool segments.
  static constexpr Sm90F3GemmLayout kLayout=Sm90F3MakeGemmLayout(TM,TN,Slots,
      false,0,Chain::kNeedsResidual,kVectors,Chain::kNeedsStats,
      2*Split::kWgCols,Pool::kArea?Pool::kImages:0,TK,Side::kStatParts,TmaStore);
  static constexpr int kLayoutBytes=kLayout.end;
  static constexpr int kSideOffset=kLayout.side_base;
  static constexpr int kPoolOffset=kLayout.pool_base;
  static_assert(Sm90F3Budget::Fits(kLayoutBytes),
      "an F3 GEMM stage must fit the shared budget");
  // expect_tx must equal the bytes actually transferred, and only the row
  // statistics shrink on a short tail, so the count is per task.
  __device__ static __forceinline__ int SideBytes(int live_rows) {
    constexpr int aligned_parts=M%2?(Side::kStatParts+1)/2:Side::kStatParts;
    return (Chain::kNeedsResidual?kResidualBoxes*TM*128:0)+4*TN*kVectors+
        (Chain::kNeedsStats?aligned_parts*(live_rows/2)*16:0);
  }

  __device__ static __forceinline__ void ProduceSide(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,int m_tile,int n_tile,int live_rows) {
    if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
      Base::Wait(&ring.side_empty,phase.side_empty&1u);
      phase.side_empty^=1u;
      // Bulk sizes are multiples of 16 B. Publish an odd final 8 B pair
      // through the leader's barrier arrival without reading past the buffer.
      if constexpr(Chain::kNeedsStats) {
        #pragma unroll
        for(int part=0;part<Side::kStatParts;++part) {
          // An odd M misaligns odd part bases by8B; those short unit tails
          // use generic pair copies before expect_tx instead of invalid bulk.
          int first=(M%2 && part%2)?0:live_rows-(live_rows&1);
          for(int row=first;row<live_rows;++row) {
            float a,b;
            Async::LoadPartial2(p.epilogue.stats+2*(std::size_t(part)*M+m_tile*TM+row),a,b);
            auto* tail=reinterpret_cast<float*>(side+Side::kRowStatBase)+2*(part*TM+row);
            tail[0]=a;tail[1]=b;
          }
        }
      }
      Base::ExpectTx(&ring.side_full,unsigned(SideBytes(live_rows)));
      if constexpr(Chain::kNeedsResidual) {
        // Out-of-range columns and rows are filled with zeros by the TMA and
        // still count toward the transaction.
        #pragma unroll
        for(int box=0;box<kResidualBoxes;++box)
          Base::Tensor2D(side+Side::kResidualBase+std::size_t(box)*TM*128,
              p.residual,n_tile*TN+box*64,m_tile*TM,&ring.side_full);
      }
      #pragma unroll
      for(int v=0;v<kVectors;++v)
        if(p.epilogue.vectors[v])
          Base::Bulk(side+Side::kVectorBase+std::size_t(v)*TN*4,
              p.epilogue.vectors[v]+std::size_t(n_tile)*TN,4*TN,&ring.side_full);
      if constexpr(Chain::kNeedsStats) {
        #pragma unroll
        for(int part=0;part<Side::kStatParts;++part)
          if(live_rows>=2 && !(M%2 && part%2))
            Base::Bulk(side+Side::kRowStatBase+part*TM*8,
                p.epilogue.stats+(std::size_t(part)*M+m_tile*TM)*2,
                unsigned((live_rows/2)*16),&ring.side_full);
      }
    }
  }

  // ---- epilogue (§6.4 steps 2-6) ------------------------------------------

  // This element's coordinates inside the CTA tile.
  __device__ static __forceinline__ void Coordinate(int wg,int sub,int row,int col,
      int& tile_row,int& tile_col) {
    tile_row=Split::ARows(wg)+sub*64+row;
    tile_col=Split::BRows(wg)+col;
  }

  template<class Tile,class Owned>
  __device__ static __forceinline__ void StoreQuad(Tile const& tile,Owned const& owned,
      Element* output,int stride,int wg,int sub,int m_tile,int n_tile,
      int live_rows,int width,int group,char* staging=nullptr) {
    int quad=int(threadIdx.x)&3;
    CUTE_UNROLL
    for(int row=0;row<2;++row) {
      unsigned words[4];
      CUTE_UNROLL
      for(int g=0;g<4;++g) {
        int i=group*16+g*4+row*2;
        words[g]=unsigned(Element(tile(i)).raw())|
            (unsigned(Element(tile(i+1)).raw())<<16);
      }
      // A4x4 transpose within each quad: destination lane owns one8-column
      // group, with its four packed column pairs in source-lane order.
      unsigned x=__shfl_xor_sync(0xffffffffu,quad&1?words[0]:words[1],1);
      unsigned y=__shfl_xor_sync(0xffffffffu,quad&1?words[2]:words[3],1);
      if(quad&1) {words[0]=x;words[2]=y;} else {words[1]=x;words[3]=y;}
      x=__shfl_xor_sync(0xffffffffu,quad&2?words[0]:words[2],2);
      y=__shfl_xor_sync(0xffffffffu,quad&2?words[1]:words[3],2);
      if(quad&2) {words[0]=x;words[1]=y;} else {words[2]=x;words[3]=y;}
      int tile_row,tile_col;
      Coordinate(wg,sub,cute::get<0>(owned(group*16+row*2)),
          cute::get<1>(owned(group*16+row*2)),tile_row,tile_col);
      tile_col+=quad*6;
      if constexpr(TmaStore) {
        // TMA's horizontal store granule may cover a partial logical tail.
        // Only complete64-column boxes use it; the partial box keeps the
        // exact generic predicates. This also avoids staging fully OOB boxes.
        if(n_tile*TN+(group/2)*64+64<=NStore) {
        int row=tile_row-Split::ARows(wg)-sub*64,col=tile_col%64;
        auto* address=staging+row*128+(((col/8)^(row%8))*16);
        Async::StoreShared128(address,words[0],words[1],words[2],words[3]);
        continue;
        }
      }
      if(tile_row>=live_rows || tile_col>=width)continue;
      auto* address=output+std::size_t(m_tile*TM+tile_row)*stride+n_tile*TN+tile_col;
      if(tile_col+7<width && (reinterpret_cast<std::uintptr_t>(address)&15u)==0)
        Async::Store128(address,words[0],words[1],words[2],words[3]);
      else {
        CUTE_UNROLL
        for(int pair=0;pair<4;++pair) {
          if(tile_col+pair*2+1<width)Async::Store32(address+pair*2,words[pair]);
          else if(tile_col+pair*2<width) {
            Element value=Element::bitcast(static_cast<unsigned short>(words[pair]));
            address[pair*2]=value;
          }
        }
      }
    }
  }

  template<class Accum>
  __device__ static __forceinline__ void Epilogue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,Accum& accumulators,int wg,
      int m_tile,int n_tile,int live_rows,int width,Sm90F3Header* header,
      Element* arena_base,int* store_count=nullptr) {
    using namespace cute;
    Mma mma;
    auto owned=mma.get_slice(int(Sm90F3Async::ThreadIndex()&127u)).partition_C(
        make_identity_tensor(cute::Shape<cute::_64,cute::Int<Split::kWgCols>>{}));
    if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
      TM_F3_TRACE_BEGIN(kSideWait);
      Base::Wait(&ring.side_full,phase.side_full&1u);
      phase.side_full^=1u;
      TM_F3_TRACE_END(kSideWait);
    }
    TM_F3_TRACE_BEGIN(kEpilogue);
    auto const& e=p.epilogue;
    constexpr int kElements=int(cute::size(Fragment{}));
    // Every SM90 GMMA FP32 accumulator lays its fastest value mode out as the
    // column pair (col, col + 1) of one row, so elements 2j and 2j + 1 always
    // share a row and differ by one column. `dnn3_f3_layout_test` asserts that
    // for every tile this round instantiates; the store relies on it, and the
    // static step keeps the loop fully unrolled (DT-SPILL).
    static_assert(kElements%4==0,"a GMMA fragment pairs columns across two rows");
    // DT-SPILL: run the chain and the store one 64-column block at a time, so
    // a block's accumulator registers die at its store instead of staying
    // live across the whole tile.
    constexpr int kBlocks=(Split::kWgCols+63)/64;
    // DT-SPILL: a128-register accumulator spills in RN50 B16 with quad
    // temporaries. Retain its4B path; smaller fragments use16B stores.
    constexpr bool kVectorStore=TmaStore || (Split::kWgCols>=32 && Split::kWgCols%32==0 &&
        Split::kWgRows*Split::kWgCols/128<128);
    constexpr int kRegionColumns=kVectorStore?32:64;
    constexpr int kRegions=(Split::kWgCols+kRegionColumns-1)/kRegionColumns;
    auto* output=e.output;
    int stride=e.output_stride;
    RowAccumulator row_accum;
    float means[Split::kSubTiles][2]{},inverses[Split::kSubTiles][2]{};
    if constexpr(Chain::kNeedsStats) {
      #pragma unroll
      for(int s=0;s<Split::kSubTiles;++s) {
        #pragma unroll
        for(int half=0;half<2;++half) {
          int row,col;
          Coordinate(wg,s,get<0>(owned(2*half)),get<1>(owned(2*half)),row,col);
          Sm90F3RowNorm<TM,Side::kStatParts>(Side::RowStat(side),row,e.norm_width,e.epsilon,
              means[s][half],inverses[s][half]);
        }
      }
    }
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s) {
      auto& tile=accumulators.tile[s];
      #pragma unroll
      for(int block=0;block<kBlocks;++block) {
        char* staging=nullptr;
        if constexpr(TmaStore) {
          bool full_box=n_tile*TN+block*64+64<=NStore;
          int q=store_count?(full_box?(*store_count)++:*store_count):s*kBlocks+block;
          if constexpr(Chain::kNeedsResidual)
            staging=side+Side::kResidualBase+(block*TM+Split::ARows(wg)+s*64)*128;
          else {
            if(q>=2) {
              if((threadIdx.x&127)==0)Base::template BulkWait<1,true>();
              WgBarrier(wg);
            }
            staging=reinterpret_cast<char*>(arena_base)+kLayout.store_base+wg*16384+(q%2)*8192;
          }
        }
        if constexpr(Chain::kWritesStats) {
          row_accum.sum[s][0]=row_accum.sum[s][1]=0;
          row_accum.square[s][0]=row_accum.square[s][1]=0;
        }
        CUTE_UNROLL
        for(int half=0;half<(kVectorStore && Split::kWgCols>=64?2:1);++half) {
        int group=kVectorStore?block*2+half:block;
        if(group>=kRegions)continue;
        CUTE_UNROLL
        for(int i=0;i<kElements;i+=4) {
          if(get<1>(owned(i))/kRegionColumns!=group)continue;
          int tile_row,tile_col;
          Coordinate(wg,s,get<0>(owned(i)),get<1>(owned(i)),tile_row,tile_col);
          float columns[2][4]{};
          #pragma unroll
          for(int v=0;v<kVectors;++v) {
            auto pair=*reinterpret_cast<float2 const*>(Side::Vector(side,v)+tile_col);
            columns[0][v]=pair.x;columns[1][v]=pair.y;
          }
          #pragma unroll
          for(int row=0;row<2;++row) {
            int j=i+2*row,live_row=tile_row+8*row;
            float mean=means[s][row],inverse=inverses[s][row];
            tile(j)=Chain::Apply(tile(j),live_row,tile_col,side,mean,inverse,e,columns[0]);
            tile(j+1)=Chain::Apply(tile(j+1),live_row,tile_col+1,side,mean,inverse,e,columns[1]);
            if constexpr(Chain::kWritesStats) {
              if(tile_col<width) {
                row_accum.sum[s][row]+=tile(j);
                row_accum.square[s][row]+=tile(j)*tile(j);
              }
              if(tile_col+1<width) {
                row_accum.sum[s][row]+=tile(j+1);
                row_accum.square[s][row]+=tile(j+1)*tile(j+1);
              }
            }
#if TILEMEGA_TRACE_STAGE != 1
            if constexpr(!kVectorStore) {
              if(live_row>=live_rows || tile_col>=width)continue;
              auto* address=output+std::size_t(m_tile*TM+live_row)*stride+n_tile*TN+tile_col;
              if(tile_col+1<width) {
                Element low(tile(j)),high(tile(j+1));
                Async::Store32(address,unsigned(low.raw())|(unsigned(high.raw())<<16));
              }else *address=Element(tile(j));
            }
#endif
          }
        }
        TM_F3_TRACE_BEGIN(kStore);
        // Quad transposition changes lane ownership. Finish every source
        // residual read before any lane overwrites that same shared box.
        if constexpr(TmaStore && Chain::kNeedsResidual)WgBarrier(wg);
        if constexpr(kVectorStore)
          StoreQuad(tile,owned,output,stride,wg,s,m_tile,n_tile,live_rows,width,group,staging);
#if TILEMEGA_TRACE_STAGE == 1
        else {
        CUTE_UNROLL
        for(int i=0;i<kElements;i+=2) {
          if(get<1>(owned(i))/kRegionColumns!=group)continue;
          int tile_row,tile_col;
          Coordinate(wg,s,get<0>(owned(i)),get<1>(owned(i)),tile_row,tile_col);
          if(tile_row>=live_rows || tile_col>=width)continue;
          auto* address=output+std::size_t(m_tile*TM+tile_row)*stride+n_tile*TN+tile_col;
          if(tile_col+1<width) {
            Element low(tile(i)),high(tile(i+1));
            Async::Store32(address,unsigned(low.raw())|(unsigned(high.raw())<<16));
          }else *address=Element(tile(i));
        }
        }
#endif
        TM_F3_TRACE_END(kStore);
        }
        if constexpr(TmaStore) {
          Base::FenceProxyAsyncShared();
          WgBarrier(wg);
          if((threadIdx.x&127)==0 && n_tile*TN+block*64+64<=NStore) {
            Async::TensorStore2D(p.output_map,n_tile*TN+block*64,
                m_tile*TM+Split::ARows(wg)+s*64,staging);
            Base::BulkCommit();
          }
        }
        RowStatsBlock(p,row_accum,owned,wg,m_tile,n_tile,live_rows,header,s,block);
      }
    }
    PoolPartials(p,reinterpret_cast<char*>(arena_base),accumulators,owned,wg,
        m_tile,n_tile,live_rows,width);
    TM_F3_TRACE_END(kEpilogue);
    if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
      if constexpr(TmaStore && Chain::kNeedsResidual)
        if((threadIdx.x&127)==0)Base::template BulkWait<0,true>();
      Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
      if((tilemega::backend::Sm90F3Async::ThreadIndex()&31)==0)Base::Arrive(&ring.side_empty);
    }
  }

  __device__ static __forceinline__ void WgBarrier(int wg) {
    if(wg==0)Async::template NamedBarrier<Async::kBarWg0,128>();
    else Async::template NamedBarrier<Async::kBarWg1,128>();
  }

  template<class Owned>
  __device__ static __forceinline__ void RowStatsBlock(Sm90F3GemmOperands const& p,
      RowAccumulator const& partial,Owned const& owned,int wg,int m_tile,int n_tile,
      int live_rows,Sm90F3Header* header,int sub,int block) {
    if constexpr(Chain::kWritesStats) {
      static_assert(TN%64==0,"statistics writers own whole64-column blocks");
      using namespace cute;
      constexpr bool kMerge=TM==64 && TN==64;
      float sums[2]={partial.sum[sub][0],partial.sum[sub][1]};
      float squares[2]={partial.square[sub][0],partial.square[sub][1]};
      #pragma unroll
      for(int r=0;r<2;++r) {
        sums[r]+=__shfl_xor_sync(0xffffffffu,sums[r],1);
        sums[r]+=__shfl_xor_sync(0xffffffffu,sums[r],2);
        squares[r]+=__shfl_xor_sync(0xffffffffu,squares[r],1);
        squares[r]+=__shfl_xor_sync(0xffffffffu,squares[r],2);
        if constexpr(kMerge) {
          int row,col;
          Coordinate(wg,sub,get<0>(owned(2*r)),get<1>(owned(2*r)),row,col);
          if(wg==1 && (threadIdx.x&3)==0 && row<live_rows) {
            header->row_stats[row][0]=sums[r];header->row_stats[row][1]=squares[r];
          }
        }
      }
      if constexpr(kMerge)Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
      #pragma unroll
      for(int r=0;r<2;++r) {
        int row,col;
        Coordinate(wg,sub,get<0>(owned(2*r)),get<1>(owned(2*r)),row,col);
        if((threadIdx.x&3)==0 && row<live_rows && (!kMerge || wg==0)) {
          float sum=sums[r],square=squares[r];
          if constexpr(kMerge) {
            sum+=header->row_stats[row][0];square+=header->row_stats[row][1];
          }
          int n64=n_tile*(TN/64)+(TM==64?wg*(Split::kWgCols/64):0)+block;
          int global_row=m_tile*TM+row;
          Async::StorePartial2(p.epilogue.row_partials+
              (std::size_t(n64)*M+global_row)*2,sum,square);
        }
      }
      if constexpr(kMerge)Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
    }
  }

  // The GAP partial sums (§6.4 step 4). Eight lanes of a warp hold the same
  // column, so they reduce through lane bits 2, 3 and 4; the warp's result
  // lands in the pool scratch, and after the consumer barrier one thread per
  // (column, image) sums the four warps of its warpgroup and publishes the
  // slot. Each (image, block, column) has exactly one writing sub-tile.
  template<class Accum,class Owned>
  __device__ static __forceinline__ void PoolPartials(Sm90F3GemmOperands const& p,
      char* arena,Accum& accumulators,Owned const& owned,int wg,int m_tile,int n_tile,
      int live_rows,int width) {
    if constexpr(Pool::kArea>0) {
      using namespace cute;
      auto* scratch=reinterpret_cast<float*>(arena+kPoolOffset)+
          wg*4*Split::kWgCols*Pool::kImages;
      int warp=(int(threadIdx.x)%128)>>5,lane=int(threadIdx.x)&31;
      constexpr int kElements=int(cute::size(Fragment{}));
      constexpr int kImages=Pool::kImages;
      auto slot=[&](int w,int col,int image) {
        return scratch+(std::size_t(w)*Split::kWgCols+col)*kImages+image;
      };
      #pragma unroll
      for(int s=0;s<Split::kSubTiles;++s) {
        auto& tile=accumulators.tile[s];
        // The sub-tile's own 64-row block, and the first image it touches.
        int row_base=m_tile*TM+Split::ARows(wg)+s*64;
        int block=row_base/64,first=row_base/Pool::kArea;
        // CLayout_64xN orders each quartet as two columns over two rows.
        // Reduce a thread's two rows before the eight-lane column reduction.
        CUTE_UNROLL
        for(int group=0;group<kElements/4;++group) {
          #pragma unroll
          for(int half=0;half<2;++half) {
            int i=group*4+half;
            int column=get<1>(owned(i));
            int tile_col=Split::BRows(wg)+column;
            float sums[kImages];
            #pragma unroll
            for(int j=0;j<kImages;++j)sums[j]=0.f;
            #pragma unroll
            for(int r=0;r<2;++r) {
              int index=i+2*r,row=row_base+get<0>(owned(index));
              if(row-m_tile*TM<live_rows && tile_col<width) {
                int image=row/Pool::kArea-first;
                #pragma unroll
                for(int j=0;j<kImages;++j)
                  if(image==j)sums[j]+=tile(index);
              }
            }
            Flush(slot,warp,column,lane,sums,Split::kWgCols);
          }
        }
        Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
        // One writer per (column, image): the warpgroup's first warp.
        if(warp==0)
          for(int col=lane;col<Split::kWgCols;col+=32) {
            if(Split::BRows(wg)+col>=width)continue;
            #pragma unroll
            for(int j=0;j<kImages;++j) {
              float total=0;
              #pragma unroll
              for(int w=0;w<4;++w)total+=*slot(w,col,j);
              int image=first+j;
              if(image*Pool::kArea>=M)continue;
              int part=block-(image*Pool::kArea)/64;
              if(part<0 || part>=Pool::kParts)continue;
              int channel=n_tile*TN+Split::BRows(wg)+col;
              Async::StoreRelaxed(p.epilogue.channel_partial+
                  (std::size_t(image)*Pool::kParts+part)*Pool::kChannels+channel,total);
            }
          }
        Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
      }
    }
  }

  // Reduce one column's partials over the eight lanes that share it (bits 2,
  // 3 and 4 of the lane index) and leave the warp's result in its own slot.
  template<class Slot>
  __device__ static __forceinline__ void Flush(Slot const& slot,int warp,int column,
      int lane,float (&sums)[Pool::kImages>0?Pool::kImages:1],int width) {
    if constexpr(Pool::kArea>0) {
      #pragma unroll
      for(int j=0;j<Pool::kImages;++j) {
        float value=sums[j];
        #pragma unroll
        for(int mask:{4,8,16})value+=__shfl_xor_sync(0xffffffffu,value,mask);
        if((lane&28)==0 && column<width)*slot(warp,column,j)=value;
      }
    }
  }

  // ---- role entry points (§5.4) -------------------------------------------

  // How many tasks this CTA owns, in the legacy order so a task's identity is
  // independent of the skeleton: n_tile fastest, then m_tile, then the slice.
  __device__ static __forceinline__ int LocalTasks() {
    return kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
  }
  __device__ static __forceinline__ int Task(int local) {
    return int(blockIdx.x)+local*int(gridDim.x);
  }
  __device__ static __forceinline__ int LiveRows(int m_tile) {
    int rows=M-m_tile*TM;
    return rows<TM?rows:TM;
  }
  __device__ static __forceinline__ int Width(int n_tile) {
    int width=NStore-n_tile*TN;
    if(width>TN)width=TN;
    return width<0?0:width;
  }

  // P0 issues A/B, P1 issues side operands independently; `prearmed` is how
  // many of the first task's k blocks the previous stage already armed (§5.6).
  __device__ static __forceinline__ void Produce(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase,int prearmed) {
    if(int(threadIdx.x)==288) {
      Base::ProxyAsyncGlobalFence();
      for(int local=0;local<LocalTasks();++local) {
        int task=Task(local),m_tile=MTile(task),n_tile=NTile(task);
        ProduceSide(p,ring,side,phase,m_tile,n_tile,LiveRows(m_tile));
      }
    }
    if constexpr(Mode==Sm90F3Mode::kGather) {
      if(threadIdx.x<288 || threadIdx.x>=320)ProduceGather(p,ring,arena,side,phase);
      else __syncwarp();
      return;
    }
    if(int(threadIdx.x)!=256) {__syncwarp();return;}
    // The activation was written by other CTAs with ordinary stores before the
    // gate, so the async proxy must be made to observe them.
    Base::ProxyAsyncGlobalFence();
    int tasks=LocalTasks();
    for(int local=0;local<tasks;++local) {
      int task=Task(local);
      for(int k=0;k<kIterations;++k) {
        int slot=(local*kIterations+k)%Slots;
        bool skip=local==0 && k<prearmed;
        ProduceTaskK(p,ring,arena,phase,task,k,slot,skip);
      }
    }
    __syncwarp();
  }

  // ---- gather mode (§6.6) -------------------------------------------------

  // The stem's input is physically eight channels, so an im2col TMA would move
  // a 64-channel box of mostly padding. Instead P0, P2 and P3 gather with 96 lanes
  // the taps with `cp.async`: k = (r*S + s)*8 + c, so one 16 B copy is one
  // tap's eight channels and a 64-wide k block is eight taps.
  __device__ static __forceinline__ void ProduceGather(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase) {
    static_assert(Mode!=Sm90F3Mode::kGather || Conv::kC<=8,
        "gather mode is for a physically eight-channel input");
    int lane=int(threadIdx.x)<288?int(threadIdx.x)-256:int(threadIdx.x)-288;
    Base::ProxyAsyncGlobalFence();
    int tasks=LocalTasks();
    for(int local=0;local<tasks;++local) {
      int task=Task(local),m_tile=MTile(task),n_tile=NTile(task);
      for(int k=0;k<kIterations;++k) {
        int global_k=(Slice(task)*kIterations+k)*kPanels;
        int slot=(local*kIterations+k)%Slots;
        // P0's leader owns the release phase; all three gather warps rendezvous.
        if(tilemega::backend::Sm90F3Async::ThreadIndex()==256) {
          Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
          phase.empty^=1u<<slot;
        }
        Async::template NamedBarrier<Async::kBarGather,96>();
        // Close the producer's own async-copy lifetime before reusing a slot.
        // One committed group per k block preserves the Slots-deep window.
        Async::template GatherWait<Slots-1>();
        auto* panel=reinterpret_cast<char*>(SlotA(arena,slot));
        #pragma unroll
        for(int kb=0;kb<kPanels;++kb) {
        for(int m=lane;m<TM;m+=Sm90F3Ring::kGatherLanes) {
          int row=m_tile*TM+m;
          int image=row/(Conv::kP*Conv::kQ),pixel=row%(Conv::kP*Conv::kQ);
          int q=pixel%Conv::kQ,r0=pixel/Conv::kQ;
          #pragma unroll
          for(int j=0;j<8;++j) {
            int tap=(global_k+kb)*8+j;
            int rr=tap/Conv::kS,ss=tap%Conv::kS;
            int y=r0*Conv::kStrideH-Conv::kPadH+rr*Conv::kDilationH;
            int x=q*Conv::kStrideW-Conv::kPadW+ss*Conv::kDilationW;
            bool valid=tap<kGatherTaps && row<M &&
                x>=0 && x<Conv::kW && y>=0 && y<Conv::kH;
            auto const* source=p.input+
                ((std::size_t(image)*Conv::kH+y)*Conv::kW+x)*Conv::kC;
            // The panel row is 128 B and SW128 swizzled exactly as a TMA box
            // would leave it, so the consumer's descriptor is unchanged.
            Base::Copy16(panel+std::size_t(kb)*TM*128+std::size_t(m)*128+
                ((j^(m%8))*16),source,valid);
          }
        }
        }
        Async::GatherCommit();
        // Every lane completes its own copies into the slot's gather barrier.
        Base::CompleteCopies(ring.gfull+slot);
        if(tilemega::backend::Sm90F3Async::ThreadIndex()==256) {
          Base::ExpectTx(ring.gfull+slot,2*TN*TK);
          Base::Bulk(SlotB(arena,slot),
              p.weight+(std::size_t(n_tile)*(K/64)+global_k)*TN*64,2*TN*TK,ring.gfull+slot);
        }
      }
    }
    Async::template GatherWait<0>();
    __syncwarp();
  }

  __device__ static __forceinline__ void Consume(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase,
      Sm90F3Header* header,int wg) {
    Accumulators accum;
    int store_count=0;
    // Cached per-shape task counts otherwise stay live across BERT's layers.
    for(int local=0,task=int(blockIdx.x);task<kTasks;++local,task+=int(gridDim.x)) {
      int m_tile=MTile(task),n_tile=NTile(task);
      TM_F3_TRACE_BEGIN(kMainLoop);
      MainLoop(ring,arena,phase,accum,wg,(local*kIterations)%Slots,kIterations);
      TM_F3_TRACE_END(kMainLoop);
      if constexpr(SplitK==1) {
        Epilogue(p,ring,side,phase,accum,wg,m_tile,n_tile,LiveRows(m_tile),Width(n_tile),header,arena,&store_count);
      }else {
        SplitKEpilogue(p,ring,side,phase,accum,wg,task,m_tile,n_tile,header,arena,&store_count);
      }
    }
    if constexpr(TmaStore)
      if((threadIdx.x&127)==0) {
        Base::template BulkWait<0,false>();
        Base::ProxyAsyncGlobalFence();
      }
  }

  // ---- split-K (§6.5) -----------------------------------------------------

  // Each slice stores its FP32 partial; the tile's last arriver reads every
  // slice back in a fixed order and runs the whole chain once, which is what
  // lets a deferred-normalization stage split its K at all (§4 F-5).
  template<class Accum>
  __device__ static __forceinline__ void SplitKEpilogue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,Accum& accumulators,int wg,
      int task,int m_tile,int n_tile,Sm90F3Header* header,Element* arena,int* store_count=nullptr) {
    using namespace cute;
    int tile=m_tile*kTiles+n_tile,slice=Slice(task);
    TM_F3_TRACE_BEGIN(kSplitK);
    constexpr int kFragment=int(size(Fragment{}));
    constexpr int kElements=kFragment*Split::kSubTiles;
    static_assert(kElements%4==0 && kElements*256==TM*TN);
    int thread=int(threadIdx.x);
    auto* base=p.epilogue.partial+
        ((std::size_t(tile)*SplitK+slice)*256+thread)*kElements;
    #pragma unroll
    for(int i=0;i<kElements;i+=4)
      Async::StorePartial4(base+i,
          accumulators.tile[i/kFragment](i%kFragment),
          accumulators.tile[(i+1)/kFragment]((i+1)%kFragment),
          accumulators.tile[(i+2)/kFragment]((i+2)%kFragment),
          accumulators.tile[(i+3)/kFragment]((i+3)%kFragment));
    Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
    if(tilemega::backend::Sm90F3Async::ThreadIndex()==0) {
      auto old=Async::CounterAdd(p.epilogue.counters+tile,1);
      header->last_arriver=int(old+1)==SplitK;
      if(header->last_arriver)p.epilogue.counters[tile]=0;
    }
    Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
    if(!header->last_arriver) {
      // A non-last slice still owes the producer its side slot.
      if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
        Base::Wait(&ring.side_full,phase.side_full&1u);
        phase.side_full^=1u;
        __syncwarp();
        if((tilemega::backend::Sm90F3Async::ThreadIndex()&31)==0)Base::Arrive(&ring.side_empty);
      }
      TM_F3_TRACE_END(kSplitK);
      return;
    }
    constexpr int kChunk=(64/SplitK/4*4<kElements?64/SplitK/4*4:kElements);
    static_assert(kChunk>=4);
    #pragma unroll
    for(int begin=0;begin<kElements;begin+=kChunk) {
      float loaded[SplitK][kChunk];
      #pragma unroll
      for(int sl=0;sl<SplitK;++sl) {
        #pragma unroll
        for(int c=0;c<kChunk && begin+c<kElements;c+=4)
          Async::LoadPartial4(p.epilogue.partial+
              ((std::size_t(tile)*SplitK+sl)*256+thread)*kElements+begin+c,
              loaded[sl][c],loaded[sl][c+1],loaded[sl][c+2],loaded[sl][c+3]);
      }
      #pragma unroll
      for(int c=0;c<kChunk && begin+c<kElements;++c) {
        float total=0;
        #pragma unroll
        for(int sl=0;sl<SplitK;++sl)total+=loaded[sl][c];
        accumulators.tile[(begin+c)/kFragment]((begin+c)%kFragment)=total;
      }
    }
    TM_F3_TRACE_END(kSplitK);
    Epilogue(p,ring,side,phase,accumulators,wg,m_tile,n_tile,LiveRows(m_tile),
        Width(n_tile),header,arena,store_count);
  }
};

}  // namespace tilemega::backend
