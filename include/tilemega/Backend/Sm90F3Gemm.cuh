// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Epilogue.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <tilemega/Backend/Sm90Gmma.h>
#include <cute/tensor.hpp>
#include <type_traits>

// The F3 forward GEMM/convolution body (DNN3 prompt §6). Two consumer
// warpgroups share one CTA tile, a single producer warp feeds an eight-deep
// ring, and the epilogue runs in registers over the producer's prefetched
// side data. The main loop keeps one WGMMA batch in flight (`wait<1>`), which
// is CUTLASS's K_PIPE_MMAS=1 (§6.3, appendix B).
namespace tilemega::backend {

enum class Sm90F3Mode {kNormal, kGather, kSwapAB};

struct Sm90F3GemmOperands {
  CUtensorMap const* activation;     // 2D {K, M} dense, or 4D im2col NHWC
  CUtensorMap const* residual;       // 2D {N, M} of the residual buffer
  cutlass::bfloat16_t const* weight;
  cutlass::bfloat16_t const* input;  // gather mode reads the image directly
  Sm90F3EpilogueOperands epilogue;
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
  static_assert(TN==32 || TN==64 || TN==128 || TN==256);
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
         int NStore,class Chain,class Conv=Sm90F3NoConv,class Pool=Sm90F3NoPool>
struct Sm90F3Gemm {
  using Element=cutlass::bfloat16_t;
  using Async=Sm90F3Async;
  using Base=Sm90F3Async::Base;
  using Split=Sm90F3Split<TM,TN>;
  using Side=typename Chain::Layout;

  static constexpr int kThreads=288;
  static constexpr int kSlots=Slots;
  static constexpr int kIterations=K/(SplitK*64);
  static_assert(Slots>=2 && Slots<=8);
  static_assert(K%(SplitK*64)==0,"split-K divides the contraction into whole k blocks");
  static_assert(N%TN==0,"N is padded to a whole number of TN tiles when the plan is built");
  static_assert(NStore>=1 && NStore<=N);
  static constexpr int kTiles=N/TN,kMTiles=(M+TM-1)/TM;
  static constexpr int kOutputTiles=kMTiles*kTiles,kTasks=kOutputTiles*SplitK;
  static constexpr int kSlotBytes=Sm90F3SlotBytes(TM,TN);
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

  __device__ static __forceinline__ int NTile(int task) {return task%kTiles;}
  __device__ static __forceinline__ int MTile(int task) {return (task/kTiles)%kMTiles;}
  __device__ static __forceinline__ int Slice(int task) {return task/kOutputTiles;}

  // A slot's A panel and B page.
  __device__ static __forceinline__ Element* SlotA(Element* arena,int slot) {
    return arena+std::size_t(slot)*(kSlotBytes/2);
  }
  __device__ static __forceinline__ Element* SlotB(Element* arena,int slot) {
    return SlotA(arena,slot)+std::size_t(TM)*64;
  }

  // ---- producer ------------------------------------------------------------

  __device__ static __forceinline__ void IssueA(Element* a,Sm90F3GemmOperands const& p,
      int m_tile,int k,std::uint64_t* barrier) {
    if constexpr(Mode==Sm90F3Mode::kNormal) {
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

  // Dense A is one 2D box; an im2col A is TM/64 boxes. Either way the slot's
  // transaction count is the whole slot.
  __device__ static __forceinline__ void ProduceTaskK(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,Sm90F3Phase& phase,int task,int k_index,int slot,
      bool skip_b) {
    int n_tile=NTile(task),m_tile=MTile(task),slice=Slice(task);
    int k=(slice*kIterations+k_index)*64;
    if(!skip_b) {
      Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
      phase.empty^=1u<<slot;
      Base::ExpectTx(ring.full+slot,kSlotBytes);
      Base::Bulk(SlotB(arena,slot),
          p.weight+(std::size_t(n_tile)*(K/64)+k/64)*TN*64,2*TN*64,ring.full+slot);
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
      auto* b=SlotB(arena,slot);
      #pragma unroll
      for(int s=0;s<Split::kSubTiles;++s) {
        auto panel=make_tensor(make_smem_ptr(
            a+std::size_t(Split::ARows(wg)+s*64)*64),Sm90KLayout<64,64>{});
        auto page=make_tensor(make_smem_ptr(
            b+std::size_t(Split::BRows(wg))*64),Sm90KLayout<Split::kWgCols,64>{});
        auto ra=thread.partition_fragment_A(panel);
        auto rb=thread.partition_fragment_B(page);
        cute::warpgroup_fence_operand(accum[s]);
        cute::warpgroup_arrive();
        CUTE_UNROLL
        for(int step=0;step<size<2>(ra);++step)
          cute::gemm(mma,ra(cute::_,cute::_,step),rb(cute::_,cute::_,step),accum[s]);
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
    if((threadIdx.x&31)==0)Base::Arrive(ring.empty+slot);
  }

  // ---- producer side prefetch (§6.4 step 1) -------------------------------

  static constexpr int kVectors=Side::kVectorBytes/(4*TN);
  static constexpr int kResidualBoxes=(TN+63)/64;
  // The stage's own arena layout, from the same function the solver prices it
  // with (§5.2 item 4). A GAP producer's reduction area is added by codegen
  // through the chain's pool segments.
  static constexpr Sm90F3GemmLayout kLayout=Sm90F3MakeGemmLayout(TM,TN,Slots,
      false,0,Chain::kNeedsResidual,kVectors,Chain::kNeedsStats,
      2*Split::kWgCols,Pool::kArea?Pool::kImages:0);
  static constexpr int kLayoutBytes=kLayout.end;
  static constexpr int kSideOffset=kLayout.side_base;
  static constexpr int kPoolOffset=kLayout.pool_base;
  static_assert(Sm90F3Budget::Fits(kLayoutBytes),
      "an F3 GEMM stage must fit the shared budget");
  // expect_tx must equal the bytes actually transferred, and only the row
  // statistics shrink on a short tail, so the count is per task.
  __device__ static __forceinline__ int SideBytes(int live_rows) {
    return (Chain::kNeedsResidual?kResidualBoxes*TM*128:0)+4*TN*kVectors+
        (Chain::kNeedsStats?live_rows*2*4:0);
  }

  __device__ static __forceinline__ void ProduceSide(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,int m_tile,int n_tile,int live_rows) {
    if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
      Base::Wait(&ring.side_empty,phase.side_empty&1u);
      phase.side_empty^=1u;
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
      if constexpr(Chain::kNeedsStats)
        Base::Bulk(side+Side::kRowStatBase,
            p.epilogue.stats+std::size_t(m_tile)*TM*2,unsigned(live_rows*2*4),
            &ring.side_full);
    }
  }

  // ---- epilogue (§6.4 steps 2-6) ------------------------------------------

  // This element's coordinates inside the CTA tile.
  __device__ static __forceinline__ void Coordinate(int wg,int sub,int row,int col,
      int& tile_row,int& tile_col) {
    tile_row=Split::ARows(wg)+sub*64+row;
    tile_col=Split::BRows(wg)+col;
  }

  template<class Accum>
  __device__ static __forceinline__ void Epilogue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,Accum& accumulators,int wg,
      int m_tile,int n_tile,int live_rows,int width,Sm90F3Header* header,
      Element* arena_base) {
    using namespace cute;
    Mma mma;
    auto owned=mma.get_slice(int(threadIdx.x)%128).partition_C(
        make_identity_tensor(cute::Shape<cute::_64,cute::Int<Split::kWgCols>>{}));
    if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
      Base::Wait(&ring.side_full,phase.side_full&1u);
      phase.side_full^=1u;
    }
    auto const& e=p.epilogue;
    constexpr int kElements=int(cute::size(Fragment{}));
    // Every SM90 GMMA FP32 accumulator lays its fastest value mode out as the
    // column pair (col, col + 1) of one row, so elements 2j and 2j + 1 always
    // share a row and differ by one column. `dnn3_f3_layout_test` asserts that
    // for every tile this round instantiates; the store relies on it, and the
    // static step keeps the loop fully unrolled (DT-SPILL).
    static_assert(kElements%2==0,"a GMMA accumulator holds whole column pairs");
    // DT-SPILL: run the chain and the store one 64-column block at a time, so
    // a block's accumulator registers die at its store instead of staying
    // live across the whole tile.
    constexpr int kBlocks=(Split::kWgCols+63)/64;
    auto* output=e.output;
    int stride=e.output_stride;
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s) {
      auto& tile=accumulators.tile[s];
      #pragma unroll
      for(int block=0;block<kBlocks;++block) {
        int cached=-1;float mean=0,inverse=0;
        CUTE_UNROLL
        for(int i=0;i<kElements;i+=2) {
          if(get<1>(owned(i))/64!=block)continue;
          int tile_row,tile_col;
          Coordinate(wg,s,get<0>(owned(i)),get<1>(owned(i)),tile_row,tile_col);
          if constexpr(Chain::kNeedsStats)
            if(tile_row!=cached) {
              cached=tile_row;
              Sm90F3RowNorm(Side::RowStat(side),tile_row,e.norm_width,e.epsilon,mean,inverse);
            }
          tile(i)=Chain::Apply(tile(i),tile_row,tile_col,side,mean,inverse,e);
          tile(i+1)=Chain::Apply(tile(i+1),tile_row,tile_col+1,side,mean,inverse,e);
          if(tile_row>=live_rows || tile_col>=width)continue;
          auto* address=output+std::size_t(m_tile*TM+tile_row)*stride+n_tile*TN+tile_col;
          if(tile_col+1<width) {
            Element low(tile(i)),high(tile(i+1));
            Async::Store32(address,unsigned(low.raw())|(unsigned(high.raw())<<16));
          }else *address=Element(tile(i));
        }
      }
    }
    PoolPartials(p,reinterpret_cast<char*>(arena_base),accumulators,owned,wg,
        m_tile,n_tile,live_rows,width);
    if constexpr(Chain::kNeedsResidual || kVectors>0 || Chain::kNeedsStats) {
      Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
      if((threadIdx.x&31)==0)Base::Arrive(&ring.side_empty);
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

  // The producer warp: the leader issues every transfer, and `prearmed` is how
  // many of the first task's k blocks the previous stage already armed (§5.6).
  __device__ static __noinline__ void Produce(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase,int prearmed) {
    if constexpr(Mode==Sm90F3Mode::kGather) {ProduceGather(p,ring,arena,side,phase);return;}
    if(int(threadIdx.x)!=256) {__syncwarp();return;}
    // The activation was written by other CTAs with ordinary stores before the
    // gate, so the async proxy must be made to observe them.
    Base::ProxyAsyncGlobalFence();
    int tasks=LocalTasks();
    for(int local=0;local<tasks;++local) {
      int task=Task(local),m_tile=MTile(task),n_tile=NTile(task);
      for(int k=0;k<kIterations;++k) {
        int slot=(local*kIterations+k)%Slots;
        bool skip=local==0 && k<prearmed;
        ProduceTaskK(p,ring,arena,phase,task,k,slot,skip);
      }
      // The side data is issued after this task's k blocks: the epilogue only
      // needs it at the end, and issuing it earlier would park the producer on
      // `side_empty` while the previous task's epilogue still held it (§6.3).
      ProduceSide(p,ring,side,phase,m_tile,n_tile,LiveRows(m_tile));
    }
    __syncwarp();
  }

  // ---- gather mode (§6.6) -------------------------------------------------

  // The stem's input is physically eight channels, so an im2col TMA would move
  // a 64-channel box of mostly padding. Instead all 32 producer lanes gather
  // the taps with `cp.async`: k = (r*S + s)*8 + c, so one 16 B copy is one
  // tap's eight channels and a 64-wide k block is eight taps.
  __device__ static __forceinline__ void ProduceGather(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase) {
    static_assert(Mode!=Sm90F3Mode::kGather || Conv::kC<=8,
        "gather mode is for a physically eight-channel input");
    int lane=int(threadIdx.x)-256;
    Base::ProxyAsyncGlobalFence();
    int tasks=LocalTasks();
    for(int local=0;local<tasks;++local) {
      int task=Task(local),m_tile=MTile(task),n_tile=NTile(task);
      for(int k=0;k<kIterations;++k) {
        int slot=(local*kIterations+k)%Slots;
        // Lane 0 alone owns the slot's release phase; the warp then gathers.
        if(lane==0) {
          Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
          phase.empty^=1u<<slot;
        }
        __syncwarp();
        // Close the producer's own async-copy lifetime before reusing a slot.
        // One committed group per k block preserves the Slots-deep window.
        Async::template GatherWait<Slots-1>();
        auto* panel=reinterpret_cast<char*>(SlotA(arena,slot));
        for(int m=lane;m<TM;m+=Sm90F3Ring::kGatherLanes) {
          int row=m_tile*TM+m;
          int image=row/(Conv::kP*Conv::kQ),pixel=row%(Conv::kP*Conv::kQ);
          int q=pixel%Conv::kQ,r0=pixel/Conv::kQ;
          #pragma unroll
          for(int j=0;j<8;++j) {
            int tap=k*8+j;
            int rr=tap/Conv::kS,ss=tap%Conv::kS;
            int y=r0*Conv::kStrideH-Conv::kPadH+rr*Conv::kDilationH;
            int x=q*Conv::kStrideW-Conv::kPadW+ss*Conv::kDilationW;
            bool valid=tap<kGatherTaps && row<M &&
                x>=0 && x<Conv::kW && y>=0 && y<Conv::kH;
            auto const* source=p.input+
                ((std::size_t(image)*Conv::kH+y)*Conv::kW+x)*Conv::kC;
            // The panel row is 128 B and SW128 swizzled exactly as a TMA box
            // would leave it, so the consumer's descriptor is unchanged.
            Base::Copy16(panel+std::size_t(m)*128+((j^(m%8))*16),source,valid);
          }
        }
        Async::GatherCommit();
        // Every lane completes its own copies into the slot's gather barrier.
        Base::CompleteCopies(ring.gfull+slot);
        if(lane==0) {
          Base::ExpectTx(ring.gfull+slot,2*TN*64);
          Base::Bulk(SlotB(arena,slot),
              p.weight+(std::size_t(n_tile)*(K/64)+k)*TN*64,2*TN*64,ring.gfull+slot);
        }
      }
      if(lane==0)ProduceSide(p,ring,side,phase,m_tile,n_tile,LiveRows(m_tile));
    }
    Async::template GatherWait<0>();
    __syncwarp();
  }

  __device__ static __forceinline__ void Consume(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase,
      Sm90F3Header* header,int wg) {
    int tasks=LocalTasks();
    Accumulators accum;
    for(int local=0;local<tasks;++local) {
      int task=Task(local),m_tile=MTile(task),n_tile=NTile(task);
      MainLoop(ring,arena,phase,accum,wg,(local*kIterations)%Slots,kIterations);
      if constexpr(SplitK==1) {
        Epilogue(p,ring,side,phase,accum,wg,m_tile,n_tile,LiveRows(m_tile),Width(n_tile),header,arena);
      }else {
        SplitKEpilogue(p,ring,side,phase,accum,wg,task,m_tile,n_tile,header,arena);
      }
    }
  }

  // ---- split-K (§6.5) -----------------------------------------------------

  // Each slice stores its FP32 partial; the tile's last arriver reads every
  // slice back in a fixed order and runs the whole chain once, which is what
  // lets a deferred-normalization stage split its K at all (§4 F-5).
  template<class Accum>
  __device__ static __forceinline__ void SplitKEpilogue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,Accum& accumulators,int wg,
      int task,int m_tile,int n_tile,Sm90F3Header* header,Element* arena) {
    using namespace cute;
    Mma mma;
    auto owned=mma.get_slice(int(threadIdx.x)%128).partition_C(
        make_identity_tensor(cute::Shape<cute::_64,cute::Int<Split::kWgCols>>{}));
    int tile=m_tile*kTiles+n_tile,slice=Slice(task);
    auto* base=p.epilogue.partial+(std::size_t(tile)*SplitK+slice)*TM*TN;
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s) {
      auto& values=accumulators.tile[s];
      CUTE_UNROLL
      for(int i=0;i<int(size(values));i+=2) {
        int row,col;
        Coordinate(wg,s,get<0>(owned(i)),get<1>(owned(i)),row,col);
        Async::StorePartial2(base+std::size_t(row)*TN+col,values(i),values(i+1));
      }
    }
    Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
    if(threadIdx.x==0) {
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
        if((threadIdx.x&31)==0)Base::Arrive(&ring.side_empty);
      }
      return;
    }
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s) {
      auto& values=accumulators.tile[s];
      CUTE_UNROLL
      for(int i=0;i<int(size(values));i+=2) {
        int row,col;
        Coordinate(wg,s,get<0>(owned(i)),get<1>(owned(i)),row,col);
        float total0=0,total1=0;
        // Ascending slice order, including this slice's own stored copy, so
        // repeated launches agree bitwise.
        #pragma unroll 1
        for(int sl=0;sl<SplitK;++sl) {
          float a,b;
          Async::LoadPartial2(p.epilogue.partial+
              (std::size_t(tile)*SplitK+sl)*TM*TN+std::size_t(row)*TN+col,a,b);
          total0+=a;total1+=b;
        }
        values(i)=total0;values(i+1)=total1;
      }
    }
    Epilogue(p,ring,side,phase,accumulators,wg,m_tile,n_tile,LiveRows(m_tile),
        Width(n_tile),header,arena);
  }
};

}  // namespace tilemega::backend
