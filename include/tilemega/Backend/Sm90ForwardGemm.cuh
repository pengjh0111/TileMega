// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseRoles.cuh>
#include <tilemega/Codegen/executor/Async.cuh>

// Native DNN forward GEMM/conv body (prompt §10 N-1 item 2). One warpgroup of
// consumers runs WGMMA m64nTNk16 over a Slots-deep ring; one producer warp
// issues the weights (K-major tile pages, bulk) and the activation (a 2D
// tensor map for dense A, a TMA im2col 4D map for strided or k x k conv).
// Split-K writes fp32 partials and counts a per-tile last arriver, which sums
// them in fixed order and runs the epilogue chain: no separate combine stage.
namespace tilemega::backend {

enum class Sm90ForwardAMode {kDense,kIm2col};

struct Sm90ForwardGemmOperands {
  CUtensorMap const* activation;   // 2D {K, M} dense, or 4D im2col NHWC
  cutlass::bfloat16_t const* weight;
  cutlass::bfloat16_t* output;
  float* partial;                  // split-K: [tiles][SplitK][TM][TN] fp32
  std::uint64_t* counters;         // split-K: one per output tile
  Sm90ForwardEpilogueOperands epilogue;
  int output_stride;               // channels of one output row
};

// A dense stage's A has no convolution geometry; a conv stage's comes from the
// plan's ConvDesc and its input layout, as compile-time constants.
struct Sm90ForwardNoConv {
  static constexpr int kQ=1,kP=1,kC=64,kS=1,kR=1,kStrideW=1,kStrideH=1,
      kPadW=0,kPadH=0,kHaloW=0,kHaloH=0,kDilationW=1,kDilationH=1;
};

// N is the weight's padded extent (a whole number of TN tiles); NStore is how
// many of the chain's output columns the model actually keeps.
template<int TM,int TN,int TK,int Slots,int SplitK,Sm90ForwardAMode AMode,int M,int N,int K,
         int NStore,class Chain,class Conv=Sm90ForwardNoConv>
struct Sm90ForwardGemm {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Gmma=Sm90ForwardGmma<arch::Sm90,TM,TN,TK>;
  static_assert(TM==64 || TM==128);
  static_assert(TN==32 || TN==64 || TN==128 || TN==256);
  static_assert(TK==64 && Slots>=2 && Slots<=8);
  static_assert(SplitK>=1 && K%(SplitK*TK)==0,"split-K divides the contraction into whole k tiles");
  static_assert(N%TN==0,"N is padded to a multiple of TN when the plan is built");
  static_assert(NStore>=1 && NStore<=N,"the stored width is part of the padded extent");
  static_assert(M>=1,"a stage has at least one row");
  static_assert(AMode==Sm90ForwardAMode::kDense || Conv::kC%TK==0,
      "an im2col stage reads whole 64-channel blocks");
  static constexpr int kThreads=160,kIterations=K/(SplitK*TK);
  // One task per (M tile, N tile, k slice); the M tiles of a conv stage walk
  // the flattened NPQ pixel space, as the im2col map's traversal does.
  static constexpr int kTiles=N/TN,kMTiles=(M+TM-1)/TM;
  static constexpr int kOutputTiles=kMTiles*kTiles,kTasks=kOutputTiles*SplitK;
  static constexpr int kSlotBytes=2*TK*(TM+TN);
  // The accumulator tile the chain rewrites, then the per-row (mu, 1/sigma)
  // a deferred normalization needs and the split-K last-arriver flag.
  static constexpr bool kNeedsStats=Chain::kNeedsStats;
  static constexpr int kStatBytes=kNeedsStats?8*TM:0;
  static constexpr int kScratchBytes=4*TM*TN+kStatBytes+16;
  static_assert(!kNeedsStats || SplitK==1,
      "a deferred normalization reads the whole row, so its stage cannot split K");
  // 1 KB header, the ring, then that scratch.
  static constexpr Sm90DenseLayout kLayout{1024,1024+2*Slots*TK*TM,
      1024+Slots*kSlotBytes,(1024+Slots*kSlotBytes+kScratchBytes+1023)/1024*1024};
  static constexpr int kSharedBytes=kLayout.end;

  // Byte offset of element (row, column) of a 64-column SW128 A panel.
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return (col/64)*TM*128+row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }

  __device__ static __forceinline__ int NTile(int task) {return task%kTiles;}
  __device__ static __forceinline__ int MTile(int task) {return (task/kTiles)%kMTiles;}
  __device__ static __forceinline__ int Slice(int task) {return task/kOutputTiles;}

  __device__ static __forceinline__ void IssueA(Element* a,
      Sm90ForwardGemmOperands const& p,int m_tile,int k,std::uint64_t* barrier) {
    if constexpr(AMode==Sm90ForwardAMode::kDense) {
      Async::Tensor2D(a,p.activation,k,m_tile*TM,barrier);
    }else {
      // The k panel selects the (r, s) tap and the 64-channel block; the base
      // pixel is the tile's first output pixel, in NPQ raster order.
      constexpr int kBlocks=Conv::kC/TK;
      int panel=k/TK,tap=panel/kBlocks,c=(panel%kBlocks)*TK;
      int row=m_tile*TM,image=row/(Conv::kP*Conv::kQ),pixel=row%(Conv::kP*Conv::kQ);
      int w=(pixel%Conv::kQ)*Conv::kStrideW+Conv::kHaloW-Conv::kPadW;
      int h=(pixel/Conv::kQ)*Conv::kStrideH+Conv::kHaloH-Conv::kPadH;
      Async::TensorIm2col4D(a,p.activation,c,w,h,image,
          std::uint16_t((tap%Conv::kS)*Conv::kDilationW),
          std::uint16_t((tap/Conv::kS)*Conv::kDilationH),barrier);
    }
  }

  __device__ static __noinline__ void Produce(Sm90ForwardGemmOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase) {
    Sm90DenseRoles::Producer();
    if(threadIdx.x!=128) {__syncwarp();return;}
    Async::ProxyAsyncGlobalFence();
    for(int sequence=0;sequence<tasks*kIterations;++sequence) {
      int slot=sequence%Slots,local=sequence/kIterations,iteration=sequence%kIterations;
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int n_tile=NTile(task),m_tile=MTile(task),slice=Slice(task);
      int k=(slice*kIterations+iteration)*TK;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      Async::ExpectTx(state.full+slot,kSlotBytes);
      // Weights: the tile's K-major pages are contiguous per (tile, k).
      Async::Bulk(b+slot*TN*TK,p.weight+(std::size_t(n_tile)*(K/TK)+k/TK)*TN*TK,2*TN*TK,
          state.full+slot);
      IssueA(a+slot*TM*TK,p,m_tile,k,state.full+slot);
      phase^=1u<<slot;
    }
    if(tasks)for(int slot=0;slot<Slots;++slot)state.producer_generations[slot]=(phase>>slot)&1;
    __syncwarp();
  }

  __device__ static __noinline__ void Consume(Sm90ForwardGemmOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,char* scratch,int tasks,unsigned phase) {
    Sm90DenseRoles::Consumer();
    using namespace cute;
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int n_tile=NTile(task),m_tile=MTile(task),slice=Slice(task);
      int tile=m_tile*kTiles+n_tile;
      int rows=M-m_tile*TM<TM?M-m_tile*TM:TM;
      // The padded tail of N has weights but no output columns.
      int width=NStore-n_tile*Chain::kWidth;
      if(width>Chain::kWidth)width=Chain::kWidth;
      auto accum=partition_fragment_C(typename Gmma::Mma{},Shape<Int<TM>,Int<TN>>{});
      clear(accum);
      // Deferred LayerNorm: the consumer sums the A rows it already reads for
      // the MMA, so the statistics cost no extra global traffic (N-1 item 6).
      float sum=0,squares=0;
      for(int iteration=0;iteration<kIterations;++iteration) {
        int slot=(local*kIterations+iteration)%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        Gmma::Accumulate(b+slot*TN*TK,a+slot*TM*TK,accum);
        // WGMMA reads the slot asynchronously: its commit must land before the
        // slot is released for the producer's next async write.
        Sm90Gmma<arch::Sm90>::template Wait<0>();
        if constexpr(kNeedsStats) {
          // Thread r owns row r of this tile; the slot is still full here.
          int row=int(threadIdx.x);
          if(row<TM) {
            auto const* panel=reinterpret_cast<char const*>(a+slot*TM*TK);
            #pragma unroll 8
            for(int col=0;col<TK;++col) {
              float v=float(*reinterpret_cast<Element const*>(panel+Swizzled(row,col)));
              sum+=v;squares+=v*v;
            }
          }
        }
        Async::FenceProxyAsyncShared();
        Async::Arrive(state.empty+slot);
      }
      float* tile_values=Gmma::Materialize(accum,scratch);
      // The chain addresses this tile's rows: its residual, row statistics and
      // output all start at the tile's first row.
      auto epilogue=p.epilogue;
      if(epilogue.residual)epilogue.residual+=std::size_t(m_tile)*TM*epilogue.residual_stride;
      if(epilogue.row_stats)epilogue.row_stats+=std::size_t(m_tile)*TM*2;
      if constexpr(kNeedsStats) {
        auto* stats=reinterpret_cast<float*>(scratch+4*TM*TN);
        if(int(threadIdx.x)<TM) {
          float mean=sum/float(K),variance=squares/float(K)-mean*mean;
          stats[2*int(threadIdx.x)]=mean;
          stats[2*int(threadIdx.x)+1]=rsqrtf(variance+p.epilogue.epsilon);
        }
        codegen::executor::ComputeSync();
        epilogue.stats=stats;
      }
      auto* output=p.output+std::size_t(m_tile)*TM*p.output_stride;
      if constexpr(SplitK==1) {
        Chain::Finish(tile_values,rows,n_tile*TN,width>0?width:0,epilogue,output,p.output_stride);
      }else {
        // Each slice stores its partial; the last arriver of the tile sums the
        // slices in fixed order and runs the chain once.
        // The partials cross SMs, whose L1 caches are not coherent: both sides
        // use GPU-scope relaxed accesses, published by the counter's release.
        for(int i=int(threadIdx.x);i<TM*TN;i+=128) {
          auto* address=p.partial+(std::size_t(tile)*SplitK+slice)*TM*TN+i;
          asm volatile("st.relaxed.gpu.global.f32 [%0], %1;"::"l"(address),
              "f"(tile_values[i]):"memory");
        }
        codegen::executor::ComputeSync();
        auto* flag=reinterpret_cast<int*>(scratch+kScratchBytes-16);
        if(threadIdx.x==0) {
          unsigned long long old;
          asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;":"=l"(old)
              :"l"(p.counters+tile):"memory");
          *flag=int(old+1)==SplitK;
          if(*flag)p.counters[tile]=0;
        }
        codegen::executor::ComputeSync();
        if(*flag) {
          for(int i=int(threadIdx.x);i<TM*TN;i+=128) {
            float value=0;
            #pragma unroll
            for(int s=0;s<SplitK;++s) {
              auto const* address=p.partial+(std::size_t(tile)*SplitK+s)*TM*TN+i;
              float part;
              asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(part)
                  :"l"(address):"memory");
              value+=part;
            }
            tile_values[i]=value;
          }
          codegen::executor::ComputeSync();
          Chain::Finish(tile_values,rows,n_tile*TN,width>0?width:0,epilogue,output,
              p.output_stride);
        }
      }
    }
    if(threadIdx.x==0 && tasks)
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
  }

  __device__ static __forceinline__ void Run(Sm90ForwardGemmOperands const& p,char* shared,
      Sm90DenseRingState& state) {
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*TM*TK;
    auto* scratch=shared+kLayout.scratch;
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    int tasks=kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,tasks,phase);
    else Produce(p,state,a,b,tasks,phase);
    __syncthreads();
  }
};

}  // namespace tilemega::backend
