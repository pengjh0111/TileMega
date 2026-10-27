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
  float* partial;                  // split-K: [tiles][TM][TN] fp32
  std::uint64_t* counters;         // split-K: one per output tile
  Sm90ForwardEpilogueOperands epilogue;
  int output_stride;               // channels of the output row
  // im2col: the spatial geometry the map was built for.
  int width,height,pad_w,pad_h,stride_w,stride_h,kernel_w,kernel_h;
};

template<int TM,int TN,int TK,int Slots,int SplitK,Sm90ForwardAMode AMode,int N,int K,class Chain>
struct Sm90ForwardGemm {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Gmma=Sm90ForwardGmma<arch::Sm90,TM,TN,TK>;
  static_assert(TM==64 || TM==128);
  static_assert(TN==32 || TN==64 || TN==128 || TN==256);
  static_assert(TK==64 && Slots>=2 && Slots<=8);
  static_assert(SplitK>=1 && K%(SplitK*TK)==0,"split-K divides the contraction into whole k tiles");
  static_assert(N%TN==0,"N is padded to a multiple of TN when the plan is built");
  static constexpr int kThreads=160,kIterations=K/(SplitK*TK);
  static constexpr int kTiles=N/TN;
  static constexpr int kSlotBytes=2*TK*(TM+TN);
  static constexpr int kScratchBytes=4*TM*TN;
  // 1 KB header, the ring, then the fp32 accumulator tile the chain rewrites.
  static constexpr Sm90DenseLayout kLayout{1024,1024+2*Slots*TK*TM,
      1024+Slots*kSlotBytes,(1024+Slots*kSlotBytes+kScratchBytes+1023)/1024*1024};
  static constexpr int kSharedBytes=kLayout.end;

  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return (col/64)*TM*128+row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }

  // Task t of a split-K stage: output tile t % kTiles, k slice t / kTiles.
  __device__ static __forceinline__ int Tile(int task) {return task%kTiles;}
  __device__ static __forceinline__ int Slice(int task) {return task/kTiles;}

  __device__ static __noinline__ void Produce(Sm90ForwardGemmOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase) {
    Sm90DenseRoles::Producer();
    if(threadIdx.x!=128) {__syncwarp();return;}
    Async::ProxyAsyncGlobalFence();
    for(int sequence=0;sequence<tasks*kIterations;++sequence) {
      int slot=sequence%Slots,local=sequence/kIterations,iteration=sequence%kIterations;
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int tile=Tile(task),slice=Slice(task),k=(slice*kIterations+iteration)*TK;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      Async::ExpectTx(state.full+slot,kSlotBytes);
      // Weights: the tile's K-major pages are contiguous per (tile, k).
      Async::Bulk(b+slot*TN*TK,p.weight+(std::size_t(tile)*(K/TK)+k/TK)*TN*TK,2*TN*TK,
          state.full+slot);
      if constexpr(AMode==Sm90ForwardAMode::kDense)
        Async::Tensor2D(a+slot*TM*TK,p.activation,k,0,state.full+slot);
      else {
        // im2col: the box walks the (r, s) offsets of one k panel; the map's
        // base coordinates are the tile's first output pixel.
        int channels=p.kernel_w*p.kernel_h;
        int offset=(k/TK)%channels,r=offset/p.kernel_w,s=offset%p.kernel_w;
        Async::TensorIm2col4D(a+slot*TM*TK,p.activation,k/TK/channels*TK,
            -p.pad_w+s,-p.pad_h+r,0,std::uint16_t(s),std::uint16_t(r),state.full+slot);
      }
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
      int tile=Tile(task),slice=Slice(task);
      auto accum=partition_fragment_C(typename Gmma::Mma{},Shape<Int<TM>,Int<TN>>{});
      clear(accum);
      for(int iteration=0;iteration<kIterations;++iteration) {
        int slot=(local*kIterations+iteration)%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        Gmma::Accumulate(b+slot*TN*TK,a+slot*TM*TK,accum);
        // WGMMA reads the slot asynchronously: its commit must land before the
        // slot is released for the producer's next async write.
        Sm90Gmma<arch::Sm90>::template Wait<0>();
        Async::FenceProxyAsyncShared();
        Async::Arrive(state.empty+slot);
      }
      float* tile_values=Gmma::Materialize(accum,scratch);
      if constexpr(SplitK==1) {
        Chain::Finish(tile_values,TM,tile*TN,Chain::kWidth,p.epilogue,p.output,p.output_stride);
      }else {
        // Each slice stores its partial; the last arriver of the tile sums the
        // slices in fixed order and runs the chain once.
        for(int i=int(threadIdx.x);i<TM*TN;i+=128)
          p.partial[(std::size_t(tile)*SplitK+slice)*TM*TN+i]=tile_values[i];
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
            float sum=0;
            #pragma unroll
            for(int s=0;s<SplitK;++s)sum+=p.partial[(std::size_t(tile)*SplitK+s)*TM*TN+i];
            tile_values[i]=sum;
          }
          codegen::executor::ComputeSync();
          Chain::Finish(tile_values,TM,tile*TN,Chain::kWidth,p.epilogue,p.output,p.output_stride);
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
    int total=kTiles*SplitK;
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,tasks,phase);
    else Produce(p,state,a,b,tasks,phase);
    __syncthreads();
  }
};

}  // namespace tilemega::backend
