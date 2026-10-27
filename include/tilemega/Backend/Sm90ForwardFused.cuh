// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardDepthwise.cuh>
#include <tilemega/Backend/Sm90ForwardGemm.cuh>

// Fused DNN forward bodies (prompt §10 N-1 item 4).
//
// DWPW (MBV1): one task computes a spatial tile's depthwise over all C_in,
// streaming C_in in blocks into shared memory, and consumes it immediately as
// the pointwise GEMM's A operand. C_out <= 256 keeps one task over all N;
// wider layers split N and recompute the depthwise, which the plan limits to
// four blocks (beyond that the plan emits DW + G as separate stages).
//
// IR (MBV2 inverted residual): a task owns an output spatial tile with halo
// and walks C_exp in blocks of 64: expand (WGMMA) -> depthwise 3x3 (CUDA
// cores) -> the project partial accumulates across blocks in registers, then
// bias + residual and one write. Tiles below G/2 split C_exp across tasks and
// reduce the project partial through a last arriver.
namespace tilemega::backend {

struct Sm90ForwardFusedOperands {
  CUtensorMap const* activation;        // 4D NHWC tiled map with halo
  cutlass::bfloat16_t const* depthwise; // [3][3][C]
  cutlass::bfloat16_t const* pointwise; // K-major tile pages
  cutlass::bfloat16_t const* expand;    // IR only: the expand projection
  cutlass::bfloat16_t* output;
  float* partial;                       // split-C_exp partials
  std::uint64_t* counters;
  Sm90ForwardEpilogueOperands epilogue;
  int width,height,channels,out_channels,output_width,output_height,stride;
};

// TW x TH output pixels; CB channels of depthwise per block; TN output
// channels per task. The pointwise contracts the depthwise result, so the
// GEMM's M is the tile's pixel count padded to 64.
template<int TW,int TH,int CB,int TN,int Stride,int Slots,class Chain>
struct Sm90ForwardDwPw {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(TW*TH<=64 && CB%64==0 && (Stride==1 || Stride==2));
  static_assert(TN==32 || TN==64 || TN==128 || TN==256);
  static constexpr int kThreads=160,kRows=64;
  static constexpr int kHaloW=(TW-1)*Stride+3,kHaloH=(TH-1)*Stride+3;
  static constexpr int kHaloElements=kHaloW*kHaloH*CB;
  using Gmma=Sm90ForwardGmma<arch::Sm90,kRows,TN,64>;
  // One slot holds a halo block; the depthwise result lands in the A panel
  // the WGMMA reads, so a block is live until its MMA commits.
  static constexpr int kSlotBytes=2*kHaloElements;
  static constexpr int kPanelBytes=2*kRows*CB;
  static constexpr int kWeightBytes=2*9*CB;
  static constexpr int kScratchBytes=4*kRows*TN;
  static constexpr Sm90DenseLayout kLayout{1024,1024,
      1024+Slots*kSlotBytes+kPanelBytes+kWeightBytes,
      (1024+Slots*kSlotBytes+kPanelBytes+kWeightBytes+kScratchBytes+1023)/1024*1024};
  static constexpr int kSharedBytes=kLayout.end;

  __device__ static __forceinline__ void Depthwise(Element const* halo,Element const* weights,
      Element* panel,int rows) {
    for(int i=int(threadIdx.x);i<rows*CB;i+=128) {
      int channel=i%CB,pixel=i/CB,x=pixel%TW,y=pixel/TW;
      float sum=0;
      #pragma unroll
      for(int r=0;r<3;++r)
        #pragma unroll
        for(int s=0;s<3;++s) {
          int hx=x*Stride+s,hy=y*Stride+r;
          sum+=float(halo[(hy*kHaloW+hx)*CB+channel])*float(weights[(r*3+s)*CB+channel]);
        }
      // ReLU6 is MBV1's depthwise activation; the pointwise chain owns the rest.
      panel[pixel*CB+channel]=Element(fminf(fmaxf(sum,0.0f),6.0f));
    }
    codegen::executor::ComputeSync();
  }

  __device__ static __forceinline__ void Run(Sm90ForwardFusedOperands const& p,char* shared,
      Sm90DenseRingState& state) {
    using namespace cute;
    auto* halo=reinterpret_cast<Element*>(shared+1024);
    auto* panel=halo+Slots*kHaloElements;
    auto* weights=panel+kRows*CB;
    auto* scratch=shared+kLayout.scratch;
    int tiles_w=(p.output_width+TW-1)/TW,tiles_h=(p.output_height+TH-1)/TH;
    int splits=p.out_channels/TN;
    int total=tiles_w*tiles_h*splits;
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    int blocks=p.channels/CB;
    if(threadIdx.x>=128) {
      Sm90DenseRoles::Producer();
      if(threadIdx.x==128) {
        Async::ProxyAsyncGlobalFence();
        for(int local=0;local<tasks;++local) {
          int task=int(blockIdx.x)+local*int(gridDim.x);
          int tile=task/splits,tile_x=(tile%tiles_w)*TW,tile_y=(tile/tiles_w)*TH;
          for(int block=0;block<blocks;++block) {
            int slot=(local*blocks+block)%Slots;
            Async::Wait(state.empty+slot,(phase>>slot)&1);
            Async::ExpectTx(state.full+slot,kSlotBytes+(block?0:kWeightBytes));
            Async::Tensor4D(halo+slot*kHaloElements,p.activation,block*CB,
                tile_x*Stride-1,tile_y*Stride-1,0,state.full+slot);
            if(!block)Async::Bulk(weights,p.depthwise,kWeightBytes,state.full+slot);
            phase^=1u<<slot;
          }
        }
        if(tasks)for(int slot=0;slot<Slots;++slot)state.producer_generations[slot]=(phase>>slot)&1;
      }
      __syncwarp();
    }else {
      Sm90DenseRoles::Consumer();
      for(int local=0;local<tasks;++local) {
        int task=int(blockIdx.x)+local*int(gridDim.x);
        int tile=task/splits,split=task%splits;
        int tile_x=(tile%tiles_w)*TW,tile_y=(tile/tiles_w)*TH;
        int rows=min(TW*TH,max(0,(p.output_width-tile_x))*max(0,(p.output_height-tile_y)));
        auto accum=partition_fragment_C(typename Gmma::Mma{},Shape<Int<kRows>,Int<TN>>{});
        clear(accum);
        for(int block=0;block<blocks;++block) {
          int slot=(local*blocks+block)%Slots;
          Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
          Depthwise(halo+slot*kHaloElements,weights,panel,rows);
          // The panel is a generic-store producer for WGMMA's shared reads.
          Async::FenceProxyAsyncShared();
          codegen::executor::ComputeSync();
          #pragma unroll
          for(int k=0;k<CB/64;++k)
            Gmma::Accumulate(p.pointwise+((std::size_t(split)*(p.channels/64))+block*(CB/64)+k)*TN*64,
                panel+k*kRows*64,accum);
          Sm90Gmma<arch::Sm90>::template Wait<0>();
          codegen::executor::ComputeSync();
          Async::Arrive(state.empty+slot);
        }
        float* values=Gmma::Materialize(accum,scratch);
        Chain::Finish(values,rows,split*TN,Chain::kWidth,p.epilogue,
            p.output+(std::size_t(tile_y)*p.output_width+tile_x)*p.out_channels,p.out_channels);
      }
      if(threadIdx.x==0 && tasks)
        for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
