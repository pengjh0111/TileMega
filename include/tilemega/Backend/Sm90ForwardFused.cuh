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
  // The depthwise weight is the plan's KRSC packing [C][3][3][pack], whose
  // inner axis is the convolution's single input channel padded to `pack`.
  cutlass::bfloat16_t const* depthwise;
  int depthwise_pack;
  cutlass::bfloat16_t const* pointwise; // K-major tile pages
  cutlass::bfloat16_t const* expand;    // IR only: the expand projection
  cutlass::bfloat16_t* output;
  float* partial;                       // split-C_exp partials
  std::uint64_t* counters;
  Sm90ForwardEpilogueOperands epilogue;
  // The depthwise's own per-channel vectors (its folded BN), which its
  // register-resident chain reads.
  float const* dw_vectors[4];
  int width,height,channels,out_channels,output_width,output_height,stride;
  // The output row's own pitch, which is the buffer's padded channel block
  // and is wider than `out_channels` whenever the plan pads them.
  int output_stride;
  int padded_channels;   // the input's padded channel extent, which the pages cover
};

// TW x TH output pixels; CB channels of depthwise per block; TN output
// channels per task. The pointwise contracts the depthwise result, so the
// GEMM's M is the tile's pixel count padded to 64.
template<int TW,int TH,int CB,int TN,int Stride,int Slots,class Chain,int Pack=8,
         class DwChain=Sm90ForwardElementChain<>>
struct Sm90ForwardDwPw {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(TW*TH<=64 && CB%64==0 && (Stride==1 || Stride==2));
  static_assert(TN==32 || TN==64 || TN==128 || TN==256);
  static constexpr int kThreads=160,kRows=64;
  static constexpr int kHaloW=(TW-1)*Stride+3,kHaloH=(TH-1)*Stride+3;
  static constexpr int kHaloElements=kHaloW*kHaloH*CB;
  using Gmma=Sm90ForwardGmma<arch::Sm90,kRows,TN,64>;
  // One slot holds this C_in block's halo and its pointwise pages: both are
  // WGMMA operands, so the block stays live until its MMA commits. The
  // depthwise result lands in the A panel in the same 128-byte swizzle the
  // pages carry, because WGMMA reads both from shared memory.
  static constexpr int kPageElements=TN*CB;
  static constexpr int kSlotBytes=2*(kHaloElements+kPageElements+CB*9*Pack);
  static constexpr int kPanelBytes=2*kRows*CB;
  // This block's depthwise filters, in the plan's KRSC packing: the channels
  // of one C block, each with 3 x 3 x Pack taps.
  static_assert(Pack>=1 && Pack<=8);
  static constexpr int kFilterElements=CB*9*Pack;
  static constexpr int kWeightBytes=2*kFilterElements;
  static constexpr int kScratchBytes=4*kRows*TN;
  // Every WGMMA operand tile must start at a multiple of the 128-byte
  // swizzle's 1 KB period: a merely 128-byte-aligned base shifts the XOR
  // pattern and the MMA reads the wrong k columns for some rows.
  static constexpr int kAligned=1024;
  static constexpr int kPagesOffset=
      (1024+Slots*2*kHaloElements+kAligned-1)/kAligned*kAligned;
  static constexpr int kFiltersOffset=
      (kPagesOffset+Slots*2*kPageElements+kAligned-1)/kAligned*kAligned;
  static constexpr int kPanelOffset=
      (kFiltersOffset+Slots*2*kFilterElements+kAligned-1)/kAligned*kAligned;
  static constexpr Sm90DenseLayout kLayout{1024,kPagesOffset,
      kPanelOffset+kPanelBytes,
      (kPanelOffset+kPanelBytes+kScratchBytes+kAligned-1)/kAligned*kAligned};
  static constexpr int kSharedBytes=kLayout.end;

  // Byte offset of (row, column) inside one 64-column SW128 panel.
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }

  // `tile_w` is this tile's own width, which is narrower than TW at the right
  // edge of an image whose width is not a whole number of tiles; a task's rows
  // are its tile's pixels in that tile's own raster order.
  __device__ static __forceinline__ void Depthwise(Element const* halo,Element const* weights,
      Element* panel,int rows,int tile_w,int channels,int block,
      float const* const (&dw_vectors)[4]) {
    auto* base=reinterpret_cast<char*>(panel);
    for(int i=int(threadIdx.x);i<kRows*CB;i+=128) {
      int channel=i%CB,pixel=i/CB,x=tile_w>0?pixel%tile_w:0,y=tile_w>0?pixel/tile_w:0;
      float sum=0;
      // Channels past the image's own are the plan's zero padding, and their
      // weights are outside the packed KRSC extent.
      if(pixel<rows && block*CB+channel<channels) {
        auto const* filter=weights+std::size_t(channel)*9*Pack;
        #pragma unroll
        for(int r=0;r<3;++r)
          #pragma unroll
          for(int s=0;s<3;++s) {
            int hx=x*Stride+s,hy=y*Stride+r;
            sum+=float(halo[(hy*kHaloW+hx)*CB+channel])*float(filter[(r*3+s)*Pack]);
          }
      }
      // The plan's depthwise epilogue (its folded BN and activation) applies
      // here, before the result becomes the pointwise's A. The A panel of k
      // block (channel / 64) is swizzled like any WGMMA operand; the rows
      // past this tile's pixels contribute zero.
      if(pixel<rows && block*CB+channel<channels)
        sum=DwChain::Apply(sum,block*CB+channel,dw_vectors);
      auto* target=base+(channel/64)*kRows*128+Swizzled(pixel,channel);
      *reinterpret_cast<Element*>(target)=Element(sum);
    }
    codegen::executor::ComputeSync();
  }

  __device__ static __forceinline__ void Run(Sm90ForwardFusedOperands const& p,char* shared,
      Sm90DenseRingState& state) {
    using namespace cute;
    auto* halo=reinterpret_cast<Element*>(shared+1024);
    auto* pages=reinterpret_cast<Element*>(shared+kPagesOffset);
    auto* filters=reinterpret_cast<Element*>(shared+kFiltersOffset);
    auto* panel=reinterpret_cast<Element*>(shared+kPanelOffset);
    auto* scratch=shared+kLayout.scratch;
    int tiles_w=(p.output_width+TW-1)/TW,tiles_h=(p.output_height+TH-1)/TH;
    // The weight is packed into whole TN-wide tiles, so a projection narrower
    // than TN still has one tile and the epilogue stores only its own columns.
    int splits=(p.out_channels+TN-1)/TN;
    int total=tiles_w*tiles_h*splits;
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    int blocks=(p.padded_channels+CB-1)/CB;
    if(threadIdx.x>=128) {
      Sm90DenseRoles::Producer();
      if(threadIdx.x==128) {
        Async::ProxyAsyncGlobalFence();
        for(int local=0;local<tasks;++local) {
          int task=int(blockIdx.x)+local*int(gridDim.x);
          int tile=task/splits,tile_x=(tile%tiles_w)*TW,tile_y=(tile/tiles_w)*TH;
          for(int block=0;block<blocks;++block) {
            int slot=(local*blocks+block)%Slots;
            int split=task%splits;
            // This block's own channels: the last block of a layer whose C is
            // not a whole number of blocks carries fewer filters.
            int channels=p.channels-block*CB;
            if(channels>CB)channels=CB;
            if(channels<0)channels=0;
            int filter_bytes=2*channels*9*Pack;
            Async::Wait(state.empty+slot,(phase>>slot)&1);
            Async::ExpectTx(state.full+slot,2*(kHaloElements+kPageElements)+filter_bytes);
            Async::Tensor4D(halo+slot*kHaloElements,p.activation,block*CB,
                tile_x*Stride-1,tile_y*Stride-1,0,state.full+slot);
            // This block's pointwise pages: TN x 64 pages of the K panels the
            // block covers, contiguous per (output tile, k panel).
            Async::Bulk(pages+slot*kPageElements,
                p.pointwise+(std::size_t(split)*(p.padded_channels/64)+block*(CB/64))*TN*64,
                2*kPageElements,state.full+slot);
            // This block's filters are a contiguous KRSC slice.
            if(filter_bytes)
              Async::Bulk(filters+slot*kFilterElements,
                  p.depthwise+std::size_t(block)*CB*9*Pack,filter_bytes,state.full+slot);
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
        int tile_w=min(TW,max(0,p.output_width-tile_x));
        int tile_h=min(TH,max(0,p.output_height-tile_y));
        int rows=tile_w*tile_h;
        auto accum=partition_fragment_C(typename Gmma::Mma{},Shape<Int<kRows>,Int<TN>>{});
        clear(accum);
        // The depthwise runs on CUDA cores between the MMAs, so the
        // accumulator is fenced around that register work; without the fence
        // the compiler may reuse the accumulator's registers.
        Gmma::Gmma::Fence(accum);
        for(int block=0;block<blocks;++block) {
          int slot=(local*blocks+block)%Slots;
          Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
          Depthwise(halo+slot*kHaloElements,filters+slot*kFilterElements,panel,rows,
              tile_w,p.channels,block,p.dw_vectors);
          // The panel is a generic-store producer for WGMMA's shared reads.
          Async::FenceProxyAsyncShared();
          codegen::executor::ComputeSync();
          #pragma unroll
          for(int k=0;k<CB/64;++k)
            Gmma::Accumulate(pages+slot*kPageElements+k*TN*64,panel+k*kRows*64,accum);
          Sm90Gmma<arch::Sm90>::template Wait<0>();
          Gmma::Gmma::Fence(accum);
          codegen::executor::ComputeSync();
          Async::Arrive(state.empty+slot);
        }
        float* values=Gmma::Materialize(accum,scratch);
        // The task's rows are the pixels of its TW x TH tile, so the store
        // walks the tile inside the image.
        auto epilogue=p.epilogue;
        epilogue.tile_width=tile_w;epilogue.image_width=p.output_width;
        // The residual is this tile's own pixels, like the output.
        if(epilogue.residual)
          epilogue.residual+=(std::size_t(tile_y)*p.output_width+tile_x)*epilogue.residual_stride;
        int width=p.out_channels-split*Chain::kWidth;
        if(width>Chain::kWidth)width=Chain::kWidth;
        Chain::Finish(values,rows,split*TN,width>0?width:0,epilogue,
            p.output+(std::size_t(tile_y)*p.output_width+tile_x)*p.output_stride,
            p.output_stride);
      }
      if(threadIdx.x==0 && tasks)
        for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
