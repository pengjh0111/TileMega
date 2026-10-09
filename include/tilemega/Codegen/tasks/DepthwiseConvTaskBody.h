// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/DmEpilogueValue.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cstddef>

namespace tilemega::codegen {
struct DepthwiseConvOperands {
  cutlass::bfloat16_t const* input=nullptr;
  cutlass::bfloat16_t const* weight=nullptr;
  cutlass::bfloat16_t* output=nullptr;
  ConvDesc convolution{};
  DmBufferLayout input_layout{},output_layout{};
  // Depthwise KRSC has one logical input channel per output filter.
  unsigned weight_channel_pitch=8;
  DmBufferView buffers{};
  DmEpilogueChain chain{};
  float* channel_partials=nullptr;
  // [image, output row band, channel], excluding unwritten alignment lanes.
  DmBufferLayout partial_layout{};
};

template<class Arch,int RowBand=1,int ChannelTile=64,
         class Program=DmEpilogueProgram<>>
struct DepthwiseConvTaskBody {
  static_assert(arch::Caps<Arch>::kCpAsync && RowBand>0);
  static_assert(ChannelTile==32 || ChannelTile==64 || ChannelTile==128 || ChannelTile==256);
  using E=cutlass::bfloat16_t;
  static constexpr int kThreads=128;
  static constexpr bool kGated=backend::DmEpilogueWalk<Program>::kGates!=0;
  static_assert(backend::DmEpilogueWalk<Program>::kGates<=1);
  static constexpr unsigned kBands=kGated?2:1;
  __host__ __device__ static unsigned InputRows(ConvDesc const& c) {
    return (RowBand-1)*c.stride_h+(c.r-1)*c.dilation_h+1;
  }
  __host__ __device__ static std::size_t SharedBytes(DepthwiseConvOperands const& p) {
    auto input=std::size_t(InputRows(p.convolution))*p.input_layout.physical[2]*ChannelTile*kBands*sizeof(E);
    return input>kThreads*8*sizeof(float)?input:kThreads*8*sizeof(float);
  }
  __host__ __device__ static unsigned Count(DepthwiseConvOperands const& p) {
    auto const& c=p.convolution;auto channels=c.c/kBands;
    return c.n*((c.p+RowBand-1)/RowBand)*((channels+ChannelTile-1)/ChannelTile);
  }
  struct Visitor {
    float first,second;
    unsigned channel,partner;
    DepthwiseConvOperands const& p;
    template<unsigned Position,class Step,bool Gated>
    __device__ void Apply() {
      using K=DmEpilogueKind;
      static_assert(Step::kKind==K::kBias || Step::kKind==K::kScale ||
                    Step::kKind==K::kActivation || Step::kKind==K::kGatePair);
      static_assert(Step::kKind!=K::kGatePair || Step::kGate==DmGatePair::kSimpleGate);
      auto const& op=p.chain.operations[Position];
      if(op.kind!=Step::kKind || op.activation!=Step::kActivation || op.gate!=Step::kGate ||
         op.unit!=Step::kUnit || op.input_rounding!=Step::kInputRounding ||
         op.output_rounding!=Step::kOutputRounding) {asm volatile("trap;");return;}
      backend::DmEpilogueInputs a,b;
      if constexpr(Step::kKind==K::kBias || Step::kKind==K::kScale) {
        auto id=op.parameter[0];
        if(id>=p.buffers.count || !p.buffers.dtypes || p.buffers.dtypes[id]!=1 ||
           !p.buffers.data || !p.buffers.data[id]) {asm volatile("trap;");return;}
        auto value=static_cast<float const*>(p.buffers.data[id]);
        if constexpr(Step::kKind==K::kBias) {
          a.bias=value[channel];if constexpr(kGated && !Gated)b.bias=value[partner];
        }else {
          a.scale=value[channel];if constexpr(kGated && !Gated)b.scale=value[partner];
        }
      }
      first=backend::DmEpilogueValue<Arch,Step>::Apply(first,a,second);
      if constexpr(kGated && !Gated && Step::kKind!=K::kGatePair) {
        second=backend::DmEpilogueValue<Arch,Step>::Apply(second,b);
      }
    }
  };
  __device__ static void Run(DepthwiseConvOperands const& p,unsigned task,char* scratch) {
    using namespace executor;
    auto const& c=p.convolution;auto const& in=p.input_layout;auto const& out=p.output_layout;
    unsigned channels=c.c/kBands,blocks=(channels+ChannelTile-1)/ChannelTile;
    unsigned row_bands=(c.p+RowBand-1)/RowBand;
    unsigned image=task/(blocks*row_bands),band=(task/blocks)%row_bands;
    unsigned first_channel=(task%blocks)*ChannelTile;
    if(!p.input || !p.weight || !p.output || !scratch || !blocks || image>=c.n ||
       in.rank!=4 || out.rank!=4 || in.kind!=DmLayout::kNHWC || out.kind!=DmLayout::kNHWC ||
       in.strides[3]!=1 || in.strides[2]%8 || in.physical[3]%8 ||
       p.chain.count!=Program::kCount || p.chain.side_count ||
       p.chain.store_rounding!=DmRounding::kBF16 || (kGated && c.c%16)) {
      asm volatile("trap;");return;
    }
    unsigned input_rows=InputRows(c),input_width=in.physical[2];
    auto* tile=reinterpret_cast<E*>(scratch);
    auto tile_elements=std::size_t(input_rows)*input_width*ChannelTile*kBands;
    auto* sums=reinterpret_cast<float*>(tile);
    long first_y=long(band)*RowBand*c.stride_h-c.pad_h+in.halo_top;
    // Complete channel vectors are copied for each strip; out-of-image tail
    // strips are zero-filled without forming an invalid global pointer.
    for(std::size_t vector=ComputeThread();vector<tile_elements/8;vector+=kThreads) {
      auto element=vector*8;
      unsigned part=element/(std::size_t(input_rows)*input_width*ChannelTile);
      auto local=element%(std::size_t(input_rows)*input_width*ChannelTile);
      unsigned x=(local/ChannelTile)%input_width,y=local/(ChannelTile*input_width);
      unsigned channel=first_channel+local%ChannelTile+part*channels;
      long source_y=first_y+y;
      bool valid=source_y>=0 && source_y<long(in.physical[1]) && channel<in.physical[3] &&
          (part==0?channel<channels:channel<c.c);
      unsigned count=valid?min(8u,(part==0?channels:c.c)-channel):0;
      auto* source=valid?p.input+image*in.strides[0]+source_y*in.strides[1]+x*in.strides[2]+channel:p.input;
      Async<Arch>::Copy16Bytes(tile+element,source,count*sizeof(E));
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");ComputeSync();
    constexpr unsigned groups=ChannelTile/8,spatial_lanes=kThreads/groups;
    unsigned vector=ComputeThread()%groups,spatial=ComputeThread()/groups;
    float partial[8]={};
    for(unsigned pixel=spatial;pixel<RowBand*c.q;pixel+=spatial_lanes) {
      unsigned y=pixel/c.q,x=pixel%c.q,output_y=band*RowBand+y;
      if(output_y>=c.p)continue;
      long input_x=long(x)*c.stride_w-c.pad_w+in.halo_left;
      if(input_x<0 || input_x+long((c.s-1)*c.dilation_w)>=long(input_width)) {
        asm volatile("trap;");return;
      }
      float accum[kBands][8]={};
      for(unsigned r=0;r<c.r;++r)for(unsigned s=0;s<c.s;++s) {
        auto at=((y*c.stride_h+r*c.dilation_h)*input_width+input_x+s*c.dilation_w)*ChannelTile+vector*8;
        #pragma unroll
        for(unsigned part=0;part<kBands;++part) {
          #pragma unroll
          for(unsigned lane=0;lane<8;++lane) {
            unsigned channel=first_channel+vector*8+lane+part*channels;
            if(first_channel+vector*8+lane<channels) {
              float a=float(tile[part*(tile_elements/kBands)+at+lane]);
              float b=float(p.weight[(std::size_t(channel)*c.r*c.s+r*c.s+s)*p.weight_channel_pitch]);
              accum[part][lane]=fmaf(a,b,accum[part][lane]);
            }
          }
        }
      }
      #pragma unroll
      for(unsigned lane=0;lane<8;++lane) {
        unsigned channel=first_channel+vector*8+lane;
        if(channel>=channels)continue;
        Visitor value{accum[0][lane],kGated?accum[kBands-1][lane]:0,channel,channel+channels,p};
        backend::DmEpilogueWalk<Program>::Run(value);
        // Convolution/chain remain FP32 until the chain's declared rounding
        // points; pooling side sums observe the final stored BF16 value.
        E stored(value.first);
        auto at=image*out.strides[0]+(output_y+out.halo_top)*out.strides[1]+(x+out.halo_left)*out.strides[2]+channel;
        p.output[at]=stored;partial[lane]+=float(stored);
      }
    }
    if(p.channel_partials) {
      // The strip's last read precedes reuse of the same storage for sums.
      ComputeSync();
      #pragma unroll
      for(unsigned lane=0;lane<8;++lane)sums[ComputeThread()*8+lane]=partial[lane];
      ComputeSync();
      if(unsigned(ComputeThread())<groups) {
        #pragma unroll
        for(unsigned lane=0;lane<8;++lane) {
          unsigned channel=first_channel+ComputeThread()*8+lane;
          if(channel>=channels)continue;
          float sum=0;
          for(unsigned source=ComputeThread();source<kThreads;source+=groups)sum+=sums[source*8+lane];
          auto const& l=p.partial_layout;
          p.channel_partials[image*l.strides[0]+band*l.strides[1]+channel*l.strides[2]]=sum;
        }
      }
    }
    ComputeSync();
  }
};
} // namespace tilemega::codegen
