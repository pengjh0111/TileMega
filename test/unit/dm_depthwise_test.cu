// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/DepthwiseConvTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>
#include <string>
#ifndef TILEMEGA_DEPTHWISE_TEST_TMA
#define TILEMEGA_DEPTHWISE_TEST_TMA 0
#endif
using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,std::conditional_t<TILEMEGA_DEPTHWISE_TEST_TMA,arch::Sm90,arch::Sm89>,arch::CurrentArch>;
template<DmEpilogueKind Kind,DmActivation Activation=DmActivation::kRelu>
using Step=DmEpilogueStep<Kind,Activation,DmGatePair::kSimpleGate,16,
    DmRounding::kFP32,DmRounding::kFP32,DmWriteKind::kDense,1>;
using Bias=Step<DmEpilogueKind::kBias>;
using Relu6=Step<DmEpilogueKind::kActivation,DmActivation::kRelu6>;
using Gate=Step<DmEpilogueKind::kGatePair>;
using Plain=DmEpilogueProgram<>;
using Biased=DmEpilogueProgram<Bias>;
using Activated=DmEpilogueProgram<Bias,Relu6>;
using Gated=DmEpilogueProgram<Bias,Gate>;
template<class T>T* Managed(std::size_t count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(T))==cudaSuccess);return p;
}
DmBufferLayout Image(unsigned batch,unsigned h,unsigned w,unsigned c,unsigned halo) {
  DmBufferLayout l;l.kind=DmLayout::kNHWC;l.rank=4;
  l.logical[0]=l.physical[0]=batch;l.logical[1]=h;l.logical[2]=w;l.logical[3]=c;
  l.physical[1]=h+2*halo;l.physical[2]=w+2*halo;l.physical[3]=(c+7)/8*8;
  l.strides[3]=1;l.strides[2]=l.physical[3];l.strides[1]=l.physical[2]*l.strides[2];
  l.strides[0]=l.physical[1]*l.strides[1];l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=halo;return l;
}
std::uint64_t Offset(DmBufferLayout const& l,unsigned b,unsigned y,unsigned x,unsigned c) {
  return b*l.strides[0]+(y+l.halo_top)*l.strides[1]+(x+l.halo_left)*l.strides[2]+c;
}
template<int Rows,int Channels,class Program>
__global__ void Run(DepthwiseConvOperands p) {
#if TILEMEGA_DEPTHWISE_TEST_TMA
  extern __shared__ __align__(128) char scratch[];
#else
  extern __shared__ __align__(16) char scratch[];
#endif
  DepthwiseConvTaskBody<Arch,Rows,Channels,Program,bool(TILEMEGA_DEPTHWISE_TEST_TMA)>::Run(p,blockIdx.x,scratch);
}
#if TILEMEGA_DEPTHWISE_TEST_TMA
template<int Rows,int Channels,class Program>
__global__ void Legacy(DepthwiseConvOperands p) {
  extern __shared__ __align__(128) char scratch[];
  DepthwiseConvTaskBody<Arch,Rows,Channels,Program,false>::Run(p,blockIdx.x,scratch);
}
#endif
template<int Rows,int Channels,class Program>
unsigned Check(unsigned input_channels,unsigned h,unsigned w,unsigned stride,unsigned dilation,unsigned batch) {
  constexpr bool gated=DepthwiseConvTaskBody<Arch,Rows,Channels,Program>::kGated;
  auto channels=input_channels/(gated?2:1);
  ConvDesc c;c.n=batch;c.h=h;c.w=w;c.c=c.k=input_channels;c.r=c.s=3;
  c.pad_h=c.pad_w=dilation;c.dilation_h=c.dilation_w=dilation;
  c.stride_h=c.stride_w=stride;c.p=(h+stride-1)/stride;c.q=(w+stride-1)/stride;
  auto in=Image(batch,h,w,input_channels,dilation),out=Image(batch,c.p,c.q,channels,1);
  auto input_count=batch*in.strides[0],output_count=batch*out.strides[0];
  auto* input=Managed<E>(input_count);auto* output=Managed<E>(output_count+32);
  auto* weight=Managed<E>(input_channels*9*8);auto* bias=Managed<float>(input_channels);
  for(unsigned i=0;i<input_count;++i)input[i]=E(0);
  for(unsigned n=0;n<batch;++n)for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
    for(unsigned ch=0;ch<input_channels;++ch)
      input[Offset(in,n,y,x,ch)]=E(float(int((n*31+y*17+x*13+ch*7)%83)-41)*.0625f);
  for(unsigned ch=0;ch<input_channels;++ch) {
    bias[ch]=float(int(ch%17)-8)*.125f;
    for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s)for(unsigned k=0;k<8;++k)
      weight[(ch*9+r*3+s)*8+k]=E(k?0:float(int((ch*11+r*5+s*3)%19)-9)*.03125f);
  }
  void** pointers=Managed<void*>(1);pointers[0]=bias;
  unsigned* dtypes=Managed<unsigned>(1);dtypes[0]=1;
  unsigned bands=(c.p+Rows-1)/Rows;
  DmBufferLayout parts;parts.rank=3;parts.logical[0]=batch;parts.logical[1]=bands;parts.logical[2]=channels;
  parts.strides[2]=1;parts.strides[1]=channels+3;parts.strides[0]=bands*parts.strides[1];
  auto* partial=Managed<float>(batch*parts.strides[0]+16);
  DepthwiseConvOperands p{input,weight,output+16,c,in,out,8,{pointers,nullptr,dtypes,1}};
  p.chain.count=Program::kCount;p.channel_partials=partial+8;p.partial_layout=parts;
  if constexpr(Program::kCount) {
    auto& op=p.chain.operations[0];op.kind=DmEpilogueKind::kBias;op.parameter[0]=0;
    op.gate=DmGatePair::kSimpleGate;
  }
  if constexpr(Program::kCount==2) {
    auto& op=p.chain.operations[1];op.kind=gated?DmEpilogueKind::kGatePair:DmEpilogueKind::kActivation;
    op.activation=gated?DmActivation::kRelu:DmActivation::kRelu6;op.gate=DmGatePair::kSimpleGate;
  }
#if TILEMEGA_DEPTHWISE_TEST_TMA
  auto* maps=Managed<executor::TensorMap>(batch);
  auto shape=executor::TensorMap3DShape::Depthwise(c,in,Rows,Channels);
  for(unsigned image=0;image<batch;++image)
    assert(executor::EncodeTensorMap3D(maps[image],input+image*in.strides[0],shape)==CUDA_SUCCESS);
  p.input_tensor_maps=maps;
  auto* legacy_output=Managed<E>(output_count+32);auto* legacy_partial=Managed<float>(batch*parts.strides[0]+16);
#endif
  auto shared=DepthwiseConvTaskBody<Arch,Rows,Channels,Program,bool(TILEMEGA_DEPTHWISE_TEST_TMA)>::SharedBytes(p);
  int limit=0;assert(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,0)==cudaSuccess);
  if(shared>unsigned(limit)) {cudaFree(input);cudaFree(output);cudaFree(weight);cudaFree(bias);
    cudaFree(pointers);cudaFree(dtypes);cudaFree(partial);return 0;}
  assert(cudaFuncSetAttribute(Run<Rows,Channels,Program>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared)==cudaSuccess);
  std::vector<bool> written(output_count),partial_written(batch*parts.strides[0]);
  std::vector<double> sums(batch*bands*channels);
  auto dot=[&](unsigned image,unsigned y,unsigned x,unsigned channel) {
    double value=0;
    for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s) {
      long iy=long(y)*stride+r*dilation-dilation,ix=long(x)*stride+s*dilation-dilation;
      if(iy>=0 && iy<long(h) && ix>=0 && ix<long(w))
        value+=double(float(input[Offset(in,image,iy,ix,channel)]))*float(weight[(channel*9+r*3+s)*8]);
    }
    if constexpr(Program::kCount)value+=bias[channel];
    return value;
  };
  for(unsigned epoch=0;epoch<3;++epoch) {
    for(unsigned i=0;i<output_count+32;++i)output[i]=E(-12345);
    for(unsigned i=0;i<batch*parts.strides[0]+16;++i)partial[i]=-12345;
    Run<Rows,Channels,Program><<<DepthwiseConvTaskBody<Arch,Rows,Channels,Program>::Count(p),128,shared>>>(p);
    assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
#if TILEMEGA_DEPTHWISE_TEST_TMA
    auto reference=p;reference.output=legacy_output+16;reference.channel_partials=legacy_partial+8;
    for(unsigned i=0;i<output_count+32;++i)legacy_output[i]=E(-12345);
    for(unsigned i=0;i<batch*parts.strides[0]+16;++i)legacy_partial[i]=-12345;
    auto legacy_shared=DepthwiseConvTaskBody<Arch,Rows,Channels,Program,false>::SharedBytes(p);
    assert(cudaFuncSetAttribute(Legacy<Rows,Channels,Program>,cudaFuncAttributeMaxDynamicSharedMemorySize,legacy_shared)==cudaSuccess);
    Legacy<Rows,Channels,Program><<<DepthwiseConvTaskBody<Arch,Rows,Channels,Program>::Count(p),128,legacy_shared>>>(reference);
    assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
    assert(!std::memcmp(output,legacy_output,(output_count+32)*sizeof(E)));
    assert(!std::memcmp(partial,legacy_partial,(batch*parts.strides[0]+16)*sizeof(float)));
#endif

    std::fill(sums.begin(),sums.end(),0);
    for(unsigned image=0;image<batch;++image)for(unsigned y=0;y<c.p;++y)for(unsigned x=0;x<c.q;++x)
      for(unsigned channel=0;channel<channels;++channel) {
        double expected=dot(image,y,x,channel);
        if constexpr(gated)expected*=dot(image,y,x,channel+channels);
        else if constexpr(Program::kCount==2)expected=std::max(0.0,std::min(6.0,expected));
        auto at=Offset(out,image,y,x,channel);float actual=float(output[16+at]);
        assert(std::abs(actual-expected)<=.016+.016*std::abs(expected));written[at]=true;
        sums[(image*bands+y/Rows)*channels+channel]+=actual;
      }
    for(unsigned image=0;image<batch;++image)for(unsigned band=0;band<bands;++band)
      for(unsigned channel=0;channel<channels;++channel) {
        auto at=image*parts.strides[0]+band*parts.strides[1]+channel;
        auto expected=sums[(image*bands+band)*channels+channel];
        assert(std::abs(partial[8+at]-expected)<=1e-5+1e-5*std::abs(expected));partial_written[at]=true;
      }
    for(unsigned i=0;i<output_count;++i)if(!written[i])assert(output[16+i]==E(-12345));
    for(unsigned i=0;i<batch*parts.strides[0];++i)if(!partial_written[i])assert(partial[8+i]==-12345);
    for(unsigned i=0;i<16;++i)assert(output[i]==E(-12345) && output[16+output_count+i]==E(-12345));
    for(unsigned i=0;i<8;++i)assert(partial[i]==-12345 && partial[8+batch*parts.strides[0]+i]==-12345);
  }
  // Input padding, including the channel remainder, is immutable.
  for(unsigned n=0;n<batch;++n)for(unsigned y=0;y<in.physical[1];++y)
    for(unsigned x=0;x<in.physical[2];++x)for(unsigned ch=0;ch<in.physical[3];++ch)
      if(y<dilation || y>=h+dilation || x<dilation || x>=w+dilation || ch>=input_channels)
        assert(input[n*in.strides[0]+y*in.strides[1]+x*in.strides[2]+ch]==E(0));
#if TILEMEGA_DEPTHWISE_TEST_TMA
  cudaFree(maps);cudaFree(legacy_output);cudaFree(legacy_partial);
#endif
  cudaFree(input);cudaFree(output);cudaFree(weight);cudaFree(bias);cudaFree(pointers);cudaFree(dtypes);cudaFree(partial);
  return 1;
}
int main(int argc,char** argv) {
#if TILEMEGA_DEPTHWISE_TEST_TMA
  if(argc>1 && std::string(argv[1])=="--host-only") {
    ConvDesc c;c.c=c.k=64;c.r=c.s=3;c.pad_h=c.pad_w=1;
    auto l=Image(1,256,256,64,1);auto shape=executor::TensorMap3DShape::Depthwise(c,l,1,32);
    assert(shape.box_width==129 && shape.box_height==3);
    int rejected=0;
    for(int i=0;i<5;++i)try {
      auto invalid=shape;
      if(i==0)invalid.box_width=257;if(i==1)invalid.box_height=257;
      if(i==2)invalid.pixel_stride_bytes=129;if(i==3)invalid.box_channels=16;
      if(i==4)invalid.channels=19;
      invalid.Validate(reinterpret_cast<void*>(16));
    }catch(std::invalid_argument const&){++rejected;}
    assert(rejected==5);std::puts("depthwise TMA 3D host shape and five invalid contracts PASS");return 0;
  }
#endif
  if(argc>1 && std::string(argv[1])=="--smoke") {
    unsigned cases=Check<1,32,Activated>(32,112,112,1,1,1);
    cases+=Check<2,64,Plain>(19,7,11,2,2,1);
    cases+=Check<2,64,Gated>(128,7,11,1,1,1);
    std::printf("DEPTHWISE smoke cases=%u exact reference/halo/tail/partial PASS\n",cases);return 0;
  }
  unsigned cases=0;
  // MobileNetV1/V2 channel/height/stride families, including narrow tails.
  unsigned geometries[][3]={{32,112,1},{32,112,2},{64,56,1},{64,56,2},{96,112,2},
      {128,56,1},{128,56,2},{144,56,1},{144,56,2},{192,28,1},{192,28,2},
      {256,28,1},{256,28,2},{384,14,1},{512,14,1},{512,14,2},{512,7,1},
      {576,14,1},{576,14,2},{960,7,1},{1024,7,1}};
  for(auto const& g:geometries)cases+=Check<1,32,Activated>(g[0],g[1],g[1],g[2],1,1);
  for(unsigned c:{64,128,256,512,1024}) {
    unsigned h=16384/c;cases+=Check<1,32,Gated>(c,h,h,1,1,1);
  }
  for(unsigned stride:{1,2})for(unsigned dilation:{1,2})for(unsigned batch:{1,2}) {
    cases+=Check<2,64,Plain>(19,7,11,stride,dilation,batch);
    cases+=Check<3,128,Biased>(144,7,11,stride,dilation,batch);
    cases+=Check<2,256,Activated>(256,7,11,stride,dilation,batch);
    cases+=Check<2,64,Gated>(128,7,11,stride,dilation,batch);
  }
  assert(cases>=58);
  std::printf("DEPTHWISE cases=%u epochs=3 tma=%d exact_SIMT_bias_relu6_simple_gate_partials_halo_canaries PASS\n",cases,TILEMEGA_DEPTHWISE_TEST_TMA);
}
