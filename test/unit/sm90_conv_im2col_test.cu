// SPDX-License-Identifier: BSD-3-Clause
#ifndef TILEMEGA_CONV_TMA_IM2COL
#define TILEMEGA_CONV_TMA_IM2COL 1
#endif
#include <tilemega/Backend/Sm90Conv.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/executor/TensorIm2col.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#ifndef TILEMEGA_TEST_PIPELINE_STAGES
#define TILEMEGA_TEST_PIPELINE_STAGES 2
#endif
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using Arch=tilemega::arch::Sm90;
void Cuda(cudaError_t status) {
  if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(2);}
}
template<int N>
__global__ void Run(ServingGemmOperands p,ConvDesc conv,DmBufferLayout layout,float* out) {
  extern __shared__ __align__(1024) char memory[];
  for(int m=0;m<(p.m+63)/64;++m)for(int n=0;n<(p.n+N-1)/N;++n) {
    conv.input_layout=0;p.convolutions=&conv;p.access.a=DmAAccess::kIm2Col;p.access.conv=0;
    p.dm_buffers.layouts=&layout;p.dm_buffers.count=1;
    p.output=reinterpret_cast<Element*>(out);p.output_stride=p.n;
    p.chain.store_rounding=DmRounding::kFP32;
    using Spec=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kDense,1,DmRounding::kFP32>;
    ServingGemmTaskBody<Arch,64,N,64,TILEMEGA_TEST_PIPELINE_STAGES,true>::template RunDmResolved<Spec>(p,m,n,memory);
  }
}
float A(int n,int h,int w,int c){return ((n*13+h*7+w*3+c*11)%17-8)/64.f;}
float B(int n,int k){return ((n*5+k*13)%19-9)/64.f;}
template<int N>
void Check(int batch,int height,int width,int channels,int filter,int stride,int dilation,int split) {
  ConvDesc conv;conv.n=batch;conv.h=height;conv.w=width;conv.c=channels;conv.k=N+7;
  conv.r=conv.s=filter;conv.pad_h=conv.pad_w=dilation*(filter-1)/2;
  conv.stride_h=conv.stride_w=stride;conv.dilation_h=conv.dilation_w=dilation;
  conv.p=(height+2*conv.pad_h-dilation*(filter-1)-1)/stride+1;
  conv.q=(width+2*conv.pad_w-dilation*(filter-1)-1)/stride+1;
  DmBufferLayout layout;layout.kind=DmLayout::kNHWC;layout.rank=4;
  layout.halo_top=layout.halo_bottom=conv.pad_h;layout.halo_left=layout.halo_right=conv.pad_w;
  int physical[]={batch,height+2*int(conv.pad_h),width+2*int(conv.pad_w),channels};
  int logical[]={batch,height,width,channels};
  for(int i=0;i<4;++i){layout.physical[i]=physical[i];layout.logical[i]=logical[i];}
  layout.strides[3]=1;for(int i=2;i>=0;--i)layout.strides[i]=layout.strides[i+1]*physical[i+1];
  int reduction=channels*filter*filter,rows=batch*conv.p*conv.q,cols=conv.k,kt=reduction/64,nt=(cols+N-1)/N;
  split=std::min(split,kt);
  Element *a,*b;float* out;executor::TensorMap* maps;
  Cuda(cudaMallocManaged(&a,batch*layout.strides[0]*2));
  std::size_t weights=TILEMEGA_NONPAGED_TILED?std::size_t(nt)*kt*N*64:std::size_t(cols)*reduction;
  Cuda(cudaMallocManaged(&b,weights*2));Cuda(cudaMallocManaged(&out,rows*cols*4));
  Cuda(cudaMallocManaged(&maps,2*sizeof(*maps)));
  for(std::size_t i=0;i<batch*layout.strides[0];++i)a[i]=Element(0);
  for(int n=0;n<batch;++n)for(int h=0;h<height;++h)for(int w=0;w<width;++w)for(int c=0;c<channels;++c)
    a[n*layout.strides[0]+(h+conv.pad_h)*layout.strides[1]+(w+conv.pad_w)*layout.strides[2]+c]=Element(A(n,h,w,c));
  for(std::size_t i=0;i<weights;++i)b[i]=Element(0);
  auto bl=cute::as_position_independent_swizzle_layout(tilemega::backend::Sm90KLayout<N,64>{});
  for(int n=0;n<cols;++n)for(int k=0;k<reduction;++k) {
    std::size_t offset=n*reduction+k;
#if TILEMEGA_NONPAGED_TILED
    offset=(std::size_t(n/N)*kt+k/64)*N*64+bl(n%N,k%64);
#endif
    b[offset]=Element(B(n,k));
  }
  auto code=executor::EncodeTensorIm2col(maps[0],a,executor::TensorIm2colShape::Build(conv,layout));
  if(code!=CUDA_SUCCESS){std::fprintf(stderr,"im2col encode error %d\n",int(code));std::exit(4);}
#if !TILEMEGA_NONPAGED_TILED
  code=executor::EncodeTensorMap(maps[1],b,{std::uint64_t(reduction),std::uint64_t(cols),std::uint64_t(reduction)*2,64,N});
  if(code!=CUDA_SUCCESS){std::fprintf(stderr,"weight encode error %d\n",int(code));std::exit(4);}
#endif
  ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=b;p.activation_tensor_map=maps;p.tensor_map=maps+1;
  p.m=rows;p.n=cols;p.k_total=p.k_total_full=reduction;
  using Body=tilemega::backend::Sm90Conv<Arch,64,N,64,TILEMEGA_TEST_PIPELINE_STAGES>;
  Cuda(cudaFuncSetAttribute(Run<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes));
  for(int part=0;part<split;++part) {
    p.k_begin=(part*kt/split)*64;p.k_count=((part+1)*kt/split)*64-p.k_begin;
    Run<N><<<1,128,Body::kSharedBytes>>>(p,conv,layout,out);Cuda(cudaDeviceSynchronize());
    for(int row=0;row<rows;++row)for(int n=0;n<cols;++n) {
      int image=row/(conv.p*conv.q),y=(row/conv.q)%conv.p,x=row%conv.q;float expected=0;
      for(int k=p.k_begin;k<p.k_begin+p.k_count;++k) {
        int tap=k/channels,c=k%channels,h=y*stride-int(conv.pad_h)+(tap/filter)*dilation;
        int w=x*stride-int(conv.pad_w)+(tap%filter)*dilation;
        if(h>=0 && h<height && w>=0 && w<width)expected+=A(image,h,w,c)*B(n,k);
      }
      if(out[row*cols+n]!=expected){std::fprintf(stderr,"conv N%d B%d H%d W%d C%d R%d stride%d dilation%d split%d part%d [%d,%d] %.9g != %.9g\n",N,batch,height,width,channels,filter,stride,dilation,split,part,row,n,out[row*cols+n],expected);std::exit(3);}
    }
  }
  for(void* pointer:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(out),static_cast<void*>(maps)})Cuda(cudaFree(pointer));
}
int main(int argc,char** argv) {
  if(argc>1 && std::string(argv[1])=="--host-only") {
    ConvDesc conv;conv.h=conv.w=8;conv.c=64;conv.k=64;conv.p=conv.q=8;
    DmBufferLayout layout;layout.kind=DmLayout::kNHWC;layout.rank=4;
    std::uint32_t shape[]={1,8,8,64};for(int i=0;i<4;++i)layout.logical[i]=layout.physical[i]=shape[i];
    layout.strides[3]=1;layout.strides[2]=64;layout.strides[1]=512;layout.strides[0]=4096;
    auto map=executor::TensorIm2colShape::Build(conv,layout);
    int rejected=0;for(int kind=0;kind<4;++kind)try {
      auto invalid=map;if(kind==0)invalid.dimensions[0]=32;if(kind==1)invalid.strides[0]=129;
      if(kind==2)invalid.lower[0]=128;if(kind==3)invalid.traversal[1]=9;
      invalid.Validate(reinterpret_cast<void*>(16));
    }catch(std::invalid_argument const&){++rejected;}
    if(rejected!=4)return 5;
    std::puts("im2col host geometry and four invalid contracts PASS");return 0;
  }
  if(argc==2 && std::string(argv[1])=="--pipeline-smoke") {
    Check<64>(1,9,11,64,3,1,1,3);
    Check<128>(1,9,11,128,1,1,1,2);
    std::puts("im2col pipeline: short and split slices PASS");return 0;
  }
  for(int channels:{64,128})for(int filter:{1,3})for(int stride:{1,2})for(int batch:{1,2}) {
    Check<64>(batch,9,11,channels,filter,stride,1,1);
    Check<128>(batch,9,11,channels,filter,stride,1,2);
  }
  Check<64>(1,9,11,64,3,1,2,3);
  std::printf("TMA im2col + WGMMA 33 convolution configurations PASS (tiledB=%d)\n",TILEMEGA_NONPAGED_TILED);
}
