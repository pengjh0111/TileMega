// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Backend/ServingConv.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef DM_TEST_TM
#define DM_TEST_TM 16
#endif
#ifndef DM_TEST_TN
#define DM_TEST_TN 16
#endif
#ifndef DM_TEST_TK
#define DM_TEST_TK 16
#endif
#ifndef DM_TEST_STAGES
#define DM_TEST_STAGES 3
#endif
using namespace tilemega;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
using Body=backend::ServingConv<Arch,DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES>;
using Packing=typename Body::Gemm;
struct Operands {E const* a=nullptr;E const* b=nullptr;E const* weight_base=nullptr;int m=0,n=0;};
__global__ void Run(Operands operands,codegen::ConvDesc conv,codegen::DmBufferLayout layout,
    backend::ConvIterationGeometry geometry,int chunks,float* output,bool tiled) {
  extern __shared__ __align__(16) char shared[];
  int iterations=geometry.iterations/chunks;
  float* tile=tiled?Body::template Run<true>(operands,conv,layout,geometry,
      blockIdx.x,blockIdx.y,std::uint64_t(blockIdx.z)*iterations,iterations,shared):
      Body::template Run<false>(operands,conv,layout,geometry,
      blockIdx.x,blockIdx.y,std::uint64_t(blockIdx.z)*iterations,iterations,shared);
  for(int i=threadIdx.x;i<DM_TEST_TM*DM_TEST_TN;i+=128) {
    int m=blockIdx.x*DM_TEST_TM+i/DM_TEST_TN,n=blockIdx.y*DM_TEST_TN+i%DM_TEST_TN;
    if(m<operands.m && n<operands.n)
      output[(std::size_t(blockIdx.z)*operands.m+m)*operands.n+n]=tile[i];
  }
}
template<class T>T* Managed(std::size_t count) {
  T* out=nullptr;assert(cudaMallocManaged(&out,count*sizeof(T))==cudaSuccess);return out;
}
float Input(unsigned image,int y,int x,unsigned c) {
  return (int((image*19+y*13+x*7+c*3)%29)-14)*0.03125f;
}
float Weight(unsigned n,unsigned r,unsigned s,unsigned c) {
  return (int((n*17+r*11+s*5+c*7)%23)-11)*0.015625f;
}
int main() {
  struct Shape {unsigned c,r,s,stride,dilation,pad;};
  std::vector<Shape> shapes={{3,7,7,2,1,3},{3,3,3,2,1,1},{3,3,3,1,1,1},
      {24,3,3,1,1,1},{24,2,2,2,1,0},{24,1,1,2,1,0},
      {27,3,3,1,2,2},{64,3,3,2,1,1},{64,1,1,1,1,0},
      {4,3,2,1,1,1},{8,1,3,1,2,2},{16,2,3,2,2,1}};
  unsigned cases=0;
  assert(cudaFuncSetAttribute(Run,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes)==cudaSuccess);
  for(auto shape:shapes)for(unsigned batch:{1,2})for(unsigned columns:{3,19})
    for(unsigned rgb_pad:{4,8}) {
      if(shape.c>4 && rgb_pad!=8)continue;
      unsigned cp=shape.c<=4?rgb_pad:(shape.c+7)/8*8;
      if(cp<DM_TEST_TK && DM_TEST_TK%cp)continue;
      codegen::ConvDesc conv;conv.n=batch;conv.h=7;conv.w=9;conv.c=shape.c;conv.k=columns;
      conv.r=shape.r;conv.s=shape.s;conv.stride_h=conv.stride_w=shape.stride;
      conv.dilation_h=conv.dilation_w=shape.dilation;conv.pad_h=conv.pad_w=shape.pad;
      conv.p=(conv.h+2*conv.pad_h-conv.dilation_h*(conv.r-1)-1)/conv.stride_h+1;
      conv.q=(conv.w+2*conv.pad_w-conv.dilation_w*(conv.s-1)-1)/conv.stride_w+1;
      codegen::DmBufferLayout layout;layout.kind=codegen::DmLayout::kNHWC;layout.rank=4;
      layout.logical[0]=layout.physical[0]=batch;layout.logical[1]=conv.h;
      layout.logical[2]=conv.w;layout.logical[3]=shape.c;layout.physical[3]=cp;
      layout.halo_top=layout.halo_bottom=layout.halo_left=layout.halo_right=shape.pad;
      layout.physical[1]=conv.h+2*shape.pad;layout.physical[2]=conv.w+2*shape.pad;
      layout.strides[3]=1;layout.strides[2]=(cp+7)/8*8;
      layout.strides[1]=layout.physical[2]*layout.strides[2];layout.strides[0]=layout.physical[1]*layout.strides[1];
      auto geometry=backend::ConvIterationGeometry::Build(conv,layout,DM_TEST_TK);
      auto input_size=batch*layout.strides[0],weight_pitch=std::uint64_t(conv.r)*conv.s*cp;
      auto* a=Managed<E>(input_size);auto* b=Managed<E>(columns*weight_pitch);
      for(std::size_t i=0;i<input_size;++i)a[i]=E(0);
      for(unsigned n=0;n<batch;++n)for(unsigned h=0;h<conv.h;++h)
        for(unsigned w=0;w<conv.w;++w)for(unsigned c=0;c<shape.c;++c)
          a[n*layout.strides[0]+(h+shape.pad)*layout.strides[1]+(w+shape.pad)*layout.strides[2]+c]=E(Input(n,h,w,c));
      for(unsigned n=0;n<columns;++n)for(unsigned r=0;r<conv.r;++r)
        for(unsigned s=0;s<conv.s;++s)for(unsigned c=0;c<cp;++c)
          b[n*weight_pitch+(r*conv.s+s)*cp+c]=E(c<shape.c?Weight(n,r,s,c):0);
      int m=batch*conv.p*conv.q,nt=(columns+DM_TEST_TN-1)/DM_TEST_TN;
      auto* packed=Managed<E>(nt*geometry.iterations*DM_TEST_TN*DM_TEST_TK);
      for(int tn=0;tn<nt;++tn)for(std::uint64_t it=0;it<geometry.iterations;++it)
        for(int n=0;n<DM_TEST_TN;++n)for(int k=0;k<DM_TEST_TK;++k) {
          auto point=geometry.At(it,k);unsigned column=tn*DM_TEST_TN+n;
          auto at=(std::uint64_t(tn)*geometry.iterations+it)*DM_TEST_TN*DM_TEST_TK+
              typename Packing::LayoutB{}(n,k);
          packed[at]=column<columns && point.valid?
              b[column*weight_pitch+(point.r*conv.s+point.s)*cp+point.c]:E(0);
        }
      Operands operands{a,b,packed,m,int(columns)};
      std::vector<float> expected(m*columns);
      for(int row=0;row<m;++row)for(unsigned n=0;n<columns;++n) {
        unsigned image=row/(conv.p*conv.q),pixel=row%(conv.p*conv.q);
        float sum=0;
        for(unsigned r=0;r<conv.r;++r)for(unsigned s=0;s<conv.s;++s)for(unsigned c=0;c<shape.c;++c) {
          int h=(pixel/conv.q)*conv.stride_h+r*conv.dilation_h-conv.pad_h;
          int w=(pixel%conv.q)*conv.stride_w+s*conv.dilation_w-conv.pad_w;
          if(h>=0 && h<int(conv.h) && w>=0 && w<int(conv.w))
            sum+=float(E(Input(image,h,w,c)))*float(E(Weight(n,r,s,c)));
        }
        expected[row*columns+n]=sum;
      }
      for(unsigned chunks:{1,2,4,8}) {
        if(geometry.iterations%chunks)continue;
        auto* output=Managed<float>(std::size_t(chunks)*m*columns+32);
        for(bool tiled:{false,true}) {
          for(std::size_t i=0;i<std::size_t(chunks)*m*columns+32;++i)output[i]=-12345;
          Run<<<dim3((m+DM_TEST_TM-1)/DM_TEST_TM,nt,chunks),128,Body::kSharedBytes>>>(
              operands,conv,layout,geometry,chunks,output+16,tiled);
          assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
          for(int i=0;i<16;++i)assert(output[i]==-12345 && output[std::size_t(chunks)*m*columns+16+i]==-12345);
          for(int i=0;i<m*int(columns);++i) {
            float sum=0;for(unsigned chunk=0;chunk<chunks;++chunk)sum+=output[16+std::size_t(chunk)*m*columns+i];
            auto reference=expected[i];
            if(!std::isfinite(sum) || std::abs(sum-reference)>0.016f+0.016f*std::abs(reference)) {
              std::fprintf(stderr,"c=%u r=%u s=%u cp=%u stride=%u dilation=%u batch=%u n=%u split=%u tiled=%d element=%d actual=%g ref=%g\n",
                  shape.c,shape.r,shape.s,cp,shape.stride,shape.dilation,batch,columns,chunks,tiled,i,sum,reference);
              return 3;
            }
          }
          ++cases;
        }
        cudaFree(output);
      }
      cudaFree(a);cudaFree(b);cudaFree(packed);
    }
  std::printf("CONV_OPERANDS tile=%dx%dx%d stages=%d cases=%u FP32_oracle_dense_tiled_split_halo_canaries PASS\n",
      DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES,cases);
}
