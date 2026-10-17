// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using Op=tilemega::backend::ServingEpilogueOp;
using Arch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
    tilemega::arch::Sm90,tilemega::arch::CurrentArch>;
void CheckCuda(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<int N,int K,bool Paged,int Page=16384,int S=2>
__global__ void Run(ServingGemmOperands p) {
  extern __shared__ __align__(1024) char memory[];
  if constexpr(Paged) {
    using Body=PagedGemmTaskBody<Arch,16,N,K,Page,5,false,true>;
    using Ring=typename Body::Ring;
    constexpr int pool=(1024+std::max(Body::kActivationBytes,Body::kScratchBytes)+1023)/1024*1024;
    Ring ring{reinterpret_cast<typename Ring::Slot*>(memory),memory+pool};
    ring.Initialize();std::uint64_t sequence=0;
    for(int m=0;m<(p.m+15)/16;++m)for(int n=0;n<(p.n+N-1)/N;++n) {
      if(executor::IsCompute())Body::Run(p,m,n,ring,sequence,memory+1024);
      else Body::Load(p,n,ring,sequence);
    }
  }else {
    using Body=ServingGemmTaskBody<Arch,16,N,K,S,true>;
    for(int m=0;m<(p.m+15)/16;++m)for(int n=0;n<(p.n+N-1)/N;++n)
      Body::Run(p,m,n,memory);
  }
}
float A(int row,int k) {return ((row*13+k*7)%29-14)*0.015625f;}
float B(int row,int k) {return ((row*17+k*11)%31-15)*0.015625f;}
template<int N,int K,bool Paged,int Page=16384,int S=2>
void Check(int rows,int cols,int reduction,int begin,Op op) {
  using Body=PagedGemmTaskBody<Arch,16,Paged?N:64,Paged?K:64,Page,5,false,true>;
  int pitch=(reduction+7)&~7,kt=(reduction+K-1)/K,nt=(cols+N-1)/N;
  Element *a,*b,*out,*residual;float *partial,*best;int* indices;
  std::size_t weight_elements=Paged||TILEMEGA_NONPAGED_TILED?std::size_t(nt)*kt*N*K:std::size_t(cols)*pitch;
  CheckCuda(cudaMallocManaged(&a,rows*pitch*2));CheckCuda(cudaMallocManaged(&b,weight_elements*2));
  CheckCuda(cudaMallocManaged(&out,rows*cols*2));CheckCuda(cudaMallocManaged(&residual,rows*cols*2));
  CheckCuda(cudaMallocManaged(&partial,rows*cols*4));CheckCuda(cudaMallocManaged(&best,rows*nt*4));
  CheckCuda(cudaMallocManaged(&indices,rows*nt*4));
  for(int row=0;row<rows;++row)for(int k=0;k<pitch;++k)a[row*pitch+k]=Element(k<reduction?A(row,k):0);
  for(std::size_t i=0;i<weight_elements;++i)b[i]=Element(0);
  for(int n=0;n<cols;++n)for(int k=0;k<reduction;++k) {
    std::size_t off=n*pitch+k;
    if(Paged||TILEMEGA_NONPAGED_TILED) {
      int rn=n%N,rk=k%K;
      off=(std::size_t(n/N)*kt+k/K)*N*K+512*(rn/8+(N/8)*(rk/64))+
          64*(rn%8)+8*(((rk%64)/8)^(rn%8))+rk%8;
    }
    b[off]=Element(B(n,k));
  }
  for(int i=0;i<rows*cols;++i){out[i]=Element(-99);partial[i]=-99;residual[i]=Element((i%13-6)*0.03125f);}
  ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=b;p.output=out;p.partial=partial;
  p.residual=residual;p.argmax_value=best;p.argmax_index=indices;p.epilogue=op;
  p.m=rows;p.n=cols;p.k_total=p.k_total_full=reduction;p.k_begin=begin;p.k_count=reduction-begin;
  p.a_row_stride=p.b_row_stride=pitch;p.partial_stride=cols;
  p.output_stride=op==Op::kArgmaxPartial?nt:op==Op::kSwiGLU?cols/2:cols;
  constexpr int pool=(1024+std::max(Body::kActivationBytes,Body::kScratchBytes)+1023)/1024*1024;
  int bytes=Paged?pool+5*Page:ServingGemmTaskBody<Arch,16,N,K,S,true>::kSharedBytes;
  CheckCuda(cudaFuncSetAttribute(Run<N,K,Paged,Page,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  Run<N,K,Paged,Page,S><<<1,Paged?160:128,bytes>>>(p);CheckCuda(cudaDeviceSynchronize());
  auto acc=[&](int row,int col) {float sum=0;for(int k=begin;k<reduction;++k)sum+=A(row,k)*B(col,k);return sum;};
  auto fail=[&](int row,int col,float actual,float expected) {
    std::fprintf(stderr,"WGMMA %s S%d N%d K%d M%d N%d K%d begin%d op%d [%d,%d]: %.9g != %.9g\n",
        Paged?"pages":"dense",S,N,K,rows,cols,reduction,begin,int(op),row,col,actual,expected);std::exit(3);
  };
  for(int row=0;row<rows;++row) {
    if(op==Op::kArgmaxPartial) {
      for(int tile=0;tile<nt;++tile) {
        float expected=-INFINITY;int index=-1;
        for(int col=tile*N;col<std::min(cols,(tile+1)*N);++col) {
          float value=float(Element(acc(row,col)));
          if(value>expected){expected=value;index=col;}
        }
        if(best[row*nt+tile]!=expected || indices[row*nt+tile]!=index)
          fail(row,tile,best[row*nt+tile],expected);
      }
    }else if(op==Op::kSwiGLU) {
      for(int col=0;col<cols/2;++col) {
        int gate=32*(col/16)+col%16;float g=float(Element(acc(row,gate)));
        float expected=float(Element(float(Element(g/(1+std::exp(-g))))*float(Element(acc(row,gate+16)))));
        float actual=float(out[row*(cols/2)+col]);
        float ulp=expected==0?std::ldexp(1.0f,-133):std::ldexp(1.0f,std::ilogb(std::abs(expected))-7);
        if(std::abs(actual-expected)>ulp)fail(row,col,actual,expected);
      }
    }else for(int col=0;col<cols;++col) {
      float expected=acc(row,col),actual;
      if(op==Op::kPartial)actual=partial[row*cols+col];
      else {
        expected=float(Element(expected));
        if(op==Op::kResidual)expected=float(Element(expected+float(residual[row*cols+col])));
        actual=float(out[row*cols+col]);
      }
      if(actual!=expected)fail(row,col,actual,expected);
    }
  }
  for(void* ptr:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(out),
      static_cast<void*>(residual),static_cast<void*>(partial),static_cast<void*>(best),static_cast<void*>(indices)})CheckCuda(cudaFree(ptr));
}
template<int N,int K,bool Paged,int Page=16384,int S=2>
void Matrix() {
  for(int rows:{1,16,19})for(int k:{2001,2048})for(int begin:{0,128})
    for(Op op:{Op::kStore,Op::kResidual,Op::kPartial,Op::kArgmaxPartial,Op::kSwiGLU})
      Check<N,K,Paged,Page,S>(rows,op==Op::kSwiGLU?96:73,k,begin,op);
}
int main() {
#if defined(TILEMEGA_SM90_WIDE_TEST) && TILEMEGA_SM90_WIDE_TEST
  Matrix<64,64,true,32768>();Matrix<128,128,true,32768>();Matrix<256,64,true,32768>();
  std::puts("WGMMA 32KiB pages: 180 shape/residue/split-offset/epilogue cases PASS");
#else
  Matrix<64,64,true>();Matrix<128,64,true>();Matrix<64,128,true>();
  Matrix<64,64,false>();Matrix<128,64,false>();Matrix<256,128,false>();
  Matrix<64,64,false,16384,3>();Matrix<64,64,false,16384,8>();
  Matrix<128,64,false,16384,4>();Matrix<64,128,false,16384,6>();
  std::puts("WGMMA paged/nonpaged and 3/4/6/8-slot nonpaged: 600 shape/residue/split-offset/epilogue cases PASS");
#endif
}
