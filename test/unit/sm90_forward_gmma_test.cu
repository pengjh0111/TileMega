// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#ifndef TILEMEGA_TEST_PIPELINE_STAGES
#define TILEMEGA_TEST_PIPELINE_STAGES 2
#endif
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using Op=tilemega::backend::ServingEpilogueOp;
using Arch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
    tilemega::arch::Sm90,tilemega::arch::CurrentArch>;
void CheckCuda(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<int M,int N>
__global__ void Run(ServingGemmOperands p) {
  extern __shared__ __align__(1024) char memory[];
  using Body=ServingGemmTaskBody<Arch,M,N,64,TILEMEGA_TEST_PIPELINE_STAGES,true>;
  for(int m=0;m<(p.m+M-1)/M;++m)for(int n=0;n<(p.n+N-1)/N;++n)
    Body::Run(p,m,n,memory);
}
float A(int row,int k) {return ((row*13+k*7)%29-14)*0.015625f;}
float B(int row,int k) {return ((row*17+k*11)%31-15)*0.015625f;}
template<int M,int N>
void Check(int rows,int reduction,int begin,Op op,bool tma) {
  using Body=ServingGemmTaskBody<Arch,M,N,64,TILEMEGA_TEST_PIPELINE_STAGES,true>;
  int cols=op==Op::kSwiGLU?N+32:N+13;
  int pitch=(reduction+7)&~7,kt=(reduction+63)/64,nt=(cols+N-1)/N;
  Element *a,*b,*out,*residual;float *partial,*best;int* indices;
  executor::TensorMap* maps;
  std::size_t weights=TILEMEGA_NONPAGED_TILED?std::size_t(nt)*kt*N*64:std::size_t(cols)*pitch;
  CheckCuda(cudaMallocManaged(&a,rows*pitch*2));CheckCuda(cudaMallocManaged(&b,weights*2));
  CheckCuda(cudaMallocManaged(&out,rows*cols*2));CheckCuda(cudaMallocManaged(&residual,rows*cols*2));
  CheckCuda(cudaMallocManaged(&partial,rows*cols*4));CheckCuda(cudaMallocManaged(&best,rows*nt*4));
  CheckCuda(cudaMallocManaged(&indices,rows*nt*4));CheckCuda(cudaMallocManaged(&maps,2*sizeof(*maps)));
  for(int row=0;row<rows;++row)for(int k=0;k<pitch;++k)a[row*pitch+k]=Element(k<reduction?A(row,k):0);
  for(std::size_t i=0;i<weights;++i)b[i]=Element(0);
  for(int n=0;n<cols;++n)for(int k=0;k<reduction;++k) {
    std::size_t off=n*pitch+k;
#if TILEMEGA_NONPAGED_TILED
    int rn=n%N,rk=k%64;
    off=(std::size_t(n/N)*kt+k/64)*N*64+512*(rn/8)+64*(rn%8)+
        8*(((rk%64)/8)^(rn%8))+rk%8;
#endif
    b[off]=Element(B(n,k));
  }
  for(int i=0;i<rows*cols;++i){out[i]=Element(-99);partial[i]=-99;residual[i]=Element((i%13-6)*0.03125f);}
  if(tma) {
    auto code=executor::EncodeTensorMap(maps[0],a,{std::uint64_t(reduction),std::uint64_t(rows),std::uint64_t(pitch)*2,64,M});
    if(code!=CUDA_SUCCESS){std::fprintf(stderr,"A map encoding failed: %d\n",int(code));std::exit(4);}
#if !TILEMEGA_NONPAGED_TILED
    code=executor::EncodeTensorMap(maps[1],b,{std::uint64_t(reduction),std::uint64_t(cols),std::uint64_t(pitch)*2,64,N});
    if(code!=CUDA_SUCCESS){std::fprintf(stderr,"B map encoding failed: %d\n",int(code));std::exit(4);}
#endif
  }
  ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=b;p.output=out;p.partial=partial;
  p.residual=residual;p.argmax_value=best;p.argmax_index=indices;p.epilogue=op;
  p.m=rows;p.n=cols;p.k_total=p.k_total_full=reduction;p.k_begin=begin;p.k_count=reduction-begin;
  p.a_row_stride=p.b_row_stride=pitch;p.partial_stride=cols;
  p.output_stride=op==Op::kArgmaxPartial?nt:op==Op::kSwiGLU?cols/2:cols;
  p.activation_tensor_map=tma?maps:nullptr;p.tensor_map=tma?maps+1:nullptr;p.tensor_k_begin=begin;
  CheckCuda(cudaFuncSetAttribute(Run<M,N>,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes));
  Run<M,N><<<1,128,Body::kSharedBytes>>>(p);CheckCuda(cudaDeviceSynchronize());
  auto acc=[&](int row,int col) {float sum=0;for(int k=begin;k<reduction;++k)sum+=A(row,k)*B(col,k);return sum;};
  auto fail=[&](int row,int col,float actual,float expected) {
    std::fprintf(stderr,"forward WGMMA M%d N%d rows%d K%d begin%d op%d TMA%d [%d,%d]: %.9g != %.9g\n",
        M,N,rows,reduction,begin,int(op),int(tma),row,col,actual,expected);std::exit(3);
  };
  for(int row=0;row<rows;++row) {
    if(op==Op::kArgmaxPartial) {
      for(int tile=0;tile<nt;++tile) {
        float expected=-INFINITY;int index=-1;
        for(int col=tile*N;col<std::min(cols,(tile+1)*N);++col) {
          float value=float(Element(acc(row,col)));
          if(value>expected){expected=value;index=col;}
        }
        if(best[row*nt+tile]!=expected || indices[row*nt+tile]!=index)fail(row,tile,best[row*nt+tile],expected);
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
  for(void* ptr:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(out),static_cast<void*>(residual),
      static_cast<void*>(partial),static_cast<void*>(best),static_cast<void*>(indices),static_cast<void*>(maps)})CheckCuda(cudaFree(ptr));
}
template<int M,int N>
void Matrix() {
  for(int rows:{1,M+1})for(int k:{2001,2048})for(int begin:{0,128})for(bool tma:{false,true})
    for(Op op:{Op::kStore,Op::kResidual,Op::kPartial,Op::kArgmaxPartial,Op::kSwiGLU})Check<M,N>(rows,k,begin,op,tma);
}
int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="--pipeline-smoke") {
    Check<64,64>(65,257,0,Op::kStore,false);
    Check<64,64>(65,64,0,Op::kStore,true);
    Check<64,64>(65,2001,128,Op::kPartial,true);
    Check<128,64>(129,257,0,Op::kStore,false);
    std::puts("forward pipeline: short/tail/split-offset TMA and cp.async PASS");return 0;
  }
  Matrix<64,64>();Matrix<64,128>();Matrix<64,256>();Matrix<128,64>();Matrix<128,128>();
  std::puts("forward WGMMA: 400 TMA/cp.async/shape/residue/split-offset/epilogue cases PASS");
}
