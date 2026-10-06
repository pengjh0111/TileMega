// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingGemvTaskBody.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace tilemega;
using Element=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm80,arch::CurrentArch>;
using Op=backend::ServingEpilogueOp;
__host__ __device__ float A(int r,int k){return float((r*7+k)%13-6)*.0625f;}
__host__ __device__ float B(int n,int k){return float((n*3+k*5)%17-8)*.03125f;}
template<int N,int K,bool Tiled>__global__ void Probe(codegen::ServingGemmOperands p){
  using Body=codegen::ServingGemvTaskBody<Arch,N,K,Tiled>;
  alignas(16) __shared__ char scratch[Body::kSharedBytes];Body::Run(p,0,blockIdx.x,scratch);
}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* Alloc(int n){T* p;Check(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<int TN,int TK,bool Tiled>void Run(int m,int k,Op op){
  int n=op==Op::kSwiGLU?TN*2:TN+3,kp=(k+7)/8*8,kt=(k+TK-1)/TK,nt=(n+TN-1)/TN;
  int width=op==Op::kSwiGLU?n/2:n;
  auto* a=Alloc<Element>(m*kp);auto* b=Alloc<Element>(n*kp);auto* packed=Alloc<Element>(nt*kt*TN*TK);
  auto* output=Alloc<Element>(m*width);auto* residual=Alloc<Element>(m*width);
  auto* partial=Alloc<float>(m*n);auto* values=Alloc<float>(m*nt);auto* indices=Alloc<int>(m*nt);
  for(int r=0;r<m;++r)for(int kk=0;kk<kp;++kk)a[r*kp+kk]=Element(kk<k?A(r,kk):0);
  for(int c=0;c<n;++c)for(int kk=0;kk<kp;++kk)b[c*kp+kk]=Element(kk<k?B(c,kk):0);
  for(int i=0;i<nt*kt*TN*TK;++i)packed[i]=Element(0);
  for(int c=0;c<n;++c)for(int kk=0;kk<k;++kk)packed[(c/TN*kt+kk/TK)*TN*TK+backend::ServingGemv<Arch,TN,TK,Tiled>::PackedIndex(c%TN,kk%TK)]=Element(B(c,kk));
  for(int i=0;i<m*width;++i)residual[i]=Element(.25f);
  codegen::ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=packed;p.output=output;p.residual=residual;p.partial=partial;
  p.argmax_value=values;p.argmax_index=indices;p.m=m;p.n=n;p.k_total=p.k_count=p.k_total_full=k;p.a_row_stride=p.b_row_stride=kp;
  p.output_stride=op==Op::kArgmaxPartial?nt:width;p.partial_stride=n;p.epilogue=op;
  Probe<TN,TK,Tiled><<<nt,128>>>(p);Check(cudaDeviceSynchronize());
  auto dot=[&](int r,int c){float s=0;for(int kk=0;kk<k;++kk)s+=A(r,kk)*B(c,kk);return s;};
  for(int r=0;r<m;++r){
    if(op==Op::kArgmaxPartial){for(int t=0;t<nt;++t){float best=-INFINITY;int index=INT32_MAX;
      for(int c=t*TN;c<std::min(n,(t+1)*TN);++c){float v=float(Element(dot(r,c)));if(v>best){best=v;index=c;}}
      if(values[r*nt+t]!=best || indices[r*nt+t]!=index)std::exit(3);}continue;}
    for(int c=0;c<width;++c){float expected=dot(r,c);
      if(op==Op::kSwiGLU){int g=32*(c/16)+c%16;float x=float(Element(dot(r,g))),y=float(Element(dot(r,g+16)));expected=float(Element(float(Element(x/(1+std::exp(-x))))*y));}
      else if(op==Op::kResidual)expected=float(Element(float(Element(expected))+.25f));
      else if(op==Op::kStore)expected=float(Element(expected));
      float actual=op==Op::kPartial?partial[r*n+c]:float(output[r*width+c]);
      if(std::abs(actual-expected)>1e-5f*(1+std::abs(expected))){std::fprintf(stderr,"GEMV mismatch TN%d TK%d tiled%d M%d K%d op%d r%d c%d %.8g %.8g\n",TN,TK,Tiled,m,k,int(op),r,c,actual,expected);std::exit(3);}}
  }
  for(void* ptr:{(void*)a,(void*)b,(void*)packed,(void*)output,(void*)residual,(void*)partial,(void*)values,(void*)indices})Check(cudaFree(ptr));
}
template<int TN,int TK,bool Tiled>void Matrix(){for(int m:{1,4})for(int k:{128,257})for(Op op:{Op::kStore,Op::kResidual,Op::kPartial,Op::kArgmaxPartial,Op::kSwiGLU}){if(TN<32 && op==Op::kSwiGLU)continue;Run<TN,TK,Tiled>(m,k,op);}}
int main(){
  Matrix<8,64,false>();Matrix<8,64,true>();Matrix<16,128,false>();Matrix<16,128,true>();Matrix<32,64,false>();Matrix<32,64,true>();
  std::puts("GEMV: position-coded M/N/K tails, both weight layouts and all legal epilogues PASS");
}
