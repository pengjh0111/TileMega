// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using Element=cutlass::bfloat16_t;
using namespace tilemega::codegen;
template<int N,int K,int Page,int Pages>
__global__ void RunPaged(ServingGemmOperands p) {
  using Body=PagedGemmTaskBody<tilemega::arch::Sm80,16,N,K,Page,Pages>;
  using Ring=typename Body::Ring;
  extern __shared__ __align__(1024) char storage[];
  constexpr int work=1024;
  constexpr int pool=(work+(Body::kActivationBytes>Body::kScratchBytes?Body::kActivationBytes:Body::kScratchBytes)+1023)/1024*1024;
  Ring ring{reinterpret_cast<typename Ring::Slot*>(storage),storage+pool};
  ring.Initialize();std::uint64_t sequence=0;
  for(int m=0;m<(p.m+15)/16;++m)for(int n=0;n<(p.n+N-1)/N;++n) {
    if(!executor::IsCompute())Body::Load(p,n,ring,sequence);
    else Body::Run(p,m,n,ring,sequence,storage+work);
  }
}
void CheckCuda(cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(code));std::exit(2);}}
template<int N,int K,int Page,int Pages>
void Check(int rows,int columns,int reduction) {
  using Body=PagedGemmTaskBody<tilemega::arch::Sm80,16,N,K,Page,Pages>;
  int pitch=(reduction+7)&~7;Element *a=nullptr,*b=nullptr,*out=nullptr;
  CheckCuda(cudaMallocManaged(&a,rows*pitch*2));CheckCuda(cudaMallocManaged(&b,columns*pitch*2));CheckCuda(cudaMallocManaged(&out,rows*columns*2));
  for(int row=0;row<rows;++row)for(int k=0;k<pitch;++k)
    a[row*pitch+k]=Element(k<reduction?((row*13+k*7)%29-14)*0.015625f:123.0f);
  for(int n=0;n<columns;++n)for(int k=0;k<pitch;++k)
    b[n*pitch+k]=Element(k<reduction?((n*17+k*11)%31-15)*0.015625f:123.0f);
  ServingGemmOperands p;p.a=a;p.b=b;p.output=out;p.m=rows;p.n=columns;p.k_total=reduction;p.k_count=reduction;
  p.a_row_stride=p.b_row_stride=pitch;p.output_stride=columns;
  constexpr int pool=(1024+(Body::kActivationBytes>Body::kScratchBytes?Body::kActivationBytes:Body::kScratchBytes)+1023)/1024*1024;
  constexpr int bytes=pool+Page*Pages;
  CheckCuda(cudaFuncSetAttribute(RunPaged<N,K,Page,Pages>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  RunPaged<N,K,Page,Pages><<<1,160,bytes>>>(p);CheckCuda(cudaDeviceSynchronize());
  for(int row=0;row<rows;++row)for(int n=0;n<columns;++n) {
    float ref=0;for(int k=0;k<reduction;++k)ref+=float(a[row*pitch+k])*float(b[n*pitch+k]);
    float expected=float(Element(ref)),actual=float(out[row*columns+n]);
    float ulp=expected==0?std::ldexp(1.0f,-133):std::ldexp(1.0f,std::ilogb(std::abs(expected))-7);
    if(std::abs(actual-expected)>ulp){std::fprintf(stderr,"FAIL N%d K%d page%d slots%d M%d N%d K%d [%d,%d]: %.9g %.9g\n",N,K,Page,Pages,rows,columns,reduction,row,n,actual,expected);std::exit(3);}
  }
  CheckCuda(cudaFree(a));CheckCuda(cudaFree(b));CheckCuda(cudaFree(out));
}
int main(){
  for(int m:{1,3,17})for(int n:{31,73})for(int k:{2001,2048}){
    Check<32,64,8192,3>(m,n,k);
    Check<128,64,8192,3>(m,n,k);
    Check<128,128,16384,3>(m,n,k);
    Check<256,64,16384,3>(m,n,k);
  }
  // This crosses the physical page boundary with valid output columns.
  Check<256,64,16384,3>(3,193,2048);
  std::puts("paged GEMM: 49 residue/cross-task/grouped-page/multi-page cases pass");
}
