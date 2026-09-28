// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
using Element=cutlass::bfloat16_t;
using namespace tilemega::codegen;
using TestArch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
    tilemega::arch::Sm80,tilemega::arch::CurrentArch>;
template<int N,int K,int Page,int Pages,bool ForceSm80=false>
__global__ void RunPaged(ServingGemmOperands p) {
  using Body=PagedGemmTaskBody<TestArch,16,N,K,Page,Pages,ForceSm80>;
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
template<int N,int K,int Page,int Pages>
__global__ void RunPagedMany(ServingGemmOperands p) {
  using Body=PagedGemmTaskBody<TestArch,16,N,K,Page,Pages>;
  using Ring=typename Body::Ring;
  extern __shared__ __align__(1024) char storage[];
  constexpr int pool=(1024+(Body::kActivationBytes>Body::kScratchBytes?
      Body::kActivationBytes:Body::kScratchBytes)+1023)/1024*1024;
  Ring ring{reinterpret_cast<typename Ring::Slot*>(storage),storage+pool};
  ring.Initialize();std::uint64_t sequence=0;
  p.output+=std::size_t(blockIdx.x)*16*N;
  if(executor::IsCompute())Body::Run(p,0,0,ring,sequence,storage+1024);
  else Body::Load(p,0,ring,sequence);
}
void CheckCuda(cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(code));std::exit(2);}}
template<int N,int K,int Page,int Pages,bool ForceSm80=false>
void Check(int rows,int columns,int reduction) {
  using Body=PagedGemmTaskBody<TestArch,16,N,K,Page,Pages,ForceSm80>;
  int pitch=(reduction+7)&~7;Element *a=nullptr,*b=nullptr,*out=nullptr;
  int kt=(reduction+K-1)/K,nt=(columns+N-1)/N;
  CheckCuda(cudaMallocManaged(&a,rows*pitch*2));
  CheckCuda(cudaMallocManaged(&b,std::size_t(nt)*kt*N*K*2));
  CheckCuda(cudaMallocManaged(&out,rows*columns*2));
  for(int row=0;row<rows;++row)for(int k=0;k<pitch;++k)
    a[row*pitch+k]=Element(k<reduction?((row*13+k*7)%29-14)*0.015625f:123.0f);
  for(std::size_t i=0;i<std::size_t(nt)*kt*N*K;++i)b[i]=Element(0.0f);
  for(int n=0;n<columns;++n)for(int k=0;k<reduction;++k) {
    int rn=n%N,rk=k%K;
    int off=512*(rn/8+(N/8)*(rk/64))+64*(rn%8)+
            8*(((rk%64)/8)^(rn%8))+(rk%8);
    b[(std::size_t(n/N)*kt+k/K)*N*K+off]=
        Element(((n*17+k*11)%31-15)*0.015625f);
  }
  ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=b;p.output=out;
  p.m=rows;p.n=columns;p.k_total=reduction;p.k_total_full=reduction;
  p.k_count=reduction;p.a_row_stride=pitch;p.b_row_stride=pitch;
  p.output_stride=columns;
  constexpr int pool=(1024+(Body::kActivationBytes>Body::kScratchBytes?Body::kActivationBytes:Body::kScratchBytes)+1023)/1024*1024;
  constexpr int bytes=pool+Page*Pages;
  CheckCuda(cudaFuncSetAttribute(RunPaged<N,K,Page,Pages,ForceSm80>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  RunPaged<N,K,Page,Pages,ForceSm80><<<1,160,bytes>>>(p);CheckCuda(cudaDeviceSynchronize());
  for(int row=0;row<rows;++row)for(int n=0;n<columns;++n) {
    float ref=0;for(int k=0;k<reduction;++k)
      ref+=float(a[row*pitch+k])*float(Element(((n*17+k*11)%31-15)*0.015625f));
    float expected=float(Element(ref)),actual=float(out[row*columns+n]);
    float ulp=expected==0?std::ldexp(1.0f,-133):std::ldexp(1.0f,std::ilogb(std::abs(expected))-7);
    if(std::abs(actual-expected)>ulp){std::fprintf(stderr,"FAIL N%d K%d page%d slots%d M%d N%d K%d [%d,%d]: %.9g %.9g\n",N,K,Page,Pages,rows,columns,reduction,row,n,actual,expected);std::exit(3);}
  }
  CheckCuda(cudaFree(a));CheckCuda(cudaFree(b));CheckCuda(cudaFree(out));
}
int main(){
  for(int m:{1,16})for(int n:{31,73})for(int k:{2001,2048}){
    Check<32,64,8192,3>(m,n,k);
    Check<128,64,16384,3>(m,n,k);
    Check<64,128,16384,3>(m,n,k);
    Check<64,64,8192,3>(m,n,k);
  }
  Check<32,64,8192,3,true>(1,31,2001);
  // The body calibration runs one CTA on every SM. A single-CTA test cannot
  // expose occupancy-dependent page-ring stalls, so exercise that shape too.
  {
    using Body=PagedGemmTaskBody<TestArch,16,32,64,16384,5>;
    Element *a=nullptr,*b=nullptr,*out=nullptr;
    float* argmax_value=nullptr;int* argmax_index=nullptr;
    CheckCuda(cudaMalloc(&a,16*128*2));CheckCuda(cudaMalloc(&b,32*128*2));
    int device=0,sms=0;CheckCuda(cudaGetDevice(&device));
    CheckCuda(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    CheckCuda(cudaMalloc(&out,std::size_t(sms)*16*32*2));
    CheckCuda(cudaMalloc(&argmax_value,std::size_t(sms)*16*sizeof(float)));
    CheckCuda(cudaMalloc(&argmax_index,std::size_t(sms)*16*sizeof(int)));
    CheckCuda(cudaMemset(a,0,16*128*2));CheckCuda(cudaMemset(b,0,32*128*2));
    ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=b;p.output=out;
    p.m=16;p.n=32;p.k_total=p.k_total_full=p.k_count=128;
    p.a_row_stride=128;p.output_stride=32;p.partial_stride=32;
    p.residual=out;p.argmax_value=argmax_value;p.argmax_index=argmax_index;
    constexpr int pool=(1024+(Body::kActivationBytes>Body::kScratchBytes?
        Body::kActivationBytes:Body::kScratchBytes)+1023)/1024*1024;
    constexpr int bytes=pool+5*16384;
    CheckCuda(cudaFuncSetAttribute(RunPagedMany<32,64,16384,5>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    RunPagedMany<32,64,16384,5><<<sms,160,bytes>>>(p);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaFree(a));CheckCuda(cudaFree(b));CheckCuda(cudaFree(out));
    CheckCuda(cudaFree(argmax_value));CheckCuda(cudaFree(argmax_index));
  }
  std::puts("paged GEMM: 33 tile-page/residue/current-arch/forced-sm80 cases pass");
}
