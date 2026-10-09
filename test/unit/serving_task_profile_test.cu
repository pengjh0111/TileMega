// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_TRACE_TASK 1
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/ServingGemvTaskBody.h>
#include <cstdio>
#include <cstdlib>
using namespace tilemega;
using E=cutlass::bfloat16_t;
#ifdef TILEMEGA_ARCH_ID
using Arch=arch::CurrentArch;
#else
using Arch=arch::Sm80;
#endif
template<class Body> __global__ void Probe(codegen::ServingGemmOperands p) {
  __shared__ __align__(16) char work[Body::kSharedBytes];
  if(threadIdx.x==0)p.profile->run_begin=codegen::TaskProfileNow(p.profile);
  Body::Run(p,0,0,work);
  if(threadIdx.x==0)p.profile->run_end=codegen::TaskProfileNow(p.profile);
}
void Check(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* Allocate(int n){T* p;Check(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<class Body>void Run() {
  auto* a=Allocate<E>(128);auto* b=Allocate<E>(32*128);auto* out=Allocate<E>(32);
  auto* profile=Allocate<codegen::ServingTaskProfile>(1);*profile={};
  for(int i=0;i<128;++i)a[i]=E(1);
  for(int i=0;i<32*128;++i)b[i]=E(1);
  codegen::ServingGemmOperands p;p.a=a;p.b=b;p.output=out;p.profile=profile;
  p.m=1;p.n=32;p.k_count=p.k_total=128;p.output_stride=32;
  Probe<Body><<<1,128>>>(p);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  codegen::ServingTaskProfile observed;
  Check(cudaMemcpy(&observed,profile,sizeof(observed),cudaMemcpyDeviceToHost));
  if(!(observed.run_begin<=observed.first_ready && observed.first_ready<observed.run_end)) {
    std::fprintf(stderr,"profile begin=%llu first=%llu end=%llu\n",observed.run_begin,observed.first_ready,observed.run_end);std::exit(3);
  }
  for(int i=0;i<32;++i)if(float(out[i])!=128){std::fprintf(stderr,"output %d = %g, expected 128\n",i,float(out[i]));std::exit(4);}
  Check(cudaFree(a));Check(cudaFree(b));Check(cudaFree(out));Check(cudaFree(profile));
}
int main(){Run<codegen::ServingGemmTaskBody<Arch,16,32,64,2>>();
  Run<codegen::ServingGemvTaskBody<Arch,32,64>>();
  std::puts("PASS first operand readiness lies within both GEMM and GEMV tasks");}
