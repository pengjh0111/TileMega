// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace tilemega::codegen::executor;
template<int Bytes,int Pages>
__global__ void CopyPages(unsigned const* input,unsigned* output,int rounds) {
  extern __shared__ __align__(1024) char storage[];
  using Ring=PageRing<Bytes,Pages>;
  constexpr int offset=1024;
  Ring ring{reinterpret_cast<typename Ring::Slot*>(storage),storage+offset};
  ring.Initialize();
  if(!IsCompute()) {
    for(int r=0;r<rounds;++r) {
      ring.AcquireEmpty(r);
      for(int byte=LoaderLane()*16;byte<Bytes;byte+=kLoaderThreads*16)
        Ring::Copy::Copy16(ring.Page(r)+byte,input+(r*Bytes+byte)/4);
      ring.PublishCopies(r);
    }
  }else {
    for(int r=0;r<rounds;++r) {
      ring.AwaitFull(r);
      auto* data=reinterpret_cast<unsigned const*>(ring.Page(r));
      for(int i=ComputeThread();i<Bytes/4;i+=kComputeThreads)
        output[r*Bytes/4+i]=data[i];
      ring.Release(r);
    }
  }
}
template<int Bytes,int Pages> void Check() {
  constexpr int rounds=103;unsigned *in=nullptr,*out=nullptr;
  auto check=[](cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(code));std::exit(2);}};
  check(cudaMallocManaged(&in,rounds*Bytes));check(cudaMallocManaged(&out,rounds*Bytes));
  for(int i=0;i<rounds*Bytes/4;++i){in[i]=(i*31337u)^0x5a5a1234u;out[i]=0;}
  check(cudaFuncSetAttribute(CopyPages<Bytes,Pages>,cudaFuncAttributeMaxDynamicSharedMemorySize,1024+Bytes*Pages));
  CopyPages<Bytes,Pages><<<1,kComputeThreads+kLoaderThreads,1024+Bytes*Pages>>>(in,out,rounds);
  check(cudaDeviceSynchronize());
  for(int i=0;i<rounds*Bytes/4;++i)if(in[i]!=out[i]){std::fprintf(stderr,"page mismatch %d %u %u\n",i,in[i],out[i]);std::exit(3);}
  check(cudaFree(in));check(cudaFree(out));
}
int main(){Check<8192,1>();Check<8192,3>();Check<16384,2>();std::puts("page ring cp.async: 3 cases / 103 wraps pass");}
