// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseLastArriver.cuh>
#include <tilemega/Codegen/executor/Async.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<int S,bool Bulk> __global__ __launch_bounds__(128)
void test(unsigned* partial,std::uint64_t* count,std::uint64_t* grid,unsigned* errors) {
 __shared__ unsigned last;
 __shared__ __align__(1024) unsigned char storage[1024+S*128*4];
 auto* barrier=reinterpret_cast<std::uint64_t*>(storage);
 auto* copy=reinterpret_cast<unsigned*>(storage+1024);
 unsigned phase=0;
 if(threadIdx.x==0){Async::Init(barrier,1);Async::InitFence();}
 __syncthreads();
 for(unsigned iteration=0;iteration<2000;++iteration) {
  partial[blockIdx.x*128+threadIdx.x]=(iteration+1)*131+blockIdx.x*17+threadIdx.x;
  if(tilemega::backend::Sm90DenseLastArriver::Publish<S>(count,&last)) {
   if constexpr(Bulk) {
    if(threadIdx.x==0) {
     Async::ProxyAsyncGlobalFence();Async::ExpectTx(barrier,S*128*4);
     Async::Bulk(copy,partial,S*128*4,barrier);
    }
    Async::Wait(barrier,phase);phase^=1;
   }
   for(int rank=0;rank<S;++rank) {
    auto value=Bulk?copy[rank*128+threadIdx.x]:partial[rank*128+threadIdx.x];
    if(value!=(iteration+1)*131+rank*17+threadIdx.x)atomicAdd(errors,1);
   }
  }
  __syncthreads();
  if(threadIdx.x==0) {
   asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(grid),"l"(1ull):"memory");
   unsigned long long seen;
   do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(grid):"memory");}
   while(seen<static_cast<unsigned long long>(iteration+1)*S);
  }
  __syncthreads();
 }
 if(threadIdx.x==0)Async::Invalidate(barrier);
}
template<int S,bool Bulk> unsigned run() {
 unsigned *p,*errors,host;std::uint64_t *counter,*grid;
 check(cudaMalloc(&p,S*128*4));check(cudaMalloc(&errors,4));
 check(cudaMalloc(&counter,8));check(cudaMalloc(&grid,8));
 check(cudaMemset(counter,0,8));check(cudaMemset(grid,0,8));check(cudaMemset(errors,0,4));
 test<S,Bulk><<<S,128>>>(p,counter,grid,errors);check(cudaGetLastError());check(cudaDeviceSynchronize());
 check(cudaMemcpy(&host,errors,4,cudaMemcpyDeviceToHost));
 check(cudaFree(p));check(cudaFree(errors));check(cudaFree(counter));check(cudaFree(grid));return host;
}
int main() {
 try {check(cudaSetDevice(0));unsigned errors=0;
 errors+=run<2,false>();errors+=run<4,false>();errors+=run<16,false>();
 errors+=run<2,true>();errors+=run<4,true>();errors+=run<16,true>();
 std::printf("{\"errors\":%u,\"pass\":%s,\"cases\":6,\"rounds_per_case\":2000}\n",errors,errors?"false":"true");return errors?1:0;
 }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
