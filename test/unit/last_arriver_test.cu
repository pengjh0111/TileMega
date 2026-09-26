// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/LastArriverTaskBody.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace tilemega::codegen;
namespace backend=tilemega::backend;
using E=cutlass::bfloat16_t;
constexpr int M=3,N=64;
__global__ void Split(float* partial,E* output,unsigned* ticket,int splits,unsigned* last_count) {
  if(!executor::IsCompute())return;
  __shared__ unsigned last;__shared__ __align__(16) float scratch[16*64];
  for(int i=ComputeThread();i<M*N;i+=kComputeThreads)
    partial[blockIdx.x*M*N+i]=float((int(blockIdx.x)*17+i)%71-35)/64.0f;
  bool final=LastArriverGemmTaskBody<16,64,backend::ServingEpilogueOp::kStore>::Run(
      ticket,splits,&last,partial,splits,0,0,M,N,N,N,output,(E const*)nullptr,
      (float*)nullptr,(int*)nullptr,scratch);
  if(final && ComputeThread()==0)atomicAdd(last_count,1u);
}
__global__ void SplitReference(float const* partial,E* output,int splits) {
  __shared__ __align__(16) float scratch[16*64];
  ServingGemmCombineTaskBody<16,64,backend::ServingEpilogueOp::kStore>::Run(
      partial,splits,0,0,M,N,N,N,output,nullptr,nullptr,nullptr,scratch);
}
template<int D,int Q>
__global__ void Attn(float* partial,float* lse,E* output,unsigned* ticket,int past,int extent,int capacity,unsigned* last_count) {
  if(!executor::IsCompute())return;
  __shared__ unsigned last;
  if(int(blockIdx.x)*extent>=past+1)return;
  for(int i=ComputeThread();i<Q*D;i+=kComputeThreads)partial[blockIdx.x*Q*D+i]=float((int(blockIdx.x)*13+i)%63-31)/32.0f;
  if(ComputeThread()<Q)lse[blockIdx.x*Q+ComputeThread()]=float(int(blockIdx.x)%5)/4.0f;
  bool final=LastArriverAttentionTaskBody<D,Q,1>::Run(ticket,&last,blockIdx.x,partial,lse,output,0,0,1,capacity,extent,past);
  if(final && ComputeThread()==0)atomicAdd(last_count,1u);
}
template<int D,int Q>
__global__ void AttnReference(float const* partial,float const* lse,E* output,int past,int extent,int capacity) {
  AttentionMergeTaskBody<D,Q,1>::Run(partial,lse,output,0,0,1,capacity,extent,past);
}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
void Equal(E const* a,E const* b,int n){if(std::memcmp(a,b,n*sizeof(E))){std::fprintf(stderr,"last-arriver output mismatch\n");std::exit(3);}}
int main(){
  float *partial,*lse;E *out,*ref;unsigned *ticket,*count;
  Check(cudaMallocManaged(&partial,32*4*128*sizeof(float)));Check(cudaMallocManaged(&lse,32*4*sizeof(float)));
  Check(cudaMallocManaged(&out,4*128*sizeof(E)));Check(cudaMallocManaged(&ref,4*128*sizeof(E)));
  Check(cudaMallocManaged(&ticket,sizeof(unsigned)));Check(cudaMallocManaged(&count,sizeof(unsigned)));*ticket=0;*count=0;
  int cases=0;
  for(int threads:{128,160})for(int splits:{1,2,4,8,16,32}) {
    for(int repeat=0;repeat<7;++repeat){Split<<<splits,threads>>>(partial,out,ticket,splits,count);SplitReference<<<1,128>>>(partial,ref,splits);Check(cudaDeviceSynchronize());Equal(out,ref,M*N);if(*ticket!=0 || *count!=unsigned(++cases))return 4;}
  }
  for(int threads:{128,160})for(int past:{1,63,64,65,1086})for(int extent:{64,256,1088}) {
    constexpr int cap=1088;int blocks=(cap+extent-1)/extent;
    Attn<64,4><<<blocks,threads>>>(partial,lse,out,ticket,past,extent,cap,count);AttnReference<64,4><<<1,128>>>(partial,lse,ref,past,extent,cap);Check(cudaDeviceSynchronize());Equal(out,ref,4*64);if(*ticket!=0 || *count!=unsigned(++cases))return 5;
    Attn<128,2><<<blocks,threads>>>(partial,lse,out,ticket,past,extent,cap,count);AttnReference<128,2><<<1,128>>>(partial,lse,ref,past,extent,cap);Check(cudaDeviceSynchronize());Equal(out,ref,2*128);if(*ticket!=0 || *count!=unsigned(++cases))return 6;
  }
  std::printf("last-arriver development check: %d exact standalone-reduction comparisons; tickets reset; one reducer per output\n",cases);
  Check(cudaFree(partial));Check(cudaFree(lse));Check(cudaFree(out));Check(cudaFree(ref));Check(cudaFree(ticket));Check(cudaFree(count));
}
