// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static void Check(cudaError_t s,const char* site) {
  if(s!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",site,cudaGetErrorString(s));std::exit(2);}
}
__global__ void First(unsigned* a,unsigned* step) {a[threadIdx.x]=*step+threadIdx.x;}
__global__ void Second(unsigned* a,unsigned* b) {b[threadIdx.x]=a[threadIdx.x]*3+1;}
__global__ void Finish(unsigned* b,unsigned* step,unsigned* bad) {
  if(b[threadIdx.x]!=(*step+threadIdx.x)*3+1)atomicAdd(bad,1u);
  __syncthreads();if(threadIdx.x==0)++*step;
}
int main() {
  unsigned *a,*b,*step,*bad;Check(cudaMalloc(&a,256*4),"alloc");Check(cudaMalloc(&b,256*4),"alloc");
  Check(cudaMalloc(&step,4),"alloc");Check(cudaMalloc(&bad,4),"alloc");
  Check(cudaMemset(step,0,4),"clear");Check(cudaMemset(bad,0,4),"clear");
  cudaStream_t stream;Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"nonblocking stream");
  cudaGraph_t graph;cudaGraphExec_t exec;
  Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"begin capture");
  First<<<1,256,0,stream>>>(a,step);Second<<<1,256,0,stream>>>(a,b);Finish<<<1,256,0,stream>>>(b,step,bad);
  Check(cudaStreamEndCapture(stream,&graph),"end capture");
  Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
  for(int i=0;i<2000;++i)Check(cudaGraphLaunch(exec,stream),"graph replay");
  Check(cudaStreamSynchronize(stream),"sync");
  unsigned hstep=0,hbad=0;Check(cudaMemcpy(&hstep,step,4,cudaMemcpyDeviceToHost),"read");Check(cudaMemcpy(&hbad,bad,4,cudaMemcpyDeviceToHost),"read");
  std::printf("{\"pass\":%s,\"replays\":%u,\"mismatches\":%u,\"kernels_per_graph\":3}\n",hstep==2000&&!hbad?"true":"false",hstep,hbad);
  cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);cudaStreamDestroy(stream);cudaFree(a);cudaFree(b);cudaFree(step);cudaFree(bad);
  return hstep==2000&&!hbad?0:1;
}
