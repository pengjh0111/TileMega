// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/MonotonicLastArriver.cuh>
#include <cstdio>
#include <cstdlib>
using tilemega::codegen::executor::MonotonicLastArriver;
static void Check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(2);}}
__global__ void Produce(unsigned long long* tickets,int* partial,int* output,
    int* completions,unsigned producers,unsigned live,unsigned iteration,unsigned mode) {
  __shared__ unsigned last;
  if(blockIdx.x<live)partial[blockIdx.x*128+threadIdx.x]=int(iteration+1)+int(blockIdx.x)+int(threadIdx.x);
  bool reduced=MonotonicLastArriver::Run(tickets+mode,producers,iteration,&last,[&]{
    int sum=0;for(unsigned c=0;c<live;++c)sum+=partial[c*128+threadIdx.x];
    output[threadIdx.x]=sum;
  });
  if(reduced && threadIdx.x==0)atomicAdd(completions,1);
}
int main(){
  unsigned long long* tickets;int *partial,*output,*completions;
  Check(cudaMallocManaged(&tickets,2*sizeof(*tickets)));Check(cudaMallocManaged(&partial,9*128*sizeof(int)));
  Check(cudaMallocManaged(&output,128*sizeof(int)));Check(cudaMallocManaged(&completions,sizeof(int)));
  Check(cudaMemset(tickets,0,2*sizeof(*tickets)));*completions=0;
  for(unsigned iteration=0;iteration<32;++iteration)for(unsigned mode=0;mode<2;++mode){
    unsigned live=1+(iteration*3+mode)%9;
    Produce<<<9,128>>>(tickets,partial,output,completions,9,live,iteration,mode);Check(cudaDeviceSynchronize());
    for(int t=0;t<128;++t){int expected=0;for(unsigned c=0;c<live;++c)expected+=iteration+1+c+t;
      if(output[t]!=expected)std::exit(3);}
    if(tickets[mode]!=(iteration+1)*9 || *completions!=int(iteration*2+mode+1))std::exit(4);
  }
  std::puts("Monotonic LA: 32 iterations x 2 independent modes, varying live chunks PASS (single process)");
  Check(cudaFree(tickets));Check(cudaFree(partial));Check(cudaFree(output));Check(cudaFree(completions));
}
