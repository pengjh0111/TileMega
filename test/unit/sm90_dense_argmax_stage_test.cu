// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace tilemega;
using BF=cutlass::bfloat16_t;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Binding {
  void* buffers[5];int prompt_len;
  template<unsigned I> __device__ BF* Get() const {return static_cast<BF*>(buffers[I]);}
};
__device__ void join(unsigned long long* counter,unsigned epoch) {
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
    unsigned long long seen;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
    while(seen<static_cast<unsigned long long>(epoch)*gridDim.x);
  }
  __syncthreads();
}
template<int Batch,bool Embed>
__global__ __launch_bounds__(160,2) void exercise(Binding b,int* reference,
    std::uint64_t* last,unsigned long long* barrier,unsigned* errors) {
  __shared__ __align__(1024) char scratch[4096];
  __shared__ backend::Sm90DenseRingState ring;
  backend::Sm90DenseOperands operands{};
  operands.output=b.Get<0>();operands.argmax_index=reinterpret_cast<int*>(b.Get<1>());
  backend::Sm90DenseHeadOperands head{reference,last,b.Get<3>(),b.Get<4>(),0,b.prompt_len};
  using Stage=codegen::native_dense::Argmax<Batch,128,1088,0,1,2,3,4,Embed>;
  for(unsigned iteration=0;iteration<256;++iteration) {
    int past=iteration%32;head.past=past;
    float best[Batch];int indices[Batch];
    for(int row=0;row<Batch;++row) {
      best[row]=float((blockIdx.x+row+iteration)%17);
      indices[row]=(blockIdx.x*3+row+iteration)%64;
      if((blockIdx.x+iteration)%29==0)best[row]=NAN;
    }
    // Delay a rotating producer so both arrival orders and the barrier's
    // partial publication are exercised over repeated counter generations.
    if(blockIdx.x==iteration%gridDim.x) {
      auto start=clock64();while(clock64()-start<2048) {}
    }
    if(threadIdx.x<128)
      backend::Sm90DenseHeadFinish<Batch,128,1088,false>{&head}(
          operands,ring,best,indices,scratch,sizeof(scratch));
    join(barrier,3*iteration+1);
    if(threadIdx.x<128)
      backend::Sm90DenseHeadPartialFinish<Batch>{}(
          operands,ring,best,indices,scratch,sizeof(scratch));
    join(barrier,3*iteration+2);
    Stage::template Run<0>(b,past,scratch,0,ring,backend::Sm90DenseNoTail{});
    if(blockIdx.x<Batch && threadIdx.x==0) {
      int row=blockIdx.x,next=row*1088+past+1;
      int token=reinterpret_cast<int*>(b.Get<2>())[next];
      if(token!=reference[next] || token<0 || token>=64)atomicAdd(errors,1);
      if constexpr(Embed)for(int col=0;col<128;++col)
        if(b.Get<4>()[row*128+col].raw()!=b.Get<3>()[token*128+col].raw())atomicAdd(errors,1);
    }
    join(barrier,3*iteration+3);
  }
}
template<int Batch,bool Embed> unsigned run(int grid) {
  Binding b{};b.prompt_len=8;
  std::size_t sizes[]={std::size_t(Batch*grid*4),std::size_t(Batch*grid*4),
      std::size_t(Batch*1088*4),64*128*2,std::size_t(Batch*128*2)};
  for(int i=0;i<5;++i)check(cudaMalloc(&b.buffers[i],sizes[i]));
  std::vector<int> tokens(Batch*1088,7);
  std::vector<BF> table(64*128);
  for(unsigned i=0;i<table.size();++i)table[i]=BF(float(int(i%37)-18)/32);
  check(cudaMemcpy(b.buffers[2],tokens.data(),sizes[2],cudaMemcpyHostToDevice));
  check(cudaMemcpy(b.buffers[3],table.data(),sizes[3],cudaMemcpyHostToDevice));
  int* reference;std::uint64_t* last;unsigned long long* barrier;unsigned *errors,host;
  check(cudaMalloc(&reference,sizes[2]));check(cudaMalloc(&last,8));
  check(cudaMalloc(&barrier,8));check(cudaMalloc(&errors,4));
  check(cudaMemcpy(reference,tokens.data(),sizes[2],cudaMemcpyHostToDevice));
  check(cudaMemset(last,0,8));check(cudaMemset(barrier,0,8));check(cudaMemset(errors,0,4));
  exercise<Batch,Embed><<<grid,160>>>(b,reference,last,barrier,errors);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(&host,errors,4,cudaMemcpyDeviceToHost));
  for(auto p:b.buffers)check(cudaFree(p));
  check(cudaFree(reference));check(cudaFree(last));check(cudaFree(barrier));check(cudaFree(errors));
  return host;
}
int main() {try {
  check(cudaSetDevice(0));unsigned errors=0;
  for(int grid:{132,264}) {
    errors+=run<1,false>(grid);errors+=run<16,false>(grid);
    errors+=run<1,true>(grid);errors+=run<16,true>(grid);
  }
  std::printf("{\"pass\":%s,\"errors\":%u,\"cases\":8,\"rounds\":256}\n",
      errors?"false":"true",errors);return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
