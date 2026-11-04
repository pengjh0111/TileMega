// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

using namespace tilemega;
using BF=cutlass::bfloat16_t;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Binding {
  void* buffers[3];
  template<unsigned I> __device__ BF* Get(int) const {return static_cast<BF*>(buffers[I]);}
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
template<int Batch,int Q,int Chunks>
__global__ __launch_bounds__(160,2) void exercise(Binding b,BF* reference,
    std::uint64_t* arrivals,unsigned long long* barrier,unsigned* errors) {
  using Body=backend::Sm90DenseAttentionBody<Batch,8,128,Q,0,Chunks,64,2,0,1088>;
  using Stage=codegen::native_dense::AttentionMerge<Batch,8,128,Q,Chunks,0,1,2>;
  extern __shared__ __align__(1024) char shared[];
  __shared__ backend::Sm90DenseOperandBank operands;
  __shared__ backend::Sm90DenseRingState ring;
  auto& storage=*reinterpret_cast<typename Body::Storage*>(shared+1024);
  auto* partial=reinterpret_cast<float*>(b.Get<0>(0));
  auto* lse=reinterpret_cast<float*>(b.Get<1>(0));
  if(threadIdx.x==0)Stage::PublishOperands(b,0,0,0,operands,0);
  __syncthreads();
  auto ref=operands.attention;ref.output=reference;
  for(unsigned iteration=0;iteration<128;++iteration) {
    if(threadIdx.x<128)for(int t=blockIdx.x;t<Batch*8*Chunks;t+=gridDim.x) {
      int chunk=t%Chunks,bg=((t/Chunks)%Batch)*8+t/(Chunks*Batch);
      for(int i=threadIdx.x;i<Q*128;i+=128)
        partial[(bg*Chunks+chunk)*Q*128+i]=float((bg*7+chunk*11+i+int(iteration))%101-50)/32;
      for(int q=threadIdx.x;q<Q;q+=128)
        lse[(bg*Chunks+chunk)*Q+q]=((bg+iteration)%13==0 || (chunk+iteration)%7==0)?
            -INFINITY:float((bg+chunk*3+q+int(iteration))%29-14);
      // Rotate the delayed writer and exercise empty chunks and all-masked rows.
      if(chunk==iteration%Chunks) {
        auto start=clock64();while(clock64()-start<1024) {}
      }
      bool last=backend::Sm90DenseLastArriver::Publish<Chunks>(arrivals+bg,&storage.last);
      if(last)Body::Merge(ref,storage,bg);
      codegen::executor::ComputeSync();
    }
    join(barrier,3*iteration+1);
    Stage::template Run<0>(b,0,shared,0,ring,backend::Sm90DenseNoTail{},operands,false);
    join(barrier,3*iteration+2);
    for(int i=blockIdx.x*160+threadIdx.x;i<Batch*8*Q*128;i+=gridDim.x*160)
      if(b.Get<2>(0)[i].raw()!=reference[i].raw())atomicAdd(errors,1);
    join(barrier,3*iteration+3);
  }
}
template<int Batch,int Q,int Chunks> unsigned run(int grid) {
  Binding b{};
  using Body=backend::Sm90DenseAttentionBody<Batch,8,128,Q,0,Chunks,64,2,0,1088>;
  std::size_t sizes[]={Batch*8*Chunks*Q*128*4,Batch*8*Chunks*Q*4,Batch*8*Q*128*2};
  for(int i=0;i<3;++i)check(cudaMalloc(&b.buffers[i],sizes[i]));
  BF* reference;std::uint64_t* arrivals;unsigned long long* barrier;unsigned *errors,host;
  check(cudaMalloc(&reference,sizes[2]));check(cudaMalloc(&arrivals,Batch*8*8));
  check(cudaMalloc(&barrier,8));check(cudaMalloc(&errors,4));
  check(cudaMemset(arrivals,0,Batch*8*8));check(cudaMemset(barrier,0,8));check(cudaMemset(errors,0,4));
  check(cudaFuncSetAttribute(exercise<Batch,Q,Chunks>,cudaFuncAttributeMaxDynamicSharedMemorySize,
      Body::kSharedBytes));
  exercise<Batch,Q,Chunks><<<grid,160,Body::kSharedBytes>>>(b,reference,arrivals,barrier,errors);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(&host,errors,4,cudaMemcpyDeviceToHost));
  for(auto p:b.buffers)check(cudaFree(p));
  check(cudaFree(reference));check(cudaFree(arrivals));check(cudaFree(barrier));check(cudaFree(errors));
  return host;
}
int main() {try {
  check(cudaSetDevice(0));unsigned errors=0;
  for(int grid:{132,264}) {
    errors+=run<1,2,16>(grid);errors+=run<16,2,2>(grid);
    errors+=run<1,8,64>(grid);errors+=run<16,8,16>(grid);
  }
  std::printf("{\"pass\":%s,\"errors\":%u,\"cases\":8,\"rounds\":128}\n",
      errors?"false":"true",errors);return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
