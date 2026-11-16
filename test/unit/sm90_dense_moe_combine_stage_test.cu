// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
using namespace tilemega;
using BF=cutlass::bfloat16_t;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Binding {
  void* buffers[2];
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
template<int Batch,int Experts,int TN,bool FloatWeights>
__global__ __launch_bounds__(160,2) void exercise(Binding b,BF* reference,float* ref_partial,
    float* logits,std::uint64_t* arrivals,unsigned long long* barrier,unsigned* errors) {
  constexpr int N=256,TopK=8,Tiles=N/TN;
  using Ref=backend::Sm90DenseMoeGemm<Batch,N,64,TN,64,2,4,1,
      backend::Sm90DenseMoeMode::kDown,16,Experts,TopK,FloatWeights>;
  using Candidate=backend::Sm90DenseMoeGemm<Batch,N,64,TN,64,2,4,1,
      backend::Sm90DenseMoeMode::kDown,16,Experts,TopK,FloatWeights,0,1,false,false,false,true>;
  using Route=backend::Sm90DenseLocalRoute<Batch,Experts,TopK,FloatWeights>;
  using Stage=codegen::native_dense::MoeCombine<Batch,N,TN,TopK,0,1>;
  __shared__ backend::Sm90DenseLocalSchedule<Batch,Experts,TopK> routing;
  __shared__ backend::Sm90DenseOperandBank operands;
  __shared__ backend::Sm90DenseRingState ring;
  __shared__ float tile[16*TN];
  __shared__ int flag;
  __shared__ __align__(1024) char shared[1024];
  if(threadIdx.x==0)Stage::PublishOperands(b,0,0,0,operands,0);
  __syncthreads();
  backend::Sm90DenseMoeOperands ref{};ref.output=ref_partial;ref.x=reference;ref.counter=arrivals;
  auto candidate=ref;candidate.output=b.Get<0>(0);candidate.x=b.Get<1>(0);
  for(unsigned iteration=0;iteration<128;++iteration) {
    for(int i=blockIdx.x*160+threadIdx.x;i<Batch*N;i+=gridDim.x*160)
      reference[i]=candidate.x[i]=BF(float((i+int(iteration))%37-18)/8);
    for(int i=blockIdx.x*160+threadIdx.x;i<Batch*Experts;i+=gridDim.x*160)
      logits[i]=float(BF(float((i*17+int(iteration)*23)%101-50)/16));
    join(barrier,4*iteration+1);
    if(threadIdx.x<128) {
      Route::Build(logits,routing);
      for(int t=blockIdx.x;t<routing.active*Tiles;t+=gridDim.x) {
        for(int i=threadIdx.x;i<16*TN;i+=128)
          tile[i]=float((i+t*13+int(iteration))%101-50)/32;
        codegen::executor::ComputeSync();
        if(t%Tiles==int(iteration)%Tiles) {
          auto start=clock64();while(clock64()-start<1024) {}
        }
        Ref::Epilogue(ref,tile,nullptr,t,&flag,&routing);
        codegen::executor::ComputeSync();
        Candidate::Epilogue(candidate,tile,nullptr,t,&flag,&routing);
        codegen::executor::ComputeSync();
      }
    }
    join(barrier,4*iteration+2);
    Stage::template Run<0>(b,0,shared,0,ring,backend::Sm90DenseNoTail{},operands,false);
    join(barrier,4*iteration+3);
    for(int i=blockIdx.x*160+threadIdx.x;i<Batch*N;i+=gridDim.x*160)
      if(reference[i].raw()!=candidate.x[i].raw())atomicAdd(errors,1);
    for(int i=blockIdx.x*160+threadIdx.x;i<Batch*TopK*N;i+=gridDim.x*160)
      if(__float_as_uint(ref_partial[i])!=__float_as_uint(static_cast<float*>(candidate.output)[i]))
        atomicAdd(errors,1);
    if(blockIdx.x==0 && threadIdx.x<Tiles && arrivals[threadIdx.x])atomicAdd(errors,1);
    join(barrier,4*iteration+4);
  }
}
template<int Batch,int Experts,int TN,bool FloatWeights> unsigned run(int grid) {
  Binding b{};BF* reference;float *ref_partial,*logits;
  std::uint64_t* arrivals;unsigned long long* barrier;unsigned *errors,host;
  check(cudaMalloc(&b.buffers[0],Batch*8*256*4));check(cudaMalloc(&b.buffers[1],Batch*256*2));
  check(cudaMalloc(&reference,Batch*256*2));check(cudaMalloc(&ref_partial,Batch*8*256*4));
  check(cudaMalloc(&logits,Batch*Experts*4));check(cudaMalloc(&arrivals,(256/TN)*8));
  check(cudaMalloc(&barrier,8));check(cudaMalloc(&errors,4));
  check(cudaMemset(arrivals,0,(256/TN)*8));check(cudaMemset(barrier,0,8));check(cudaMemset(errors,0,4));
  exercise<Batch,Experts,TN,FloatWeights><<<grid,160>>>(b,reference,ref_partial,logits,arrivals,barrier,errors);
  check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(&host,errors,4,cudaMemcpyDeviceToHost));
  for(auto p:b.buffers)check(cudaFree(p));
  check(cudaFree(reference));check(cudaFree(ref_partial));check(cudaFree(logits));
  check(cudaFree(arrivals));check(cudaFree(barrier));check(cudaFree(errors));return host;
}
int main() {try {
  check(cudaSetDevice(0));unsigned errors=0;
  for(int grid:{132,264}) {
    errors+=run<1,64,32,false>(grid);errors+=run<16,64,128,true>(grid);
    errors+=run<1,128,128,true>(grid);errors+=run<16,128,32,false>(grid);
  }
  std::printf("{\"pass\":%s,\"errors\":%u,\"cases\":8,\"rounds\":128}\n",errors?"false":"true",errors);
  return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
