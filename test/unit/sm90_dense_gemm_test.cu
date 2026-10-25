// SPDX-License-Identifier: BSD-3-Clause
// D3 GEMM check (prompt §5): TMA activation, bulk weights, one-arrival ring.
// Each repeat rewrites the activation with generic stores from every CTA and
// passes a grid barrier before the TMA reads it, so a missing proxy fence or
// a stale slot shows up as a sign error. All products are dyadic: exact.
#ifndef TEST_BATCH
#define TEST_BATCH 1
#endif
#define TILEMEGA_SWIGLU_U 8
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
constexpr int B=TEST_BATCH,kGrid=4,kRepeats=4;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}

__device__ void GridBarrier(unsigned long long* counter,unsigned long long round) {
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    unsigned long long v;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<(round+1)*gridDim.x);
  }
  __syncthreads();
}

template<int N,int K,int TN,int TK,int WN,int S1>
__global__ __launch_bounds__(160) void evaluate(Sm90DenseOperands first,Sm90DenseOperands second,
    BF const* source,BF* activation,unsigned long long* counter) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  using A=Sm90DenseGemm<B,N,K,TN,TK,S1,WN,4/WN,ServingEpilogueOp::kStore,8,false>;
  using C=Sm90DenseGemm<B,N,K,TN,TK,2,WN,4/WN,ServingEpilogueOp::kStore,8,false>;
  for(int repeat=0;repeat<kRepeats;++repeat) {
    for(int i=blockIdx.x*160+threadIdx.x;i<B*K;i+=gridDim.x*160)
      activation[i]=repeat%2?BF(-float(source[i])):source[i];
    GridBarrier(counter,2*repeat);
    A::template Run<0>(first,shared);
    C::template Run<0>(second,shared);
    GridBarrier(counter,2*repeat+1);
  }
  state.Invalidate();
}

template<int TN,int WN,int TK,int K=1024>
int run() {
  constexpr int N=TN*25;
  // Alternating 3- and 2-slot rings exercise phase carry-over; the widest
  // tiles fit the shared budget only with two slots.
  constexpr int S1=Sm90DenseGemm<B,N,K,TN,TK,3,WN,4/WN,ServingEpilogueOp::kStore,8,false>::kSharedBytes<=232448?3:2;
  using A=Sm90DenseGemm<B,N,K,TN,TK,S1,WN,4/WN,ServingEpilogueOp::kStore,8,false>;
  constexpr int rows=Sm90DenseLayout::ActivationRows(B);
  std::vector<BF> input(B*K),weight(std::size_t(N)*K),output(B*N);
  for(int b=0;b<B;++b)for(int k=0;k<K;++k)input[b*K+k]=BF((b+1)*(k%7+1)/64.0f);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    int row=n%TN,col=k%TK;
    size_t permutation=512*(row/8+(TN/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
    weight[(size_t(n/TN)*(K/TK)+k/TK)*TN*TK+permutation]=BF((n%29-14)*(k%11+1)/256.0f);
  }
  BF *src,*act,*w,*o,*other;unsigned long long* counter;CUtensorMap* map;
  check(cudaMalloc(&src,input.size()*2));check(cudaMalloc(&act,input.size()*2));
  check(cudaMalloc(&w,weight.size()*2));check(cudaMalloc(&o,output.size()*2));
  check(cudaMalloc(&other,output.size()*2));check(cudaMalloc(&counter,8));check(cudaMalloc(&map,128));
  check(cudaMemset(counter,0,8));check(cudaMemset(act,0,input.size()*2));
  check(cudaMemcpy(src,input.data(),input.size()*2,cudaMemcpyHostToDevice));
  check(cudaMemcpy(w,weight.data(),weight.size()*2,cudaMemcpyHostToDevice));
  CUtensorMap host;
  tilemega::codegen::executor::EncodeTiledMap<3>(host,act,
      {{64,std::uint64_t(B),K/64},{2ull*K,128},{64,std::uint32_t(rows),TK/64}});
  check(cudaMemcpy(map,&host,sizeof host,cudaMemcpyHostToDevice));
  Sm90DenseOperands p{map,w,nullptr,o,nullptr,0.f};auto q=p;q.output=other;
  std::fprintf(stderr,"case B%d TN%d WN%d TK%d shared%d\n",B,TN,WN,TK,A::kSharedBytes);
  auto kernel=evaluate<N,K,TN,TK,WN,S1>;
  check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
  kernel<<<kGrid,160,A::kSharedBytes>>>(p,q,src,act,counter);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  int errors=0;
  long long products=0;for(int k=0;k<K;++k)products+=(k%7+1)*(k%11+1);
  for(BF* device:{o,other}) {
    check(cudaMemcpy(output.data(),device,output.size()*2,cudaMemcpyDeviceToHost));
    for(int b=0;b<B;++b)for(int n=0;n<N;++n) {
      // The final repeat is odd: the activation was negated before its TMA.
      BF expected(-float(products*(b+1)*(n%29-14))/16384.0f);
      if(float(output[b*N+n])!=float(expected)) {
        if(errors<3)std::fprintf(stderr,"B%d TN%d WN%d TK%d b%d n%d got %f expected %f\n",
            B,TN,WN,TK,b,n,float(output[b*N+n]),float(expected));
        ++errors;
      }
    }
  }
  for(void* ptr:{(void*)src,(void*)act,(void*)w,(void*)o,(void*)other,(void*)counter,(void*)map})
    check(cudaFree(ptr));
  return errors;
}

int main() {
  try {
    check(cudaSetDevice(0));int errors=0,cases=0;
    auto add=[&](int e) {errors+=e;++cases;};
    // Every TN of the five dense models' families (prompt appendix A), each
    // (wn, wk) the tile allows, and shallow/deep TK slots.
    add(run<8,1,64>());add(run<8,1,256>());add(run<8,1,1024>());add(run<8,1,2048,2048>());
    add(run<16,1,128>());add(run<16,2,512>());
    add(run<24,1,256>());add(run<32,1,64>());add(run<32,4,256>());add(run<32,2,1024>());
    add(run<40,1,128>());add(run<48,2,256>());add(run<56,1,512>());
    add(run<64,4,64>());add(run<64,2,512>());add(run<80,2,256>());add(run<96,4,128>());
    add(run<112,2,256>());add(run<128,4,64>());add(run<128,1,256>());
    add(run<192,4,128>());add(run<224,4,64>());add(run<256,4,128>());
    add(run<448,4,64>());add(run<512,4,64>());
    std::printf("{\"batch\":%d,\"cases\":%d,\"errors\":%d,\"pass\":%s,\"repeats\":%d}\n",B,cases,errors,
        errors?"false":"true",kRepeats);
    return errors?1:0;
  }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
