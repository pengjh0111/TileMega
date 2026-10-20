// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DN_VECTOR_SUMS 1
#define TILEMEGA_SWIGLU_U 8
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<int B,int TN,int WN,int WK>
__global__ __launch_bounds__(160) void evaluate(Sm90DenseOperands first,Sm90DenseOperands second) {
 extern __shared__ __align__(1024) char shared[];
 auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
 using A=Sm90DenseGemm<B,TN*25,1024,TN,64,3,WN,WK,ServingEpilogueOp::kStore,8>;
 using C=Sm90DenseGemm<B,TN*25,1024,TN,64,2,WN,WK,ServingEpilogueOp::kStore,8>;
 for(int repeat=0;repeat<4;++repeat) {A::Run(first,shared,1e-6f);C::Run(second,shared,1e-6f);}
 state.Invalidate();
}
template<int B,int TN,int WN,int WK> int run() {
 constexpr int K=1024,N=TN*25,TK=64;
 using A=Sm90DenseGemm<B,N,K,TN,TK,3,WN,WK,ServingEpilogueOp::kStore,8>;
 std::vector<BF> input(B*K),weight(N*K),output(B*N);
 for(int b=0;b<B;++b)for(int k=0;k<K;++k)input[b*K+k]=BF((b+1)*0.25f);
 for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
  int row=n%TN,col=k%TK;
  size_t permutation=512*(row/8+(TN/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
  size_t index=((size_t(n/TN)*(K/TK)+k/TK)*TN*TK)+permutation;
  weight[index]=BF((n%29-14)/128.0f);
 }
 BF *a,*w,*o,*other;
 check(cudaMalloc(&a,input.size()*2));check(cudaMalloc(&w,weight.size()*2));
 check(cudaMalloc(&o,output.size()*2));check(cudaMalloc(&other,output.size()*2));
 check(cudaMemcpy(a,input.data(),input.size()*2,cudaMemcpyHostToDevice));
 check(cudaMemcpy(w,weight.data(),weight.size()*2,cudaMemcpyHostToDevice));
 Sm90DenseOperands p{a,w,nullptr,o,nullptr,nullptr,nullptr};auto q=p;q.output=other;
 check(cudaFuncSetAttribute(evaluate<B,TN,WN,WK>,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
 evaluate<B,TN,WN,WK><<<4,160,A::kSharedBytes>>>(p,q);check(cudaGetLastError());check(cudaDeviceSynchronize());
 int errors=0;
 for(BF* device:{o,other}) {
  check(cudaMemcpy(output.data(),device,output.size()*2,cudaMemcpyDeviceToHost));
  for(int b=0;b<B;++b)for(int n=0;n<N;++n) {
   BF expected(K*(b+1)*0.25f*((n%29-14)/128.0f));
   if(float(output[b*N+n])!=float(expected)) {
    if(errors<3)std::fprintf(stderr,"B%d TN%d b%d n%d got %f expected %f\n",B,TN,b,n,float(output[b*N+n]),float(expected));
    ++errors;
   }
  }
 }
 check(cudaFree(a));check(cudaFree(w));check(cudaFree(o));check(cudaFree(other));
 return errors;
}
int main() {
 try {
  check(cudaSetDevice(0));int errors=0;
  errors+=run<1,8,1,4>();errors+=run<16,24,1,4>();errors+=run<1,48,2,2>();
  errors+=run<16,128,4,1>();errors+=run<1,224,4,1>();
  std::printf("{\"errors\":%d,\"pass\":%s,\"cases\":5,\"slot_sequence\":[3,2,3,2,3,2,3,2]}\n",errors,errors?"false":"true");
  return errors?1:0;
 }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
