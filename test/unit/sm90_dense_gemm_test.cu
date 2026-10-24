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
template<int TK>
int check_layout() {
 using G=Sm90DenseGemm<1,8,TK,8,TK,2,1,4,ServingEpilogueOp::kStore,4>;
 using A=typename G::LayoutA;constexpr auto inverse=cute::right_inverse(A{});
 int errors=0;
 for(int r=0;r<16;++r)for(int c=0;c<TK;++c) {
  int logical=inverse(A{}(r,c));errors+=(logical%16!=r || logical/16!=c);
 }
 for(int batch:{1,2,4,8,16})for(int i=0;i<16*TK;i+=8)
  for(int j=1;j<8;++j)errors+=((inverse(i+j)%16>=batch)!=(inverse(i)%16>=batch));
 return errors;
}
template<int B,int TN,int WN,int WK,int TK,int Reg>
__global__ __launch_bounds__(160) void evaluate(Sm90DenseOperands first,Sm90DenseOperands second) {
 extern __shared__ __align__(1024) char shared[];
 auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
 using A=Sm90DenseGemm<B,TN*25,1024,TN,TK,3,WN,WK,ServingEpilogueOp::kStore,8,Reg>;
 using C=Sm90DenseGemm<B,TN*25,1024,TN,TK,2,WN,WK,ServingEpilogueOp::kStore,8,Reg>;
 for(int repeat=0;repeat<4;++repeat) {A::template Run<false,false>(first,shared);C::template Run<false,false>(second,shared);}
 state.Invalidate();
}
template<int B,int TN,int WN,int WK,int TK=64,int Reg=(TK/(16*WK)<4?TK/(16*WK):4)>
int run() {
 constexpr int K=1024,N=TN*25;
 using A=Sm90DenseGemm<B,N,K,TN,TK,3,WN,WK,ServingEpilogueOp::kStore,8,Reg>;
 std::vector<BF> input(B*K),weight(N*K),output(B*N);
 for(int b=0;b<B;++b)for(int k=0;k<K;++k)input[b*K+k]=BF((b+1)*(k%7+1)/64.0f);
 for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
  int row=n%TN,col=k%TK;
  size_t permutation=512*(row/8+(TN/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
  size_t index=((size_t(n/TN)*(K/TK)+k/TK)*TN*TK)+permutation;
  weight[index]=BF((n%29-14)*(k%11+1)/256.0f);
 }
 BF *a,*w,*o,*other;
 check(cudaMalloc(&a,input.size()*2));check(cudaMalloc(&w,weight.size()*2));
 check(cudaMalloc(&o,output.size()*2));check(cudaMalloc(&other,output.size()*2));
 check(cudaMemcpy(a,input.data(),input.size()*2,cudaMemcpyHostToDevice));
 check(cudaMemcpy(w,weight.data(),weight.size()*2,cudaMemcpyHostToDevice));
 Sm90DenseOperands p{a,w,nullptr,o,nullptr,nullptr,nullptr};auto q=p;q.output=other;
 check(cudaFuncSetAttribute(evaluate<B,TN,WN,WK,TK,Reg>,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
 evaluate<B,TN,WN,WK,TK,Reg><<<4,160,A::kSharedBytes>>>(p,q);check(cudaGetLastError());check(cudaDeviceSynchronize());
 int errors=0;
 for(BF* device:{o,other}) {
  check(cudaMemcpy(output.data(),device,output.size()*2,cudaMemcpyDeviceToHost));
  // Dyadic products and their sum fit exactly in FP32. Distinct K periods
  // expose a stale/misaligned lookahead fragment without a numerical tolerance.
  long long products=0;for(int k=0;k<K;++k)products+=(k%7+1)*(k%11+1);
  for(int b=0;b<B;++b)for(int n=0;n<N;++n) {
   BF expected(float(products*(b+1)*(n%29-14))/16384.0f);
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
  errors+=run<1,16,2,2,256,4>();errors+=run<1,8,1,4,256,4>();
  errors+=run<1,8,1,4,1024,1>();errors+=run<1,24,1,4,512,1>();
  errors+=run<16,16,2,2,1024,1>();errors+=run<16,16,2,2,1024,4>();
  errors+=run<2,16,1,4,256,1>();errors+=run<4,24,1,4,256,1>();errors+=run<8,8,1,4,256,1>();
  errors+=check_layout<64>()+check_layout<128>()+check_layout<192>()+check_layout<256>();
  errors+=check_layout<384>()+check_layout<448>()+check_layout<512>()+check_layout<768>();
  errors+=check_layout<896>()+check_layout<1024>()+check_layout<1536>()+
      check_layout<1792>()+check_layout<2048>();
  std::printf("{\"errors\":%d,\"pass\":%s,\"cases\":14,\"layout_cases\":13,\"register_stages\":[1,2,4],\"slot_sequence\":[3,2,3,2,3,2,3,2]}\n",errors,errors?"false":"true");
  return errors?1:0;
 }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
