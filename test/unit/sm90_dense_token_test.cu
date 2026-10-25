// SPDX-License-Identifier: BSD-3-Clause
// The embedding stage copies table[tokens[b][past]] into row b exactly; the
// argmax runs in the lm_head last arriver (D9, sm90_dense_epilogue_test).
#include <tilemega/Backend/Sm90DenseToken.cuh>
#include <vector>
#include <cstdio>
#include <stdexcept>

using BF=cutlass::bfloat16_t;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> T* upload(std::vector<T> const& values) {
  T* out=nullptr;check(cudaMalloc(&out,values.size()*sizeof(T)));
  check(cudaMemcpy(out,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));return out;
}
template<int Width>
__global__ void run(int const* tokens,BF const* table,BF* output,int past) {
  if(threadIdx.x<128)tilemega::backend::Sm90DenseEmbedding<Width,64,1088>::Run(tokens,table,output,blockIdx.x,past);
}
template<int Batch,int Width>
int test() {
  std::vector<BF> table(64*Width),output(Batch*Width);
  for(unsigned i=0;i<table.size();++i)table[i]=BF(float(int(i%17)-8)/16);
  std::vector<int> tokens(Batch*1088,-1);
  for(int b=0;b<Batch;++b)for(int p:{64,575,1086})tokens[b*1088+p]=(b*3+7)%64;
  auto* t=upload(tokens);auto *w=upload(table),*o=upload(output);int errors=0;
  for(int past:{64,575,1086}) {
    run<Width><<<Batch,160>>>(t,w,o,past);check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),o,output.size()*2,cudaMemcpyDeviceToHost));
    for(int b=0;b<Batch;++b) {
      int id=tokens[b*1088+past];
      for(int d=0;d<Width;++d)errors+=output[b*Width+d].raw()!=table[id*Width+d].raw();
    }
  }
  for(void* p:{(void*)t,(void*)w,(void*)o})check(cudaFree(p));
  return errors;
}
int main() {try {
  int errors=test<1,1024>()+test<16,1024>()+test<1,3072>()+test<16,3072>();
  std::printf("{\"pass\":%s,\"errors\":%d,\"cases\":12}\n",errors?"false":"true",errors);
  return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
