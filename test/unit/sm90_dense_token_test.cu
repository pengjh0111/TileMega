// SPDX-License-Identifier: BSD-3-Clause
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
template<int Batch,int Width,int Parts>
__global__ void run(int* tokens,BF const* table,BF* output,
                    float const* values,int const* indices,int past) {
  using Argmax=tilemega::backend::Sm90DenseArgmax<Parts,1088>;
  __shared__ typename Argmax::Storage storage;
  if(threadIdx.x<128) {
    tilemega::backend::Sm90DenseEmbedding<Width,64,1088>::Run(tokens,table,output,blockIdx.x,past);
    Argmax::Run(values,indices,tokens,storage,blockIdx.x,past);
  }
}
template<int Batch,int Width,int Parts>
int test() {
  std::vector<BF> table(64*Width),output(Batch*Width);
  for(unsigned i=0;i<table.size();++i)table[i]=BF(float(int(i%17)-8)/16);
  std::vector<int> tokens(Batch*1088,-1),indices(Batch*Parts);
  std::vector<float> values(Batch*Parts,-2);
  for(int b=0;b<Batch;++b) {
    for(int p:{64,575,1086})tokens[b*1088+p]=(b*3+7)%64;
    for(int i=0;i<Parts;++i)indices[b*Parts+i]=i;
    values[b*Parts+17]=values[b*Parts+Parts-2]=7;
    indices[b*Parts+17]=9;indices[b*Parts+Parts-2]=5;
  }
  auto *t=upload(tokens),*i=upload(indices);auto *w=upload(table),*o=upload(output);
  auto *v=upload(values);int errors=0;
  auto expected_tokens=tokens;
  for(int past:{64,575,1086}) {
    run<Batch,Width,Parts><<<Batch,160>>>(t,w,o,v,i,past);check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<int> actual(tokens.size());
    check(cudaMemcpy(actual.data(),t,actual.size()*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(output.data(),o,output.size()*2,cudaMemcpyDeviceToHost));
    for(int b=0;b<Batch;++b)expected_tokens[b*1088+past+1]=5;
    errors+=actual!=expected_tokens;
    for(int b=0;b<Batch;++b) {
      errors+=actual[b*1088+past+1]!=5;
      int id=tokens[b*1088+past];
      for(int d=0;d<Width;++d)errors+=output[b*Width+d].raw()!=table[id*Width+d].raw();
    }
  }
  void* pointers[]={t,i,w,o,v};
  for(void* p:pointers)check(cudaFree(p));
  return errors;
}
int main() {try {
  int errors=test<1,1024,1187>()+test<16,1024,1187>()+
             test<1,3072,1002>()+test<16,3072,1002>();
  std::printf("{\"pass\":%s,\"errors\":%d,\"cases\":12}\n",errors?"false":"true",errors);
  return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
