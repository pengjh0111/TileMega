// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Backend/Sm90GemvBulk.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cstdlib>
using Arch=tilemega::arch::Sm90;
using E=cutlass::bfloat16_t;
using P=tilemega::codegen::ServingGemmOperands;
void Check(cudaError_t error){if(error!=cudaSuccess){std::cerr<<cudaGetErrorString(error)<<'\n';std::exit(2);}}
template<bool Bulk,int N,int K>
__global__ void Run(P p,float* output,bool tiled){
 extern __shared__ __align__(16) char memory[];
 using Impl=std::conditional_t<Bulk,tilemega::backend::Sm90GemvBulk<Arch,16,N,K>,tilemega::backend::ServingGemv<Arch,16,N,K>>;
 auto* tile=Impl::template Dense<false>(p,0,blockIdx.x,memory,tiled);
 for(int i=threadIdx.x;i<16*N;i+=blockDim.x)output[blockIdx.x*16*N+i]=tile[i];
}
template<int N,int K>
int Cases(){
 using Direct=tilemega::backend::ServingGemv<Arch,16,N,K>;
 using Bulk=tilemega::backend::Sm90GemvBulk<Arch,16,N,K>;
 Check(cudaFuncSetAttribute(Run<true,N,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bulk::kSharedBytes));
 int done=0;
 for(int rows:{1,2,4})for(int count:{K,7*K,17*K,65*K})for(int begin:{0,K})for(bool tiled:{false,true}){
  int n=2*N+3,total=begin+count+K,pitch=total+8,kt=(total+K-1)/K,tiles=(n+N-1)/N;
  std::vector<E> a(rows*pitch),b(n*pitch),packed(tiles*kt*N*K,E(0));
  for(std::size_t i=0;i<a.size();++i)a[i]=E(float(int(i%31)-15)/32);
  for(std::size_t i=0;i<b.size();++i)b[i]=E(float(int((i*7)%37)-18)/64);
  for(int col=0;col<n;++col)for(int k=0;k<total;++k)
   packed[((col/N)*kt+k/K)*N*K+Direct::PackedIndex(col%N,k%K)]=b[col*pitch+k];
  E *ga,*gb,*gp;float *go,*gr;
  Check(cudaMalloc(&ga,a.size()*sizeof(E)));Check(cudaMalloc(&gb,b.size()*sizeof(E)));Check(cudaMalloc(&gp,packed.size()*sizeof(E)));
  Check(cudaMemcpy(ga,a.data(),a.size()*sizeof(E),cudaMemcpyHostToDevice));Check(cudaMemcpy(gb,b.data(),b.size()*sizeof(E),cudaMemcpyHostToDevice));Check(cudaMemcpy(gp,packed.data(),packed.size()*sizeof(E),cudaMemcpyHostToDevice));
  auto bytes=tiles*16*N*sizeof(float);Check(cudaMalloc(&go,bytes));Check(cudaMalloc(&gr,bytes));
  P p{};p.a=ga;p.b=gb;p.weight_base=gp;p.m=rows;p.n=n;p.k_total=total;p.k_total_full=total;p.a_row_stride=pitch;p.b_row_stride=pitch;p.k_begin=begin;p.k_count=count;
  Run<false,N,K><<<tiles,128,Direct::kSharedBytes>>>(p,gr,tiled);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  Run<true,N,K><<<tiles,128,Bulk::kSharedBytes>>>(p,go,tiled);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<float> expected(bytes/sizeof(float)),actual(expected.size());
  Check(cudaMemcpy(expected.data(),gr,bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(actual.data(),go,bytes,cudaMemcpyDeviceToHost));
  if(std::memcmp(expected.data(),actual.data(),bytes)){
   for(std::size_t i=0;i<actual.size();++i)if(std::memcmp(&actual[i],&expected[i],sizeof(float))){std::cerr<<"mismatch TN="<<N<<" TK="<<K<<" rows="<<rows<<" count="<<count<<" begin="<<begin<<" tiled="<<tiled<<" index="<<i<<" actual="<<actual[i]<<" reference="<<expected[i]<<'\n';break;}
   return -1;
  }
  Check(cudaFree(ga));Check(cudaFree(gb));Check(cudaFree(gp));Check(cudaFree(go));Check(cudaFree(gr));++done;
 }
 return done;
}
int main(){int sum=0;
#define CASE(N,K) {int count=Cases<N,K>();if(count<0)return 1;sum+=count;}
 CASE(8,32) CASE(8,64) CASE(8,128) CASE(16,32) CASE(16,64) CASE(16,128) CASE(32,32) CASE(32,64) CASE(32,128)
 std::cout<<"{\"bitwise_cases\":"<<sum<<",\"passed\":true,\"scope\":\"bulk ring vs unchanged direct SIMT arithmetic; not E2E\"}\n";
}
