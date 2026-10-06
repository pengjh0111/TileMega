// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingGemm.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace tilemega;
using Element=cutlass::bfloat16_t;
template<class T>T* Allocate(int n){T* p=nullptr;if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)std::exit(2);return p;}
template<int TN,backend::ServingEpilogueOp Op>
__global__ void Probe(Element* a,Element* b,Element const* residual,float const* norm,
    float* ssa,float* ssb,int m,int n,bool dn) {
  using Config=backend::ServingGemmConfig<arch::CurrentArch,16,TN,128,4>;
  using Ep=backend::ServingEpilogue<Op,16,TN>;
  __shared__ __align__(16) char memory[4*16*TN+512];
  auto mma=typename Config::TiledMma{};
  auto acc=cute::partition_fragment_C(mma,cute::take<0,2>(typename Config::TileShape{}));
  auto coords=mma.get_thread_slice(threadIdx.x).partition_C(cute::make_identity_tensor(cute::Shape<cute::_16,cute::Int<TN>>{}));
  float* tile=reinterpret_cast<float*>(memory);
  for(int i=threadIdx.x;i<16*TN;i+=128)tile[i]=float((i/TN)*13+(i%TN+int(blockIdx.x)*TN)%37-18)*.0625f;
  for(int i=0;i<cute::size(acc);++i)acc(i)=float(int(cute::get<0>(coords(i)))*13+(int(cute::get<1>(coords(i)))+int(blockIdx.x)*TN)%37-18)*.0625f;
  codegen::executor::ComputeSync();
  int stride=Op==backend::ServingEpilogueOp::kSwiGLU?n/2:n;
  Ep::template RunFromTile<false>(tile,0,blockIdx.x,m,n,stride,n,a,residual,nullptr,nullptr,nullptr,dn?norm:nullptr,ssa,64,1e-5f);
  codegen::executor::ComputeSync();
  Ep::Run(acc,mma,memory,0,blockIdx.x,m,n,stride,n,b,residual,nullptr,nullptr,nullptr,dn?norm:nullptr,ssb,64,1e-5f);
}
template<int TN,backend::ServingEpilogueOp Op>
void Check(int m,int n,bool dn) {
  int stride=Op==backend::ServingEpilogueOp::kSwiGLU?n/2:n;
  auto* a=Allocate<Element>(m*stride);auto* b=Allocate<Element>(m*stride);auto* residual=Allocate<Element>(m*stride);
  auto* norm=Allocate<float>(m*2);auto* ssa=Allocate<float>(m*(n/32));auto* ssb=Allocate<float>(m*(n/32));
  for(int i=0;i<m*stride;++i){a[i]=b[i]=Element(0);residual[i]=Element(float(i%11-5)*.125f);}
  for(int i=0;i<m*2;++i)norm[i]=float(i%3+1)*16;
  for(int i=0;i<m*(n/32);++i)ssa[i]=ssb[i]=-1;
  for(int repeat=0;repeat<3;++repeat) {
    Probe<TN,Op><<<(n+TN-1)/TN,128>>>(a,b,residual,norm,Op==backend::ServingEpilogueOp::kResidual?ssa:nullptr,Op==backend::ServingEpilogueOp::kResidual?ssb:nullptr,m,n,dn);
    auto code=cudaDeviceSynchronize();if(code!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(code));std::exit(3);}
    if(std::memcmp(a,b,m*stride*sizeof(Element)) || (Op==backend::ServingEpilogueOp::kResidual && std::memcmp(ssa,ssb,m*(n/32)*sizeof(float)))) {
      std::fprintf(stderr,"fragment mismatch TN=%d op=%d M=%d N=%d DN=%d\n",TN,int(Op),m,n,dn);std::exit(4);
    }
  }
  cudaFree(a);cudaFree(b);cudaFree(residual);cudaFree(norm);cudaFree(ssa);cudaFree(ssb);
}
template<int TN>void Shapes(){for(int m:{1,16})for(bool dn:{false,true})for(int n:{TN,TN+32}) {
  Check<TN,backend::ServingEpilogueOp::kStore>(m,n,dn);
  Check<TN,backend::ServingEpilogueOp::kResidual>(m,n,dn);
  Check<TN,backend::ServingEpilogueOp::kSwiGLU>(m,n,dn);
}}
int main(){Shapes<32>();Shapes<64>();Shapes<128>();std::puts("fragment epilogue: 72 shapes x 3 repeats bitwise PASS");}
