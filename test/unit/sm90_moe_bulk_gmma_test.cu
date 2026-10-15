// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MOE_ACTIVATION_BULK 1
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#define TILEMEGA_ARCH_ID 900
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;using Arch=tilemega::arch::Sm90;
using Spec=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kDense,1,DmRounding::kFP32>;
void Cuda(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* Allocate(std::size_t n){T* p;Cuda(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<int N,int K,bool Paged>
__global__ void Run(ServingGemmOperands p) {
  constexpr int Page=N*K*2>16384?32768:16384;
  using P=PagedGemmTaskBody<Arch,16,N,K,Page,5,false,true>;
  using D=ServingGemmTaskBody<Arch,16,N,K,2,true>;
  extern __shared__ __align__(1024) char memory[];
  constexpr int pool=(1024+P::kDmWorkspaceBytes+1023)/1024*1024;
  typename P::Ring ring{reinterpret_cast<typename P::Ring::Slot*>(memory),memory+pool};
  if constexpr(Paged)ring.Initialize();
  std::uint64_t sequence=0;
  for(int m=0;m<(p.m+15)/16;++m)for(int n=0;n<(p.n+N-1)/N;++n) {
    auto tile=p;if(!tilemega::backend::ResolveDmMoeTile(tile,m,16))continue;
    if constexpr(Paged) {
      if(executor::IsCompute())P::template RunDm<Spec>(tile,m,n,ring,sequence,memory+1024);
      else P::Load(tile,n,ring,sequence);
    }else D::template RunDm<Spec>(tile,m,n,memory);
  }
}
float A(int row,int k){return ((row*13+k*7)%29-14)/64.f;}
float B(int expert,int col,int k){return ((expert*11+col*17+k*13)%31-15)/64.f;}
template<int N,int K,bool Paged>
void Case(int rows,int reduction,int begin,bool expert) {
  constexpr int Page=N*K*2>16384?32768:16384;
  using P=PagedGemmTaskBody<Arch,16,N,K,Page,5,false,true>;
  using D=ServingGemmTaskBody<Arch,16,N,K,2,true>;
  int source_rows=23,cols=N+13,pitch=(reduction+7)&~7,kt=(reduction+K-1)/K,nt=(cols+N-1)/N;
  int experts=expert?3:1,virtual_rows=expert?96:rows;
  bool packed=Paged || TILEMEGA_NONPAGED_TILED;
  std::size_t expert_stride=packed?std::size_t(nt)*kt*N*K:std::size_t(cols)*pitch;
  auto* a=Allocate<E>(source_rows*pitch);auto* b=Allocate<E>(experts*expert_stride);
  auto* out=Allocate<float>(virtual_rows*cols);auto* indices=Allocate<std::uint32_t>(virtual_rows);
  auto* blocks=Allocate<MoeBindingRecord>(3);auto* bindings=Allocate<MoeBindingRow>(24);
  for(int r=0;r<source_rows;++r)for(int k=0;k<pitch;++k)a[r*pitch+k]=E(k<reduction?A(r,k):0);
  for(std::size_t i=0;i<experts*expert_stride;++i)b[i]=E(0);
  auto layout=cute::as_position_independent_swizzle_layout(tilemega::backend::Sm90KLayout<N,K>{});
  for(int e=0;e<experts;++e)for(int n=0;n<cols;++n)for(int k=0;k<reduction;++k) {
    std::size_t offset=n*pitch+k;
    if(packed)offset=(std::size_t(n/N)*kt+k/K)*N*K+layout(n%N,k%K);
    b[e*expert_stride+offset]=E(B(e,n,k));
  }
  for(int i=0;i<virtual_rows*cols;++i)out[i]=NAN;
  for(int i=0;i<virtual_rows;++i)indices[i]=(i*7+3)%source_rows;
  blocks[0]={2,0,19,1};blocks[1]={0,19,5,1};blocks[2]={999,999,999,0};
  for(int i=0;i<24;++i)bindings[i]={std::uint32_t((i*5+7)%source_rows),std::uint32_t(i%2),0,0,0};
  ServingGemmOperands p;p.a=a;p.b=p.weight_base=b;p.m=virtual_rows;p.n=cols;
  p.output=reinterpret_cast<E*>(out);p.output_stride=cols;p.chain.store_rounding=DmRounding::kFP32;
  p.k_total=p.k_total_full=reduction;p.k_begin=begin;p.k_count=reduction-begin;
  p.a_row_stride=p.b_row_stride=pitch;p.access.a=DmAAccess::kRowGather;p.rows=indices;
  if(expert) {
    p.access.b=DmBAccess::kExpertIndirect;p.access.block_rows=32;p.access.binding_blocks=3;
    p.access.binding_rows=24;p.access.experts=3;p.access.expert_stride=expert_stride;
    p.binding=blocks;p.rows=bindings;
  }
  constexpr int pool=(1024+P::kDmWorkspaceBytes+1023)/1024*1024;
  int bytes=Paged?pool+5*Page:D::kSharedBytes;
  Cuda(cudaFuncSetAttribute(Run<N,K,Paged>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  Run<N,K,Paged><<<1,Paged?160:128,bytes>>>(p);Cuda(cudaDeviceSynchronize());
  for(int r=0;r<virtual_rows;++r)for(int n=0;n<cols;++n) {
    bool active=true;int src=indices[r],e=0;
    if(expert) {
      auto block=blocks[r/32];active=block.valid && r%32<int(block.row_count);e=block.expert;
      if(active)src=bindings[block.row_begin+r%32].token;
    }
    if(!active) {if(!std::isnan(out[r*cols+n])){std::fprintf(stderr,"wrote padded expert row\n");std::exit(3);}continue;}
    float expected=0;for(int k=begin;k<reduction;++k)expected+=A(src,k)*B(e,n,k);
    if(out[r*cols+n]!=expected) {
      std::fprintf(stderr,"MoE bulk N%d K%d paged%d packed%d rows%d reduction%d begin%d expert%d [%d,%d] %.9g != %.9g\n",
          N,K,Paged,packed,virtual_rows,reduction,begin,expert,r,n,out[r*cols+n],expected);std::exit(3);
    }
  }
  for(void* ptr:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(out),static_cast<void*>(indices),
      static_cast<void*>(blocks),static_cast<void*>(bindings)})Cuda(cudaFree(ptr));
}
template<int N,int K,bool Paged>void Cases(){for(int reduction:{1536,1539})for(int begin:{0,128}){
  for(int rows:{1,16,19})Case<N,K,Paged>(rows,reduction,begin,false);
  Case<N,K,Paged>(0,reduction,begin,true);
}}
template<int N,int K>void Shapes(){Cases<N,K,false>();Cases<N,K,true>();}
int main(){Shapes<64,64>();Shapes<128,64>();Shapes<64,128>();Shapes<128,128>();
  std::puts("MoE activation bulk: 128 gather/expert/paged/tail/split-offset cases PASS");}
