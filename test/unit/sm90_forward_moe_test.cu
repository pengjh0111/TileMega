// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>

using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=arch::Sm90;
using Spec=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kDense,1,DmRounding::kFP32>;
template<class T>T* Allocate(std::size_t count) {
  T* result=nullptr;assert(cudaMallocManaged(&result,count*sizeof(T))==cudaSuccess);return result;
}
float Activation(unsigned row,unsigned k) {return (int((row*13+k*7)%29)-14)/64.f;}
float Weight(unsigned expert,unsigned n,unsigned k) {return (int((expert*11+n*17+k*13)%31)-15)/64.f;}
template<int M>
__global__ void Run(ServingGemmOperands p) {
  extern __shared__ __align__(1024) char scratch[];
  using Body=ServingGemmTaskBody<Arch,M,64,64,3,true>;
  for(int m=0;m<(p.m+M-1)/M;++m) {
    Body::template RunDm<Spec>(p,m,0,scratch);
    executor::ComputeSync();
  }
}
template<int M>void Check(bool gather,int begin) {
  constexpr unsigned K=256,N=64,Experts=3,VirtualRows=3*M;
  unsigned source_rows=gather?23:VirtualRows;
  auto* a=Allocate<E>(source_rows*K);auto* b=Allocate<E>(Experts*N*K);
  auto* output=Allocate<float>(VirtualRows*N+16);
  auto* records=Allocate<MoeBindingRecord>(3);auto* rows=Allocate<MoeBindingRow>(24);
  records[0]={2,0,19,1};records[1]={0,19,5,1};records[2]={999,999,999,0};
  for(unsigned i=0;i<24;++i)rows[i]={unsigned((i*5+7)%23),i%2,0,0,0};
  for(unsigned row=0;row<source_rows;++row)for(unsigned k=0;k<K;++k)a[row*K+k]=E(Activation(row,k));
  auto layout=cute::as_position_independent_swizzle_layout(backend::Sm90KLayout<64,64>{});
  for(unsigned e=0;e<Experts;++e)for(unsigned n=0;n<N;++n)for(unsigned k=0;k<K;++k) {
    unsigned offset=n*K+k;
#if TILEMEGA_NONPAGED_TILED
    offset=(k/64)*N*64+layout(n,k%64);
#endif
    b[e*N*K+offset]=E(Weight(e,n,k));
  }
  for(unsigned i=0;i<VirtualRows*N+16;++i)output[i]=NAN;
  ServingGemmOperands p;p.a=a;p.b=p.weight_base=b;p.output=reinterpret_cast<E*>(output+8);
  p.m=VirtualRows;p.n=N;p.k_total=p.k_total_full=K;p.k_begin=begin;p.k_count=K-begin;
  p.a_row_stride=p.b_row_stride=K;p.output_stride=N;p.chain.store_rounding=DmRounding::kFP32;
  p.access.a=gather?DmAAccess::kRowGather:DmAAccess::kDense;
  p.access.b=DmBAccess::kExpertIndirect;p.access.block_rows=M;p.access.binding_blocks=3;
  p.access.binding_rows=24;p.access.experts=Experts;p.access.expert_stride=N*K;
  p.binding=records;p.rows=rows;
  using Body=ServingGemmTaskBody<Arch,M,64,64,3,true>;
  assert(cudaFuncSetAttribute(Run<M>,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes)==cudaSuccess);
  Run<M><<<1,128,Body::kSharedBytes>>>(p);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  for(unsigned row=0;row<VirtualRows;++row)for(unsigned n=0;n<N;++n) {
    auto const block=records[row/M];float actual=output[8+row*N+n];
    bool active=block.valid && row%M<block.row_count;
    if(!active) {assert(std::isnan(actual));continue;}
    unsigned source=gather?rows[block.row_begin+row%M].token:row;
    float expected=0;
    for(unsigned k=begin;k<K;++k)expected+=Activation(source,k)*Weight(block.expert,n,k);
    if(actual!=expected) {
      std::fprintf(stderr,"forward expert M%d gather%d begin%d [%u,%u]: %.9g != %.9g\n",
          M,int(gather),begin,row,n,actual,expected);assert(false);
    }
  }
  for(unsigned i=0;i<8;++i)assert(std::isnan(output[i]) && std::isnan(output[8+VirtualRows*N+i]));
  for(void* ptr:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(output),
      static_cast<void*>(records),static_cast<void*>(rows)})assert(cudaFree(ptr)==cudaSuccess);
}
int main() {
  for(bool gather:{false,true})for(int begin:{0,128}) {Check<64>(gather,begin);Check<128>(gather,begin);}
  std::puts("forward expert: 8 gathered/dense/padded/empty/offset cases PASS");
}
