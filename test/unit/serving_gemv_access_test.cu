// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=arch::CurrentArch;
struct NoGate { bool enabled=false; __device__ void Wait(unsigned) const {} };
void Check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
template<class T>T* Alloc(std::size_t count) {T* p;Check(cudaMallocManaged(&p,count*sizeof(T)));return p;}
float A(int row,int k){return ((row*7+k*3)%17-8)*.0625f;}
float B(int expert,int col,int k){return ((expert*11+col*3+k*5)%19-9)*.03125f;}
template<int N,int K,int Page=0>
__global__ void Probe(ServingGemmOperands p,float* out,bool tiled,bool binding) {
  using G=backend::ServingGemv<Arch,16,N,K>;
  if(binding && !backend::ResolveDmMoeTile(p,0,16))return;
  if constexpr(Page==0) {
    __shared__ __align__(16) char shared[G::kSharedBytes];
    auto* values=G::Dense(p,0,blockIdx.x,shared,tiled);
    for(int i=threadIdx.x;i<4*N;i+=128)out[blockIdx.x*4*N+i]=values[i];
  }else {
    using Ring=executor::PageRing<Page,32768/Page,Arch>;
    constexpr int pool=((1024+G::kSharedBytes+1023)/1024)*1024;
    __shared__ __align__(1024) char shared[pool+32768];
    Ring ring{reinterpret_cast<typename Ring::Slot*>(shared),shared+pool};
    ring.Initialize();std::uint64_t sequence=0;
    if(executor::IsCompute()) {
      auto* values=G::Paged(p,0,blockIdx.x,ring,sequence,shared+1024,NoGate{});
      for(int i=executor::ComputeThread();i<4*N;i+=128)out[blockIdx.x*4*N+i]=values[i];
    }else {
      int kt=(p.k_total_full+K-1)/K;
      int begin=(p.k_begin/K)*N*K,total=((p.k_count+K-1)/K)*N*K;
      auto* source=p.weight_base+std::uint64_t(blockIdx.x)*kt*N*K+begin;
      for(int off=0;off<total;off+=Page/2) {
        ring.AcquireEmpty(sequence);
        for(int vector=executor::LoaderLane();vector<Page/16;vector+=32) {
          int offset=off+vector*8,valid=offset<total?min(8,total-offset):0;
          Ring::Copy::Copy16Bytes(ring.Page(sequence)+vector*16,
              source+(valid?offset:0),valid*sizeof(E));
        }
        ring.PublishCopies(sequence++);
      }
    }
  }
}
template<int N,int K,int Page=0>
void Run(int m,int length,int chunks,bool tiled,bool binding) {
  int n=N+3,pitch=(length+7)/8*8,kt=(length+K-1)/K,nt=(n+N-1)/N;
  std::size_t expert_stride=tiled?nt*kt*N*K:n*pitch;
  auto* a=Alloc<E>(6*pitch);
  auto* b=Alloc<E>(3*n*pitch);
  auto* packed=Alloc<E>(3*nt*kt*N*K);
  auto* output=Alloc<float>(nt*4*N);
  auto* rows=Alloc<MoeBindingRow>(4);auto* blocks=Alloc<MoeBindingRecord>(1);
  for(int row=0;row<6;++row)for(int k=0;k<pitch;++k)a[row*pitch+k]=E(k<length?A(row,k):0);
  for(int i=0;i<3*nt*kt*N*K;++i)packed[i]=E(0);
  for(int e=0;e<3;++e)for(int col=0;col<n;++col)for(int k=0;k<pitch;++k) {
    b[(e*n+col)*pitch+k]=E(k<length?B(e,col,k):0);
    if(k<length)packed[e*nt*kt*N*K+(col/N*kt+k/K)*N*K+
        backend::ServingGemv<Arch,16,N,K>::PackedIndex(col%N,k%K)]=E(B(e,col,k));
  }
  unsigned order[4]={3,1,4,0};
  for(int row=0;row<4;++row)rows[row]={order[row],0,0,0,0};
  blocks[0]={2,0,unsigned(m),1};
  ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=packed;p.m=m;p.n=n;
  p.k_total=p.k_total_full=length;p.a_row_stride=p.b_row_stride=pitch;
  if(binding) {
    p.access.a=DmAAccess::kRowGather;p.access.b=DmBAccess::kExpertIndirect;
    p.access.binding_blocks=1;p.access.binding_rows=4;p.access.experts=3;p.access.block_rows=16;
    p.access.expert_stride=expert_stride;p.binding=blocks;p.rows=rows;
    // Dense and tiled expert strides differ; both pointers must use the
    // selected representation when the common binding adapter resolves B.
    if(tiled)p.b=packed;else p.weight_base=b;
  }
  for(int chunk=0;chunk<chunks;++chunk) {
    int split=((length+chunks*K-1)/(chunks*K))*K;
    p.k_begin=chunk*split;p.k_count=std::min(split,length-p.k_begin);
    if(p.k_count<=0)continue;
    Probe<N,K,Page><<<nt,Page?160:128>>>(p,output,tiled,binding);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    for(int t=0;t<nt;++t)for(int row=0;row<4;++row)for(int col=0;col<N;++col) {
      float expected=0;
      if(row<m && t*N+col<n)for(int k=p.k_begin;k<p.k_begin+p.k_count;++k)
        expected=std::fma(A(binding?order[row]:row,k),B(binding?2:0,t*N+col,k),expected);
      float actual=output[t*4*N+row*N+col];
      if(std::abs(actual-expected)>1e-6f*(1+std::abs(expected))) {
        std::fprintf(stderr,"GEMV access mismatch N%d K%d P%d M%d length%d split%d tiled%d binding%d row%d col%d: %.8g %.8g\n",
            N,K,Page,m,length,chunks,tiled,binding,row,t*N+col,actual,expected);std::exit(3);
      }
    }
  }
  for(void* ptr:{(void*)a,(void*)b,(void*)packed,(void*)output,(void*)rows,(void*)blocks})Check(cudaFree(ptr));
}
template<int N,int K>void Matrix() {
  static_assert(backend::ServingGemv<Arch,16,N,K>::kColumns*
                backend::ServingGemv<Arch,16,N,K>::kUnroll==16);
  for(int m:{1,4})for(int length:{129,257,2305})for(int split:{1,2})for(bool binding:{false,true}) {
    Run<N,K>(m,length,split,false,binding);
    Run<N,K>(m,length,split,true,binding);
    Run<N,K,8192>(m,length,split,true,binding);
    Run<N,K,16384>(m,length,split,true,binding);
  }
}
int main() {
  Matrix<8,32>();Matrix<8,64>();Matrix<8,128>();
  Matrix<16,32>();Matrix<16,64>();Matrix<16,128>();
  Matrix<32,32>();Matrix<32,64>();Matrix<32,128>();
  Matrix<64,32>();Matrix<64,64>();Matrix<64,128>();
  std::puts("GEMV: four TN, three TK, dense/packed/paged/expert, row/K/N tails and split PASS");
}
