// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#ifndef DM_TEST_TILED
#define DM_TEST_TILED 0
#endif
#define TILEMEGA_NONPAGED_TILED DM_TEST_TILED
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <tilemega/Codegen/tasks/ServingDmGemmCombineTaskBody.h>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef DM_TEST_TM
#define DM_TEST_TM 16
#endif
#ifndef DM_TEST_TN
#define DM_TEST_TN 32
#endif
#ifndef DM_TEST_TK
#define DM_TEST_TK 16
#endif
#ifndef DM_TEST_SPLIT
#define DM_TEST_SPLIT 1
#endif
using namespace tilemega;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
using K=codegen::DmEpilogueKind;
using A=codegen::DmActivation;
using G=codegen::DmGatePair;
using R=codegen::DmRounding;
using W=codegen::DmWriteKind;
using Norm=codegen::DmEpilogueStep<K::kDeferredRMSNorm,A::kRelu,G::kSwiGLU,16,
    R::kFP32,R::kBF16,W::kDense,1>;
using Gate=codegen::DmEpilogueStep<K::kGatePair,A::kRelu,G::kSwiGLU,16,
    R::kBF16,R::kBF16,W::kDense,1>;
using GateSpec=codegen::DmEpilogueSpec<codegen::DmEpilogueProgram<Norm,Gate>>;
using DownSpec=codegen::DmEpilogueSpec<codegen::DmEpilogueProgram<>,W::kRowScatter>;
constexpr unsigned H=72,I=48,O=19,T=17,Experts=9,TopK=8,DownTM=32;

template<int M>
struct Bodies {
  using Dense=codegen::ServingGemmTaskBody<Arch,M,DM_TEST_TN,DM_TEST_TK,3>;
  using Paged=codegen::PagedGemmTaskBody<Arch,M,DM_TEST_TN,DM_TEST_TK,8192,2>;
  using Packing=backend::ServingDmGemm<Arch,M,DM_TEST_TN,DM_TEST_TK,3>;
  using Ring=typename Paged::Ring;
  static constexpr int Pool=(1024+Paged::kDmWorkspaceBytes+1023)/1024*1024;
  static constexpr int Shared=Pool+16384;
};
template<int M,class Spec>
__global__ void Dense(codegen::ServingGemmOperands p) {
  extern __shared__ __align__(16) char shared[];
  Bodies<M>::Dense::template RunDm<Spec>(p,blockIdx.x,blockIdx.y,shared);
}
template<int M,class Spec>
__global__ void Combine(codegen::ServingGemmOperands p,float const* partials,int chunks) {
  extern __shared__ __align__(16) char shared[];
  backend::DmEpilogueArguments operands;
  if(!codegen::ResolveDmCombineOperands(p,blockIdx.x,M,&operands))return;
  codegen::DmCombineRunner<Arch,M,DM_TEST_TN>{partials,chunks,int(blockIdx.x),int(blockIdx.y),
      shared,operands,p.m}.template Run<Spec>();
}
template<int M,class Spec>
__global__ void Paged(codegen::ServingGemmOperands p,unsigned long long* consumed) {
  extern __shared__ __align__(1024) char shared[];
  using B=Bodies<M>;typename B::Ring ring{
      reinterpret_cast<typename B::Ring::Slot*>(shared),shared+B::Pool};
  ring.Initialize();std::uint64_t sequence=0;
  for(unsigned tm=0;tm<(p.m+M-1)/M;++tm) {
    auto bound=p;
    if(!backend::ResolveDmMoeTile(bound,tm,M))continue;
    for(unsigned tn=0;tn<(p.n+DM_TEST_TN-1)/DM_TEST_TN;++tn) {
      if(codegen::executor::IsCompute())
        B::Paged::template RunDm<Spec>(bound,tm,tn,ring,sequence,shared+1024);
      else B::Paged::Load(bound,tn,ring,sequence);
    }
  }
  if(threadIdx.x==0)consumed[0]=sequence;
  if(threadIdx.x==128)consumed[1]=sequence;
}
template<class V>V* Managed(std::size_t count) {
  V* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(V))==cudaSuccess);return p;
}
std::size_t PackedStride(unsigned n,unsigned k) {
  return std::size_t((n+DM_TEST_TN-1)/DM_TEST_TN)*
      ((k+DM_TEST_TK-1)/DM_TEST_TK)*DM_TEST_TN*DM_TEST_TK;
}
template<int M>
E* Packed(E const* source,unsigned n,unsigned k) {
  auto stride=PackedStride(n,k);auto* output=Managed<E>(Experts*stride);
  unsigned kt=(k+DM_TEST_TK-1)/DM_TEST_TK;
  for(unsigned e=0;e<Experts;++e)for(unsigned nt=0;nt<(n+DM_TEST_TN-1)/DM_TEST_TN;++nt)
    for(unsigned t=0;t<kt;++t)for(unsigned col=0;col<DM_TEST_TN;++col)
      for(unsigned c=0;c<DM_TEST_TK;++c) {
        auto at=e*stride+(std::size_t(nt)*kt+t)*DM_TEST_TN*DM_TEST_TK+
            typename Bodies<M>::Packing::LayoutB{}(col,c);
        output[at]=nt*DM_TEST_TN+col<n && t*DM_TEST_TK+c<k?
            source[(e*n+nt*DM_TEST_TN+col)*k+t*DM_TEST_TK+c]:E(0);
      }
  return output;
}
template<int M,class Spec>
void LaunchBody(codegen::ServingGemmOperands p,bool paged,unsigned long long* consumed) {
  if(paged) {
    assert(cudaFuncSetAttribute(Paged<M,Spec>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bodies<M>::Shared)==cudaSuccess);
    Paged<M,Spec><<<1,160,Bodies<M>::Shared>>>(p,consumed);
  }else {
    assert(cudaFuncSetAttribute(Dense<M,Spec>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bodies<M>::Dense::kSharedBytes)==cudaSuccess);
    Dense<M,Spec><<<dim3((p.m+M-1)/M,(p.n+DM_TEST_TN-1)/DM_TEST_TN),128,
        Bodies<M>::Dense::kSharedBytes>>>(p);
  }
  assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
  if(paged)assert(consumed[0]==consumed[1] && consumed[0]>0);
}
template<int M,class Spec>
void Launch(codegen::ServingGemmOperands p,bool paged,unsigned long long* consumed) {
  p.access.expert_stride=paged || DM_TEST_TILED?PackedStride(p.n,p.k_total):p.n*p.k_total;
  p.m=p.access.binding_blocks*((p.access.block_rows+M-1)/M)*M;
  if constexpr(DM_TEST_SPLIT==1) {
    LaunchBody<M,Spec>(p,paged,consumed);
  }else {
    unsigned kt=(p.k_total+DM_TEST_TK-1)/DM_TEST_TK,chunks=std::min<unsigned>(DM_TEST_SPLIT,kt);
    auto count=std::size_t(chunks)*p.m*p.n;auto* allocation=Managed<float>(count+32);
    std::fill(allocation,allocation+count+32,-123.0f);auto* partials=allocation+16;
    for(unsigned j=0;j<chunks;++j) {
      auto q=p;q.k_begin=(j*kt/chunks)*DM_TEST_TK;
      q.k_count=std::min<unsigned>(p.k_total,((j+1)*kt/chunks)*DM_TEST_TK)-q.k_begin;
      q.output=reinterpret_cast<E*>(partials+std::size_t(j)*p.m*p.n);q.output_stride=p.n;
      q.chain={};q.chain.store_rounding=R::kFP32;q.access.write={};q.dm_partial_rows=true;
      using Partial=codegen::DmEpilogueSpec<codegen::DmEpilogueProgram<>,W::kDense,1,R::kFP32>;
      LaunchBody<M,Partial>(q,paged,consumed);
    }
    constexpr unsigned shared=(M*DM_TEST_TN+2*M)*sizeof(float);
    assert(cudaFuncSetAttribute(Combine<M,Spec>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared)==cudaSuccess);
    Combine<M,Spec><<<dim3((p.m+M-1)/M,(p.n+DM_TEST_TN-1)/DM_TEST_TN),128,shared>>>(p,partials,chunks);
    assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
    for(unsigned i=0;i<16;++i)assert(allocation[i]==-123.0f && allocation[count+16+i]==-123.0f);
    assert(cudaFree(allocation)==cudaSuccess);
  }
}
int main() {
  auto* input=Managed<E>(T*H);
  auto* gu=Managed<E>(Experts*2*I*H);auto* down=Managed<E>(Experts*O*I);
  auto* stats=Managed<float>(T*2*2);
  for(unsigned t=0;t<T;++t) {
    for(unsigned c=0;c<H;++c)input[t*H+c]=E((int((t*13+c*7)%37)-18)*.03125f);
    for(unsigned part=0;part<2;++part) {
      float sum=0,square=0;
      for(unsigned c=part*36;c<(part+1)*36;++c) {
        float value=float(input[t*H+c]);sum+=value;square+=value*value;
      }
      stats[(t*2+part)*2]=sum;stats[(t*2+part)*2+1]=square;
    }
  }
  for(unsigned e=0;e<Experts;++e)for(unsigned n=0;n<2*I;++n)for(unsigned c=0;c<H;++c)
    gu[(e*2*I+n)*H+c]=E((int((e*17+n*5+c*11)%41)-20)*.0078125f);
  for(unsigned e=0;e<Experts;++e)for(unsigned n=0;n<O;++n)for(unsigned c=0;c<I;++c)
    down[(e*O+n)*I+c]=E((int((e*13+n*3+c*7)%31)-15)*.015625f);
  auto* gu_packed=Packed<DM_TEST_TM>(gu,2*I,H);auto* down_packed=Packed<DownTM>(down,O,I);
  auto* layouts=Managed<codegen::DmBufferLayout>(1);*layouts={};layouts->rank=3;
  layouts->logical[0]=T;layouts->logical[1]=2;layouts->logical[2]=2;
  layouts->strides[0]=4;layouts->strides[1]=2;layouts->strides[2]=1;
  auto* types=Managed<unsigned>(1);*types=1;
  auto* buffers=Managed<void*>(1);buffers[0]=stats;
  auto* consumed=Managed<unsigned long long>(2);
  unsigned cases=0;
  for(unsigned bm:{1,16,32,64,128}) {
    // Include a stale invalid tail and active blocks with BM remainders.
    unsigned capacity=T*TopK+Experts;
    auto* bindings=Managed<codegen::MoeBindingRecord>(capacity);
    auto* rows=Managed<codegen::MoeBindingRow>(T*TopK);
    unsigned used=0,position=0;
    if(bm==1) {
      for(unsigned t=0;t<T;++t)for(unsigned rank=0;rank<TopK;++rank) {
        unsigned e=(t+rank)%Experts;
        rows[position]={t,rank,0,0,0};bindings[used++]={e,position++,1,1};
      }
    }else for(unsigned e=0;e<Experts;++e) {
      std::vector<codegen::MoeBindingRow> selected;
      for(unsigned t=0;t<T;++t)for(unsigned rank=0;rank<TopK;++rank)
        if((t+rank)%Experts==e)selected.push_back({t,rank,0,0,0});
      for(unsigned start=0;start<selected.size();start+=bm) {
        unsigned count=std::min<unsigned>(bm,selected.size()-start);
        bindings[used++]={e,position,count,1};
        for(unsigned i=0;i<count;++i)rows[position++]=selected[start+i];
      }
    }
    assert(position==T*TopK);
    for(unsigned v=used;v<capacity;++v)bindings[v]={0xffffffffu,0xffffffffu,0xffffffffu,0};
    auto* hidden=Managed<E>(std::size_t(capacity)*bm*I+32);
    auto* output=Managed<E>(T*TopK*O+32);
    codegen::ServingGemmOperands p;p.a=input;p.b=gu;p.weight_base=gu_packed;
    p.output=hidden+16;p.n=2*I;p.k_total=p.k_count=p.k_total_full=H;
    p.output_stride=I;p.a_row_stride=p.b_row_stride=H;p.norm_k=H;p.norm_eps=1e-6;
    p.access.a=codegen::DmAAccess::kRowGather;p.access.b=codegen::DmBAccess::kExpertIndirect;
    p.access.binding_blocks=capacity;p.access.binding_rows=T*TopK;
    p.access.experts=Experts;p.access.block_rows=bm;p.binding=bindings;p.rows=rows;
    p.chain.count=2;p.chain.operations[0].kind=K::kDeferredRMSNorm;
    p.chain.operations[0].parameter[0]=0;p.chain.operations[0].output_rounding=R::kBF16;
    p.chain.operations[1].kind=K::kGatePair;p.chain.operations[1].input_rounding=R::kBF16;
    p.chain.operations[1].output_rounding=R::kBF16;
    p.dm_buffers={buffers,layouts,types,1};
    std::vector<E> expected_hidden(std::size_t(capacity)*bm*I,E(-123));
    std::vector<E> expected(T*TopK*O);
    for(unsigned v=0;v<used;++v)for(unsigned r=0;r<bindings[v].row_count;++r) {
      auto row=rows[bindings[v].row_begin+r];unsigned e=bindings[v].expert;
      float inverse=1/std::sqrt((stats[row.token*4+1]+stats[row.token*4+3])/H+1e-6f);
      for(unsigned n=0;n<I;++n) {
        unsigned col=2*16*(n/16)+n%16;float gate=0,up=0;
        for(unsigned c=0;c<H;++c) {
          gate+=float(input[row.token*H+c])*float(gu[(e*2*I+col)*H+c]);
          up+=float(input[row.token*H+c])*float(gu[(e*2*I+col+16)*H+c]);
        }
        gate=float(E(gate*inverse));up=float(E(up*inverse));
        expected_hidden[(v*bm+r)*I+n]=E(float(E(gate/(1+std::exp(-gate))))*up);
      }
      for(unsigned n=0;n<O;++n) {
        float sum=0;
        for(unsigned c=0;c<I;++c)sum+=float(expected_hidden[(v*bm+r)*I+c])*float(down[(e*O+n)*I+c]);
        expected[(row.token*TopK+row.rank)*O+n]=E(sum);
      }
    }
    std::vector<E> first;
    for(unsigned epoch=0;epoch<3;++epoch)for(bool paged:{false,true}) {
      for(std::size_t i=0;i<std::size_t(capacity)*bm*I+32;++i)hidden[i]=E(-123);
      for(unsigned i=0;i<T*TopK*O+32;++i)output[i]=E(-123);
      Launch<DM_TEST_TM,GateSpec>(p,paged,consumed);
      auto q=p;q.a=hidden+16;q.b=down;q.weight_base=down_packed;q.output=output+16;
      q.n=O;q.k_total=q.k_count=q.k_total_full=I;q.a_row_stride=q.b_row_stride=I;q.output_stride=O;
      q.access.a=codegen::DmAAccess::kDense;q.access.write.kind=W::kRowScatter;q.access.routing_topk=TopK;
      q.chain={};
      Launch<DownTM,DownSpec>(q,paged,consumed);
      for(unsigned v=0;v<capacity;++v)for(unsigned r=0;r<bm;++r)for(unsigned c=0;c<I;++c) {
        auto at=(std::size_t(v)*bm+r)*I+c;
        float reference=float(expected_hidden[at]),actual=float(hidden[16+at]);
        assert(std::abs(actual-reference)<=.016f+.016f*std::abs(reference));
      }
      for(unsigned i=0;i<T*TopK*O;++i)
        assert(std::abs(float(output[16+i])-float(expected[i]))<=.016f+.016f*std::abs(float(expected[i])));
      for(unsigned i=0;i<16;++i)assert(hidden[i]==E(-123) &&
          hidden[std::size_t(capacity)*bm*I+16+i]==E(-123) && output[i]==E(-123) && output[T*TopK*O+16+i]==E(-123));
      if(first.empty())first.assign(output+16,output+16+T*TopK*O);
      else for(unsigned i=0;i<T*TopK*O;++i)assert(first[i]==output[16+i]);
      ++cases;
    }
    cudaFree(bindings);cudaFree(rows);cudaFree(hidden);cudaFree(output);
  }
  for(void* pointer:{static_cast<void*>(input),static_cast<void*>(gu),static_cast<void*>(down),
      static_cast<void*>(gu_packed),static_cast<void*>(down_packed),static_cast<void*>(stats),
      static_cast<void*>(layouts),static_cast<void*>(types),static_cast<void*>(buffers),static_cast<void*>(consumed)})
    assert(cudaFree(pointer)==cudaSuccess);
  std::printf("DM_MOE_EXPERT cases=%u split=%u slots_groups_norm_scatter_empty_dense_pages PASS\n",cases,unsigned(DM_TEST_SPLIT));
}
