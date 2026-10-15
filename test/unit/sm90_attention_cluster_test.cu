// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_ARCH_ID 900
#define TILEMEGA_ATTENTION_IMPL 4
#include <tilemega/Codegen/tasks/ClusterAttentionMergeTaskBody.h>
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using Arch=tilemega::arch::Sm90;
constexpr int Ranks=TILEMEGA_SERVING_CLUSTER_DIM,Capacity=1024;
static void Check(cudaError_t status) {
  if(status!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status));std::exit(2);}
}
template<class T>T* Allocate(std::size_t count) {
  T* pointer=nullptr;Check(cudaMallocManaged(&pointer,count*sizeof(T)));return pointer;
}
template<int D,int Q>struct Case {
  using Cluster=ClusterAttentionMergeTaskBody<Arch,D,Q,Ranks,false>;
  using Attention=typename Cluster::Attention;
  using Mailbox=typename Cluster::Mailbox;
  static constexpr std::size_t Offset=(sizeof(typename Attention::SharedStorage)+1023)&~std::size_t(1023);
  static constexpr std::size_t Shared=Offset+sizeof(typename Mailbox::Storage);
};
template<int D,int Q>__global__ void ClusterKernel(ServingAttentionOperands p,int episodes,bool reverse) {
  using C=Case<D,Q>;
  extern __shared__ __align__(1024) unsigned char memory[];
  auto& shared=*reinterpret_cast<typename C::Attention::SharedStorage*>(memory);
  auto& mailbox=*reinterpret_cast<typename C::Mailbox::Storage*>(memory+C::Offset);
  C::Mailbox::Init(mailbox);
  int cluster=blockIdx.x/Ranks,clusters=gridDim.x/Ranks,rank=blockIdx.x%Ranks;
  for(int episode=0;episode<episodes;++episode) {
    for(int ordinal=cluster;ordinal<p.batch;ordinal+=clusters) {
      int batch=reverse?p.batch-1-ordinal:ordinal;
      C::Cluster::Run(p,shared,mailbox,batch,0,rank);
    }
  }
  C::Mailbox::Close();
}
template<int D,int Q>int Test() {
  using C=Case<D,Q>;
  Check(cudaFuncSetAttribute(ClusterKernel<D,Q>,cudaFuncAttributeMaxDynamicSharedMemorySize,C::Shared));
  int active=0;Check(executor::ServingActiveClusters(&active,ClusterKernel<D,Q>,128,C::Shared,Ranks*2));
  if(active<=0){std::fprintf(stderr,"no resident c%d cluster for D%d Q%d shared%zu\n",Ranks,D,Q,C::Shared);return 1;}
  cudaFuncAttributes attributes{};Check(cudaFuncGetAttributes(&attributes,ClusterKernel<D,Q>));
  std::printf("RESOURCE c%d D%d Q%d shared%zu regs%d local%zu active_clusters%d\n",Ranks,D,Q,C::Shared,attributes.numRegs,attributes.localSizeBytes,active);
  constexpr int Batches=3,Extent=Capacity/Ranks;
  auto* qkv=Allocate<Element>(Batches*(Q+2)*D);
  auto* key=Allocate<Element>(Batches*Capacity*D);auto* value=Allocate<Element>(Batches*Capacity*D);
  auto* cosine=Allocate<Element>(Capacity*D);auto* sine=Allocate<Element>(Capacity*D);
  auto* context=Allocate<Element>(Batches*Q*D);auto* maps=Allocate<executor::TensorMap>(2);
  executor::TensorMapShape shape{D,Batches*Capacity,D*2,64,64};
  if(executor::EncodeTensorMap(maps[0],key,shape)!=CUDA_SUCCESS ||
     executor::EncodeTensorMap(maps[1],value,shape)!=CUDA_SUCCESS)return 2;
  for(int i=0;i<Capacity*D;++i){cosine[i]=Element(1);sine[i]=Element(0);}
  for(int i=0;i<Batches*(Q+2)*D;++i)qkv[i]=Element(.02f*(i%13-6));
  for(int batch=0;batch<Batches;++batch)for(int d=0;d<D;++d)
    qkv[(batch*(Q+2)+Q+1)*D+d]=Element(.75f+.01f*(d%7));
  int failures=0,cases=0;
  // Empty higher ranks, a boundary current token, non-divisible tails and all ranks active.
  for(int past:{0,14,63,64,Extent-1,Extent,Extent+17,Capacity-2,Capacity-1}) {
    for(int batch=0;batch<Batches;++batch)for(int pos=0;pos<Capacity;++pos)for(int d=0;d<D;++d) {
      auto index=(batch*Capacity+pos)*D+d;
      key[index]=Element(.018f*(pos%17-8)+.01f*(d%11-5)+.005f*batch);
      value[index]=Element(float(pos)/256+.03f*(d%13)+.01f*batch);
    }
    ServingAttentionOperands p{qkv,key,value,cosine,sine,nullptr,nullptr,context,nullptr,nullptr,
      Batches,1,Capacity,past,Extent,1e-6f};
    p.key_tensor_map=maps;p.value_tensor_map=maps+1;
    Check(executor::LaunchServing(ClusterKernel<D,Q>,std::min(active,2)*Ranks,128,C::Shared,nullptr,false,p,13,false));
    Check(cudaDeviceSynchronize());
    std::vector<Element> first(context,context+Batches*Q*D);
    Check(executor::LaunchServing(ClusterKernel<D,Q>,Ranks,128,C::Shared,nullptr,false,p,13,true));
    Check(cudaDeviceSynchronize());
    bool exact=std::memcmp(context,first.data(),first.size()*sizeof(Element))==0;
    bool pass=exact;float error=0;
    for(int batch=0;batch<Batches;++batch)for(int q=0;q<Q;++q) {
      std::vector<float> scores(past+1);float maximum=-INFINITY;
      for(int pos=0;pos<=past;++pos) {
        float dot=0;for(int d=0;d<D;++d)
          dot+=float(qkv[(batch*(Q+2)+q)*D+d])*float(key[(batch*Capacity+pos)*D+d]);
        scores[pos]=dot/std::sqrt(float(D));maximum=std::max(maximum,scores[pos]);
      }
      float sum=0;for(auto& score:scores){score=std::exp(score-maximum);sum+=score;}
      for(int d=0;d<D;++d) {
        float expected=0;for(int pos=0;pos<=past;++pos)expected+=scores[pos]*float(value[(batch*Capacity+pos)*D+d]);
        expected/=sum;float got=float(context[(batch*Q+q)*D+d]);
        error=std::max(error,std::abs(expected-got));
        pass=pass && std::isfinite(got) && std::abs(expected-got)<.006f*(1+std::abs(expected));
      }
    }
    ++cases;failures+=!pass;
    std::printf("CASE c%d D%d Q%d past%d exact_schedule%d fp32_error%g pass%d\n",Ranks,D,Q,past,exact,error,pass);
  }
  void* allocations[]={qkv,key,value,cosine,sine,context,maps};
  for(void* pointer:allocations)Check(cudaFree(pointer));
  std::printf("RESULT c%d D%d Q%d cases%d failures%d\n",Ranks,D,Q,cases,failures);
  return failures;
}
int main() {
  int failures=Test<64,1>()+Test<64,3>()+Test<64,8>()+Test<128,1>()+Test<128,3>()+Test<128,8>();
  if(failures)return 1;
  std::printf("PASS c%d weighted split-KV merge, empty splits, 13 episodes and exact schedule variants\n",Ranks);
}
