// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_QKV_EPILOGUE 1
#define TILEMEGA_ATTENTION_IMPL 4
#include <tilemega/Codegen/tasks/ServingQkvEpilogue.h>
#include <tilemega/Codegen/tasks/IndependentAttentionTaskBody.h>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using Arch=tilemega::arch::Sm90;
constexpr int Capacity=160;
void Cuda(cudaError_t result) {
  if(result!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(result));std::exit(2);}
}
template<class T>T* Allocate(std::size_t count) {T* p=nullptr;Cuda(cudaMallocManaged(&p,count*sizeof(T)));return p;}
__global__ void Fuse(QkvEpilogueDesc p,Element* qkv,int batch,int columns,int tn,int past) {
  ServingQkvEpilogue<Arch>::Run(p,qkv,batch,columns,blockIdx.y,blockIdx.x,16,tn,past);
}
template<int D,int Q,bool Norm>
__global__ void Reference(QkvEpilogueDesc p,Element const* qkv,Element* expected,int batch,int past) {
  using Base=FusedAttentionTaskBody<Arch,D,Q,1,16,32,Norm>;
  int warp=threadIdx.x/32,lane=threadIdx.x%32;
  int columns=p.heads_kv*(Q+2)*D;
  for(int head=warp;head<batch*p.heads_kv*(Q+2);head+=4) {
    int kind=head%(Q+2),group=head/(Q+2);
    auto* source=qkv+head*D;
    if(kind<Q)Base::RotateRow(source,p.q_norm,p.cosine+past*D,p.sine+past*D,p.epsilon,
        [&](int d,uint4 value){tilemega::backend::StoreGlobal16(expected+head*D+d,value);});
    else if(kind==Q)Base::RotateRow(source,p.k_norm,p.cosine+past*D,p.sine+past*D,p.epsilon,
        [&](int d,uint4 value){tilemega::backend::StoreGlobal16(p.key_cache+(group*Capacity+past)*D+d,value);});
    else if(lane*8<D)tilemega::backend::StoreGlobal16(p.value_cache+(group*Capacity+past)*D+lane*8,
        tilemega::backend::LoadGlobal16(source+lane*8));
  }
}
template<int D,int Q,bool Norm>
__global__ void Attend(ServingAttentionOperands p) {
  using Body=IndependentAttentionTaskBody<Arch,D,Q,Norm,4*D*64,AttentionPagePolicy::WarpGroup>;
  extern __shared__ __align__(1024) char memory[];
  Body::Run(p,*reinterpret_cast<typename Body::SharedStorage*>(memory),blockIdx.y,blockIdx.x,0,0);
}
void Equal(void const* a,void const* b,std::size_t bytes,char const* what,int d,int q,bool norm,int batch,int heads,int tn,int past) {
  if(std::memcmp(a,b,bytes)) {
    auto* x=static_cast<unsigned short const*>(a);auto* y=static_cast<unsigned short const*>(b);
    std::size_t bad=0;while(bad<bytes/2 && x[bad]==y[bad])++bad;
    std::fprintf(stderr,"FAIL %s D%d Q%d norm%d B%d heads%d TN%d past%d index%zu %04x != %04x\n",
        what,d,q,norm,batch,heads,tn,past,bad,x[bad],y[bad]);std::exit(3);
  }
}
template<int D,int Q,bool Norm>int Check() {
  using Body=IndependentAttentionTaskBody<Arch,D,Q,Norm,4*D*64,AttentionPagePolicy::WarpGroup>;
  Cuda(cudaFuncSetAttribute(Attend<D,Q,Norm>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Body::SharedStorage)));
  int cases=0;
  for(int batch:{1,16})for(int heads:{1,2}) {
    int columns=heads*(Q+2)*D,qkv_count=batch*columns,cache_count=batch*heads*Capacity*D;
    auto* raw=Allocate<Element>(qkv_count);auto* fused=Allocate<Element>(qkv_count);auto* expected=Allocate<Element>(qkv_count);
    auto* key=Allocate<Element>(cache_count);auto* value=Allocate<Element>(cache_count);
    auto* refkey=Allocate<Element>(cache_count);auto* refvalue=Allocate<Element>(cache_count);
    auto* cosine=Allocate<Element>(Capacity*D);auto* sine=Allocate<Element>(Capacity*D);
    auto* qnorm=Allocate<Element>(D);auto* knorm=Allocate<Element>(D);
    auto* context=Allocate<Element>(batch*heads*Q*D);auto* refcontext=Allocate<Element>(batch*heads*Q*D);
    auto* maps=Allocate<executor::TensorMap>(4);
    for(int i=0;i<qkv_count;++i)raw[i]=Element(.03125f*((i*7)%29-14));
    for(int i=0;i<Capacity*D;++i) {
      cosine[i]=Element(std::cos(.013f*(i/D+1)*(i%D+1)));
      sine[i]=Element(std::sin(.013f*(i/D+1)*(i%D+1)));
    }
    for(int d=0;d<D;++d){qnorm[d]=Element(.75f+.03125f*(d%11));knorm[d]=Element(.5f+.03125f*(d%13));}
    executor::TensorMapShape shape{D,std::uint64_t(batch*heads*Capacity),D*2,64,64};
    void* sources[]={key,value,refkey,refvalue};
    for(int i=0;i<4;++i)if(executor::EncodeTensorMap(maps[i],sources[i],shape)!=CUDA_SUCCESS)std::exit(4);
    QkvEpilogueDesc desc{key,value,cosine,sine,Norm?qnorm:nullptr,Norm?knorm:nullptr,heads,D,Q,Capacity,1e-6f};
    for(int tn:{D,2*D})for(int past:{0,31,63,64,127}) {
      std::memcpy(fused,raw,qkv_count*2);std::memcpy(expected,raw,qkv_count*2);
      for(int i=0;i<cache_count;++i) {
        key[i]=refkey[i]=Element(.015625f*((i*3)%23-11));value[i]=refvalue[i]=Element(.03125f*((i*5)%19-9));
      }
      desc.Validate(columns,tn,1,1);
      auto refdesc=desc;refdesc.key_cache=refkey;refdesc.value_cache=refvalue;
      Reference<D,Q,Norm><<<1,128>>>(refdesc,raw,expected,batch,past);
      Fuse<<<dim3((columns+tn-1)/tn,(batch+15)/16),128>>>(desc,fused,batch,columns,tn,past);
      Cuda(cudaDeviceSynchronize());
      Equal(fused,expected,qkv_count*2,"rounded Q",D,Q,Norm,batch,heads,tn,past);
      Equal(key,refkey,cache_count*2,"K cache",D,Q,Norm,batch,heads,tn,past);
      Equal(value,refvalue,cache_count*2,"V cache",D,Q,Norm,batch,heads,tn,past);
      ServingAttentionOperands p{raw,refkey,refvalue,cosine,sine,Norm?qnorm:nullptr,Norm?knorm:nullptr,
          refcontext,nullptr,nullptr,batch,heads,Capacity,past,Capacity,1e-6f};
      p.key_tensor_map=maps+2;p.value_tensor_map=maps+3;
      Attend<D,Q,Norm><<<dim3(heads,batch),128,sizeof(typename Body::SharedStorage)>>>(p);
      p.qkv=fused;p.key_cache=key;p.value_cache=value;p.context=context;
      p.key_tensor_map=maps;p.value_tensor_map=maps+1;p.qkv_preprocessed=true;
      Attend<D,Q,Norm><<<dim3(heads,batch),128,sizeof(typename Body::SharedStorage)>>>(p);
      Cuda(cudaDeviceSynchronize());
      Equal(context,refcontext,batch*heads*Q*D*2,"attention output",D,Q,Norm,batch,heads,tn,past);
      Equal(key,refkey,cache_count*2,"attention K cache",D,Q,Norm,batch,heads,tn,past);
      Equal(value,refvalue,cache_count*2,"attention V cache",D,Q,Norm,batch,heads,tn,past);
      ++cases;
    }
    for(void* p:{static_cast<void*>(raw),static_cast<void*>(fused),static_cast<void*>(expected),static_cast<void*>(key),static_cast<void*>(value),static_cast<void*>(refkey),static_cast<void*>(refvalue),static_cast<void*>(cosine),static_cast<void*>(sine),static_cast<void*>(qnorm),static_cast<void*>(knorm),static_cast<void*>(context),static_cast<void*>(refcontext),static_cast<void*>(maps)})Cuda(cudaFree(p));
  }
  return cases;
}
int HostContracts() {
  Element* pointer=reinterpret_cast<Element*>(16);
  QkvEpilogueDesc p{pointer,pointer,pointer,pointer,nullptr,nullptr,2,128,3,160,1e-6f};
  p.Validate(1280,256,1,1);
  int rejected=0;
  for(int i=0;i<6;++i)try {
    auto invalid=p;int columns=1280,tn=256,split=1,seq=1;
    if(i==0)tn=64;if(i==1)split=2;if(i==2)seq=64;if(i==3)columns=1024;
    if(i==4)invalid.q_norm=pointer;if(i==5)invalid.capacity=0;
    invalid.Validate(columns,tn,split,seq);
  }catch(std::invalid_argument const&){++rejected;}
  if(rejected!=6)return 5;
  std::puts("QKV epilogue six invalid geometry/rounding contracts PASS");return 0;
}
int main(int argc,char** argv) {
  if(int failed=HostContracts())return failed;
  if(argc>1 && std::string(argv[1])=="--host-only")return 0;
  int cases=Check<64,2,false>()+Check<64,2,true>()+Check<64,3,false>()+Check<64,3,true>()+
      Check<128,4,false>()+Check<128,4,true>()+Check<128,8,false>()+Check<128,8,true>();
  std::printf("QKV epilogue exact BF16 Q/cache and WGMMA attention output %d cases PASS\n",cases);
}
