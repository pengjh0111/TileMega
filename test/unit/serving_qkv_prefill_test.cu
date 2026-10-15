// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_QKV_EPILOGUE 1
#include <tilemega/Codegen/tasks/ServingQkvEpilogue.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;using Arch=tilemega::arch::Sm90;
constexpr int Capacity=256;
void Check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}}
template<class T>T* Allocate(std::size_t n){T* p;Check(cudaMallocManaged(&p,n*sizeof(T)));return p;}
__global__ void Fuse(QkvEpilogueDesc p,E* qkv,int rows,int columns,int tm,int tn,int past){
  ServingQkvEpilogue<Arch>::Run(p,qkv,rows,columns,blockIdx.y,blockIdx.x,tm,tn,past);
}
template<int D,int Q,bool Norm>
__global__ void Reference(QkvEpilogueDesc p,E const* raw,E* expected,int rows,int past){
  using Base=FusedAttentionTaskBody<Arch,D,Q,1,16,32,Norm>;
  int columns=p.heads_kv*(Q+2)*D,lane=threadIdx.x%32;
  for(int item=threadIdx.x/32;item<rows*p.heads_kv*(Q+2);item+=4){
    int row=item/(p.heads_kv*(Q+2)),head=item%(p.heads_kv*(Q+2));
    int batch=row/p.sequence,token=row%p.sequence,group=head/(Q+2),kind=head%(Q+2);
    int pos=past+token;auto* source=raw+std::size_t(row)*columns+head*D;
    auto offset=((std::size_t(batch)*p.heads_kv+group)*Capacity+pos)*D;
    if(kind<Q)Base::RotateRow(source,p.q_norm,p.cosine+pos*D,p.sine+pos*D,p.epsilon,
      [&](int d,uint4 value){tilemega::backend::StoreGlobal16(expected+std::size_t(row)*columns+head*D+d,value);});
    else if(kind==Q)Base::RotateRow(source,p.k_norm,p.cosine+pos*D,p.sine+pos*D,p.epsilon,
      [&](int d,uint4 value){tilemega::backend::StoreGlobal16(p.key_cache+offset+d,value);});
    else if(lane*8<D)tilemega::backend::StoreGlobal16(p.value_cache+offset+lane*8,tilemega::backend::LoadGlobal16(source+lane*8));
  }
}
template<int D,int Q,bool Norm>int Cases(){
  int cases=0;
  for(int tokens:{17,64,73})for(int batch:{1,2})for(int tm:{16,64})for(int tn:{D,2*D})for(int past:{0,65}){
    constexpr int heads=2,columns=heads*(Q+2)*D;int rows=batch*tokens;
    std::size_t count=std::size_t(rows)*columns,cache=std::size_t(batch)*heads*Capacity*D;
    auto* raw=Allocate<E>(count);auto* fused=Allocate<E>(count);auto* expected=Allocate<E>(count);
    auto* key=Allocate<E>(cache);auto* value=Allocate<E>(cache);auto* refkey=Allocate<E>(cache);auto* refvalue=Allocate<E>(cache);
    auto* cos=Allocate<E>(Capacity*D);auto* sin=Allocate<E>(Capacity*D);auto* qnorm=Allocate<E>(D);auto* knorm=Allocate<E>(D);
    for(std::size_t i=0;i<count;++i)raw[i]=fused[i]=expected[i]=E((int(i*7%29)-14)*.03125f);
    for(std::size_t i=0;i<cache;++i){key[i]=refkey[i]=E((int(i*3%23)-11)*.015625f);value[i]=refvalue[i]=E((int(i*5%19)-9)*.03125f);}
    for(int i=0;i<Capacity*D;++i){cos[i]=E(std::cos(.003f*(i/D+1)*(i%D+1)));sin[i]=E(std::sin(.003f*(i/D+1)*(i%D+1)));}
    for(int d=0;d<D;++d){qnorm[d]=E(.75f+(d%11)*.03125f);knorm[d]=E(.5f+(d%13)*.03125f);}
    QkvEpilogueDesc desc{key,value,cos,sin,Norm?qnorm:nullptr,Norm?knorm:nullptr,heads,D,Q,Capacity,1e-6f};desc.sequence=tokens;desc.Validate(columns,tn,1,tokens);
    auto ref=desc;ref.key_cache=refkey;ref.value_cache=refvalue;
    Reference<D,Q,Norm><<<1,128>>>(ref,raw,expected,rows,past);
    Fuse<<<dim3((columns+tn-1)/tn,(rows+tm-1)/tm),128>>>(desc,fused,rows,columns,tm,tn,past);Check(cudaDeviceSynchronize());
    if(std::memcmp(fused,expected,count*2)||std::memcmp(key,refkey,cache*2)||std::memcmp(value,refvalue,cache*2)){
      std::fprintf(stderr,"QKV prefill D%d Q%d norm%d B%d T%d TM%d TN%d past%d mismatch\n",D,Q,Norm,batch,tokens,tm,tn,past);std::exit(3);
    }
    for(void* ptr:{(void*)raw,(void*)fused,(void*)expected,(void*)key,(void*)value,(void*)refkey,(void*)refvalue,(void*)cos,(void*)sin,(void*)qnorm,(void*)knorm})Check(cudaFree(ptr));++cases;
  }
  return cases;
}
int main(){int n=Cases<64,1,false>()+Cases<64,1,true>()+Cases<64,4,false>()+Cases<64,4,true>()+
  Cases<128,2,false>()+Cases<128,2,true>()+Cases<128,3,false>()+Cases<128,3,true>()+
  Cases<128,4,false>()+Cases<128,4,true>()+Cases<128,8,false>()+Cases<128,8,true>();
  std::printf("PASS QKV prefill exact BF16/cache %d batch/token/position/tile cases\n",n);}
