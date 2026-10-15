// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90PrefillAttentionTaskBody.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=tilemega::arch::Sm90;
void CheckCuda(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* Allocate(std::size_t n){T* p;CheckCuda(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<int D,int Q,int Tokens,bool Norm,bool Hopper>
__global__ void Run(ServingAttentionOperands p) {
  using Body=std::conditional_t<Hopper,Sm90PrefillAttentionTaskBody<Arch,D,Q,Tokens,64,64,Norm>,
      FusedAttentionTaskBody<Arch,D,Q,Tokens,64,64,Norm>>;
  extern __shared__ __align__(1024) char s[];
  for(int qb=0;qb<(Tokens*Q+63)/64;++qb) {
    Body::Run(p,*reinterpret_cast<typename Body::SharedStorage*>(s),0,0,qb,0);
    __syncthreads();
  }
}
template<int D,int Q,int Tokens,bool Norm>
void Case(int past) {
  constexpr int Capacity=512,Width=(Q+2)*D;
  auto* qkv=Allocate<E>(Tokens*Width);auto* key=Allocate<E>(Capacity*D);
  auto* value=Allocate<E>(Capacity*D);auto* cosine=Allocate<E>(Capacity*D);auto* sine=Allocate<E>(Capacity*D);
  auto* qnorm=Allocate<E>(D);auto* knorm=Allocate<E>(D);auto* out=Allocate<E>(Tokens*Q*D);
  auto* legacy=Allocate<E>(Tokens*Q*D);
  for(int i=0;i<Tokens*Width;++i)qkv[i]=E((i*13%37-18)*.03125f);
  for(int i=0;i<Capacity*D;++i) {
    key[i]=E((i*7%41-20)*.015625f);value[i]=E((i*17%43-21)*.03125f);
    int pos=i/D,d=i%D;cosine[i]=E(std::cos(.003f*(pos+1)*(d%(D/2)+1)));
    sine[i]=E(std::sin(.003f*(pos+1)*(d%(D/2)+1)));
  }
  for(int d=0;d<D;++d){qnorm[d]=E(.75f+(d%9)*.03125f);knorm[d]=E(.875f+(d%7)*.015625f);}
  for(int i=0;i<Tokens*Q*D;++i){out[i]=E(NAN);legacy[i]=E(NAN);}
  ServingAttentionOperands p{qkv,key,value,cosine,sine,qnorm,knorm,out,nullptr,nullptr,1,1,Capacity,past,Capacity,1e-6f};
  using H=Sm90PrefillAttentionTaskBody<Arch,D,Q,Tokens,64,64,Norm>;
  using L=FusedAttentionTaskBody<Arch,D,Q,Tokens,64,64,Norm>;
  CheckCuda(cudaFuncSetAttribute(Run<D,Q,Tokens,Norm,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename H::SharedStorage)));
  Run<D,Q,Tokens,Norm,true><<<1,128,sizeof(typename H::SharedStorage)>>>(p);CheckCuda(cudaDeviceSynchronize());
  p.context=legacy;
  Run<D,Q,Tokens,Norm,false><<<1,128,sizeof(typename L::SharedStorage)>>>(p);CheckCuda(cudaDeviceSynchronize());
  auto rotate=[&](E const* x,E const* norm,int pos) {
    std::vector<float> r(D);float ss=0;for(int d=0;d<D;++d)ss+=float(x[d])*float(x[d]);
    float inverse=Norm?1/std::sqrt(ss/D+1e-6f):1;
    for(int d=0;d<D;++d) {
      int partner=(d+D/2)%D;E a=x[d],b=x[partner];
      if constexpr(Norm){a=E(float(E(float(a)*inverse))*float(norm[d]));b=E(float(E(float(b)*inverse))*float(norm[partner]));}
      E first=E(float(a)*float(cosine[pos*D+d]));
      E second=E((d<D/2?-float(b):float(b))*float(sine[pos*D+d]));r[d]=float(E(float(first)+float(second)));
    }
    return r;
  };
  std::vector<std::vector<float>> keys(Tokens);
  for(int t=0;t<Tokens;++t)keys[t]=rotate(qkv+t*Width+Q*D,knorm,past+t);
  float cpu_error=0,legacy_error=0;
  // Fixed thresholds precede execution; no reference depends on produced KV.
  for(int t=0;t<Tokens;++t)for(int h=0;h<Q;++h) {
    auto query=rotate(qkv+t*Width+h*D,qnorm,past+t);
    int count=past+t+1;std::vector<float> scores(count);float peak=-INFINITY;
    for(int pos=0;pos<count;++pos) {
      float dot=0;for(int d=0;d<D;++d)dot+=query[d]*(pos<past?float(key[pos*D+d]):keys[pos-past][d]);
      scores[pos]=dot/std::sqrt(float(D));peak=std::max(peak,scores[pos]);
    }
    float sum=0;for(float& score:scores){score=std::exp(score-peak);sum+=score;}
    for(int d=0;d<D;++d) {
      float expected=0;for(int pos=0;pos<count;++pos)expected+=scores[pos]*(pos<past?float(value[pos*D+d]):float(qkv[(pos-past)*Width+(Q+1)*D+d]));
      expected/=sum;int index=(t*Q+h)*D+d;float actual=float(out[index]),old=float(legacy[index]);
      cpu_error=std::max(cpu_error,std::abs(actual-expected));legacy_error=std::max(legacy_error,std::abs(actual-old));
      if(!std::isfinite(actual) || std::abs(actual-expected)>.008f*(1+std::abs(expected)) ||
          std::abs(actual-old)>.012f*(1+std::abs(old))) {
        std::fprintf(stderr,"prefill D%d Q%d T%d norm%d past%d token%d head%d d%d: %g CPU %g legacy %g\n",
            D,Q,Tokens,Norm,past,t,h,d,actual,expected,old);std::exit(3);
      }
    }
  }
  std::printf("prefill D%d Q%d T%d norm%d past%d CPU_error=%g legacy_error=%g PASS\n",D,Q,Tokens,Norm,past,cpu_error,legacy_error);
  for(void* ptr:{static_cast<void*>(qkv),static_cast<void*>(key),static_cast<void*>(value),static_cast<void*>(cosine),
      static_cast<void*>(sine),static_cast<void*>(qnorm),static_cast<void*>(knorm),static_cast<void*>(out),static_cast<void*>(legacy)})CheckCuda(cudaFree(ptr));
}
template<int D,int Q,int T>void Cases(){for(int p:{0,65,129}){Case<D,Q,T,false>(p);Case<D,Q,T,true>(p);}}
template<int D,int Q>void Shapes(){Cases<D,Q,17>();Cases<D,Q,64>();Cases<D,Q,73>();}
int main(){Shapes<64,1>();Shapes<64,3>();Shapes<64,8>();Shapes<128,1>();Shapes<128,3>();Shapes<128,8>();
  std::puts("Hopper prefill attention: 108 causal/QKnorm/RoPE/past/query-tail cases PASS");}
