// SPDX-License-Identifier: BSD-3-Clause
// Covers both serving head dimensions, Q/K norm modes and query row tilings.
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Element=cutlass::bfloat16_t;
constexpr int S=64;
std::string dump_directory;
template<class T> T* Managed(std::size_t count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(T))==cudaSuccess);
  return p;
}
template<int D,int Q,int R,bool Norm>
__global__ void Run(tilemega::codegen::ServingAttentionOperands p) {
  using Body=tilemega::codegen::FusedAttentionTaskBody<
      tilemega::arch::Sm89,D,Q,S,R,64,Norm>;
  extern __shared__ __align__(16) unsigned char memory[];
  auto& smem=*reinterpret_cast<typename Body::SharedStorage*>(memory);
  Body::Run(p,smem,0,0,int(blockIdx.x),0);
}
template<int D,int Q,int R,bool Norm>
bool Check() {
  using Body=tilemega::codegen::FusedAttentionTaskBody<
      tilemega::arch::Sm89,D,Q,S,R,64,Norm>;
  constexpr int group_width=(Q+2)*D;
  auto* qkv=Managed<Element>(S*group_width);
  auto* key=Managed<Element>(S*D),*value=Managed<Element>(S*D);
  auto* cosine=Managed<Element>(S*D),*sine=Managed<Element>(S*D);
  auto* qnorm=Managed<Element>(D),*knorm=Managed<Element>(D);
  auto* context=Managed<Element>(S*Q*D);
  for(int d=0;d<D;++d) {
    qnorm[d]=Element(0.75f+float(d%7)/32);
    knorm[d]=Element(0.875f+float(d%5)/32);
  }
  for(int t=0;t<S;++t) {
    for(int i=0;i<group_width;++i)
      qkv[t*group_width+i]=Element(float((t*19+i*7)%43-21)/256);
    for(int d=0;d<D;++d) {
      cosine[t*D+d]=Element(1.0f);sine[t*D+d]=Element(0.0f);
      key[t*D+d]=Element(-7);value[t*D+d]=Element(-7);
    }
  }
  for(int i=0;i<S*Q*D;++i)context[i]=Element(-7);
  tilemega::codegen::ServingAttentionOperands operands{
      qkv,key,value,cosine,sine,Norm?qnorm:nullptr,Norm?knorm:nullptr,
      context,nullptr,nullptr,1,1,S,0,S,1e-6f};
  auto shared=sizeof(typename Body::SharedStorage);
  assert(cudaFuncSetAttribute(Run<D,Q,R,Norm>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,shared)==cudaSuccess);
  Run<D,Q,R,Norm><<<(Q*S+R-1)/R,128,shared>>>(operands);
  auto status=cudaDeviceSynchronize();
  if(status!=cudaSuccess) {
    std::fprintf(stderr,"prefill D=%d Q=%d R=%d norm=%d: %s\n",
        D,Q,R,Norm,cudaGetErrorString(status));return false;
  }
  std::vector<Element> query(S*Q*D),keys(S*D);
  auto normalize=[&](Element const* x,Element const* weight,int d) {
    if constexpr (!Norm)return x[d];
    float square=0;for(int i=0;i<D;++i)square+=float(x[i])*float(x[i]);
    float inverse=rsqrtf(square/D+1e-6f);
    return Element(float(Element(float(x[d])*inverse))*float(weight[d]));
  };
  for(int t=0;t<S;++t) {
    for(int h=0;h<Q;++h)
      for(int d=0;d<D;++d)
        query[(t*Q+h)*D+d]=normalize(qkv+t*group_width+h*D,qnorm,d);
    for(int d=0;d<D;++d) {
      keys[t*D+d]=normalize(qkv+t*group_width+Q*D,knorm,d);
      if(key[t*D+d]!=keys[t*D+d] ||
         value[t*D+d]!=qkv[t*group_width+(Q+1)*D+d]) {
        std::fprintf(stderr,"prefill cache mismatch D=%d Q=%d R=%d norm=%d t=%d d=%d\n",
            D,Q,R,Norm,t,d);return false;
      }
    }
  }
  for(int t=0;t<S;++t)for(int h=0;h<Q;++h) {
    std::vector<float> scores(t+1);float peak=-INFINITY;
    for(int p=0;p<=t;++p) {
      float dot=0;for(int d=0;d<D;++d)
        dot+=float(query[(t*Q+h)*D+d])*float(keys[p*D+d]);
      scores[p]=dot/std::sqrt(float(D));peak=std::max(peak,scores[p]);
    }
    float sum=0;
    for(auto& score:scores){score=std::exp(score-peak);sum+=score;}
    for(int d=0;d<D;++d) {
      float numerator=0;
      for(int p=0;p<=t;++p)
        numerator+=float(Element(scores[p]))*
            float(qkv[p*group_width+(Q+1)*D+d]);
      float expected=numerator/sum;
      float actual=float(context[(t*Q+h)*D+d]);
      float limit=std::ldexp(std::abs(expected),-7)+std::ldexp(1.0f,-8);
      if(std::abs(actual-expected)>limit) {
        std::fprintf(stderr,"prefill D=%d Q=%d R=%d norm=%d t=%d h=%d d=%d got=%g ref=%g\n",
            D,Q,R,Norm,t,h,d,actual,expected);return false;
      }
    }
  }
  if(!dump_directory.empty()) {
    auto path=std::filesystem::path(dump_directory)/
        ("D"+std::to_string(D)+"_Q"+std::to_string(Q)+"_R"+
         std::to_string(R)+"_N"+std::to_string(int(Norm))+".bin");
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<char const*>(context),S*Q*D*sizeof(Element));
    if(!file)throw std::runtime_error("cannot write prefill attention dump");
  }
  cudaFree(qkv);cudaFree(key);cudaFree(value);cudaFree(cosine);
  cudaFree(sine);cudaFree(qnorm);cudaFree(knorm);cudaFree(context);
  return true;
}
} // namespace

int main(int argc,char** argv) {
  if(argc==3 && std::string(argv[1])=="--dump-dir") {
    dump_directory=argv[2];std::filesystem::create_directories(dump_directory);
  } else if(argc!=1) return 2;
  bool okay=true;
  okay&=Check<64,4,16,false>();okay&=Check<64,4,64,false>();
  okay&=Check<64,4,16,true>();okay&=Check<64,4,64,true>();
  okay&=Check<128,2,16,false>();okay&=Check<128,2,64,false>();
  okay&=Check<128,2,16,true>();okay&=Check<128,2,64,true>();
  if(!okay)return 1;
  std::puts("serving prefill attention matrix: pass");
}
