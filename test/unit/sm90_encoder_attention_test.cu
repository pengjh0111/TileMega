// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90EncoderAttentionTaskBody.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;using Arch=tilemega::arch::Sm90;
void CheckCuda(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* Allocate(std::size_t n){T* p;CheckCuda(cudaMallocManaged(&p,sizeof(T)*n));return p;}
__global__ void Hopper(Sm90EncoderAttentionOperands p) {
  extern __shared__ __align__(1024) char storage[];
  using Body=Sm90EncoderAttentionTaskBody<Arch>;
  auto& s=*reinterpret_cast<typename Body::SharedStorage*>(storage);
  for(unsigned t=0;t<Body::Count(p.base);++t)Body::Run(p,t,s);
}
__global__ void Legacy(EncoderAttentionOperands p) {
  extern __shared__ char storage[];using Body=EncoderAttentionTaskBody<Arch,128,64,false>;
  auto& s=*reinterpret_cast<typename Body::SharedStorage*>(storage);
  for(unsigned t=0;t<Body::Count(p);++t)Body::Run(p,t,s);
}
void Case(int batch,int heads,int seed,bool tma) {
  std::size_t values=std::size_t(batch)*128*heads*192,outputs=values/3;
  auto* qkv=Allocate<E>(values);auto* out=Allocate<E>(outputs);auto* legacy=Allocate<E>(outputs);
  auto* maps=Allocate<executor::TensorMap>(batch*heads*3);
  float amplitude=seed%2?.5f:1.75f;
  for(std::size_t i=0;i<values;++i)qkv[i]=E(amplitude*std::sin(float(i%7919)*.017f+seed));
  for(std::size_t i=0;i<outputs;++i){out[i]=E(NAN);legacy[i]=E(NAN);}
  if(tma)for(int b=0;b<batch;++b)for(int h=0;h<heads;++h)for(int slot=0;slot<3;++slot) {
    auto* address=qkv+(std::size_t(b)*128*heads+h)*192+slot*64;
    auto code=executor::EncodeTensorMap(maps[(b*heads+h)*3+slot],address,{64,128,std::uint64_t(heads)*192*2,64,slot==0?64u:128u});
    if(code!=CUDA_SUCCESS){std::fprintf(stderr,"encoder TMA map failed %d\n",int(code));std::exit(4);}
  }
  EncoderAttentionOperands base{qkv,out,nullptr,unsigned(batch),unsigned(heads)};
  Sm90EncoderAttentionOperands p{base,tma?maps:nullptr};using Body=Sm90EncoderAttentionTaskBody<Arch>;
  CheckCuda(cudaFuncSetAttribute(Hopper,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes));
  Hopper<<<1,128,Body::kSharedBytes>>>(p);CheckCuda(cudaDeviceSynchronize());
  base.context=legacy;Legacy<<<1,128,sizeof(typename EncoderAttentionTaskBody<Arch,128,64,false>::SharedStorage)>>>(base);
  CheckCuda(cudaDeviceSynchronize());float cpu_error=0,legacy_error=0;
  for(int b=0;b<batch;++b)for(int h=0;h<heads;++h)for(int q=0;q<128;++q) {
    std::vector<float> score(128);float peak=-INFINITY;
    for(int k=0;k<128;++k) {
      float dot=0;for(int d=0;d<64;++d)dot+=float(qkv[((std::size_t(b)*128+q)*heads+h)*192+d])*float(qkv[((std::size_t(b)*128+k)*heads+h)*192+64+d]);
      score[k]=dot/8;peak=std::max(peak,score[k]);
    }
    float sum=0;for(float& x:score){x=std::exp(x-peak);sum+=x;}
    for(int d=0;d<64;++d) {
      float expected=0;for(int k=0;k<128;++k)expected+=score[k]*float(qkv[((std::size_t(b)*128+k)*heads+h)*192+128+d]);
      expected/=sum;auto i=((std::size_t(b)*128+q)*heads+h)*64+d;
      float got=float(out[i]),old=float(legacy[i]);cpu_error=std::max(cpu_error,std::abs(got-expected));legacy_error=std::max(legacy_error,std::abs(got-old));
      if(!std::isfinite(got) || std::abs(got-expected)>.008f*(1+std::abs(expected)) || std::abs(got-old)>.012f*(1+std::abs(old))) {
        std::fprintf(stderr,"encoder B%d H%d seed%d TMA%d query%d d%d: %g CPU %g legacy %g\n",batch,heads,seed,tma,q,d,got,expected,old);std::exit(3);
      }
    }
  }
  std::printf("encoder B%d H%d seed%d TMA%d CPU_error=%g legacy_error=%g PASS\n",batch,heads,seed,tma,cpu_error,legacy_error);
  for(void* ptr:{static_cast<void*>(qkv),static_cast<void*>(out),static_cast<void*>(legacy),static_cast<void*>(maps)})CheckCuda(cudaFree(ptr));
}
int main(){using Body=Sm90EncoderAttentionTaskBody<Arch>;
  auto k=cute::as_position_independent_swizzle_layout(typename Body::KLayout{});
  auto v=cute::as_position_independent_swizzle_layout(typename Body::VLayout{});
  for(int r=0;r<128;++r)for(int d=0;d<64;++d)if(k(r,d)!=v(d,r)){std::fprintf(stderr,"encoder layout mismatch\n");return 5;}
  for(int batch:{1,2})for(int heads:{1,3,12})for(int seed:{1,2})for(bool tma:{false,true})Case(batch,heads,seed,tma);
  std::puts("encoder WGMMA: 48 TMA/cp.async/B/head/softmax cases PASS");}
