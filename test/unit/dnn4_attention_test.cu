#include <tilemega/Backend/Sm90F3Attention.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstdint>
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
namespace ex=tilemega::codegen::executor;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
 T* data;std::size_t count;
 Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}
 ~Device(){cudaFree(data);}
 void Put(std::vector<T> const& v){Check(cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> Get(){std::vector<T> v(count);Check(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
std::uint64_t hash=14695981039346656037ull;
#ifdef DNN4_ATTENTION_PROFILE3_TEST
constexpr int kProducerRegisters=72,kConsumerRegisters=216;
#else
constexpr int kProducerRegisters=56,kConsumerRegisters=224;
#endif
template<int B,int Rows> __global__ __launch_bounds__(384,1) void AttentionKernel(
 const __grid_constant__ CUtensorMap q,const __grid_constant__ CUtensorMap k,
 const __grid_constant__ CUtensorMap v,E* output) {
 extern __shared__ __align__(1024) char shared[];
 auto* h=reinterpret_cast<Sm90F3Header*>(shared);
 if(threadIdx.x==0){h->rings[0].Initialize();Sm90F3Async::Base::InitFence();}
 __syncthreads();
 if(threadIdx.x<256 && (threadIdx.x&31)==0)h->rings[0].Prime();
 __syncthreads();
 Sm90F3AttentionOperands p{&q,&k,&v,output};Sm90F3Phase phase{};
 if(threadIdx.x>=256){Sm90F3Async::RegDealloc<kProducerRegisters>();Sm90F3Attention<B,Rows>::Produce(p,h->rings[0],shared+2048,phase);}
 else {Sm90F3Async::RegAlloc<kConsumerRegisters>();Sm90F3Attention<B,Rows>::Consume(p,h->rings[0],shared+2048,phase);}
 __syncthreads();if(threadIdx.x==0)h->rings[0].Invalidate();
}
template<int B,int Rows=(16*B<128?16*B:128)> void Case() {
 constexpr int S=128,D=64,H=12;
 std::vector<E> input(B*S*H*3*D);
 for(std::size_t i=0;i<input.size();++i)input[i]=E(float(int(i%37)-18)/32);
 Device<E> a(input.size()),o(B*S*H*D);a.Put(input);CUtensorMap maps[3]{};
 for(int part=0;part<3;++part)ex::EncodeTiledMap<3>(maps[part],a.data+part*D,
  {{D,H,B*S},{3*D*2,3*H*D*2},{64,1,unsigned(part==0?Rows:128)}});
 Check(cudaFuncSetAttribute(AttentionKernel<B,Rows>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
 AttentionKernel<B,Rows><<<132,384,Sm90F3Budget::kBudget>>>(maps[0],maps[1],maps[2],o.data);
 Check(cudaGetLastError());Check(cudaDeviceSynchronize());auto actual=o.Get();
 for(int b=0;b<B;++b)for(int head=0;head<H;++head) {
  double dot=0,aa=0,rr=0;
  for(int q=0;q<S;++q) {
   float score[S],maximum=-INFINITY,total=0;
   for(int k=0;k<S;++k) {
    float v=0;
    for(int d=0;d<D;++d)
     v+=float(input[((b*S+q)*H+head)*3*D+d])*float(input[((b*S+k)*H+head)*3*D+D+d]);
    score[k]=v*.125f;maximum=std::max(maximum,score[k]);
   }
   for(int k=0;k<S;++k){score[k]=std::exp(score[k]-maximum);total+=score[k];}
   for(int d=0;d<D;++d) {
    float ref=0;for(int k=0;k<S;++k)ref+=score[k]/total*float(input[((b*S+k)*H+head)*3*D+2*D+d]);
    float value=float(actual[((b*S+q)*H+head)*D+d]);
    if(!std::isfinite(value))throw std::runtime_error("nonfinite attention");
    dot+=value*ref;aa+=value*value;rr+=ref*ref;
   }
  }
  double cosine=dot/std::sqrt(aa*rr);
  if(cosine<.9999)throw std::runtime_error("attention head mismatch b="+std::to_string(b)+" h="+std::to_string(head)+" cosine="+std::to_string(cosine));
 }
 for(E e:actual){auto bits=e.raw();for(int byte=0;byte<2;++byte){hash^=(bits>>(byte*8))&255;hash*=1099511628211ull;}}
 std::cout<<"attention_b"<<B<<"_rows"<<Rows<<" PASS\n";
}
int main(int,char**)try{Case<1>();Case<2>();Case<4>();Case<16>();Case<16,64>();std::cout<<"output_hash="<<std::hex<<hash<<'\n';return 0;}
catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
