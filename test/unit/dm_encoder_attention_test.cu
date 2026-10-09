// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/EncoderAttentionTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
template<class T>T* Managed(std::size_t count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(T))==cudaSuccess);return p;
}
void Complete() {assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);}
template<int Sequence,int Rows,bool Masked>
__global__ void Run(EncoderAttentionOperands p) {
  __shared__ typename EncoderAttentionTaskBody<Arch,Sequence,Rows,Masked>::SharedStorage shared;
  EncoderAttentionTaskBody<Arch,Sequence,Rows,Masked>::Run(p,blockIdx.x,shared);
}
// Independent FP64 reference: full QK, conventional row softmax, then PV.
// Its ownership and layouts do not use any attention-warp helper or swizzle.
__global__ void Scores(EncoderAttentionOperands p,double* scores,unsigned sequence) {
  auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  auto size=std::size_t(p.batch)*p.heads*sequence*sequence;if(index>=size)return;
  unsigned key=index%sequence,query=(index/sequence)%sequence;
  unsigned head=(index/(sequence*sequence))%p.heads,image=index/(p.heads*sequence*sequence);
  if(p.key_padding && !p.key_padding[image*sequence+key]) {scores[index]=-INFINITY;return;}
  auto q=p.qkv+((std::size_t(image)*sequence+query)*p.heads+head)*192;
  auto k=p.qkv+((std::size_t(image)*sequence+key)*p.heads+head)*192+64;
  double sum=0;for(unsigned d=0;d<64;++d)sum+=double(float(q[d]))*float(k[d]);
  scores[index]=sum/8;
}
__global__ void Softmax(double* scores,std::size_t rows,unsigned sequence) {
  auto row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=rows)return;
  auto* values=scores+row*sequence;double maximum=-INFINITY,sum=0;
  for(unsigned key=0;key<sequence;++key)maximum=fmax(maximum,values[key]);
  for(unsigned key=0;key<sequence;++key)sum+=exp(values[key]-maximum);
  for(unsigned key=0;key<sequence;++key)values[key]=exp(values[key]-maximum)/sum;
}
__global__ void Context(EncoderAttentionOperands p,double const* scores,double* output,unsigned sequence) {
  auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  auto size=std::size_t(p.batch)*sequence*p.heads*64;if(index>=size)return;
  unsigned d=index%64,head=(index/64)%p.heads,query=(index/(64*p.heads))%sequence;
  unsigned image=index/(64*p.heads*sequence);
  auto* probability=scores+((std::size_t(image)*p.heads+head)*sequence+query)*sequence;
  double sum=0;
  for(unsigned key=0;key<sequence;++key)
    sum+=probability[key]*float(p.qkv[((std::size_t(image)*sequence+key)*p.heads+head)*192+128+d]);
  output[index]=sum;
}
template<int Sequence,int Rows,bool Masked>
unsigned Check(unsigned pattern) {
  unsigned batch=1+pattern%2,heads=pattern%2?3:2;
  auto elements=std::size_t(batch)*Sequence*heads*64;
  auto* qkv=Managed<E>(3*elements);auto* output=Managed<E>(elements+32);
  auto* reference=Managed<double>(elements);
  auto* scores=Managed<double>(std::size_t(batch)*heads*Sequence*Sequence);
  auto* mask=Managed<std::int64_t>(batch*Sequence);
  for(unsigned image=0;image<batch;++image)for(unsigned token=0;token<Sequence;++token) {
    bool valid=pattern<2 || (pattern==2?((token+image*13)%7!=0 && token<Sequence-31):token==Sequence-1);
    mask[image*Sequence+token]=valid;
    for(unsigned head=0;head<heads;++head)for(unsigned kind=0;kind<3;++kind)for(unsigned d=0;d<64;++d) {
      double value=std::sin((image*37+token*13+head*19+kind*23+d*7)*.071)*(.3+pattern*.7);
      if(kind==2)value+=double(int(token%17)-8)*.125+head*.3+image*.2;
      qkv[((std::size_t(image)*Sequence+token)*heads+head)*192+kind*64+d]=E(value);
    }
  }
  EncoderAttentionOperands p{qkv,output+16,Masked?mask:nullptr,batch,heads};
  std::size_t score_elements=std::size_t(batch)*heads*Sequence*Sequence;
  Scores<<<(score_elements+255)/256,256>>>(p,scores,Sequence);
  Softmax<<<(batch*heads*Sequence+127)/128,128>>>(scores,batch*heads*Sequence,Sequence);
  Context<<<(elements+255)/256,256>>>(p,scores,reference,Sequence);Complete();
  std::vector<E> first(elements);
  double maximum_error=0;
  for(unsigned epoch=0;epoch<3;++epoch) {
    for(std::size_t i=0;i<elements+32;++i)output[i]=E(-12345);
    Run<Sequence,Rows,Masked><<<EncoderAttentionTaskBody<Arch,Sequence,Rows,Masked>::Count(p),128>>>(p);Complete();
    for(std::size_t i=0;i<elements;++i) {
      double error=std::abs(float(output[16+i])-reference[i]);maximum_error=std::max(maximum_error,error);
      assert(error<=.016+.016*std::abs(reference[i]));
      if(epoch==0)first[i]=output[16+i];else assert(first[i]==output[16+i]);
    }
    for(unsigned i=0;i<16;++i)assert(output[i]==E(-12345) && output[16+elements+i]==E(-12345));
    if constexpr(Masked) {
      if(pattern==1) {
        Run<Sequence,Rows,false><<<EncoderAttentionTaskBody<Arch,Sequence,Rows,false>::Count(p),128>>>(p);Complete();
        for(std::size_t i=0;i<elements;++i)assert(first[i]==output[16+i]);
      }
    }
  }
  std::printf("ENCODER case S=%d Rows=%d mask=%u batch=%u heads=%u max_error=%.9g PASS\n",
      Sequence,Rows,pattern,batch,heads,maximum_error);
  cudaFree(qkv);cudaFree(output);cudaFree(reference);cudaFree(scores);cudaFree(mask);return 1;
}
template<int Sequence,int Rows>unsigned Geometry() {
  unsigned count=Check<Sequence,Rows,false>(0);
  for(unsigned pattern:{1,2,3})count+=Check<Sequence,Rows,true>(pattern);
  return count;
}
int main() {
  unsigned cases=Geometry<128,64>()+Geometry<128,128>()+Geometry<384,64>()+
      Geometry<384,128>()+Geometry<512,64>()+Geometry<512,128>();
  assert(cases==24);
  std::printf("ENCODER cases=%u epochs=3 noncausal_MHA_D64_double_buffer_FP64_mask_canaries PASS\n",cases);
}
