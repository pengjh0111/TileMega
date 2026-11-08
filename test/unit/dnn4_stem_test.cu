// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Backend/Sm90F3SwapAB.cuh>
#include <tilemega/Backend/Sm90F3Small.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <tilemega/Codegen/executor/TensorIm2col.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>

using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
namespace ex=tilemega::codegen::executor;

// tile_pages' global packing uses element addresses. CuTe's flagged shared
// layout applies its swizzle through the pointer engine instead.
constexpr int PackedIndex(int row,int col) {
  return row*64+8*((col/8)^(row%8))+col%8;
}

void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> struct Device {
  T* data=nullptr;std::size_t size;
  explicit Device(std::size_t count):size(count) {Check(cudaMalloc(&data,count*sizeof(T)));}
  ~Device() {cudaFree(data);}
  void Put(std::vector<T> const& v) {
    if(v.size()!=size)throw std::runtime_error("upload size differs");
    Check(cudaMemcpy(data,v.data(),size*sizeof(T),cudaMemcpyHostToDevice));
  }
  std::vector<T> Get() const {
    std::vector<T> v(size);Check(cudaMemcpy(v.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return v;
  }
};

std::uint64_t hash=14695981039346656037ull;
template<class T> void Hash(std::vector<T> const& v) {
  auto* bytes=reinterpret_cast<unsigned char const*>(v.data());
  for(std::size_t i=0;i<v.size()*sizeof(T);++i) {hash^=bytes[i];hash*=1099511628211ull;}
}
void Compare(std::vector<E> const& actual,std::vector<float> const& reference) {
  double dot=0,aa=0,rr=0,relative=0;std::size_t worst=0;
  for(std::size_t i=0;i<actual.size();++i) {
    double a=float(actual[i]),r=reference[i];
    if(!std::isfinite(a))throw std::runtime_error("nonfinite output");
    dot+=a*r;aa+=a*a;rr+=r*r;
    double error=std::abs(a-r)/std::max(std::abs(r),1e-12);
    if(error>relative) {relative=error;worst=i;}
  }
  double cosine=dot/std::sqrt(aa*rr);
  if(cosine<.99999 || relative>1e-2)
    throw std::runtime_error("FP32 reference mismatch: cosine="+std::to_string(cosine)+
        " max_relative="+std::to_string(relative)+" index="+std::to_string(worst)+
        " actual="+std::to_string(float(actual[worst]))+" expected="+std::to_string(reference[worst]));
  Hash(actual);
}

template<class G>
__global__ __launch_bounds__(384,1) void Body(Sm90F3GemmOperands p,
    const __grid_constant__ CUtensorMap activation,const __grid_constant__ CUtensorMap residual) {
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);
  if(threadIdx.x==0) {
    h->rings[0].Initialize();h->rings[1].Initialize();Sm90F3Async::Base::InitFence();
  }
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0) {h->rings[0].Prime();h->rings[1].Prime();}
  __syncthreads();
  p.activation=&activation;p.residual=&residual;
  Sm90F3Phase phase{};
  auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+G::kSideOffset;
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<56>();G::Produce(p,h->rings[0],arena,side,phase,0);
  }else {
    Sm90F3Async::RegAlloc<224>();G::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
  }
  __syncthreads();
  if(threadIdx.x==0) {h->rings[0].Invalidate();h->rings[1].Invalidate();}
}
template<class G> void Launch(Sm90F3GemmOperands p,CUtensorMap a,CUtensorMap r) {
  Check(cudaFuncSetAttribute(Body<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  Body<G><<<132,384,Sm90F3Budget::kBudget>>>(p,a,r);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
}
template<int B> __global__ __launch_bounds__(384,1) void Pool(Sm90F3SmallOperands p) {
  __shared__ Sm90F3Header h;Sm90F3Phase phase{};
  if(threadIdx.x<256)Sm90F3GlobalPool<B>::Consume(p,h.rings[0],nullptr,phase,&h,int(threadIdx.x)/128);
}


template<int H,int Window,int Stride,int Pad> struct StemGeometry {
  static constexpr int kW=H,kH=H,kC=8,kR=Window,kS=Window,
      kStrideW=Stride,kStrideH=Stride,kPadW=Pad,kPadH=Pad,kDilationW=1,kDilationH=1,
      kQ=(H+2*Pad-Window)/Stride+1,kP=kQ;
};
template<int B,int H,int Window,int Stride,int Pad,int TM,int TN,int Slots,int TK=64>
void StemCase() {
  using C=StemGeometry<H,Window,Stride,Pad>;
  constexpr int M=B*C::kQ*C::kP,N=TN*2,K=((Window*Window*8+63)/64)*64;
  using Chain=Sm90F3Chain<Sm90F3SideLayout<TM,TN,false,1,false>,Sm90F3Step<Sm90ForwardOp::kBias,0>>;
  using Gather=Sm90F3Gemm<TM,TN,Slots,1,Sm90F3Mode::kGather,M,N,K,N,Chain,C,Sm90F3NoPool,TK>;
  using Tma=Sm90F3Gemm<TM,TN,Slots,1,Sm90F3Mode::kTmaStem,M,N,K,N,Chain,C,Sm90F3NoPool,TK>;
  std::vector<E> input(B*H*H*8),weight(N*K),packed(N*K);
  for(int i=0;i<int(input.size());++i)input[i]=E(i%8<3?float((i*17+1)%19-9)/32:0.f);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E v(k<Window*Window*8 && k%8<3?float((n*7+k*3)%13-6)/32:0.f);
    weight[n*K+k]=v;packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=v;
  }
  Device<E> a(input.size()),w(packed.size()),o(M*N);Device<float> bias(N);
  a.Put(input);w.Put(packed);bias.Put(std::vector<float>(N,4.f));
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;p.epilogue.output=o.data;
  p.epilogue.output_stride=N;p.epilogue.vectors[0]=bias.data;
  CUtensorMap map{},unused{};tilemega::codegen::ConvDesc conv{};
  conv.n=B;conv.h=H;conv.w=H;conv.c=3;conv.k=N;conv.r=Window;conv.s=Window;
  conv.stride_h=Stride;conv.stride_w=Stride;conv.pad_h=Pad;conv.pad_w=Pad;conv.p=C::kP;conv.q=C::kQ;
  tilemega::codegen::DmBufferLayout layout{};layout.kind=tilemega::codegen::DmLayout::kNHWC;layout.rank=4;
  unsigned dims[]={B,H,H,8};
  for(int i=0;i<4;++i)layout.logical[i]=layout.physical[i]=dims[i];
  layout.strides[3]=1;layout.strides[2]=8;layout.strides[1]=H*8;layout.strides[0]=H*H*8;
  auto status=Sm90F3Async::EncodeStemMap(map,a.data,conv,layout);
  if(status!=CUDA_SUCCESS)throw std::runtime_error("stem im2col encode status="+std::to_string(status));
  Launch<Gather>(p,unused,unused);auto reference=o.Get();
  Launch<Tma>(p,map,unused);auto actual=o.Get();
  if(std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(E)))
    throw std::runtime_error("stem TMA differs bitwise from96-lane gather");
  std::vector<float> expected(M*N);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
    int image=m/(C::kQ*C::kP),pixel=m%(C::kQ*C::kP);float dot=0;
    for(int rr=0;rr<Window;++rr)for(int ss=0;ss<Window;++ss) {
      int y=(pixel/C::kQ)*Stride-Pad+rr,x=(pixel%C::kQ)*Stride-Pad+ss;
      if(y<0 || y>=H || x<0 || x>=H)continue;
      for(int c=0;c<3;++c)dot+=float(input[((image*H+y)*H+x)*8+c])*float(weight[n*K+(rr*Window+ss)*8+c]);
    }
    expected[m*N+n]=dot+4.f;
  }
  Compare(actual,expected);std::cout<<"stem_b"<<B<<"_w"<<Window<<"_tm"<<TM<<"_tk"<<TK<<" PASS\n";
}
int main(int,char**) {
  try {
    StemCase<3,9,7,2,3,128,64,4>();
    StemCase<2,17,3,1,1,64,64,8>();
    StemCase<2,17,3,1,1,128,128,3,128>();
    StemCase<3,17,3,1,1,256,64,4>();
    std::cout<<"output_hash="<<std::hex<<hash<<"\n";return 0;
  }catch(std::exception const& error) {std::cerr<<error.what()<<"\n";return 1;}
}
