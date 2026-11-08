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

template<int TM,int B,int SplitK=1> void PoolCase() {
  constexpr int area=49,M=B*area,N=128,K=64*SplitK,TN=64,parts=2;
  using Chain=Sm90F3Chain<Sm90F3SideLayout<TM,TN,true,2,false>,
      Sm90F3Step<Sm90ForwardOp::kScale,0>,Sm90F3Step<Sm90ForwardOp::kBias,1>,
      Sm90F3Step<Sm90ForwardOp::kResidual,~0u>>;
  using G=Sm90F3Gemm<TM,TN,3,SplitK,Sm90F3Mode::kNormal,M,N,K,N,Chain,
      Sm90F3NoConv,Sm90F3Pool<area,parts,N>>;
  std::vector<E> input(M*K),weight(N*K),packed(N*K),residual(M*N);
  for(int i=0;i<N*K;++i)weight[i]=E(float((i*7+3)%13-6)/32);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k)
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=weight[n*K+k];
  for(int i=0;i<M*N;++i)residual[i]=E(float((i*11)%7-3)/16);
  Device<E> a(M*K),w(N*K),r(M*N),o(M*N),pooled(B*N);
  Device<float> scale(TN*2),bias(TN*2),partial(B*parts*N+32);
  constexpr int tiles=((M+TM-1)/TM)*(N/TN);
  Device<float> split_partial(tiles*SplitK*TM*TN);
  Device<unsigned long long> counters(tiles);
  counters.Put(std::vector<unsigned long long>(tiles,0));
  w.Put(packed);r.Put(residual);scale.Put(std::vector<float>(N,.5f));bias.Put(std::vector<float>(N,.25f));
  constexpr float canary=-12345.25f;
  partial.Put(std::vector<float>(partial.size,canary));
  CUtensorMap am{},rm{};
  ex::EncodeTiledMap<2>(am,a.data,{{K,M},{K*2},{64,TM}});
  ex::EncodeTiledMap<2>(rm,r.data,{{N,M},{N*2},{64,TM}});
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;
  p.epilogue.vectors[0]=scale.data;p.epilogue.vectors[1]=bias.data;
  p.epilogue.residual=r.data;p.epilogue.output=o.data;
  p.epilogue.output_stride=N;p.epilogue.residual_stride=N;p.epilogue.channel_partial=partial.data+16;
  p.epilogue.partial=split_partial.data;p.epilogue.counters=counters.data;
  for(int repeat=0;repeat<3;++repeat) {
    for(int i=0;i<M*K;++i)input[i]=E(float((i*17+7)%19-9)*(repeat+1)/32);
    a.Put(input);Launch<G>(p,am,rm);
    std::vector<float> expected(M*N),sums(B*parts*N,0),mean(B*N,0);
    for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
      float dot=0;for(int k=0;k<K;++k)dot+=float(input[m*K+k])*float(weight[n*K+k]);
      float value=(dot*.5f+.25f)+float(residual[m*N+n]);expected[m*N+n]=value;
      int image=m/area,part=m/64-(image*area)/64;
      sums[(image*parts+part)*N+n]+=value;mean[image*N+n]+=value;
    }
    Compare(o.Get(),expected);
    if constexpr(SplitK>1)
      for(auto value:counters.Get())
        if(value!=0)throw std::runtime_error("split-K counter did not reset");
    auto actual=partial.Get();
    for(int i=0;i<16;++i)
      if(actual[i]!=canary || actual[actual.size()-16+i]!=canary)
        throw std::runtime_error("GAP wrote past the partial allocation");
    for(int image=0;image<B;++image) {
      int count=((image+1)*area-1)/64-(image*area)/64+1;
      for(int part=0;part<count;++part)for(int n=0;n<N;++n) {
        int index=(image*parts+part)*N+n;
        if(actual[16+index]!=sums[index])throw std::runtime_error("GAP partial differs from FP32 sum");
      }
    }
    Sm90F3SmallOperands small{};small.accumulator=partial.data+16;small.output=pooled.data;
    small.rows=area;small.parts=parts;small.channels=N;small.out_channels=N;
    Pool<B><<<132,384>>>(small);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    for(auto& value:mean)value/=area;Compare(pooled.Get(),mean);Hash(actual);
  }
  std::cout<<"pool_tm"<<TM<<"_b"<<B;
  if constexpr(SplitK>1)std::cout<<"_s"<<SplitK;
  std::cout<<" PASS\n";
}

template<int Window,int Stride,int Pad,int B> struct Conv {
  static constexpr int kW=9,kH=9,kC=8,kR=Window,kS=Window,
      kStrideW=Stride,kStrideH=Stride,kPadW=Pad,kPadH=Pad,kDilationW=1,kDilationH=1;
  static constexpr int kQ=(kW+2*Pad-Window)/Stride+1,kP=kQ;
};
template<int Window,int Stride,int Pad,int B> void GatherCase() {
  using C=Conv<Window,Stride,Pad,B>;
  constexpr int M=B*C::kQ*C::kP,N=64,K=((Window*Window*8+63)/64)*64;
  using Chain=Sm90F3Chain<Sm90F3SideLayout<128,64,false,1,false>,Sm90F3Step<Sm90ForwardOp::kBias,0>>;
  using G=Sm90F3Gemm<128,64,4,1,Sm90F3Mode::kGather,M,N,K,N,Chain,C>;
  std::vector<E> input(B*9*9*8),weight(N*K),packed(N*K);
  for(int i=0;i<int(input.size());++i)input[i]=E(i%8<3?float((i*17+1)%19-9)/32:0.f);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E v(k<Window*Window*8 && k%8<3?float((n*7+k*3)%13-6)/32:0.f);
    weight[n*K+k]=v;
    packed[(k/64)*N*64+PackedIndex(n,k%64)]=v;
  }
  Device<E> a(input.size()),w(packed.size()),o(M*N);Device<float> bias(N);
  a.Put(input);w.Put(packed);bias.Put(std::vector<float>(N,.5f));
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;p.epilogue.output=o.data;
  p.epilogue.output_stride=N;p.epilogue.vectors[0]=bias.data;
  CUtensorMap unused{};Launch<G>(p,unused,unused);
  std::vector<float> expected(M*N);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
    int image=m/(C::kQ*C::kP),pixel=m%(C::kQ*C::kP);float dot=0;
    for(int rr=0;rr<Window;++rr)for(int ss=0;ss<Window;++ss) {
      int y=(pixel/C::kQ)*Stride-Pad+rr,x=(pixel%C::kQ)*Stride-Pad+ss;
      if(y<0 || y>=9 || x<0 || x>=9)continue;
      for(int c=0;c<3;++c)
        dot+=float(input[((image*9+y)*9+x)*8+c])*float(weight[n*K+(rr*Window+ss)*8+c]);
    }
    expected[m*N+n]=dot+.5f;
  }
  Compare(o.Get(),expected);std::cout<<"gather"<<Window<<"_b"<<B<<" PASS\n";
}

template<int H,int C,int Window,int Stride,int Pad>
struct ConvGeometry {
    enum {kW=H,kH=H,kC=C,kR=Window,kS=Window,kStrideW=Stride,kStrideH=Stride,
          kPadW=Pad,kPadH=Pad,kDilationW=1,kDilationH=1,
          kQ=(H+2*Pad-Window)/Stride+1,kP=kQ};
};
template<int B,int H,int C,int N,int Window,int Stride,int Pad,int TM,int TN,
         int Slots,int SplitK,bool Residual=false,bool Gather=false,bool Relu=true,int TK=64>
void ConvCase() {
  using Geometry=ConvGeometry<H,C,Window,Stride,Pad>;
  constexpr int P=Geometry::kP,M=B*P*P,K=((Window*Window*C+63)/64)*64;
  constexpr int input_channels=Gather?3:C;
  using Bias=Sm90F3Step<Sm90ForwardOp::kBias,0>;
  using Activation=Sm90F3Step<Sm90ForwardOp::kActivation,0,tilemega::codegen::DmActivation::kRelu>;
  using Plain=std::conditional_t<Relu,
      Sm90F3Chain<Sm90F3SideLayout<TM,TN,false,1,false>,Bias,Activation>,
      Sm90F3Chain<Sm90F3SideLayout<TM,TN,false,1,false>,Bias>>;
  using Chain=std::conditional_t<Residual,
      Sm90F3Chain<Sm90F3SideLayout<TM,TN,true,1,false>,Bias,
          Sm90F3Step<Sm90ForwardOp::kResidual,~0u>,Activation>,
      Plain>;
  using G=Sm90F3Gemm<TM,TN,Slots,SplitK,Gather?Sm90F3Mode::kGather:Sm90F3Mode::kNormal,
      M,N,K,N,Chain,Geometry,Sm90F3NoPool,TK>;
  std::vector<E> input(std::size_t(B)*H*H*C),weight(std::size_t(N)*K),packed(weight.size());
  for(std::size_t i=0;i<input.size();++i)
    input[i]=E(i%C<input_channels?float((i*17+1)%19)/32-9.f/32:0.f);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E value(k<Window*Window*C && k%C<input_channels?float((n*7+k*3)%13-6)/32:0.f);
    weight[std::size_t(n)*K+k]=value;
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=value;
  }
  Device<E> a(input.size()),w(packed.size()),o(std::size_t(M)*N),r(Residual?std::size_t(M)*N:1);
  Device<float> bias(N),partial(SplitK>1?std::size_t((M+TM-1)/TM)*(N/TN)*SplitK*TM*TN:1);
  Device<unsigned long long> counters(((M+TM-1)/TM)*(N/TN));
  a.Put(input);w.Put(packed);bias.Put(std::vector<float>(N,.5f));
  counters.Put(std::vector<unsigned long long>(counters.size,0));
  std::vector<E> residual(r.size);
  for(std::size_t i=0;i<residual.size();++i)residual[i]=E(float(int(i%7)-3)/16);
  if constexpr(Residual)r.Put(residual);
  ex::TensorMap activation{};CUtensorMap residual_map{};
  if constexpr(!Gather) {
    tilemega::codegen::ConvDesc conv{};
    conv.n=B;conv.h=H;conv.w=H;conv.c=C;conv.k=N;conv.r=Window;conv.s=Window;
    conv.stride_h=Stride;conv.stride_w=Stride;conv.pad_h=Pad;conv.pad_w=Pad;conv.p=P;conv.q=P;
    tilemega::codegen::DmBufferLayout layout{};
    layout.kind=tilemega::codegen::DmLayout::kNHWC;layout.rank=4;
    int dims[4]={B,H,H,C};
    for(int i=0;i<4;++i)layout.logical[i]=layout.physical[i]=dims[i];
    layout.strides[3]=1;layout.strides[2]=C;layout.strides[1]=H*C;layout.strides[0]=H*H*C;
    if(ex::EncodeTensorIm2col(activation,a.data,ex::TensorIm2colShape::Build(conv,layout))!=CUDA_SUCCESS)
      throw std::runtime_error("im2col descriptor encoding failed");
  }
  if constexpr(Residual)ex::EncodeTiledMap<2>(residual_map,r.data,{{N,M},{N*2},{64,TM}},true);
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;p.epilogue.output=o.data;
  p.epilogue.output_stride=N;p.epilogue.vectors[0]=bias.data;p.epilogue.residual=r.data;
  p.epilogue.residual_stride=N;p.epilogue.partial=partial.data;p.epilogue.counters=counters.data;
  Launch<G>(p,activation.descriptor,residual_map);
  // The fixture's BF16 inputs repeat every 19 elements and weights every 13
  // columns. Cache identical FP32 dot products, including the padding mask,
  // so the full VGG B16 shape still uses a CPU oracle.
  std::unordered_map<std::uint64_t,std::array<float,13>> cache;
  std::vector<float> expected(std::size_t(M)*N);
  for(int m=0;m<M;++m) {
    int image=m/(P*P),pixel=m%(P*P),y=(pixel/P)*Stride-Pad,x=(pixel%P)*Stride-Pad;
    std::uint64_t mask=0;
    for(int rr=0;rr<Window;++rr)for(int ss=0;ss<Window;++ss)
      if(y+rr>=0 && y+rr<H && x+ss>=0 && x+ss<H)mask|=1ull<<(rr*Window+ss);
    long long base=((1LL*image*H+y)*H+x)*C;
    auto key=(mask<<5)|std::uint64_t((base%19+19)%19);
    auto found=cache.find(key);
    if(found==cache.end()) {
      std::array<float,13> dots{};
      for(int n=0;n<13;++n)for(int rr=0;rr<Window;++rr)for(int ss=0;ss<Window;++ss) {
        if(!(mask&(1ull<<(rr*Window+ss))))continue;
        for(int c=0;c<input_channels;++c)
          dots[n]+=float(input[((std::size_t(image)*H+y+rr)*H+x+ss)*C+c])*
              float(weight[n*K+(rr*Window+ss)*C+c]);
      }
      found=cache.emplace(key,dots).first;
    }
    for(int n=0;n<N;++n) {
      auto index=std::size_t(m)*N+n;float value=found->second[n%13]+.5f;
      if constexpr(Residual)value+=float(residual[index]);
      expected[index]=Relu?std::max(0.f,value):value;
    }
  }
  Compare(o.Get(),expected);
  for(auto value:counters.Get())if(value)throw std::runtime_error("convolution split-K counter did not reset");
  std::cout<<"conv_b"<<B<<"_h"<<H<<"_c"<<C<<"_n"<<N<<"_r"<<Window<<"_stride"<<Stride<<"_s"<<SplitK<<" PASS\n";
}

template<int M,int NStore,int K,int SplitK,int Kind,int TK=64,int Slots=6> void SwapCase() {
  constexpr int N=(NStore+127)/128*128,Rows=M<=8?8:16;
  constexpr int RowStride=Kind==2?128:1,Vectors=Kind==2?0:1;
  using Bias=Sm90F3Step<Sm90ForwardOp::kBias,0>;
  using Relu=Sm90F3Step<Sm90ForwardOp::kActivation,0,tilemega::codegen::DmActivation::kRelu>;
  using Tanh=Sm90F3Step<Sm90ForwardOp::kActivation,0,tilemega::codegen::DmActivation::kTanh>;
  using Side=Sm90F3SideLayout<Rows,128,false,Vectors,false>;
  using Chain=std::conditional_t<Kind==2,Sm90F3Chain<Side,Tanh>,
      std::conditional_t<Kind==1,Sm90F3Chain<Side,Bias,Relu>,Sm90F3Chain<Side,Bias>>>;
  using G=Sm90F3GemmSwapAB<Slots,SplitK,M,N,K,NStore,Chain,TK>;
  std::vector<E> input(std::size_t(M)*RowStride*K),weight(std::size_t(N)*K),packed(weight.size());
  for(std::size_t i=0;i<input.size();++i)input[i]=E(float(int(i%19)-9)/32);
  for(int n=0;n<NStore;++n)for(int k=0;k<K;++k) {
    E value(float((n%13+k*7)%13-6)/32);weight[std::size_t(n)*K+k]=value;
    packed[(std::size_t(n/128)*(K/64)+k/64)*128*64+PackedIndex(n%128,k%64)]=value;
  }
  Device<E> a(input.size()),w(packed.size()),o(M*NStore);
  Device<float> bias(N),partial(G::kTiles*SplitK*Rows*128);
  Device<unsigned long long> counters(G::kTiles);
  a.Put(input);w.Put(packed);bias.Put(std::vector<float>(N,.5f));
  counters.Put(std::vector<unsigned long long>(G::kTiles,0));
  CUtensorMap am{},unused{};
  ex::EncodeTiledMap<2>(am,a.data,{{K,M},{K*RowStride*2},{64,Rows}});
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;p.epilogue.output=o.data;
  p.epilogue.output_stride=NStore;p.epilogue.vectors[0]=bias.data;
  p.epilogue.partial=partial.data;p.epilogue.counters=counters.data;
  std::vector<float> expected(M*NStore),dots(M*13,0.f);
  for(int row=0;row<M;++row)for(int n=0;n<13;++n)for(int k=0;k<K;++k)
    dots[row*13+n]+=float(input[std::size_t(row)*RowStride*K+k])*float(weight[std::size_t(n)*K+k]);
  for(int row=0;row<M;++row)for(int n=0;n<NStore;++n) {
    float v=dots[row*13+n%13];
    if constexpr(Kind==2)v=std::tanh(v);
    else {v+=.5f;if constexpr(Kind==1)v=std::max(0.f,v);}
    expected[row*NStore+n]=v;
  }
  for(int repeat=0;repeat<3;++repeat) {
    Launch<G>(p,am,unused);Compare(o.Get(),expected);
    for(auto value:counters.Get())if(value)throw std::runtime_error("swap-AB split-K counter did not reset");
  }
  std::cout<<"swap_m"<<M<<"_n"<<NStore<<"_k"<<K<<"_s"<<SplitK<<"_kind"<<Kind<<" PASS\n";
}

template<class G>
__global__ __launch_bounds__(384,1) void SmallBody(Sm90F3SmallOperands p) {
  __shared__ Sm90F3Header header;Sm90F3Phase phase{};
  if(threadIdx.x<256)G::Consume(p,header.rings[0],nullptr,phase,&header,int(threadIdx.x)/128);
}
template<class G> void LaunchSmall(Sm90F3SmallOperands p) {
  SmallBody<G><<<132,384>>>(p);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
}
void CompareBits(std::vector<E> const& actual,std::vector<E> const& expected) {
  if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(E)))
    throw std::runtime_error("layout/maxpool output differs bitwise from CPU reference");
  Hash(actual);
}
template<int B> void LayoutCase() {
  constexpr int H=224,W=224,C=3,Pixels=H*W;
  std::vector<E> input(B*Pixels*C),expected(B*Pixels*8);
  for(std::size_t i=0;i<input.size();++i)input[i]=E(float(int(i%37)-18)/32);
  for(int b=0;b<B;++b)for(int p=0;p<Pixels;++p)for(int c=0;c<C;++c)
    expected[(b*Pixels+p)*8+c]=input[(b*C+c)*Pixels+p];
  Device<E> a(input.size()),o(expected.size());a.Put(input);
  Sm90F3SmallOperands p{};p.input=a.data;p.output=o.data;p.width=W;p.height=H;
  LaunchSmall<Sm90F3Layout<C,B>>(p);CompareBits(o.Get(),expected);
  std::cout<<"layout_b"<<B<<" PASS\n";
}
template<int B,int H,int C,int Window,int Stride,int Pad> void MaxPoolCase() {
  constexpr int P=(H+2*Pad-Window)/Stride+1;
  std::vector<E> input(std::size_t(B)*H*H*C),expected(std::size_t(B)*P*P*C);
  for(std::size_t i=0;i<input.size();++i)input[i]=E(float(int(i%37)-18)/32);
  for(int b=0;b<B;++b)for(int y=0;y<P;++y)for(int x=0;x<P;++x)for(int c=0;c<C;++c) {
    float best=-INFINITY;
    for(int r=0;r<Window;++r)for(int s=0;s<Window;++s) {
      int sy=y*Stride-Pad+r,sx=x*Stride-Pad+s;
      if(sy>=0 && sy<H && sx>=0 && sx<H)
        best=std::max(best,float(input[((std::size_t(b)*H+sy)*H+sx)*C+c]));
    }
    expected[((std::size_t(b)*P+y)*P+x)*C+c]=E(best);
  }
  Device<E> a(input.size()),o(expected.size());a.Put(input);
  Sm90F3SmallOperands p{};p.input=a.data;p.output=o.data;p.width=H;p.height=H;
  LaunchSmall<Sm90F3MaxPool<Window,Window,Stride,Pad,C,B>>(p);CompareBits(o.Get(),expected);
  std::cout<<"maxpool_b"<<B<<"_h"<<H<<"_r"<<Window<<" PASS\n";
}

int main(int argc,char** argv) try {
  std::string which=argc>1?argv[1]:"all";
  if(which!="pool" && which!="gather" && which!="split" && which!="conv" && which!="swap" && which!="small" && which!="all")
    throw std::invalid_argument("expected pool, gather, split, conv, swap, small or all");
  if(which=="pool" || which=="all") {
    PoolCase<64,1>();PoolCase<64,16>();PoolCase<128,1>();PoolCase<128,16>();
    PoolCase<256,1>();PoolCase<256,16>();
  }
  if(which=="gather" || which=="all") {
    GatherCase<7,2,3,1>();GatherCase<7,2,3,16>();GatherCase<3,1,1,1>();
    ConvCase<1,9,8,64,7,2,3,128,64,4,7,false,true,false>();
    ConvCase<1,9,8,64,3,1,1,64,32,4,2,false,true,false>();
  }
  if(which=="split" || which=="all") {
    PoolCase<64,1,8>();PoolCase<64,16,8>();PoolCase<128,1,8>();PoolCase<128,16,8>();
    PoolCase<256,1,8>();PoolCase<256,16,8>();
  }
  if(which=="conv" || which=="all") {
    ConvCase<1,56,64,128,1,2,0,128,64,3,1,true>();
    ConvCase<1,56,64,64,3,1,1,64,64,6,1,true>();
    ConvCase<1,7,512,512,3,1,1,64,64,6,8>();
    ConvCase<16,224,64,64,3,1,1,256,64,2,1>();
    ConvCase<16,56,64,256,1,1,0,128,256,2,1,false,false,false>();
    ConvCase<1,224,8,64,7,2,3,128,64,4,1,false,true>();
    ConvCase<16,224,8,64,7,2,3,128,64,4,1,false,true>();
    ConvCase<1,224,8,64,3,1,1,128,64,4,1,false,true>();
  }
  if(which=="swap" || which=="all") {
    SwapCase<1,1000,512,1,0>();SwapCase<16,1000,512,1,0>();
    SwapCase<1,1000,512,8,0>();SwapCase<16,1000,512,8,0>();
    SwapCase<1,4096,25088,1,1>();SwapCase<16,4096,25088,1,1>();
    SwapCase<1,4096,25088,8,1>();SwapCase<16,4096,25088,8,1>();
    // 768/64=12: six is the largest legal divisor below the table's S=8.
    SwapCase<1,768,768,1,2>();SwapCase<16,768,768,1,2>();
    SwapCase<1,768,768,6,2>();SwapCase<16,768,768,6,2>();
    SwapCase<1,768,768,12,2>();SwapCase<16,768,768,12,2>();
  }
  if(which=="small" || which=="all") {
    LayoutCase<1>();LayoutCase<16>();
    MaxPoolCase<1,112,64,3,2,1>();MaxPoolCase<16,112,64,3,2,1>();
    MaxPoolCase<1,224,64,2,2,0>();MaxPoolCase<16,224,64,2,2,0>();
  }
  std::cout<<"output_hash="<<std::hex<<hash<<"\n";return 0;
}catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
