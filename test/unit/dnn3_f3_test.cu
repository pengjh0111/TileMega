// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Backend/Sm90F3Small.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
__global__ __launch_bounds__(288,1) void Body(Sm90F3GemmOperands p,
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
  if(threadIdx.x>=256)G::Produce(p,h->rings[0],arena,side,phase,0);
  else G::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
  __syncthreads();
  if(threadIdx.x==0) {h->rings[0].Invalidate();h->rings[1].Invalidate();}
}
template<class G> void Launch(Sm90F3GemmOperands p,CUtensorMap a,CUtensorMap r) {
  Check(cudaFuncSetAttribute(Body<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  Body<G><<<132,288,Sm90F3Budget::kBudget>>>(p,a,r);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
}
template<int B> __global__ __launch_bounds__(288,1) void Pool(Sm90F3SmallOperands p) {
  __shared__ Sm90F3Header h;Sm90F3Phase phase{};
  if(threadIdx.x<256)Sm90F3GlobalPool<B>::Consume(p,h.rings[0],nullptr,phase,&h,int(threadIdx.x)/128);
}

template<int TM,int B> void PoolCase() {
  constexpr int area=49,M=B*area,N=128,K=64,TN=64,parts=2;
  using Chain=Sm90F3Chain<Sm90F3SideLayout<TM,TN,true,2,false>,
      Sm90F3Step<Sm90ForwardOp::kScale,0>,Sm90F3Step<Sm90ForwardOp::kBias,1>,
      Sm90F3Step<Sm90ForwardOp::kResidual,~0u>>;
  using G=Sm90F3Gemm<TM,TN,3,1,Sm90F3Mode::kNormal,M,N,K,N,Chain,
      Sm90F3NoConv,Sm90F3Pool<area,parts,N>>;
  std::vector<E> input(M*K),weight(N*K),packed(N*K),residual(M*N);
  for(int i=0;i<N*K;++i)weight[i]=E(float((i*7+3)%13-6)/32);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k)
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=weight[n*K+k];
  for(int i=0;i<M*N;++i)residual[i]=E(float((i*11)%7-3)/16);
  Device<E> a(M*K),w(N*K),r(M*N),o(M*N),pooled(B*N);
  Device<float> scale(TN*2),bias(TN*2),partial(B*parts*N+32);
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
    Pool<B><<<132,288>>>(small);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    for(auto& value:mean)value/=area;Compare(pooled.Get(),mean);Hash(actual);
  }
  std::cout<<"pool_tm"<<TM<<"_b"<<B<<" PASS\n";
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

int main(int argc,char** argv) try {
  std::string which=argc>1?argv[1]:"all";
  if(which!="pool" && which!="gather" && which!="all")throw std::invalid_argument("expected pool, gather or all");
  if(which=="pool" || which=="all") {
    PoolCase<64,1>();PoolCase<64,16>();PoolCase<128,1>();PoolCase<128,16>();
    PoolCase<256,1>();PoolCase<256,16>();
  }
  if(which=="gather" || which=="all") {
    GatherCase<7,2,3,1>();GatherCase<7,2,3,16>();GatherCase<3,1,1,1>();
  }
  std::cout<<"output_hash="<<std::hex<<hash<<"\n";return 0;
}catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
