// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90F3Program.cuh>
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
#ifdef DNN4_PP_STORE_TEST
constexpr bool kPingPong=true;
#else
constexpr bool kPingPong=false;
#endif

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



namespace f3=tilemega::codegen::forward_f3;
template<unsigned Input,unsigned Weight,unsigned Output,unsigned Map,unsigned OutMap,
         bool Residual=false> struct Bind {
  static constexpr unsigned kInput=Input,kWeight=Weight,kOutput=Output,kMap=Map,kOutputMap=OutMap,
      kResidual=Residual?5:f3::kAbsent,kResidualMap=Residual?1:f3::kAbsent,
      kVector0=f3::kAbsent,kVector1=f3::kAbsent,kVector2=f3::kAbsent,kVector3=f3::kAbsent,
      kStats=f3::kAbsent,kRowStats=f3::kAbsent,kRowPartials=f3::kAbsent,kRowCounters=f3::kAbsent,
      kChannelPartial=f3::kAbsent,kPartial=f3::kAbsent,kCounters=f3::kAbsent;
};
template<int Stride> struct Shape {
  static constexpr int kOutputStride=Stride,kResidualStride=Stride,kNormWidth=1;
  static constexpr float kEpsilon=1e-5f;
};
template<int Count> struct CrossCtaCopy {
  using Kind=CrossCtaCopy;
  static constexpr bool kMatrix=false,kUsesRing=false,kCanPreissue=false;
  static constexpr int kSharedBytes=1024,kSideOffset=0;
  template<class B> __device__ static void Produce(B const&,Sm90F3Ring&,char*,char*,Sm90F3Phase&,int) {}
  template<class B> __device__ static void Consume(B const& b,Sm90F3Ring&,char*,char*,
      Sm90F3Phase&,Sm90F3Header*,int) {
    // Shift ownership by one CTA so the generic reader crosses writer CTAs.
    int c=(int(blockIdx.x)+1)%int(gridDim.x);
    for(int i=c*256+int(threadIdx.x);i<Count;i+=int(gridDim.x)*256)
      b.template Get<3>()[i]=b.template Get<2>()[i];
  }
};
template<int M,int N,int TM,int TN,bool Residual,bool Store>
using G0=Sm90F3Gemm<TM,TN,2,1,Sm90F3Mode::kNormal,M,N,64,N-3,
    std::conditional_t<Residual,
      Sm90F3Chain<Sm90F3SideLayout<kPingPong?TM/2:TM,TN,true,0,false>,Sm90F3Step<Sm90ForwardOp::kResidual,~0u>>,
      Sm90F3Chain<Sm90F3SideLayout<kPingPong?TM/2:TM,TN,false,0,false>>>,Sm90F3NoConv,Sm90F3NoPool,64,Store,false,kPingPong>;
template<int M,int N,int TM,int TN,bool Residual,bool Store> struct Sequence {
  using First=f3::GemmStage<G0<M,N,TM,TN,Residual,Store>,Bind<0,1,2,0,2,Residual>,Shape<N>>;
  using LastBody=Sm90F3Gemm<128,64,2,1,Sm90F3Mode::kNormal,M,64,N,64,
      Sm90F3Chain<Sm90F3SideLayout<128,64,false,0,false>>>;
  using Last=f3::GemmStage<LastBody,Bind<2,6,4,3,f3::kAbsent>,Shape<64>>;
  using Program=f3::Program<1,7,5,
      f3::Placement<First,Sm90F3Budget::Offset(0,First::kSharedBytes)>,
      f3::Placement<CrossCtaCopy<M*N>,Sm90F3Budget::kArena1>,
      f3::Placement<Last,Sm90F3Budget::Offset(2,Last::kSharedBytes)>>;
};
template<int TM,int TN,bool Residual> void StoreCase() {
  constexpr int M=TM*9+13,N=TN*2,K=64;
  std::vector<E> input(M*K),weight(N*K),packed(N*K),weight2(64*N),packed2(64*N),residual(M*N);
  for(int i=0;i<M*K;++i)input[i]=E(float(i%19+1)/64);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E v(float((n*7+k*3)%13+1)/128);weight[n*K+k]=v;
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=v;
  }
  for(int n=0;n<64;++n)for(int k=0;k<N;++k) {
    E v(float((n+k)%7+1)/256);weight2[n*N+k]=v;
    packed2[(k/64)*64*64+PackedIndex(n,k%64)]=v;
  }
  for(int i=0;i<M*N;++i)residual[i]=E(float(i%7+1)/64);
  Device<E> a(input.size()),w(packed.size()),x((M+2)*N),y(M*N),z((M+2)*64),r(residual.size()),w2(packed2.size());
  a.Put(input);w.Put(packed);r.Put(residual);w2.Put(packed2);
  std::vector<E*> pointers={a.data,w.data,x.data,y.data,z.data,r.data,w2.data};
  Device<E*> buffers(pointers.size());buffers.Put(pointers);
  using Reg=typename Sequence<M,N,TM,TN,Residual,false>::Program;
  using Tma=typename Sequence<M,N,TM,TN,Residual,true>::Program;
  typename Tma::Binding b{};b.buffers=buffers.data;
  ex::EncodeTiledMap<2>(b.maps[0],a.data,{{K,M},{K*2},{64,kPingPong?TM/2:TM}});
  ex::EncodeTiledMap<2>(b.maps[1],r.data,{{N,M},{N*2},{64,kPingPong?TM/2:TM}});
  ex::EncodeTiledMap<2>(b.maps[2],x.data,{{N-3,M},{N*2},{64,64}});
  ex::EncodeTiledMap<2>(b.maps[3],x.data,{{N-3,M},{N*2},{64,128}});
  ex::EncodeTiledMap<2>(b.maps[4],z.data,{{64,M},{128},{64,64}});
  Device<f3::ForwardCounter> events(3);
  auto clear=[&]() {events.Put(std::vector<f3::ForwardCounter>(3));x.Put(std::vector<E>(x.size,E(-7)));z.Put(std::vector<E>(z.size,E(-9)));};
  auto run=[&](auto program,int epoch) {
    using P=decltype(program);
    Check(cudaFuncSetAttribute(f3::sm90_forward_f3_kernel<P>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
    f3::sm90_forward_f3_kernel<P><<<4,384,Sm90F3Budget::kBudget>>>(b,events.data,epoch);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  };
  clear();for(int epoch=1;epoch<=200;++epoch)run(Reg{},epoch);
  auto xr=x.Get(),yr=y.Get(),zr=z.Get();
  std::vector<E> actual;std::vector<float> expected;
  for(int m=0;m<M;++m)for(int n=0;n<N-3;++n) {
    float dot=0;for(int k=0;k<K;++k)dot+=float(input[m*K+k])*float(weight[n*K+k]);
    expected.push_back(dot+(Residual?float(residual[m*N+n]):0.f));actual.push_back(xr[m*N+n]);
  }
  Compare(actual,expected);
  clear();
  for(int epoch=1;epoch<=200;++epoch) {
    run(Tma{},epoch);
    auto xx=x.Get(),yy=y.Get(),zz=z.Get();
    auto same=[&](char const* label,std::vector<E> const& got,std::vector<E> const& ref) {
      for(std::size_t i=0;i<got.size();++i)
        if(got[i].raw()!=ref[i].raw())throw std::runtime_error(std::string(label)+
            " differs epoch="+std::to_string(epoch)+" index="+std::to_string(i)+
            " actual="+std::to_string(float(got[i]))+" expected="+std::to_string(float(ref[i])));
    };
    same("X",xx,xr);same("generic Y",yy,yr);same("TMA Z",zz,zr);
  }
  Hash(xr);Hash(yr);Hash(zr);
  std::cout<<"store_tm"<<TM<<"_tn"<<TN<<"_residual"<<Residual<<" PASS200\n";
}
int main(int,char**) {
  try {
#ifdef DNN4_FFN2_STORE_TEST
    StoreCase<256,64,true>();
#elif defined(DNN4_PP_STORE_TEST)
    StoreCase<256,128,false>();StoreCase<256,128,true>();
#else
    StoreCase<128,64,false>();StoreCase<128,128,false>();
    StoreCase<128,64,true>();StoreCase<256,128,true>();
    StoreCase<128,192,false>();StoreCase<128,192,true>();
#endif
    std::cout<<"output_hash="<<std::hex<<hash<<"\n";return 0;
  }catch(std::exception const& error) {std::cerr<<error.what()<<"\n";return 1;}
}
