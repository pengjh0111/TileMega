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


__device__ void StageGate(unsigned long long* events,int stage,int epoch,bool consumer) {
  if(consumer) {
    Sm90F3Async::NamedBarrier<2,256>();
    if(threadIdx.x==0) {
      Sm90F3Async::GateArrive(events+stage);
      Sm90F3Async::GateWait(events+stage,std::uint64_t(epoch)*gridDim.x);
    }
  }
  Sm90F3Async::NamedBarrier<1,384>();
}
template<class Mc,class Reg>
__global__ __launch_bounds__(384,1) void ClusterBody(Sm90F3GemmOperands p,unsigned long long* events,int epoch,
    const __grid_constant__ CUtensorMap activation) {
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);
  if(threadIdx.x==0) {
    h->rings[0].Initialize();h->rings[1].Initialize();Sm90F3Async::Base::InitFence();
  }
  __syncthreads();Sm90F3Async::Base::ClusterSync();
  if(threadIdx.x<256 && (threadIdx.x&31)==0) {
    h->rings[0].Prime();h->rings[1].Prime();
    h->rings[0].PrimeCluster();h->rings[1].PrimeCluster();
  }
  __syncthreads();Sm90F3Async::Base::ClusterSync();
  p.activation=&activation;
  Sm90F3Phase phase{};
  auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+Mc::kSideOffset;
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<56>();
    Mc::Produce(p,h->rings[0],arena,side,phase,0);
    StageGate(events,0,epoch,false);
    Reg::Produce(p,h->rings[0],arena,side,phase,0);
    StageGate(events,1,epoch,false);
    Mc::Produce(p,h->rings[0],arena,side,phase,0);
  }else {
    Sm90F3Async::RegAlloc<224>();
    Mc::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
    StageGate(events,0,epoch,true);
    Reg::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
    StageGate(events,1,epoch,true);
    Mc::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
  }
  __syncthreads();Sm90F3Async::Base::ClusterSync();
  if(threadIdx.x==0) {h->rings[0].Invalidate();h->rings[1].Invalidate();}
}

template<int TM,int TN,int Slots,int S,int TK,int MTiles> void MulticastCase() {
  constexpr int M=TM*MTiles-3,N=TN*2,K=TK*S*4;
  using Chain=Sm90F3Chain<Sm90F3SideLayout<TM,TN,false,0,false>>;
  using Mc=Sm90F3Gemm<TM,TN,Slots,S,Sm90F3Mode::kNormal,M,N,K,N,Chain,
      Sm90F3NoConv,Sm90F3NoPool,TK,false,true>;
  using Reg=Sm90F3Gemm<TM,TN,Slots,S,Sm90F3Mode::kNormal,M,N,K,N,Chain,
      Sm90F3NoConv,Sm90F3NoPool,TK>;
  static_assert(Sm90F3Budget::Fits(Mc::kLayoutBytes));
  std::vector<E> input(M*K),weight(N*K),packed(N*K);
  // Seven row patterns make the full BF16->FP32 CPU reference inexpensive.
  for(int m=0;m<M;++m)for(int k=0;k<K;++k)input[m*K+k]=E(float(((m%7)*3+k*5)%17+1)/128);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E v(float((n*5+k*3)%11+1)/128);weight[n*K+k]=v;
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=v;
  }
  Device<E> a(input.size()),w(packed.size()),out(M*N);
  Device<float> partial(std::size_t(Mc::kOutputTiles)*S*TM*TN);
  Device<unsigned long long> counters(Mc::kOutputTiles),events(2);
  events.Put(std::vector<unsigned long long>(2));
  a.Put(input);w.Put(packed);out.Put(std::vector<E>(M*N,E(-123)));
  partial.Put(std::vector<float>(partial.size));counters.Put(std::vector<unsigned long long>(counters.size));
  CUtensorMap map{};ex::EncodeTiledMap<2>(map,a.data,{{K,M},{K*2},{64,TM}});
  Sm90F3GemmOperands p{};p.weight=w.data;p.epilogue.output=out.data;
  p.epilogue.output_stride=N;p.epilogue.partial=partial.data;p.epilogue.counters=counters.data;
  auto kernel=ClusterBody<Mc,Reg>;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={2,1,1};
  cudaLaunchConfig_t config{};config.gridDim=dim3(132);config.blockDim=dim3(384);
  config.dynamicSmemBytes=Sm90F3Budget::kBudget;config.attrs=&attr;config.numAttrs=1;
  int clusters=0;Check(cudaOccupancyMaxActiveClusters(&clusters,kernel,&config));
  if(clusters<66)throw std::runtime_error("multicast residency below66 clusters");
  for(int i=0;i<5;++i) {
    Check(cudaLaunchKernelEx(&config,kernel,p,events.data,i+1,map));Check(cudaDeviceSynchronize());
  }
  std::vector<float> patterns(7*N),reference(M*N);
  for(int m=0;m<7;++m)for(int n=0;n<N;++n)
    for(int k=0;k<K;++k)patterns[m*N+n]+=float(input[m*K+k])*float(weight[n*K+k]);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n)reference[m*N+n]=patterns[(m%7)*N+n];
  Compare(out.Get(),reference);
  std::cout<<"multicast "<<TM<<'x'<<TN<<" slots="<<Slots<<" S="<<S<<" TK="<<TK
      <<" Mtiles="<<MTiles<<" clusters="<<clusters<<" PASS\n";
}
int main(int,char**) {
  try {
    MulticastCase<64,32,8,1,64,135>();
    MulticastCase<128,128,6,1,64,134>();
    MulticastCase<128,128,3,3,64,135>();
    MulticastCase<128,256,4,1,64,135>();
    MulticastCase<128,128,3,2,128,9>();
    MulticastCase<256,64,4,1,64,1>();
    std::cout<<"output_hash="<<std::hex<<hash<<"\n";return 0;
  }catch(std::exception const& error) {std::cerr<<error.what()<<"\n";return 1;}
}
