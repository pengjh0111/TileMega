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

#ifndef TILEMEGA_F3_REGISTER_BUDGET
#define TILEMEGA_F3_REGISTER_BUDGET 0
#endif

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
    Sm90F3Async::RegDealloc<(TILEMEGA_F3_REGISTER_BUDGET>=2?72:(TILEMEGA_F3_REGISTER_BUDGET?64:56))>();G::Produce(p,h->rings[0],arena,side,phase,0);
  }else {
    Sm90F3Async::RegAlloc<(TILEMEGA_F3_REGISTER_BUDGET==2?208:(TILEMEGA_F3_REGISTER_BUDGET?216:224))>();G::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
  }
  __syncthreads();
  if(threadIdx.x==0) {h->rings[0].Invalidate();h->rings[1].Invalidate();}
}
template<class G> void Launch(Sm90F3GemmOperands p,CUtensorMap a,CUtensorMap r) {
  Check(cudaFuncSetAttribute(Body<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  Body<G><<<132,384,Sm90F3Budget::kBudget>>>(p,a,r);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
}
template<class G>
__global__ __launch_bounds__(384,1) void SmallBody(Sm90F3SmallOperands p) {
  __shared__ Sm90F3Header header;Sm90F3Phase phase{};
  if(threadIdx.x<256)G::Consume(p,header.rings[0],nullptr,phase,&header,int(threadIdx.x)/128);
}
template<class G> void LaunchSmall(Sm90F3SmallOperands p) {
  SmallBody<G><<<132,384>>>(p);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
}

void CompareStats(std::vector<float> const& actual,std::vector<float> const& expected) {
  double dot=0,aa=0,rr=0;
  for(std::size_t i=0;i<expected.size();++i) {
    if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>
        1e-3f*std::max(1.f,std::abs(expected[i])))
      throw std::runtime_error("FP32 row statistics differ at "+std::to_string(i));
    dot+=double(actual[i])*expected[i];aa+=double(actual[i])*actual[i];
    rr+=double(expected[i])*expected[i];
  }
  if(dot/std::sqrt(aa*rr)<.99999)throw std::runtime_error("row statistics cosine");
  Hash(actual);
}

template<int M,int N,int K,int TM,int TN,int SplitK,int Kind,
         int Parts=(Kind==1?N:K)/64,int Slots=2,int TK=64>
void DenseCase() {
  std::cout<<"START dense m="<<M<<" n="<<N<<" k="<<K<<" tm="<<TM<<"\n";
  using Side=Sm90F3SideLayout<TM,TN,Kind==1,2,true,Parts>;
  using Norm=Sm90F3Step<Sm90ForwardOp::kDeferredLayerNorm,0,
      tilemega::codegen::DmActivation::kRelu,1>;
  using Bias=Sm90F3Step<Sm90ForwardOp::kBias,0>;
  using Residual=Sm90F3Step<Sm90ForwardOp::kResidualLN,0,
      tilemega::codegen::DmActivation::kRelu,1>;
  using Gelu=Sm90F3Step<Sm90ForwardOp::kActivation,0,
      tilemega::codegen::DmActivation::kGeluErf>;
  using Plain=std::conditional_t<Kind==0,Sm90F3Chain<Side,Norm>,
      std::conditional_t<Kind==1,Sm90F3Chain<Side,Bias,Residual>,
                         Sm90F3Chain<Side,Norm,Gelu>>>;
  using Chain=std::conditional_t<Kind==1,Sm90F3WriteRowStats<Plain>,Plain>;
  using G=Sm90F3Gemm<TM,TN,Slots,SplitK,Sm90F3Mode::kNormal,M,N,K,N,Chain,
                    Sm90F3NoConv,Sm90F3NoPool,TK>;
  constexpr int tiles=((M+TM-1)/TM)*(N/TN),mt=(M+TM-1)/TM;
  std::vector<E> input(M*K),weight(N*K),packed(N*K),residual(M*N);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E value(float((n*7+k*3)%13+1)/1024);
    weight[n*K+k]=value;
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=value;
  }
  for(int m=0;m<M;++m)for(int n=0;n<N;++n)
    residual[m*N+n]=E(float((m*11+n*5)%19)/32);
  Device<E> a(input.size()),w(packed.size()),r(residual.size()),o(M*N);
  Device<float> v0(N),v1(N),stats(2*M*Parts),row_partials(2*M*(N/64));
  Device<float> partial(SplitK*tiles*TM*TN);
  Device<unsigned long long> counters(tiles);
  w.Put(packed);r.Put(residual);
  v0.Put(std::vector<float>(N,.25f));v1.Put(std::vector<float>(N,.75f));
  counters.Put(std::vector<unsigned long long>(tiles,0));
  CUtensorMap am{},rm{};
  ex::EncodeTiledMap<2>(am,a.data,{{K,M},{K*2},{64,TM}});
  ex::EncodeTiledMap<2>(rm,r.data,{{N,M},{N*2},{64,TM}});
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;
  auto& e=p.epilogue;e.output=o.data;e.output_stride=N;e.residual=r.data;e.residual_stride=N;
  e.vectors[0]=v0.data;e.vectors[1]=v1.data;e.stats=stats.data;
  e.norm_width=Kind==1?N:K;e.epsilon=1e-5f;
  e.partial=partial.data;e.counters=counters.data;
  if constexpr(Kind==1) {
    e.row_partials=row_partials.data;
  }
  for(int repeat=0;repeat<3;++repeat) {
    for(int m=0;m<M;++m)for(int k=0;k<K;++k)
      input[m*K+k]=E(float((m*11+k*17+repeat)%19)/32);
    std::vector<float> sums(2*M,0),chunks(2*M*Parts,0),expected(M*N),expected_stats(2*M,0);
    for(int m=0;m<M;++m)for(int k=0;k<(Kind==1?N:K);++k) {
      float value=float(Kind==1?residual[m*N+k]:input[m*K+k]);
      sums[2*m]+=value;sums[2*m+1]+=value*value;
      int part=k*Parts/(Kind==1?N:K);
      chunks[(part*M+m)*2]+=value;chunks[(part*M+m)*2+1]+=value*value;
    }
    a.Put(input);stats.Put(chunks);Launch<G>(p,am,rm);
    std::array<std::array<float,13>,19> cache{};
    for(int m=0;m<19;++m)for(int n=0;n<13;++n)for(int k=0;k<K;++k)
      cache[m][n]+=float(input[m*K+k])*float(weight[n*K+k]);
    for(int m=0;m<M;++m) {
      float mean=sums[2*m]/e.norm_width;
      float inverse=1.f/std::sqrt(sums[2*m+1]/e.norm_width-mean*mean+e.epsilon);
      for(int n=0;n<N;++n) {
        float dot=cache[m%19][n%13],value;
        if constexpr(Kind==1)value=(dot+.25f)+
            (float(residual[m*N+n])-mean)*inverse*.25f+.75f;
        else value=(dot-mean*.25f)*inverse+.75f;
        if constexpr(Kind==2)value=.5f*value*(1.f+std::erf(value*.7071067811865475f));
        expected[m*N+n]=value;
        expected_stats[2*m]+=value;expected_stats[2*m+1]+=value*value;
      }
    }
    Compare(o.Get(),expected);
    if constexpr(Kind==1) {
      auto raw=row_partials.Get();std::vector<float> reduced(2*M,0);
      for(int part=0;part<N/64;++part)for(int row=0;row<M;++row) {
        reduced[2*row]+=raw[(part*M+row)*2];
        reduced[2*row+1]+=raw[(part*M+row)*2+1];
      }
      CompareStats(reduced,expected_stats);
    }
    for(auto value:counters.Get())if(value)throw std::runtime_error("dense split counter reset");
  }
  std::cout<<"dense_m"<<M<<"_n"<<N<<"_k"<<K<<"_tm"<<TM<<"_s"<<SplitK<<" PASS\n";
}

template<int B> void SmallBertCase() {
  constexpr int C=768,Rows=B*128,Words=37;
  std::vector<E> words(Words*C),types(2*C),positions(128*C),gamma(C),beta(C);
  for(std::size_t i=0;i<words.size();++i)words[i]=E(float(int(i%19)-9)/32);
  for(std::size_t i=0;i<types.size();++i)types[i]=E(float(int(i%7)-3)/32);
  for(std::size_t i=0;i<positions.size();++i)positions[i]=E(float(int(i%13)-6)/32);
  for(int c=0;c<C;++c){gamma[c]=E(.125f);beta[c]=E(.75f);}
  std::vector<long long> ids(Rows),type_ids(Rows);
  for(int row=0;row<Rows;++row){ids[row]=(row*17)%Words;type_ids[row]=row%2;}
  Device<E> w(words.size()),t(types.size()),pos(positions.size()),g(C),b(C),o(Rows*C),ln(Rows*C);
  Device<long long> i0(Rows),i1(Rows);Device<float> stats(Rows*2);
  w.Put(words);t.Put(types);pos.Put(positions);g.Put(gamma);b.Put(beta);i0.Put(ids);i1.Put(type_ids);
  Sm90F3SmallOperands p{};p.input=w.data;p.weight=t.data;p.table=pos.data;
  p.index0=i0.data;p.index1=i1.data;p.stats=stats.data;p.output=o.data;
  p.height=128;p.rows=Rows;
  LaunchSmall<Sm90F3EmbeddingSum<C>>(p);
  std::vector<float> expected(Rows*C),sums(Rows*2,0),normalized(Rows*C);
  for(int row=0;row<Rows;++row)for(int c=0;c<C;++c) {
    float value=float(words[ids[row]*C+c])+float(types[type_ids[row]*C+c])+float(positions[(row%128)*C+c]);
    expected[row*C+c]=value;sums[2*row]+=value;sums[2*row+1]+=value*value;
  }
  Compare(o.Get(),expected);CompareStats(stats.Get(),sums);
  auto embedding=o.Get();p.input=o.data;p.weight=g.data;p.table=b.data;p.output=ln.data;p.epsilon=1e-5f;
  LaunchSmall<Sm90F3LayerNorm<C,C>>(p);
  for(int row=0;row<Rows;++row) {
    float sum=0,square=0;
    for(int c=0;c<C;++c){float v=float(embedding[row*C+c]);sum+=v;square+=v*v;}
    float mean=sum/C,inverse=1.f/std::sqrt(square/C-mean*mean+p.epsilon);
    for(int c=0;c<C;++c)normalized[row*C+c]=(float(embedding[row*C+c])-mean)*inverse*.125f+.75f;
  }
  Compare(ln.Get(),normalized);std::cout<<"embedding_ln_b"<<B<<" PASS\n";
}

int main(int,char**) try {
  std::cout<<std::unitbuf;
#if TILEMEGA_F3_REGISTER_BUDGET >= 2
  DenseCase<1024,2304,768,256,96,1,0,1>();
  DenseCase<1024,2304,768,128,192,1,0,12>();
  DenseCase<1024,3072,768,128,96,1,2,12>();
  DenseCase<1024,768,3072,128,64,1,1,12>();
  DenseCase<1024,768,768,128,64,1,1,12>();
  DenseCase<2048,2304,768,128,96,1,0,12>();
  DenseCase<2048,3072,768,128,128,1,2,12>();
  DenseCase<2048,768,3072,128,128,1,1,12>();
  DenseCase<2048,768,768,128,128,1,1,12>();
#else
  DenseCase<128,2304,768,128,64,3,0>();
  DenseCase<2048,2304,768,128,128,1,0>();
  DenseCase<128,768,768,128,64,6,1>();
  DenseCase<2048,768,768,128,64,1,1>();
  DenseCase<128,768,3072,128,64,4,1>();
  DenseCase<2048,768,3072,128,64,1,1>();
  DenseCase<2048,3072,768,128,128,1,2>();
  DenseCase<2048,3072,768,128,128,1,2,12,3,128>();
  DenseCase<2048,3072,768,256,128,1,2,12,4>();
  DenseCase<128,768,768,64,128,1,1>();
  DenseCase<128,768,768,64,64,1,1,1>();
  DenseCase<131,768,256,64,64,1,1>();
  DenseCase<259,768,256,128,64,2,1>();
  DenseCase<128,768,768,64,96,1,0>();
  DenseCase<128,768,768,64,192,1,0>();
  DenseCase<128,768,768,128,96,1,2>();
  DenseCase<128,768,768,128,192,1,1>();
  DenseCase<512,768,768,256,96,1,0>();
  SmallBertCase<1>();SmallBertCase<16>();
#endif
  std::cout<<"output_hash="<<std::hex<<hash<<'\n';return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
