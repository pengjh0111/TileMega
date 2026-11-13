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
#ifdef DNN4_PP_STORE_TEST
constexpr bool kPingPong=true;
#else
constexpr bool kPingPong=false;
#endif

#if defined(DNN4_FIXED_GRID_PROFILE2_TEST) || defined(DNN4_QKV_SLOTS7_B4_TEST) || defined(DNN4_FFN1_TMA128_TEST) || defined(DNN4_FFN2_TMA128_SPLIT2_B8_TEST) || defined(DNN4_FFN1_M64N128_B8_TEST)
constexpr int kProducerRegisters=72,kConsumerRegisters=208;
#elif defined(DNN4_FIXED_GRID_TEST) || defined(DNN4_QKV_SLOTS4_B2_TEST) || defined(DNN4_QKV_TMA192_TEST) || defined(DNN4_FFN1_TMA256_TEST) || defined(DNN4_FFN2_STORE_TEST) || defined(DNN4_FFN1_REG64_TEST) || defined(DNN4_FFN2_REG64_TEST) || defined(DNN4_FFN2_TK128_S2_B2_TEST)
constexpr int kProducerRegisters=72,kConsumerRegisters=216;
#else
constexpr int kProducerRegisters=kPingPong?72:56;
constexpr int kConsumerRegisters=kPingPong?216:224;
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
    const __grid_constant__ CUtensorMap activation,const __grid_constant__ CUtensorMap residual,
    const __grid_constant__ CUtensorMap output) {
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);
  if(threadIdx.x==0) {
    h->rings[0].Initialize();h->rings[1].Initialize();Sm90F3Async::Base::InitFence();
  }
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0) {h->rings[0].Prime();h->rings[1].Prime();}
  __syncthreads();
  p.activation=&activation;p.residual=&residual;p.output_map=&output;
  Sm90F3Phase phase{};
  auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+G::kSideOffset;
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<kProducerRegisters>();G::Produce(p,h->rings[0],arena,side,phase,0);
  }else {
    Sm90F3Async::RegAlloc<kConsumerRegisters>();G::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);
  }
  __syncthreads();
  if(threadIdx.x==0) {h->rings[0].Invalidate();h->rings[1].Invalidate();}
}
template<class G> void Launch(Sm90F3GemmOperands p,CUtensorMap a,CUtensorMap r,CUtensorMap out) {
  Check(cudaFuncSetAttribute(Body<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  Body<G><<<132,384,Sm90F3Budget::kBudget>>>(p,a,r,out);
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

template<int M,int N,int K,int TM,int TN,int SplitK,int Kind,int Parts=(Kind==1?N:K)/64,int Tk=64,
         int RequestedSlots=0,int RequestedStore=-1>
void DenseCase() {
  std::cout<<"START dense m="<<M<<" n="<<N<<" k="<<K<<" tm="<<TM<<"\n";
  constexpr int Rows=kPingPong?TM/2:TM;
  using Side=Sm90F3SideLayout<Rows,TN,Kind==1,Kind==1?3:2,true,Parts>;
  using Norm=Sm90F3Step<Sm90ForwardOp::kDeferredLayerNorm,0,
      tilemega::codegen::DmActivation::kRelu,1>;
  using Bias=Sm90F3Step<Sm90ForwardOp::kBias,2>;
  using Residual=Sm90F3Step<Sm90ForwardOp::kResidualLN,0,
      tilemega::codegen::DmActivation::kRelu,1>;
  using Gelu=Sm90F3Step<Sm90ForwardOp::kActivation,0,
      tilemega::codegen::DmActivation::kGeluErf>;
  using Plain=std::conditional_t<Kind==0,Sm90F3Chain<Side,Norm>,
      std::conditional_t<Kind==1,Sm90F3Chain<Side,Bias,Residual>,
                         Sm90F3Chain<Side,Norm,Gelu>>>;
  using Chain=std::conditional_t<Kind==1,Sm90F3WriteRowStats<Plain>,Plain>;
#if defined(DNN4_QKV_SLOTS7_B4_TEST)
  constexpr int Slots=7;
  constexpr bool kTmaStore=false;
#elif defined(DNN4_QKV_SLOTS4_B2_TEST)
  constexpr int Slots=4;
  constexpr bool kTmaStore=false;
#elif defined(DNN4_FFN2_TK128_S2_B2_TEST)
  constexpr int Slots=6;
  constexpr bool kTmaStore=false;
#elif defined(DNN4_FFN1_M64N128_B8_TEST)
  constexpr int Slots=4;
  constexpr bool kTmaStore=false;
#elif defined(DNN4_QKV_TMA192_TEST)
  constexpr int Slots=4;
  constexpr bool kTmaStore=true;
#elif defined(DNN4_FFN1_TMA256_TEST)
  constexpr int Slots=3;
  constexpr bool kTmaStore=true;
#elif defined(DNN4_FFN1_TMA128_TEST) || defined(DNN4_FFN2_TMA128_SPLIT2_B8_TEST)
  constexpr int Slots=5;
  constexpr bool kTmaStore=true;
#elif defined(DNN4_FFN1_REG64_TEST) || defined(DNN4_FFN2_REG64_TEST)
  constexpr int Slots=8;
  constexpr bool kTmaStore=false;
#elif defined(DNN4_FFN2_STORE_TEST)
  constexpr int Slots=4;
  constexpr bool kTmaStore=true;
#else
  constexpr int Slots=kPingPong?(Kind==1?4:5):2;
  constexpr bool kTmaStore=true;
#endif
  using G=Sm90F3Gemm<TM,TN,RequestedSlots?RequestedSlots:Slots,SplitK,Sm90F3Mode::kNormal,
      M,N,K,N,Chain,Sm90F3NoConv,Sm90F3NoPool,Tk,
      RequestedStore<0?kTmaStore:bool(RequestedStore),false,kPingPong>;
  constexpr int tiles=((M+Rows-1)/Rows)*(N/TN);
  std::vector<E> input(M*K),weight(N*K),packed(N*K),residual(M*N);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    E value(float((n*7+k*3)%13+1)/1024);
    weight[n*K+k]=value;
    packed[((n/TN)*(K/64)+k/64)*TN*64+PackedIndex(n%TN,k%64)]=value;
  }
  for(int m=0;m<M;++m)for(int n=0;n<N;++n)
    residual[m*N+n]=E(float((m*11+n*5)%19)/32);
  Device<E> a(input.size()),w(packed.size()),r(residual.size()),o(M*N);
  Device<float> v0(N),v1(N),v2(N),stats(2*M*Parts),row_partials(2*M*(N/64));
  Device<float> partial(SplitK*tiles*TM*TN);
  Device<unsigned long long> counters(tiles);
  w.Put(packed);r.Put(residual);
  v0.Put(std::vector<float>(N,.25f));v1.Put(std::vector<float>(N,.75f));v2.Put(std::vector<float>(N,.25f));
  counters.Put(std::vector<unsigned long long>(tiles,0));
  CUtensorMap am{},rm{},om{};
  ex::EncodeTiledMap<2>(am,a.data,{{K,M},{K*2},{64,Rows}});
  ex::EncodeTiledMap<2>(rm,r.data,{{N,M},{N*2},{64,Rows}});
  ex::EncodeTiledMap<2>(om,o.data,{{N,M},{N*2},{64,64}});
  Sm90F3GemmOperands p{};p.input=a.data;p.weight=w.data;
  auto& e=p.epilogue;e.output=o.data;e.output_stride=N;e.residual=r.data;e.residual_stride=N;
  e.vectors[0]=v0.data;e.vectors[1]=v1.data;e.vectors[2]=v2.data;e.stats=stats.data;
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
    // Isolated2025.2.1 does not mark TMA global writes initialized for host
    // API shadow checking. Poison every output; the full FP32 comparison
    // below detects any unwritten element independently of that limitation.
    o.Put(std::vector<E>(o.size,E(-123)));
    a.Put(input);stats.Put(chunks);Launch<G>(p,am,rm,om);
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


#ifdef DNN4_GELU_OUTLINE_TEST
__device__ __noinline__ float OriginalScalarGelu(float x) {
  return Sm90F3GeluErf(x);
}
__global__ void GeluQuartetAccuracy(unsigned* errors) {
  constexpr unsigned count=1u<<24,total=count+65536u;
  for(unsigned base=4*(blockIdx.x*blockDim.x+threadIdx.x);base<total;
      base+=4*gridDim.x*blockDim.x) {
    float input[4],values[4];bool finite[4];
    #pragma unroll
    for(unsigned j=0;j<4;++j) {
      unsigned i=base+j,bits=i-count;
      finite[j]=i<count || (bits&0x7f80u)!=0x7f80u;
      input[j]=i<count?-8.f+16.f*(float(i)/float(count-1)):
          (finite[j]?__uint_as_float(bits<<16):0.f);
      values[j]=input[j];
    }
#if TILEMEGA_F3_GELU_OUTLINE
    auto result=Sm90F3GeluErfFour(make_float4(values[0],values[1],values[2],values[3]));
    values[0]=result.x;values[1]=result.y;values[2]=result.z;values[3]=result.w;
#else
    #pragma unroll
    for(int j=0;j<4;++j)values[j]=OriginalScalarGelu(values[j]);
#endif
    #pragma unroll
    for(int j=0;j<4;++j)if(finite[j]) {
      float old=OriginalScalarGelu(input[j]);
      float reference=.5f*input[j]*(1.f+erff(input[j]*.70710678118654752f));
      float error=fabsf(values[j]-reference)/fmaxf(1.f,fabsf(input[j]));
      if(__float_as_uint(values[j])!=__float_as_uint(old))atomicAdd(errors,1u);
      if(!isfinite(values[j]) || !(error<=1e-6f))atomicAdd(errors+1,1u);
    }
  }
}
void CheckGeluQuartet() {
  Device<unsigned> counters(2);counters.Put(std::vector<unsigned>(2,0));
  GeluQuartetAccuracy<<<132,256>>>(counters.data);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  auto result=counters.Get();
  if(result[0] || result[1])throw std::runtime_error("GELU quartet exact/domain mismatch "+
      std::to_string(result[0])+"/"+std::to_string(result[1]));
  Hash(result);
}

#endif
int main(int,char**) try {
  std::cout<<std::unitbuf;
#ifdef DNN4_GELU_OUTLINE_TEST
#ifdef DNN4_GELU_OUTLINE_B8_TEST
  CheckGeluQuartet();
  DenseCase<1024,3072,768,128,96,1,2,12,64,7,0>();
  DenseCase<1025,3072,768,128,96,1,2,12,64,7,0>();
  DenseCase<1024,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<1025,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<256,3072,768,64,128,1,2,12,128,4,0>();
  DenseCase<257,3072,768,64,128,1,2,12,128,4,0>();
  DenseCase<1024,768,3072,128,64,1,1,12,64,8,1>();
  DenseCase<1025,2304,768,128,64,1,0,12,64,8,0>();
#else
  CheckGeluQuartet();
  DenseCase<2048,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<2049,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<2048,3072,768,128,96,1,2,12,64,7,0>();
  DenseCase<2049,3072,768,128,96,1,2,12,64,7,0>();
  DenseCase<256,3072,768,64,128,1,2,12,128,4,0>();
  DenseCase<257,3072,768,64,128,1,2,12,128,4,0>();
  DenseCase<2048,768,3072,128,128,1,1,12,64,5,1>();
  DenseCase<2049,2304,768,128,96,1,0,12,64,7,0>();
#endif
#elif defined(DNN4_PIPELINE_REDUCTION_TEST)
  DenseCase<256,2304,768,64,128,1,0,1,64,4,0>();
  DenseCase<257,2304,768,64,128,1,0,1,64,4,0>();
  DenseCase<256,2304,768,128,64,1,0,12,64,3,0>();
  DenseCase<513,2304,768,128,64,1,0,12,64,3,0>();
  DenseCase<256,768,768,64,64,1,1,1,128,6,0>();
  DenseCase<256,768,768,64,64,1,1,12,128,6,0>();
  DenseCase<257,768,768,64,64,1,1,12,128,6,0>();
  DenseCase<256,3072,768,128,64,1,2,12,64,8,0>();
  DenseCase<513,3072,768,128,64,1,2,12,64,8,0>();
  DenseCase<256,768,3072,64,64,2,1,12,128,6,0>();
  DenseCase<1025,768,3072,64,64,2,1,12,128,6,0>();
#elif defined(DNN4_FULL_TILES_TEST)
  // Exact dimensions and tails exercise both sides of bounds specialization.
  DenseCase<1024,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<513,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<1024,768,3072,128,128,1,1,12,64,5,1>();
  DenseCase<513,768,3072,128,128,1,1,12,64,5,1>();
  DenseCase<512,2304,768,128,96,1,0,12,64,7,0>();
  DenseCase<512,768,3072,64,64,2,1,12,128,6,0>();
  DenseCase<257,768,3072,64,64,2,1,12,128,6,0>();
#elif defined(DNN4_FIXED_GRID_TEST)
  // Odd tails and multiple tasks cover the fixed-grid ownership boundaries.
  DenseCase<1025,3072,768,128,128,1,2,12,64,5,1>();
  DenseCase<513,768,3072,128,128,1,1,12,64,5,1>();
  DenseCase<513,2304,768,256,96,1,0,1,64,5,0>();
  DenseCase<513,2304,768,128,96,1,0,12,64,7,0>();
  DenseCase<257,768,3072,64,64,2,1,12,128,6,0>();
#elif defined(DNN4_QKV_SLOTS7_B4_TEST)
  // Seven spanning slots wrap across the twelve K panels and an odd row tail.
  DenseCase<513,2304,768,128,96,1,0,12>();
#elif defined(DNN4_QKV_SLOTS4_B2_TEST)
  // The B2 NT12 QKV ring wraps three times; the final row is partial.
  DenseCase<257,2304,768,128,64,1,0,12>();
#elif defined(DNN4_FFN2_TK128_S2_B2_TEST)
  // The accepted B2 FFN2 geometry combines two K slices, NT12 residual
  // statistics and six TK128 spanning slots, including an odd final row.
  DenseCase<257,768,3072,64,64,2,1,12,128>();
#elif defined(DNN4_FFN1_M64N128_B8_TEST)
  // N-split WGs share the same64 normalized rows; several tasks per CTA
  // exercise the side lifetime with a short final row.
  DenseCase<1025,3072,768,64,128,1,2>();
#elif defined(DNN4_FFN2_TMA128_SPLIT2_B8_TEST)
  // The odd B8 tail combines last-arriver reduction, residual staging and
  // NT12 statistics; five slots wrap repeatedly within each split slice.
  DenseCase<1025,768,3072,128,128,2,1>();
#elif defined(DNN4_QKV_TMA192_TEST)
  // The first and repeated QKV stages consume NT1 and NT12, respectively.
  DenseCase<2049,2304,768,128,192,1,0,1>();
  DenseCase<2049,2304,768,128,192,1,0,12>();
#elif defined(DNN4_FFN1_TMA256_TEST)
  // Wide N accumulator, three spanning slots and the B16 odd-tail P1 GELU.
  DenseCase<2049,3072,768,128,256,1,2>();
#elif defined(DNN4_FFN1_TMA128_TEST)
#ifdef DNN4_FFN1_TMA128_B8_TEST
  // Two tasks per CTA reuse the side buffer across the B8 odd tail.
  DenseCase<1025,3072,768,128,128,1,2>();
#else
  // M513 includes the B4 tail with five spanning slots and NT12 P1 GELU.
  DenseCase<513,3072,768,128,128,1,2>();
#endif
#elif defined(DNN4_FFN2_REG64_TEST)
  // Four split slices, NT12 residual row statistics and eight-slot wrap.
  DenseCase<257,768,3072,128,64,4,1>();
#elif defined(DNN4_FFN1_REG64_TEST)
  // M257 exercises side and eight-slot wrap beyond the B2 M256 shape.
  DenseCase<257,3072,768,128,64,1,2>();
#elif defined(DNN4_FFN2_STORE_TEST)
  // Two m64 fragments, NT12 statistics, four ring slots and an odd row tail.
  DenseCase<2049,768,3072,256,64,1,1>();
#elif defined(DNN4_PP_STORE_TEST)
  // More than two tasks per CTA exercises overlap and staging reuse; the
  // odd tail checks the independent side and ring phase streams.
  DenseCase<6145,768,768,256,128,1,2>();
  DenseCase<6145,768,768,256,128,1,1>();
#else
  DenseCase<128,768,768,128,256,1,2>();
  DenseCase<128,768,768,128,256,1,1>();
  DenseCase<512,768,768,256,128,1,2>();
  DenseCase<128,768,768,128,192,1,2>();
  DenseCase<128,768,768,128,192,1,1>();
#endif
  std::cout<<"output_hash="<<std::hex<<hash<<'\n';return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
