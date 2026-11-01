// SPDX-License-Identifier: BSD-3-Clause
// Tensor parallel device code on T real devices (SM90R5 B8.1, B8.2).
//
// AR: the real o/down GEMM body with the kResidualAllReduce epilogue, on both
// epilogue paths (register WK=1 and shared WK>1), several repeats with
// increasing epochs. Every operand is dyadic so each rank's fp32 partial is
// exact; the host reproduces appendix C.2's rounding order (bf16 per rank,
// fp32 sum in rank order, bf16, + residual, bf16) and must match bit for bit
// on every rank. Columns cover zero partials, bf16 round-to-even ties and
// residuals of +-0, the bf16 maximum and the smallest normal.
//
// Head: Sm90DenseHeadFinish<..., Tp, VocabLocal> on a 4-CTA grid per rank, fed
// per-CTA (value, local index) candidates drawn from a tiny value set so ties
// within a rank and across ranks are frequent. The token must be the global
// rule's (larger value, then smaller global index) on every rank.
//
// Usage: CUDA_VISIBLE_DEVICES=<T cards> sm90_dense_tp_test; T = visible devices.
#define TILEMEGA_SWIGLU_U 8
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseHead.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
constexpr int kGrid=4,kRepeats=3,kCapacity=1088;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}

__device__ void GridBarrier(unsigned long long* counter,unsigned long long round) {
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    unsigned long long v;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<(round+1)*gridDim.x);
  }
  __syncthreads();
}

// Host bf16 round-to-nearest-even (finite inputs).
float Bf16(float x) {
  std::uint32_t u;std::memcpy(&u,&x,4);
  u+=0x7fffu+((u>>16)&1u);u&=0xffff0000u;
  float r;std::memcpy(&r,&u,4);return r;
}

template<class G>
__global__ __launch_bounds__(160) void ArKernel(Sm90DenseOperands p,Sm90DenseTpOperands tp,
    unsigned per_region,unsigned long long* counter) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  for(int r=0;r<kRepeats;++r) {
    // A fresh region per repeat keeps the test independent of the step-level
    // reuse argument (plan §3.5 C2), which the end-to-end litmus covers.
    tp.region=unsigned(r)*per_region;tp.epoch=unsigned(r+1);
    G::template Run<0>(p,shared,nullptr,Sm90DenseNoTail{},true,Sm90DenseNoFinish{},&tp);
    GridBarrier(counter,r);
  }
  state.Invalidate();
}

template<int T,int B,int TN,int WN>
int ArCase(std::vector<int> const& devices) {
  constexpr int WK=4/WN,K=512,TK=128,N=TN*2*kGrid;
  using G=Sm90DenseGemm<B,N,K,TN,TK,2,WN,WK,ServingEpilogueOp::kResidualAllReduce,8,false>;
  static_assert(G::kRegisterEpilogue==(WK==1));
  constexpr int rows=Sm90DenseLayout::ActivationRows(B);
  std::vector<std::vector<float>> a(T,std::vector<float>(B*K)),w(T,std::vector<float>(std::size_t(N)*K));
  std::vector<float> residual(B*N);
  for(int q=0;q<T;++q) {
    for(int b=0;b<B;++b)for(int k=0;k<K;++k)
      a[q][b*K+k]=k<2?1.0f:float((b+1+q)*(k%5+1)-7)/64.0f;
    for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
      float v=float((n%13-6)*(k%7+1)+q)/256.0f;
      if(n%TN==3)v=0;                                   // zero partial
      if(n%TN==5)v=k==0?1.0f:k==1?1.0f/256:0.0f;        // 1 + 2^-8: a bf16 tie
      w[q][std::size_t(n)*K+k]=v;
    }
  }
  float const special[]={0.0f,-0.0f,65280.0f,-65280.0f,1.17549435e-38f,0.5f,-1.0f};
  for(int i=0;i<B*N;++i)residual[i]=i%9<7?special[i%9]:float((i%17)-8)/32.0f;
  // Reference x'' per (b, n), identical for every rank.
  std::vector<float> expected(B*N);
  for(int b=0;b<B;++b)for(int n=0;n<N;++n) {
    float s=0;
    for(int q=0;q<T;++q) {
      double exact=0;for(int k=0;k<K;++k)exact+=double(a[q][b*K+k])*w[q][std::size_t(n)*K+k];
      s+=Bf16(float(exact));
    }
    expected[b*N+n]=Bf16(Bf16(s)+residual[b*N+n]);
  }
  unsigned per_region=unsigned(T*B*(N/2));
  std::vector<std::uint64_t*> ar(T);std::vector<BF*> act(T),weight(T),res(T),out(T);
  std::vector<unsigned long long*> counter(T);std::vector<CUtensorMap*> map(T);
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(devices[q]));
    std::vector<BF> ha(B*K),hw(std::size_t(N)*K),hr(B*N);
    for(int i=0;i<B*K;++i)ha[i]=BF(a[q][i]);
    for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
      int row=n%TN,col=k%TK;
      std::size_t permutation=512*(row/8+(TN/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
      hw[(std::size_t(n/TN)*(K/TK)+k/TK)*TN*TK+permutation]=BF(w[q][std::size_t(n)*K+k]);
    }
    for(int i=0;i<B*N;++i)hr[i]=BF(residual[i]);
    check(cudaMalloc(&ar[q],std::size_t(kRepeats)*per_region*8));check(cudaMemset(ar[q],0,std::size_t(kRepeats)*per_region*8));
    check(cudaMalloc(&act[q],ha.size()*2));check(cudaMalloc(&weight[q],hw.size()*2));
    check(cudaMalloc(&res[q],hr.size()*2));check(cudaMalloc(&out[q],hr.size()*2));
    check(cudaMalloc(&counter[q],8));check(cudaMemset(counter[q],0,8));check(cudaMalloc(&map[q],128));
    check(cudaMemcpy(act[q],ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(weight[q],hw.data(),hw.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(res[q],hr.data(),hr.size()*2,cudaMemcpyHostToDevice));
    CUtensorMap host;
    tilemega::codegen::executor::EncodeTiledMap<3>(host,act[q],
        {{64,std::uint64_t(B),K/64},{2ull*K,128},{64,std::uint32_t(rows),TK/64}});
    check(cudaMemcpy(map[q],&host,sizeof host,cudaMemcpyHostToDevice));
    check(cudaFuncSetAttribute(ArKernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::kSharedBytes));
  }
  // All kernels are enqueued before any is waited on: each blocks in its
  // first all-reduce until every rank's has started.
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(devices[q]));
    Sm90DenseOperands p{map[q],weight[q],res[q],out[q],nullptr,0.f};
    Sm90DenseTpOperands tp{};
    for(int r=0;r<T;++r)tp.recv[r]=ar[r];
    tp.rank=q;tp.world=T;
    ArKernel<G><<<kGrid,160,G::kSharedBytes>>>(p,tp,per_region,counter[q]);
    check(cudaGetLastError());
  }
  int errors=0;
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(devices[q]));check(cudaDeviceSynchronize());
    std::vector<BF> got(B*N);check(cudaMemcpy(got.data(),out[q],got.size()*2,cudaMemcpyDeviceToHost));
    for(int i=0;i<B*N;++i) {
      float g=float(got[i]);std::uint32_t gb,eb;std::memcpy(&gb,&g,4);std::memcpy(&eb,&expected[i],4);
      if(gb!=eb) {
        if(errors<4)std::fprintf(stderr,"AR T%d B%d TN%d WK%d rank%d b%d n%d got %a expected %a\n",
            T,B,TN,WK,q,i/N,i%N,g,expected[i]);
        ++errors;
      }
    }
    for(void* ptr:{(void*)ar[q],(void*)act[q],(void*)weight[q],(void*)res[q],(void*)out[q],
                   (void*)counter[q],(void*)map[q]})check(cudaFree(ptr));
  }
  std::fprintf(stderr,"ar T%d B%d TN%d WK%d errors %d\n",T,B,TN,WK,errors);
  return errors;
}

template<class Finish,int B>
__global__ __launch_bounds__(128) void HeadKernel(Sm90DenseOperands p,Sm90DenseHeadOperands head,
    Sm90DenseTpHeadOperands tp,float const* values,int const* indices,unsigned long long* counter,
    int repeats) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  __shared__ float best[16];__shared__ int arg[16];
  for(int r=0;r<repeats;++r) {
    if(threadIdx.x<B) {
      best[threadIdx.x]=values[(std::size_t(r)*gridDim.x+blockIdx.x)*B+threadIdx.x];
      arg[threadIdx.x]=indices[(std::size_t(r)*gridDim.x+blockIdx.x)*B+threadIdx.x];
    }
    __syncthreads();
    head.past=r;tp.epoch=std::uint64_t(r)+1;
    Finish{&head,&tp}(p,state,best,arg,shared+1024,4096);
    GridBarrier(counter,r);
  }
  state.Invalidate();
}

template<int T,int B>
int HeadCase(std::vector<int> const& devices) {
  constexpr int VocabLocal=37984,repeats=64,Width=64;
  using Finish=Sm90DenseHeadFinish<B,Width,kCapacity,false,T,VocabLocal>;
  // Candidates: values from {2, 1, 0.5, +0, -0} with indices clustered in a
  // few local slots, so equal values on two ranks or two CTAs are common.
  std::vector<std::vector<float>> values(T,std::vector<float>(std::size_t(repeats)*kGrid*B));
  std::vector<std::vector<int>> indices(T,std::vector<int>(values[0].size()));
  std::uint64_t seed=0x1234567ull;
  auto next=[&] {seed=seed*6364136223846793005ull+1442695040888963407ull;return unsigned(seed>>33);};
  float const set[]={2.0f,1.0f,0.5f,0.0f,-0.0f};
  for(int q=0;q<T;++q)for(std::size_t i=0;i<values[q].size();++i) {
    values[q][i]=set[next()%5];indices[q][i]=int(next()%4)*(VocabLocal/4)+int(next()%3);
  }
  int errors=0;
  std::vector<Sm90DenseTpExchange*> exchange(T);std::vector<int*> tokens(T),arg(T);
  std::vector<float*> partials(T),dv(T);std::vector<int*> di(T);std::vector<std::uint64_t*> lm(T);
  std::vector<unsigned long long*> counter(T);
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(devices[q]));
    check(cudaMalloc(&exchange[q],sizeof(Sm90DenseTpExchange)));check(cudaMemset(exchange[q],0,sizeof(Sm90DenseTpExchange)));
    check(cudaMalloc(&tokens[q],B*kCapacity*4));check(cudaMemset(tokens[q],0xff,B*kCapacity*4));
    check(cudaMalloc(&partials[q],B*kGrid*4));check(cudaMalloc(&arg[q],B*kGrid*4));
    check(cudaMalloc(&dv[q],values[q].size()*4));check(cudaMalloc(&di[q],values[q].size()*4));
    check(cudaMemcpy(dv[q],values[q].data(),values[q].size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(di[q],indices[q].data(),indices[q].size()*4,cudaMemcpyHostToDevice));
    check(cudaMalloc(&lm[q],8));check(cudaMemset(lm[q],0,8));
    check(cudaMalloc(&counter[q],8));check(cudaMemset(counter[q],0,8));
    check(cudaFuncSetAttribute(HeadKernel<Finish,B>,cudaFuncAttributeMaxDynamicSharedMemorySize,8192));
  }
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(devices[q]));
    Sm90DenseOperands p{nullptr,nullptr,nullptr,reinterpret_cast<BF*>(partials[q]),arg[q],0.f};
    Sm90DenseHeadOperands head{tokens[q],lm[q],nullptr,nullptr,0,0};
    Sm90DenseTpHeadOperands tp{};
    for(int r=0;r<T;++r)tp.x[r]=exchange[r];
    tp.rank=q;tp.world=T;tp.vocab_local=VocabLocal;
    HeadKernel<Finish,B><<<kGrid,128,8192>>>(p,head,tp,dv[q],di[q],counter[q],repeats);
    check(cudaGetLastError());
  }
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(devices[q]));check(cudaDeviceSynchronize());
    std::vector<int> got(B*kCapacity);check(cudaMemcpy(got.data(),tokens[q],got.size()*4,cudaMemcpyDeviceToHost));
    for(int r=0;r<repeats;++r)for(int row=0;row<B;++row) {
      float value=-INFINITY;long index=1L<<40;
      for(int s=0;s<T;++s)for(int c=0;c<kGrid;++c) {
        std::size_t i=(std::size_t(r)*kGrid+c)*B+row;
        float v=values[s][i];long id=indices[s][i]+long(s)*VocabLocal;
        if(v>value || (v==value && id<index)) {value=v;index=id;}
      }
      int g=got[row*kCapacity+r+1];
      if(g!=index) {
        if(errors<4)std::fprintf(stderr,"head T%d B%d rank%d step%d row%d got %d expected %ld\n",
            T,B,q,r,row,g,index);
        ++errors;
      }
    }
    for(void* ptr:{(void*)exchange[q],(void*)tokens[q],(void*)partials[q],(void*)arg[q],(void*)dv[q],
                   (void*)di[q],(void*)lm[q],(void*)counter[q]})check(cudaFree(ptr));
  }
  std::fprintf(stderr,"head T%d B%d errors %d\n",T,B,errors);
  return errors;
}

template<int T>
int RunAll(std::vector<int> const& devices) {
  int errors=0;
  errors+=ArCase<T,1,8,1>(devices);errors+=ArCase<T,16,8,1>(devices);
  errors+=ArCase<T,1,16,2>(devices);errors+=ArCase<T,16,16,2>(devices);
  errors+=ArCase<T,1,32,2>(devices);errors+=ArCase<T,16,32,2>(devices);
  errors+=ArCase<T,1,32,4>(devices);errors+=ArCase<T,16,32,4>(devices);
  errors+=HeadCase<T,1>(devices);errors+=HeadCase<T,4>(devices);errors+=HeadCase<T,16>(devices);
  return errors;
}

int main() {
  try {
    int count=0;check(cudaGetDeviceCount(&count));
    if(count!=2 && count!=4 && count!=8) {
      std::fprintf(stderr,"skipped: expose exactly 2, 4 or 8 peer-capable devices\n");
      return 77;
    }
    std::vector<int> devices;
    for(int d=0;d<count;++d)devices.push_back(d);
    for(int d:devices) {
      check(cudaSetDevice(d));
      for(int e:devices)if(e!=d) {
        auto status=cudaDeviceEnablePeerAccess(e,0);
        if(status==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(status);
      }
    }
    int errors=count==2?RunAll<2>(devices):count==4?RunAll<4>(devices):RunAll<8>(devices);
    std::printf("{\"tp\":%d,\"errors\":%d,\"pass\":%s}\n",count,errors,errors?"false":"true");
    return errors?1:0;
  }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
