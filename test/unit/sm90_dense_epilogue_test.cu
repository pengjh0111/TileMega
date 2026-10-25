// SPDX-License-Identifier: BSD-3-Clause
// D4 epilogue check (prompt §5): consumer-side RMS, prefetched residual,
// register and shared epilogue paths against an FP64 reference of the same
// rounding sequence (ServingEpilogueValue). Grid 4 with eight tasks gives each
// CTA two tasks, the most a residual stage prefetches. Stream cases issue the
// first task's weight slots ahead of Run, as the D6 producer tail does.
// Argmax cases run the D9 head: per-CTA partials, lm_ctr and the last
// arriver's reduction to tokens, and with Embed the next embedding rows.
#ifndef TEST_BATCH
#define TEST_BATCH 1
#endif
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using Op=ServingEpilogueOp;
constexpr int B=TEST_BATCH,kGrid=4,kTasks=8;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
float Round(double x) {return float(BF(float(x)));}

template<class G,int Prefix,class Finish>
__global__ __launch_bounds__(160) void evaluate(Sm90DenseOperands p,Finish finish) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  for(int repeat=0;repeat<2;++repeat) {
    if constexpr(Prefix>0)if(threadIdx.x==128)
      G::template Preissue<Prefix>(p.weight,shared,state);
    G::template Run<Prefix,Sm90DenseNoTail,Finish>(p,shared,nullptr,{},true,finish);
  }
  state.Invalidate();
}

template<Op O,int TN,int WN,int TK,int K,int U=8,bool Stream=false,bool Embed=false>
int run() {
  constexpr int N=TN*kTasks;
  constexpr bool norm=O!=Op::kResidual;
  using G=Sm90DenseGemm<B,N,K,TN,TK,2,WN,4/WN,O,U,norm>;
  constexpr int prefix=Stream?(K/TK<2?K/TK:2)*2*TN*TK:0;
  constexpr int rows=Sm90DenseLayout::ActivationRows(B);
  std::vector<BF> act(B*K),weight(std::size_t(N)*K),packed(std::size_t(N)*K),residual(B*N);
  for(int b=0;b<B;++b)for(int k=0;k<K;++k)act[b*K+k]=BF(float(((b*37+k*11)%41)-20)/(16.0f+b));
  for(int n=0;n<N;++n)for(int k=0;k<K;++k)weight[std::size_t(n)*K+k]=BF(float(((n*13+k*7)%29)-14)/256.0f);
  for(int i=0;i<B*N;++i)residual[i]=BF(float((i*5)%17-8)/8.0f);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    int row=n%TN,col=k%TK;
    size_t permutation=512*(row/8+(TN/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
    packed[(size_t(n/TN)*(K/TK)+k/TK)*TN*TK+permutation]=weight[std::size_t(n)*K+k];
  }
  int out_count=O==Op::kSwiGLU?B*N/2:O==Op::kArgmaxPartial?B*kGrid:B*N;
  BF *a,*w,*r,*o;int* index;CUtensorMap* map;
  check(cudaMalloc(&a,act.size()*2));check(cudaMalloc(&w,packed.size()*2));
  check(cudaMalloc(&r,residual.size()*2));check(cudaMalloc(&o,std::max(out_count*4,2)));
  check(cudaMalloc(&index,std::max(out_count*4,4)));check(cudaMalloc(&map,128));
  check(cudaMemcpy(a,act.data(),act.size()*2,cudaMemcpyHostToDevice));
  check(cudaMemcpy(w,packed.data(),packed.size()*2,cudaMemcpyHostToDevice));
  check(cudaMemcpy(r,residual.data(),residual.size()*2,cudaMemcpyHostToDevice));
  CUtensorMap host;
  tilemega::codegen::executor::EncodeTiledMap<3>(host,a,
      {{64,std::uint64_t(B),K/64},{2ull*K,128},{64,std::uint32_t(rows),TK/64}});
  check(cudaMemcpy(map,&host,sizeof host,cudaMemcpyHostToDevice));
  // Residual stages read and write the stream in place, as o/down do.
  BF* output=O==Op::kResidual?r:o;
  Sm90DenseOperands p{map,w,r,output,index,1e-6f};
  // D9 head: tokens [B][1088], lm_ctr, a vocabulary of N rows of width K.
  constexpr int kPast=575;
  int* tokens;std::uint64_t* counter;BF *table,*x;Sm90DenseHeadOperands* head;
  std::vector<BF> vocabulary(std::size_t(N)*K);
  for(std::size_t i=0;i<vocabulary.size();++i)vocabulary[i]=BF(float(int(i%251)-125)/64.0f);
  check(cudaMalloc(&tokens,B*1088*4));check(cudaMemset(tokens,0xff,B*1088*4));
  check(cudaMalloc(&counter,8));check(cudaMemset(counter,0,8));
  check(cudaMalloc(&table,vocabulary.size()*2));check(cudaMalloc(&x,B*K*2));check(cudaMemset(x,0,B*K*2));
  check(cudaMemcpy(table,vocabulary.data(),vocabulary.size()*2,cudaMemcpyHostToDevice));
  check(cudaMalloc(&head,sizeof(Sm90DenseHeadOperands)));
  Sm90DenseHeadOperands host_head{tokens,counter,table,x,kPast};
  check(cudaMemcpy(head,&host_head,sizeof host_head,cudaMemcpyHostToDevice));
  using Finish=std::conditional_t<O==Op::kArgmaxPartial,Sm90DenseHeadFinish<B,K,1088,Embed>,
      Sm90DenseNoFinish>;
  Finish finish{};
  if constexpr(O==Op::kArgmaxPartial)finish.head=head;
  check(cudaFuncSetAttribute(evaluate<G,prefix,Finish>,cudaFuncAttributeMaxDynamicSharedMemorySize,
      G::kSharedBytes));
  evaluate<G,prefix,Finish><<<kGrid,160,G::kSharedBytes>>>(p,finish);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<double> acc(B*N);std::vector<double> inv(B,1.0);
  for(int b=0;b<B;++b) {
    if(norm) {
      double s=0;for(int k=0;k<K;++k)s+=double(act[b*K+k])*double(act[b*K+k]);
      inv[b]=1.0/std::sqrt(s/K+1e-6);
    }
    for(int n=0;n<N;++n) {
      double s=0;for(int k=0;k<K;++k)s+=double(act[b*K+k])*double(weight[std::size_t(n)*K+k]);
      acc[b*N+n]=s*inv[b];
    }
  }
  int errors=0;
  auto bad=[&](char const* what,int b,int n,double got,double want) {
    if(errors<3)std::fprintf(stderr,"B%d TN%d WN%d %s b%d n%d got %g expected %g\n",B,TN,WN,what,b,n,got,want);
    ++errors;
  };
  // FP32 accumulation and BF16 output: one BF16 ulp plus accumulation slack.
  auto close=[](double got,double want) {return std::fabs(got-want)<=std::fabs(want)/64+2e-3;};
  if constexpr(O==Op::kArgmaxPartial) {
    // The token is a maximal logit of the whole row (near-ties may resolve
    // either way under FP32 accumulation order).
    std::vector<int> next(B*1088);std::vector<BF> written(B*K);
    check(cudaMemcpy(next.data(),tokens,next.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(written.data(),x,written.size()*2,cudaMemcpyDeviceToHost));
    std::uint64_t arrivals=0;check(cudaMemcpy(&arrivals,counter,8,cudaMemcpyDeviceToHost));
    if(arrivals!=2*kGrid)bad("lm_ctr",0,0,double(arrivals),2*kGrid);
    for(int b=0;b<B;++b) {
      double best=-1e30;
      for(int c=0;c<N;++c)best=std::max(best,double(Round(acc[b*N+c])));
      int id=next[b*1088+kPast+1];
      if(id<0 || id>=N || !close(Round(acc[b*N+id]),best))bad("token",b,id,id>=0 && id<N?Round(acc[b*N+id]):0,best);
      if(Embed && id>=0 && id<N)
        for(int c=0;c<K;++c)
          if(written[b*K+c].raw()!=vocabulary[std::size_t(id)*K+c].raw()) {bad("embedding",b,c,float(written[b*K+c]),
              float(vocabulary[std::size_t(id)*K+c]));break;}
    }
  }else {
    int width=O==Op::kSwiGLU?N/2:N;
    std::vector<BF> got(B*width);
    check(cudaMemcpy(got.data(),output,got.size()*2,cudaMemcpyDeviceToHost));
    for(int b=0;b<B;++b)for(int c=0;c<width;++c) {
      double want;
      if constexpr(O==Op::kSwiGLU) {
        int tile=c/(TN/2),local=c%(TN/2),gate=tile*TN+2*U*(local/U)+local%U;
        double g=Round(acc[b*N+gate]),u=Round(acc[b*N+gate+U]);
        want=double(Round(Round(g/(1+std::exp(-g)))*u));
      }else if constexpr(O==Op::kResidual) {
        // Two repeats: the stream gains the projection twice.
        double once=Round(double(Round(acc[b*N+c]))+double(residual[b*N+c]));
        want=Round(double(Round(acc[b*N+c]))+once);
      }else want=Round(acc[b*N+c]);
      if(!close(float(got[b*width+c]),want))bad("value",b,c,float(got[b*width+c]),want);
    }
  }
  for(void* ptr:{(void*)a,(void*)w,(void*)r,(void*)o,(void*)index,(void*)map,(void*)tokens,(void*)counter,
                 (void*)table,(void*)x,(void*)head})check(cudaFree(ptr));
  return errors;
}

int main() {
  try {
    check(cudaSetDevice(0));int errors=0,cases=0;
    auto add=[&](int e) {errors+=e;++cases;};
    // qkv (store + RMS), o/down (residual), gate_up (SwiGLU + RMS), lm_head
    // (argmax + RMS); WN=4 takes the register path for store/residual.
    add(run<Op::kStore,32,4,256,1024>());add(run<Op::kStore,32,1,256,1024>());
    add(run<Op::kStore,40,1,128,3072>());add(run<Op::kStore,64,2,512,2048>());
    add(run<Op::kStore,128,4,64,4096>());
    add(run<Op::kResidual,8,1,1024,2048>());add(run<Op::kResidual,16,2,512,3072>());
    add(run<Op::kResidual,32,4,256,4096>());add(run<Op::kResidual,64,4,128,8192>());
    add(run<Op::kSwiGLU,24,1,128,1024,4>());add(run<Op::kSwiGLU,48,2,256,1024,8>());
    add(run<Op::kSwiGLU,64,4,128,2048,16>());add(run<Op::kSwiGLU,224,4,64,4096,16>());
    add(run<Op::kArgmaxPartial,128,4,64,1024>());add(run<Op::kArgmaxPartial,128,1,256,2048,8,false,true>());
    add(run<Op::kArgmaxPartial,256,2,64,4096,8,false,true>());
    // D6 streams: two slots of K/TK 4, a fully resident K/TK 2, and the
    // shared-epilogue SwiGLU/argmax paths.
    add(run<Op::kStore,32,4,256,1024,8,true>());add(run<Op::kResidual,8,1,1024,2048,8,true>());
    add(run<Op::kSwiGLU,224,4,64,4096,16,true>());add(run<Op::kArgmaxPartial,128,1,256,2048,8,true>());
    std::printf("{\"batch\":%d,\"cases\":%d,\"errors\":%d,\"pass\":%s}\n",B,cases,errors,
        errors?"false":"true");
    return errors?1:0;
  }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
