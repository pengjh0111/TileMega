// SPDX-License-Identifier: BSD-3-Clause
// MR3 router last arriver: packed-key top-k, both softmax modes, the
// schedule record. Random BF16 logits with forced ties and signed zeros
// against an FP64 reference (ties keep the smaller expert; -0 equals +0).
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;

template<int B,int E,bool All>
__global__ __launch_bounds__(128) void finish(Sm90DenseMoeOperands p) {
  __shared__ __align__(16) int free[8+B*8+8];
  Sm90DenseRouterFinish<B,E,8,16,true,All>{}(p,16,reinterpret_cast<char*>(free));
}

void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}

template<int B,int E,bool All>
int Run(std::mt19937& rng,int trial) {
  constexpr int K=8;
  using Schedule=Sm90DenseMoeSchedule<E>;
  std::vector<float> logits(B*E);
  std::uniform_int_distribution<int> pick(-40,40);
  for(auto& v:logits)v=float(BF(float(pick(rng))/16.0f));
  // Forced ties: copy values across experts, and signed zeros.
  for(int b=0;b<B;++b) {
    for(int t=0;t<6;++t)logits[b*E+rng()%E]=logits[b*E+rng()%E];
    logits[b*E+rng()%E]=0.0f;logits[b*E+rng()%E]=-0.0f;
    if(trial%2)for(int e=0;e<E;++e)logits[b*E+e]=float(BF(float(pick(rng)%3)));
  }
  float *dl,*dw;int *di,*ds;std::uint64_t* dc;
  Check(cudaMalloc(&dl,B*E*4));Check(cudaMalloc(&dw,B*K*4));Check(cudaMalloc(&di,B*K*4));
  Check(cudaMalloc(&ds,Schedule::kInts*4));Check(cudaMalloc(&dc,8));
  Check(cudaMemcpy(dl,logits.data(),B*E*4,cudaMemcpyHostToDevice));Check(cudaMemset(dc,0,8));
  Check(cudaMemset(ds,0xff,Schedule::kInts*4));
  Sm90DenseMoeOperands p{nullptr,nullptr,nullptr,dl,ds,dw,di,nullptr,dc,0};
  finish<B,E,All><<<1,128>>>(p);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<float> w(B*K);std::vector<int> index(B*K),schedule(Schedule::kInts);
  Check(cudaMemcpy(w.data(),dw,B*K*4,cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(index.data(),di,B*K*4,cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(schedule.data(),ds,Schedule::kInts*4,cudaMemcpyDeviceToHost));
  int errors=0;
  std::vector<int> count(E,0);
  for(int b=0;b<B;++b) {
    std::vector<int> order(E);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int x,int y) {return logits[b*E+x]>logits[b*E+y];});
    double top=logits[b*E+order[0]],sum=0;
    if(All)for(int e=0;e<E;++e)sum+=std::exp(double(logits[b*E+e])-top);
    else for(int k=0;k<K;++k)sum+=std::exp(double(logits[b*E+order[k]])-top);
    for(int k=0;k<K;++k) {
      double want=std::exp(double(logits[b*E+order[k]])-top)/sum;
      if(index[b*K+k]!=order[k] || std::fabs(w[b*K+k]-want)>1e-6+1e-5*want) {
        if(errors<4)std::fprintf(stderr,"B%d E%d all%d b%d k%d got %d %g want %d %g\n",B,E,All,b,k,
            index[b*K+k],w[b*K+k],order[k],want);
        ++errors;
      }
      ++count[order[k]];
    }
  }
  int active=0;
  for(int e=0;e<E;++e) {
    if(!count[e])continue;
    if(schedule[Schedule::Expert(active)]!=e || schedule[Schedule::Count(active)]!=count[e])++errors;
    for(int r=0;r<count[e];++r) {
      int route=schedule[Schedule::Row(active,r)];
      if(route<0 || route>=B*K || index[route]!=e)++errors;
    }
    ++active;
  }
  if(schedule[Schedule::Active()]!=active)++errors;
  cudaFree(dl);cudaFree(dw);cudaFree(di);cudaFree(ds);cudaFree(dc);
  return errors;
}

int main() {try {
  Check(cudaSetDevice(0));
  std::mt19937 rng(20261029);
  int errors=0;
  for(int trial=0;trial<20;++trial)
    errors+=Run<1,128,false>(rng,trial)+Run<16,128,false>(rng,trial)+Run<3,128,false>(rng,trial)+
        Run<1,64,true>(rng,trial)+Run<16,64,true>(rng,trial)+Run<5,64,true>(rng,trial)+
        Run<16,128,true>(rng,trial)+Run<16,64,false>(rng,trial);
  std::printf("{\"trials\":20,\"errors\":%d,\"pass\":%s}\n",errors,errors?"false":"true");
  return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
