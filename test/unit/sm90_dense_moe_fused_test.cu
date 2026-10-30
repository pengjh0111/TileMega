// SPDX-License-Identifier: BSD-3-Clause
// MR7 (prompt §4 A2): the fused expert stage M56 behind the router, in one
// grid-synchronized kernel, against the same FP64 reference as the unfused
// bodies (sm90_dense_moe_test.cu). The counter protocol is exercised under
// random arrival order: before M56 every CTA spins a pseudo-random 0-20 us
// drawn from the process seed (argv[1]), so gate_up publishers and down
// waiters meet in a different order on each of LAUNCHES launches. Every
// launch is checked, and the router's last arriver must clear gu_done.
#ifndef TEST_BATCH
#define TEST_BATCH 1
#endif
// A grid smaller than the task count gives every CTA several tasks of both
// types, which the real models do and 132 CTAs over this small shape do not.
#ifndef TEST_GRID
#define TEST_GRID 132
#endif
// TEST_H / TEST_I = 2048 / 768 is Qwen3-30B-A3B's own expert geometry.
#ifndef TEST_H
#define TEST_H 512
#endif
#ifndef TEST_I
#define TEST_I 192
#endif
// Page 64 shares the legacy prefill pages (Qwen3); 0 is OLMoE's per-TN packing.
#ifndef TEST_PAGE
#define TEST_PAGE 64
#endif
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
constexpr int B=TEST_BATCH,E=128,TopK=8,H=TEST_H,I=TEST_I,U=16,kGrid=TEST_GRID,kLaunches=20;
constexpr int TNgu=96,TKgu=128,TNdn=128,TKdn=64,kSlots=3;
using Router=Sm90DenseMoeGemm<B,E,H,8,256,2,1,4,Sm90DenseMoeMode::kRouter,U,E,TopK,false>;
using Fused=Sm90DenseMoeFusedGemm<B,2*I,H,H,TNgu,TKgu,TNdn,TKdn,kSlots,4,1,4,1,U,E,TopK,false,TEST_PAGE>;
using Schedule=Sm90DenseMoeSchedule<E>;
void Check(cudaError_t e,int line) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorString(e))+" at line "+std::to_string(line));
}
#define check(e) Check((e),__LINE__)
float Round(double x) {return float(BF(float(x)));}

__device__ void GridBarrier(U64* counter,U64 round) {
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    U64 v;do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<(round+1)*gridDim.x);
  }
  __syncthreads();
}
__global__ __launch_bounds__(160) void moe(Sm90DenseMoeOperands r,Sm90DenseMoeFusedOperands f,
    U64* barrier,int launch,unsigned const* skew_ns) {
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<Sm90DenseRingState*>(shared);
  rings[0].Initialize();rings[1].Initialize();
  Router::Run(r,shared,&rings[0]);GridBarrier(barrier,launch*2);
  // The whole CTA starts M56 late, so its producer and consumers stay together.
  U64 until=0;
  if(threadIdx.x==0) {
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(until));until+=skew_ns[blockIdx.x];
    for(U64 now=0;now<until;)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
  }
  __syncthreads();
  Fused::Run(f,shared,&rings[1]);GridBarrier(barrier,launch*2+1);
  rings[0].Invalidate();rings[1].Invalidate();
}

// Tile-pages permutation of one N x K matrix (the native weight layout).
void Pack(std::vector<BF> const& w,int n_rows,int k_cols,int tn,int tk,std::size_t base,std::vector<BF>& packed) {
  for(int n=0;n<n_rows;++n)for(int k=0;k<k_cols;++k) {
    int row=n%tn,col=k%tk;
    std::size_t perm=512*(row/8+(tn/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
    packed[base+(std::size_t(n/tn)*(k_cols/tk)+k/tk)*tn*tk+perm]=w[std::size_t(n)*k_cols+k];
  }
}
template<class T> T* Device(std::vector<T> const& host) {
  T* p;check(cudaMalloc(&p,host.size()*sizeof(T)));
  check(cudaMemcpy(p,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> std::vector<T> Host(T const* p,std::size_t n) {
  std::vector<T> host(n);check(cudaMemcpy(host.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return host;
}

int main(int argc,char** argv) {try {
  unsigned seed=argc>1?unsigned(std::strtoul(argv[1],nullptr,10)):1u;
  check(cudaSetDevice(0));
  std::vector<BF> x(B*H),router(E*H),gate_up(std::size_t(E)*2*I*H),down(std::size_t(E)*H*I);
  for(int i=0;i<B*H;++i)x[i]=BF(float(((i*37)%41)-20)/16.0f);
  for(int i=0;i<E*H;++i)router[i]=BF(float(((i*13+i/H*7)%29)-14)/64.0f);
  for(std::size_t i=0;i<gate_up.size();++i)gate_up[i]=BF(float(int((i*11)%31)-15)/128.0f);
  for(std::size_t i=0;i<down.size();++i)down[i]=BF(float(int((i*7)%23)-11)/64.0f);
  std::vector<BF> packed_router(router.size()),packed_gu(gate_up.size()),packed_dn(down.size());
  Pack(router,E,H,8,256,0,packed_router);
  for(int e=0;e<E;++e) {
    std::vector<BF> one(gate_up.begin()+std::size_t(e)*2*I*H,gate_up.begin()+std::size_t(e+1)*2*I*H);
    Pack(one,2*I,H,TEST_PAGE?64:TNgu,TEST_PAGE?64:TKgu,std::size_t(e)*2*I*H,packed_gu);
    std::vector<BF> two(down.begin()+std::size_t(e)*H*I,down.begin()+std::size_t(e+1)*H*I);
    Pack(two,H,I,TEST_PAGE?64:TNdn,TEST_PAGE?64:TKdn,std::size_t(e)*H*I,packed_dn);
  }
  BF *dx=Device(x),*dr=Device(packed_router),*dgu=Device(packed_gu),*ddn=Device(packed_dn);
  float *logits,*partial;BF *h,*weights;int *schedule,*index;std::uint64_t *router_ctr,*tile_ctr;
  unsigned *gu_done,*skew;U64* barrier;
  check(cudaMalloc(&logits,B*E*4));check(cudaMalloc(&partial,std::size_t(B)*TopK*H*4));
  check(cudaMalloc(&h,std::size_t(E)*16*I*2));check(cudaMalloc(&weights,B*TopK*2));
  check(cudaMalloc(&schedule,Schedule::kInts*4));check(cudaMalloc(&index,B*TopK*4));
  check(cudaMalloc(&router_ctr,8));check(cudaMalloc(&tile_ctr,8*(H/TNdn)));check(cudaMalloc(&barrier,8));
  check(cudaMalloc(&gu_done,4*E));check(cudaMalloc(&skew,4*kGrid));
  check(cudaMemset(router_ctr,0,8));check(cudaMemset(tile_ctr,0,8*(H/TNdn)));check(cudaMemset(barrier,0,8));
  check(cudaMemset(h,0,std::size_t(E)*16*I*2));
  // Stale counts from an imagined previous layer: the router must clear them.
  check(cudaMemset(gu_done,0x7f,4*E));
  CUtensorMap* maps;check(cudaMalloc(&maps,256));
  CUtensorMap host_maps[2];
  int rows=Sm90DenseLayout::ActivationRows(B);
  tilemega::codegen::executor::EncodeTiledMap<3>(host_maps[0],dx,
      {{64,std::uint64_t(B),H/64},{2ull*H,128},{64,std::uint32_t(rows),256/64}});
  // h [E][16][I] as {64, 16, I/64, E}; a box is one slot's [TK/64][rows][64].
  tilemega::codegen::executor::EncodeTiledMap<4>(host_maps[1],h,
      {{64,16,I/64,E},{2ull*I,128,2ull*16*I},{64,std::uint32_t(rows),TKdn/64,1}});
  check(cudaMemcpy(maps,host_maps,sizeof host_maps,cudaMemcpyHostToDevice));
  Sm90DenseMoeOperands r{maps,dr,nullptr,logits,schedule,weights,index,nullptr,router_ctr,1e-6f,gu_done};
  Sm90DenseMoeFusedOperands f{maps+1,dgu,ddn,dx,h,partial,schedule,weights,nullptr,dx,tile_ctr,gu_done,1e-6f};
  int shared=std::max(Router::kSharedBytes,Fused::kSharedBytes);
  check(cudaFuncSetAttribute(moe,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  // Reference (one launch from x).
  std::vector<double> inv(B);
  for(int b=0;b<B;++b) {
    double s=0;for(int k=0;k<H;++k)s+=double(x[b*H+k])*double(x[b*H+k]);
    inv[b]=1.0/std::sqrt(s/H+1e-6);
  }
  std::vector<std::vector<int>> top(B);std::vector<std::vector<float>> prob(B);
  for(int b=0;b<B;++b) {
    std::vector<float> l(E);
    for(int e=0;e<E;++e) {
      double s=0;for(int k=0;k<H;++k)s+=double(x[b*H+k])*double(router[e*H+k]);
      l[e]=Round(s*inv[b]);
    }
    std::vector<int> order(E);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int c) {return l[a]>l[c];});
    double sum=0;std::vector<double> p(TopK);
    for(int k=0;k<TopK;++k) {top[b].push_back(order[k]);p[k]=std::exp(double(l[order[k]])-l[order[0]]);sum+=p[k];}
    for(int k=0;k<TopK;++k)prob[b].push_back(Round(p[k]/sum));
  }
  std::vector<double> expected(B*H);
  for(int b=0;b<B;++b) {
    std::vector<double> mix(H,0);
    for(int k=0;k<TopK;++k) {
      int e=top[b][k];std::vector<float> hv(I);
      for(int c=0;c<I;++c) {
        int gate=2*U*(c/U)+c%U;double gs=0,us=0;
        for(int q=0;q<H;++q) {
          gs+=double(x[b*H+q])*double(gate_up[(std::size_t(e)*2*I+gate)*H+q]);
          us+=double(x[b*H+q])*double(gate_up[(std::size_t(e)*2*I+gate+U)*H+q]);
        }
        double gv=Round(gs*inv[b]),uv=Round(us*inv[b]);
        hv[c]=Round(Round(gv/(1+std::exp(-gv)))*uv);
      }
      for(int n=0;n<H;++n) {
        double y=0;for(int c=0;c<I;++c)y+=double(hv[c])*double(down[(std::size_t(e)*H+n)*I+c]);
        mix[n]+=double(prob[b][k])*y;
      }
    }
    for(int n=0;n<H;++n)expected[b*H+n]=Round(double(x[b*H+n])+mix[n]);
  }
  int errors=0;
  auto bad=[&](char const* what,int launch,int b,int c,double got,double want) {
    if(errors<4)std::fprintf(stderr,"B%d L%d %s b%d c%d got %g want %g\n",B,launch,what,b,c,got,want);++errors;
  };
  std::mt19937 rng(seed);std::uniform_int_distribution<unsigned> spread(0,20000);
  std::vector<unsigned> skews(kGrid);
  for(int launch=0;launch<kLaunches;++launch) {
    for(auto& s:skews)s=spread(rng);
    check(cudaMemcpy(skew,skews.data(),4*kGrid,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    moe<<<kGrid,160,shared>>>(r,f,barrier,launch,skew);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    auto got_x=Host(dx,B*H);auto counters=Host(tile_ctr,H/TNdn);
    auto got_schedule=Host(schedule,Schedule::kInts);auto done=Host(gu_done,E);
    int active=got_schedule[Schedule::Active()];
    // Each active slot's gate_up tasks each released once; nothing else moved.
    for(int a=0;a<E;++a)
      if(done[a]!=(a<active?unsigned(Fused::kTilesGu):0u))bad("gu_done",launch,0,a,done[a],a<active?Fused::kTilesGu:0);
    for(int j=0;j<H/TNdn;++j)if(counters[j])bad("ctr reset",launch,0,j,double(counters[j]),0);
    for(int b=0;b<B;++b)for(int n=0;n<H;++n) {
      double want=expected[b*H+n],got=float(got_x[b*H+n]);
      if(std::fabs(got-want)>std::fabs(want)/32+2e-2)bad("x''",launch,b,n,got,want);
    }
  }
  std::printf("{\"batch\":%d,\"seed\":%u,\"launches\":%d,\"errors\":%d,\"pass\":%s}\n",
      B,seed,kLaunches,errors,errors?"false":"true");
  return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
