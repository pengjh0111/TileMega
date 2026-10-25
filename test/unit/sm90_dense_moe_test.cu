// SPDX-License-Identifier: BSD-3-Clause
// Native MoE bodies (prompt §8 M4-M6) in one grid-synchronized kernel:
// router + top-k last arriver, grouped gate_up (cp.async row gather), down +
// per-tile combine. FP64 reference with the bodies' BF16 roundings; two
// launches exercise the monotonic router counter and the ctr[j] reset.
#ifndef TEST_BATCH
#define TEST_BATCH 1
#endif
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
constexpr int B=TEST_BATCH,E=128,TopK=8,H=512,I=192,U=16,kGrid=132;
constexpr int TNgu=96,TKgu=128,TNdn=128,TKdn=64;
using Router=Sm90DenseMoeGemm<B,E,H,8,256,2,1,4,Sm90DenseMoeMode::kRouter,U,E,TopK,false>;
using GateUp=Sm90DenseMoeGemm<B,2*I,H,TNgu,TKgu,3,4,1,Sm90DenseMoeMode::kGateUp,U,E,TopK,false>;
using Down=Sm90DenseMoeGemm<B,H,I,TNdn,TKdn,3,4,1,Sm90DenseMoeMode::kDown,U,E,TopK,false>;
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
__global__ __launch_bounds__(160) void moe(Sm90DenseMoeOperands r,Sm90DenseMoeOperands g,
    Sm90DenseMoeOperands d,U64* barrier,int launch) {
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<Sm90DenseRingState*>(shared);
  rings[0].Initialize();rings[1].Initialize();
  Router::Run(r,shared,&rings[0]);GridBarrier(barrier,launch*3);
  GateUp::Run(g,shared,&rings[1]);GridBarrier(barrier,launch*3+1);
  Down::Run(d,shared,&rings[0]);GridBarrier(barrier,launch*3+2);
  rings[0].Invalidate();rings[1].Invalidate();
}

// Tile-pages permutation of one N x K matrix (the native weight layout).
std::vector<BF> Pack(std::vector<BF> const& w,int n_rows,int k_cols,int tn,int tk,std::size_t base=0,
                     std::vector<BF>* out=nullptr) {
  std::vector<BF> local;auto& packed=out?*out:local;
  if(!out)packed.assign(std::size_t(n_rows)*k_cols,BF(0));
  for(int n=0;n<n_rows;++n)for(int k=0;k<k_cols;++k) {
    int row=n%tn,col=k%tk;
    std::size_t perm=512*(row/8+(tn/8)*(col/64))+64*(row%8)+8*(((col%64)/8)^(row%8))+col%8;
    packed[base+(std::size_t(n/tn)*(k_cols/tk)+k/tk)*tn*tk+perm]=w[std::size_t(n)*k_cols+k];
  }
  return local;
}
template<class T> T* Device(std::vector<T> const& host) {
  T* p;check(cudaMalloc(&p,host.size()*sizeof(T)));
  check(cudaMemcpy(p,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> std::vector<T> Host(T const* p,std::size_t n) {
  std::vector<T> host(n);check(cudaMemcpy(host.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return host;
}

int main() {try {
  check(cudaSetDevice(0));
  std::vector<BF> x(B*H),router(E*H),gate_up(std::size_t(E)*2*I*H),down(std::size_t(E)*H*I);
  for(int i=0;i<B*H;++i)x[i]=BF(float(((i*37)%41)-20)/16.0f);
  for(int i=0;i<E*H;++i)router[i]=BF(float(((i*13+i/H*7)%29)-14)/64.0f);
  for(std::size_t i=0;i<gate_up.size();++i)gate_up[i]=BF(float(int((i*11)%31)-15)/128.0f);
  for(std::size_t i=0;i<down.size();++i)down[i]=BF(float(int((i*7)%23)-11)/64.0f);
  std::vector<BF> packed_router=Pack(router,E,H,8,256);
  std::vector<BF> packed_gu(gate_up.size()),packed_dn(down.size());
  for(int e=0;e<E;++e) {
    std::vector<BF> one(gate_up.begin()+std::size_t(e)*2*I*H,gate_up.begin()+std::size_t(e+1)*2*I*H);
    Pack(one,2*I,H,TNgu,TKgu,std::size_t(e)*2*I*H,&packed_gu);
    std::vector<BF> two(down.begin()+std::size_t(e)*H*I,down.begin()+std::size_t(e+1)*H*I);
    Pack(two,H,I,TNdn,TKdn,std::size_t(e)*H*I,&packed_dn);
  }
  // Device state.
  BF *dx=Device(x),*dr=Device(packed_router),*dgu=Device(packed_gu),*ddn=Device(packed_dn);
  float *logits,*partial;BF *h,*weights;int *schedule,*index;std::uint64_t *router_ctr,*tile_ctr;U64* barrier;
  check(cudaMalloc(&logits,B*E*4));check(cudaMalloc(&partial,std::size_t(B)*TopK*H*4));
  check(cudaMalloc(&h,std::size_t(E)*16*I*2));check(cudaMalloc(&weights,B*TopK*2));
  check(cudaMalloc(&schedule,Schedule::kInts*4));check(cudaMalloc(&index,B*TopK*4));
  check(cudaMalloc(&router_ctr,8));check(cudaMalloc(&tile_ctr,8*(H/TNdn)));check(cudaMalloc(&barrier,8));
  check(cudaMemset(router_ctr,0,8));check(cudaMemset(tile_ctr,0,8*(H/TNdn)));check(cudaMemset(barrier,0,8));
  check(cudaMemset(h,0,std::size_t(E)*16*I*2));
  CUtensorMap* maps;check(cudaMalloc(&maps,256));
  CUtensorMap host_maps[2];
  int rows=Sm90DenseLayout::ActivationRows(B);
  tilemega::codegen::executor::EncodeTiledMap<3>(host_maps[0],dx,
      {{64,std::uint64_t(B),H/64},{2ull*H,128},{64,std::uint32_t(rows),256/64}});
  // h [E][16][I] as {64, 16, I/64, E}; a box is one slot's [TK/64][rows][64].
  tilemega::codegen::executor::EncodeTiledMap<4>(host_maps[1],h,
      {{64,16,I/64,E},{2ull*I,128,2ull*16*I},{64,std::uint32_t(rows),TKdn/64,1}});
  check(cudaMemcpy(maps,host_maps,sizeof host_maps,cudaMemcpyHostToDevice));
  Sm90DenseMoeOperands r{maps,dr,nullptr,logits,schedule,weights,index,nullptr,router_ctr,1e-6f};
  Sm90DenseMoeOperands g{nullptr,dgu,dx,h,schedule,nullptr,nullptr,nullptr,nullptr,1e-6f};
  Sm90DenseMoeOperands d{maps+1,ddn,nullptr,partial,schedule,weights,nullptr,dx,tile_ctr,1e-6f};
  int shared=std::max({Router::kSharedBytes,GateUp::kSharedBytes,Down::kSharedBytes});
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
  auto bad=[&](char const* what,int b,int c,double got,double want) {
    if(errors<4)std::fprintf(stderr,"B%d %s b%d c%d got %g want %g\n",B,what,b,c,got,want);++errors;
  };
  for(int launch=0;launch<2;++launch) {
    check(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    moe<<<kGrid,160,shared>>>(r,g,d,barrier,launch);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    auto got_index=Host(index,B*TopK);auto got_weights=Host(weights,B*TopK);
    auto got_schedule=Host(schedule,Schedule::kInts);auto got_x=Host(dx,B*H);
    auto counters=Host(tile_ctr,H/TNdn);
    std::set<int> active;
    for(int b=0;b<B;++b)for(int k=0;k<TopK;++k) {
      active.insert(top[b][k]);
      if(got_index[b*TopK+k]!=top[b][k])bad("top-k",b,k,got_index[b*TopK+k],top[b][k]);
      if(std::fabs(float(got_weights[b*TopK+k])-prob[b][k])>1e-2)bad("weight",b,k,float(got_weights[b*TopK+k]),prob[b][k]);
    }
    if(got_schedule[Schedule::Active()]!=int(active.size()))
      bad("n_active",0,0,got_schedule[Schedule::Active()],active.size());
    int previous=-1,routes=0;
    for(int a=0;a<got_schedule[Schedule::Active()];++a) {
      int e=got_schedule[Schedule::Expert(a)];
      if(e<=previous || !active.count(e))bad("schedule order",a,e,e,previous);
      previous=e;routes+=got_schedule[Schedule::Count(a)];
      for(int q=0;q<got_schedule[Schedule::Count(a)];++q) {
        int route=got_schedule[Schedule::Row(a,q)];
        if(top[route/TopK][route%TopK]!=e)bad("schedule row",a,q,route,e);
      }
    }
    if(routes!=B*TopK)bad("schedule rows",0,0,routes,B*TopK);
    for(int j=0;j<H/TNdn;++j)if(counters[j])bad("ctr reset",0,j,double(counters[j]),0);
    for(int b=0;b<B;++b)for(int n=0;n<H;++n) {
      double want=expected[b*H+n],got=float(got_x[b*H+n]);
      if(std::fabs(got-want)>std::fabs(want)/32+2e-2)bad("x''",b,n,got,want);
    }
  }
  std::printf("{\"batch\":%d,\"launches\":2,\"errors\":%d,\"pass\":%s}\n",B,errors,errors?"false":"true");
  return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
