// SPDX-License-Identifier: BSD-3-Clause
// D2 attention check (prompt §5): fp64 reference over B, D, Q, norm, S, past
// (64, block/chunk edges +-1, 1086) with position-encoded K/V; the new cache
// row must equal the reference rotation exactly, and filling rows
// [past+1, Capacity) with +-1e4 must leave every output bit unchanged, as
// must preissuing every K/V slot of each CTA's first task (D6). The cache is
// the stacked [L][2][groups][Capacity][D] layout read through 4D maps at
// layer 1; layer 0 holds +-1e4 so a wrong layer coordinate is visible (D7).
#ifndef TEST_BATCH
#define TEST_BATCH 1
#endif
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
constexpr int B=TEST_BATCH,kCap=1088;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
float R(float x) {return float(BF(x));}

template<class A,int Prefix>
__global__ __launch_bounds__(160) void evaluate(Sm90DenseAttentionOperands p) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  if constexpr(Prefix>0)if(threadIdx.x==128 && int(blockIdx.x)<A::kTasks)
    A::template Preissue<Prefix>(p.key_map,p.value_map,p.layer,p.past,shared,state);
  A::template Run<Prefix>(p,shared);
  state.Invalidate();
}

// D10b: chunk pairs merge through the leader's DSMEM inbox (cluster 2).
template<class A>
__global__ __launch_bounds__(160) void evaluate_cluster(Sm90DenseAttentionOperands p) {
  using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  auto& cluster=Sm90DenseClusterState::Of(shared);
  if(threadIdx.x==0) {
    // S - 1 = 1 peer chunk arrives on the leader's inbox.
    Async::Init(&cluster.inbox,1);cluster.inbox_generation=0;Async::InitFence();
  }
  __syncthreads();
  Async::ClusterSync();
  A::template Run<0>(p,shared);
  Async::ClusterSync();
  state.Invalidate();
}

// Host mirror of the device rotation (same BF16 roundings). `inverse` is
// supplied for a row norm, whose reduction spans the whole projected row.
template<int D,int Norm>
void Rotate(BF const* x,BF const* w,BF const* cos,BF const* sin,float eps,float* out,
            float inverse=0.0f) {
  float inv=1;
  if(Norm==1) {
    float sq=0;for(int d=0;d<D;++d)sq+=float(x[d])*float(x[d]);
    inv=1.0f/std::sqrt(sq/D+eps);
  }else if(Norm==2)inv=inverse;
  for(int d=0;d<D;++d) {
    float first=float(x[d]),second=float(x[(d+D/2)%D]);
    if(Norm) {
      first=R(R(first*inv)*float(w[d]));second=R(R(second*inv)*float(w[(d+D/2)%D]));
    }
    float left=R(first*float(cos[d])),right=R((d<D/2?-second:second)*float(sin[d]));
    out[d]=R(left+right);
  }
}

template<int Hkv,int D,int Q,int Norm,int S,int Keys,int Slots,bool Stream=false,bool Dsmem=false>
int run(int past,bool poison,std::vector<BF>* bits=nullptr) {
  using A=Sm90DenseAttentionBody<B,Hkv,D,Q,Norm,S,Keys,Slots,0,kCap,Dsmem>;
  constexpr int prefix=Stream?A::kMaxPreissue:0;
  int groups=B*Hkv;
  std::vector<BF> qkv(std::size_t(groups)*(Q+2)*D),key(std::size_t(groups)*kCap*D),value(key.size());
  constexpr int kNormQ=Norm==2?Hkv*Q*D:D,kNormK=Norm==2?Hkv*D:D;
  std::vector<BF> cos(kCap*D),sin(kCap*D),qn(kNormQ),kn(kNormK);
  for(std::size_t i=0;i<qkv.size();++i)qkv[i]=BF(float(int((i*7919)%61)-30)/32.0f);
  for(int g=0;g<groups;++g)for(int t=0;t<kCap;++t)for(int d=0;d<D;++d) {
    std::size_t i=(std::size_t(g)*kCap+t)*D+d;
    bool history=t<past;
    float k=float(int((t*31+d*7+g*13)%23)-11)/16.0f,v=float(int((t*17+d*5+g*3)%19)-9)/8.0f;
    key[i]=BF(history?k:poison?((t+d)%2?1e4f:-1e4f):0.0f);
    value[i]=BF(history?v:poison?((t+d)%2?-1e4f:1e4f):0.0f);
  }
  for(int t=0;t<kCap;++t)for(int d=0;d<D;++d) {
    double angle=t/std::pow(10000.0,double(2*(d%(D/2)))/D);
    cos[t*D+d]=BF(float(std::cos(angle)));sin[t*D+d]=BF(float(std::sin(angle)));
  }
  for(int d=0;d<kNormQ;++d)qn[d]=BF(1.0f+(d%5)/8.0f);
  for(int d=0;d<kNormK;++d)kn[d]=BF(1.0f-(d%3)/8.0f);
  auto device=[](auto const& host) {
    using T=typename std::decay_t<decltype(host)>::value_type;T* p;
    check(cudaMalloc(&p,host.size()*sizeof(T)));
    check(cudaMemcpy(p,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
  };
  constexpr int kLayers=2,kLayer=1;
  std::size_t view=key.size();
  std::vector<BF> stacked(kLayers*2*view);
  for(std::size_t i=0;i<stacked.size();++i)stacked[i]=BF(i%2?1e4f:-1e4f);
  std::copy(key.begin(),key.end(),stacked.begin()+(2*kLayer)*view);
  std::copy(value.begin(),value.end(),stacked.begin()+(2*kLayer+1)*view);
  BF *dqkv=device(qkv),*cache=device(stacked),*dc=device(cos),*ds=device(sin),*dqn=device(qn),*dkn=device(kn);
  BF *dk=cache+(2*kLayer)*view,*dv=cache+(2*kLayer+1)*view;
  BF* out;float *partial,*lse;std::uint64_t* arrivals;CUtensorMap* maps;
  check(cudaMalloc(&out,groups*Q*D*2));check(cudaMalloc(&partial,std::size_t(groups)*S*Q*D*4));
  check(cudaMalloc(&lse,groups*S*Q*4));check(cudaMalloc(&arrivals,groups*8));
  check(cudaMemset(arrivals,0,groups*8));check(cudaMalloc(&maps,256));
  CUtensorMap host[2];
  for(int i=0;i<2;++i)tilemega::codegen::executor::EncodeTiledMap<4>(host[i],cache+i*view,
      {{D,kCap,std::uint64_t(groups),kLayers},{2ull*D,2ull*kCap*D,2ull*2*view},{64,Keys,1,1}});
  check(cudaMemcpy(maps,host,sizeof host,cudaMemcpyHostToDevice));
  Sm90DenseAttentionOperands p{maps,maps+1,dqkv,dk,dv,out,dc,ds,dqn,dkn,partial,lse,arrivals,past,1e-6f,kLayer};
  if constexpr(Dsmem) {
    static_assert(S==2 && A::kTasks<=132,"one wave of chunk pairs");
    check(cudaFuncSetAttribute(evaluate_cluster<A>,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
    cudaLaunchConfig_t config{};config.gridDim=dim3(A::kTasks);config.blockDim=dim3(160);
    config.dynamicSmemBytes=A::kSharedBytes;
    cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x=2;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
    config.attrs=&attribute;config.numAttrs=1;
    check(cudaLaunchKernelEx(&config,evaluate_cluster<A>,p));
  }else {
    check(cudaFuncSetAttribute(evaluate<A,prefix>,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
    evaluate<A,prefix><<<std::min(132,A::kTasks),160,A::kSharedBytes>>>(p);
  }
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<BF> got(groups*Q*D),cache_k(key.size()),cache_v(value.size());
  check(cudaMemcpy(got.data(),out,got.size()*2,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(cache_k.data(),dk,key.size()*2,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(cache_v.data(),dv,value.size()*2,cudaMemcpyDeviceToHost));
  int errors=0;
  auto bad=[&](char const* what,int g,int q,int d,double x,double y) {
    if(errors<3)std::fprintf(stderr,"B%d D%d Q%d S%d past%d%s %s g%d q%d d%d got %g want %g\n",
        B,D,Q,S,past,poison?" poison":"",what,g,q,d,x,y);
    ++errors;
  };
  float const scale=1.0f/std::sqrt(float(D));
  for(int g=0;g<groups;++g) {
    BF const* row=qkv.data()+std::size_t(g)*(Q+2)*D;
    std::vector<float> knew(D),qrot(D);
    float inv_q=0,inv_k=0;
    if(Norm==2) {
      // The row spans every group of this token (bg = b Hkv + head group).
      BF const* token=qkv.data()+std::size_t(g-g%Hkv)*(Q+2)*D;
      double sq=0,sk=0;
      for(int h=0;h<Hkv;++h) {
        for(int i=0;i<Q*D;++i) {float v=float(token[std::size_t(h)*(Q+2)*D+i]);sq+=double(v)*v;}
        for(int i=0;i<D;++i) {float v=float(token[std::size_t(h)*(Q+2)*D+Q*D+i]);sk+=double(v)*v;}
      }
      inv_q=1.0f/std::sqrt(float(sq/(Hkv*Q*D))+1e-6f);
      inv_k=1.0f/std::sqrt(float(sk/(Hkv*D))+1e-6f);
    }
    Rotate<D,Norm>(row+Q*D,kn.data()+(Norm==2?(g%Hkv)*D:0),cos.data()+past*D,sin.data()+past*D,1e-6f,knew.data(),inv_k);
    for(int d=0;d<D;++d) {
      std::size_t i=(std::size_t(g)*kCap+past)*D+d;
      if(float(cache_k[i])!=knew[d])bad("new key",g,0,d,float(cache_k[i]),knew[d]);
      if(float(cache_v[i])!=float(row[(Q+1)*D+d]))bad("new value",g,0,d,float(cache_v[i]),float(row[(Q+1)*D+d]));
    }
    for(int q=0;q<Q;++q) {
      Rotate<D,Norm>(row+q*D,qn.data()+(Norm==2?((g%Hkv)*Q+q)*D:0),cos.data()+past*D,sin.data()+past*D,1e-6f,qrot.data(),inv_q);
      std::vector<double> score(past+1);double m=-1e300;
      for(int t=0;t<=past;++t) {
        double s=0;
        for(int d=0;d<D;++d)s+=double(qrot[d])*(t<past?double(key[(std::size_t(g)*kCap+t)*D+d]):knew[d]);
        score[t]=s*scale;m=std::max(m,score[t]);
      }
      double l=0;for(auto& s:score) {s=std::exp(s-m);l+=s;}
      double peak=0;std::vector<double> o(D,0);
      for(int d=0;d<D;++d) {
        for(int t=0;t<=past;++t)
          o[d]+=score[t]*(t<past?double(value[(std::size_t(g)*kCap+t)*D+d]):double(row[(Q+1)*D+d]));
        o[d]/=l;peak=std::max(peak,std::fabs(o[d]));
      }
      for(int d=0;d<D;++d) {
        double x=float(got[(std::size_t(g)*Q+q)*D+d]);
        // BF16 probabilities and output: a few BF16 ulps of the head's scale.
        if(std::fabs(x-o[d])>peak/48+1e-3)bad("output",g,q,d,x,o[d]);
      }
    }
  }
  if(bits)*bits=got;
  for(void* ptr:{(void*)dqkv,(void*)cache,(void*)dc,(void*)ds,(void*)dqn,(void*)dkn,(void*)out,
      (void*)partial,(void*)lse,(void*)arrivals,(void*)maps})check(cudaFree(ptr));
  return errors;
}

template<int Hkv,int D,int Q,int Norm,int S,int Keys,int Slots>
int sweep() {
  int errors=0;
  for(int past:{64,Keys-1,Keys,Keys+1,575,1023,1024,1086}) {
    std::vector<BF> clean,poisoned,streamed;
    errors+=run<Hkv,D,Q,Norm,S,Keys,Slots>(past,false,&clean);
    errors+=run<Hkv,D,Q,Norm,S,Keys,Slots>(past,true,&poisoned);
    errors+=run<Hkv,D,Q,Norm,S,Keys,Slots,true>(past,true,&streamed);
    if(std::memcmp(clean.data(),poisoned.data(),clean.size()*2)) {
      std::fprintf(stderr,"B%d D%d Q%d S%d past%d: poisoned rows changed the output\n",B,D,Q,S,past);++errors;
    }
    if(std::memcmp(clean.data(),streamed.data(),clean.size()*2)) {
      std::fprintf(stderr,"B%d D%d Q%d S%d past%d: preissue changed the output\n",B,D,Q,S,past);++errors;
    }
  }
  return errors;
}

// The DSMEM merge against the LA merge of the same chunks: equal outputs.
template<int Hkv,int D,int Q,int Norm,int Keys,int Slots>
int dsmem() {
  int errors=0;
  for(int past:{64,575,1086}) {
    std::vector<BF> la,cluster;
    errors+=run<Hkv,D,Q,Norm,2,Keys,Slots>(past,true,&la);
    errors+=run<Hkv,D,Q,Norm,2,Keys,Slots,false,true>(past,true,&cluster);
    if(std::memcmp(la.data(),cluster.data(),la.size()*2)) {
      std::fprintf(stderr,"B%d D%d Q%d past%d: DSMEM merge differs from the LA merge\n",B,D,Q,past);++errors;
    }
  }
  return errors;
}

int main() {
  try {
    check(cudaSetDevice(0));int errors=0,cases=0;
    auto add=[&](int e) {errors+=e;++cases;};
    constexpr int S0=B==1?16:B==2?8:B==4?4:B==8?2:1;
    // Dense (D, Q, norm) tuples, S0 and S0/2 (or 1), and the MoE tuple Hkv 4, Q 8.
    add(sweep<8,128,2,1,S0,64,4>());add(sweep<8,128,2,1,(S0>1?S0/2:1),64,2>());
    add(sweep<8,64,4,0,S0,64,3>());add(sweep<8,64,4,0,S0,128,2>());
    add(sweep<8,128,3,0,S0,64,2>());add(sweep<8,128,4,0,S0,64,4>());
    add(sweep<4,128,8,1,S0,64,2>());add(sweep<4,128,8,1,(S0>1?S0*2:2),64,3>());
    // MR5: one chunk (no merge) and the widest merges.
    add(sweep<4,128,8,1,1,64,4>());add(sweep<8,128,2,1,1,64,4>());
    add(sweep<4,128,8,1,(B<=2?32:4),64,2>());add(sweep<8,128,2,1,(B<=2?16:2),64,2>());
    if constexpr(B*4*2<=132)add(dsmem<4,128,8,1,64,2>());
    if constexpr(B*8*2<=132)add(dsmem<8,128,2,1,64,2>());
    // A3: OLMoE is MHA with a row norm over the whole projected row.
    if constexpr(B<=8)add(sweep<16,128,1,2,(B<=4?2:1),64,2>());
    if constexpr(B<=4)add(sweep<16,128,1,2,1,64,4>());
    std::printf("{\"batch\":%d,\"sweeps\":%d,\"errors\":%d,\"pass\":%s}\n",B,cases,errors,errors?"false":"true");
    return errors?1:0;
  }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
