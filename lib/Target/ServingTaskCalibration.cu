// SPDX-License-Identifier: BSD-3-Clause
// In-body serving calibration: one CTA, device timer around the actual body.
// xcore1000: the 100 MHz realtime counter (P-11) replaces clock64, whose rate
// follows DVFS; bodies run for microseconds, so its 10 ns step is below 1 %.
// Decode attention on xcore1000 is the simt body (attention_impl=simt).
#ifndef TILEMEGA_ATTENTION_IMPL
#define TILEMEGA_ATTENTION_IMPL 3
#endif
#include <tilemega/Target/Calibration.h>
#include <tilemega/Codegen/tasks/ServingEmbeddingTaskBody.h>
#include <tilemega/Codegen/tasks/ServingRMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/ServingArgmaxReduceTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionMergeTaskBody.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/tasks/IndependentAttentionTaskBody.h>
#include <tilemega/Codegen/tasks/mx/PrefillAttentionMma.h>
#include <tilemega/Codegen/tasks/ServingGemvTaskBody.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

namespace tilemega::calib {
namespace {
using Element=tilemega::Bf16;
using namespace tilemega::codegen;
using tilemega::backend::ServingEpilogueOp;

void Check(cudaError_t code,char const* site) {
  if(code!=cudaSuccess)throw std::runtime_error(std::string(site)+": "+cudaGetErrorString(code));
}
template<class T> T* Managed(std::size_t n) {
  T* p=nullptr;Check(cudaMallocManaged(&p,n*sizeof(T)),"allocate serving sample");
  Check(cudaMemset(p,0,n*sizeof(T)),"initialize serving sample");return p;
}
template<class T> T* Device(std::size_t n) {
  T* p=nullptr;Check(cudaMalloc(&p,n*sizeof(T)),"allocate serving input");
  Check(cudaMemset(p,0,n*sizeof(T)),"initialize serving input");return p;
}
struct Sample {double bytes=0,flops=0,ns=0;};

__device__ unsigned long long Begin(unsigned long long* ns) {
  __syncthreads();return maca::GlobalTimerNs();
}
__device__ void End(unsigned long long start,unsigned long long* ns) {
  __syncthreads();if(threadIdx.x==0)*ns=maca::GlobalTimerNs()-start;
}
__global__ void TimeEmbed(int const* tokens,Element const* table,Element* output,
    int width,unsigned long long* cycles) {
  auto start=Begin(cycles);
  ServingEmbeddingTaskBody::RunRow(tokens,table,output,0,1,0,1,width,2);
  End(start,cycles);
}
__global__ void TimeNorm(Element const* input,Element const* weight,
    Element* output,int width,unsigned long long* cycles) {
  __shared__ float warp_sums[4];auto start=Begin(cycles);
  ServingRMSNormTaskBody::RunRow(input,weight,output,0,1,0,width,1e-6f,warp_sums);
  End(start,cycles);
}
__global__ void TimeArgmax(float const* values,int const* indices,
    int* tokens,int count,unsigned long long* cycles) {
  __shared__ ServingArgmaxReduceTaskBody::SharedStorage storage;
  auto start=Begin(cycles);
  ServingArgmaxReduceTaskBody::RunRow(values,indices,tokens,0,count,1,0,&storage);
  End(start,cycles);
}
__global__ void TimeMerge(float const* partial,float const* lse,
    Element* context,int cap,int block,int past,unsigned long long* cycles) {
  auto start=Begin(cycles);
  AttentionMergeTaskBody<64,4,1>::Run(partial,lse,context,0,0,1,cap,block,past);
  End(start,cycles);
}
using X=arch::Xcore1000;
// Depth 1, the serial mainloop. Since MX-3 Stages is a real pipeline depth,
// bounded by registers (ServingGemmDepthFits), and 16x128 or 32x128 cannot
// pipeline at all within that budget, so depth 2 no longer builds for them.
// The cost model carries depth through in-flight bytes
// (ServingGemmIterationsInFlight), so the fit is of the serial body and the
// pipeline is not counted twice.
using Gemm=ServingGemmTaskBody<X,16,128,64,1>;
using NarrowGemm=ServingGemmTaskBody<X,16,32,64,1>;
using TallerGemm=ServingGemmTaskBody<X,32,128,64,1>;
// GEMV (M <= 4) is a distinct body; its fits carry the "gemv_" prefix.
using Gemv=ServingGemvTaskBody<X,64,128>;
using NarrowGemv=ServingGemvTaskBody<X,16,128>;
template<class Body>
__global__ void TimeGemm(ServingGemmOperands operands,unsigned long long* cycles) {
  extern __shared__ char storage[];auto start=Begin(cycles);
  Body::Run(operands,0,0,storage);End(start,cycles);
}
// Decode runs the SIMT body (attention_impl=simt), prefill the MMA body.
template<int D,int Q,int S,bool Norm>
using AttentionBody=std::conditional_t<S==1,IndependentAttentionTaskBody<X,D,Q,Norm>,
    FusedAttentionTaskBody<X,D,Q,S,16,64,Norm>>;
template<int D,int Q,int S,bool Norm>
__global__ void TimeAttention(ServingAttentionOperands operands,
    unsigned long long* cycles,int query_block=0) {
  using Body=AttentionBody<D,Q,S,Norm>;
  extern __shared__ __align__(16) unsigned char bytes[];
  auto& storage=*reinterpret_cast<typename Body::SharedStorage*>(bytes);
  auto start=Begin(cycles);Body::Run(operands,storage,0,0,query_block,0);End(start,cycles);
}

// Decode attention through the MMA body (attention_impl=mma16, MX3-3):
// PrefillAttentionMma at Tokens=1, one 16-row query tile, one KV block.
template<int D,int Q,bool Norm>
using DecodeMmaBody=mx::PrefillAttentionMma<D,Q,1,16,(Q>=8?64:32),Norm>;
template<int D,int Q,bool Norm>
__global__ void TimeDecodeMma(ServingAttentionOperands operands,unsigned long long* cycles) {
  using Body=DecodeMmaBody<D,Q,Norm>;
  extern __shared__ __align__(16) unsigned char bytes[];
  auto& storage=*reinterpret_cast<typename Body::SharedStorage*>(bytes);
  auto start=Begin(cycles);Body::Run(operands,storage,0,0,0,0);End(start,cycles);
}

template<class Launch>
double Measure(Launch launch,unsigned long long* ns,int repeats) {
  launch();Check(cudaDeviceSynchronize(),"warm serving body");
  std::vector<double> values;
  for(int i=0;i<std::max(3,repeats);++i) {
    launch();Check(cudaDeviceSynchronize(),"time serving body");
    values.push_back(double(*ns));
  }
  std::sort(values.begin(),values.end());return values[values.size()/2];
}

// NNLS over the three declared features, by enumerating active sets. The
// features are normalized for conditioning before the 1-3 variable solve.
TargetSpec::TaskBodyCalibration::ServingFit Fit(std::vector<Sample> const& samples) {
  using FitType=TargetSpec::TaskBodyCalibration::ServingFit;
  if(samples.empty())throw std::invalid_argument("empty serving fit");
  std::array<double,3> scale{1,1000,1000000};
  double best=std::numeric_limits<double>::infinity();
  std::array<double,3> answer{};
  for(int mask=1;mask<8;++mask) {
    int cols[3],n=0;for(int j=0;j<3;++j)if(mask&(1<<j))cols[n++]=j;
    double a[3][4]{};
    for(auto const& s:samples) {
      double x[3]={1,s.bytes/scale[1],s.flops/scale[2]};
      for(int i=0;i<n;++i) {
        a[i][n]+=x[cols[i]]*s.ns;
        for(int j=0;j<n;++j)a[i][j]+=x[cols[i]]*x[cols[j]];
      }
    }
    bool singular=false;
    for(int j=0;j<n;++j) {
      int pivot=j;for(int i=j+1;i<n;++i)if(std::abs(a[i][j])>std::abs(a[pivot][j]))pivot=i;
      if(std::abs(a[pivot][j])<1e-14){singular=true;break;}
      if(pivot!=j)for(int k=j;k<=n;++k)std::swap(a[pivot][k],a[j][k]);
      double div=a[j][j];for(int k=j;k<=n;++k)a[j][k]/=div;
      for(int i=0;i<n;++i)if(i!=j) {
        double factor=a[i][j];for(int k=j;k<=n;++k)a[i][k]-=factor*a[j][k];
      }
    }
    if(singular)continue;
    std::array<double,3> c{};bool feasible=true;
    for(int i=0;i<n;++i){c[cols[i]]=a[i][n];if(c[cols[i]]<0)feasible=false;}
    if(!feasible)continue;
    double error=0;
    for(auto const& s:samples) {
      double predicted=c[0]+c[1]*s.bytes/scale[1]+c[2]*s.flops/scale[2];
      error+=(predicted-s.ns)*(predicted-s.ns);
    }
    if(error<best){best=error;answer=c;}
  }
  if(!std::isfinite(best))throw std::runtime_error("serving NNLS failed");
  FitType fit;fit.fixed_ns=answer[0];fit.byte_ns=answer[1]/scale[1];
  fit.flop_ns=answer[2]/scale[2];fit.samples=int(samples.size());
  std::vector<double> errors;
  for(auto const& s:samples)errors.push_back(std::abs(fit.fixed_ns+
      fit.byte_ns*s.bytes+fit.flop_ns*s.flops-s.ns)/std::max(s.ns,1.));
  std::sort(errors.begin(),errors.end());fit.median_relative_error=errors[errors.size()/2];
  return fit;
}
} // namespace

void MeasureServingTaskBodies(TargetSpec& target,Options const& options,std::ostream& log) {
  if(target.arch_tag!="xcore1000")
    throw std::invalid_argument("serving TaskBody calibration is built for xcore1000");
  Check(cudaSetDevice(options.device),"select calibration device");
  auto* cycles=Managed<unsigned long long>(1);
  auto* element=Device<Element>(std::size_t(1088)*128*6);
  auto* output=Device<Element>(std::size_t(1088)*128*6);
  auto* numbers=Device<float>(std::size_t(1088)*128*6);
  auto* indices=Device<int>(std::size_t(1088)*128*6);
  auto* tokens=Device<int>(1088);
  auto* lse=Device<float>(1088);
  std::map<std::string,std::vector<Sample>> observations;
  auto record=[&](std::string const& name,double bytes,double flops,auto launch) {
    double ns=Measure(launch,cycles,std::min(9,std::max(3,options.repeats)));
    observations[name].push_back({bytes,flops,ns});
    log<<name<<'\t'<<bytes<<'\t'<<flops<<'\t'<<ns<<'\n';
  };
  log<<"kind\tbytes\tflops\tbody_ns\n";
  try {
    for(int width:{512,1024,2048,4096}) {
      record("embedding",4*width,0,[&]{TimeEmbed<<<1,128>>>(tokens,element,output,width,cycles);});
      record("rmsnorm",6*width,5*width,[&]{TimeNorm<<<1,128>>>(element,element,output,width,cycles);});
    }
    for(int count:{32,64,128,256,512,1024})
      record("argmax_reduce",8*count,2*count,[&]{TimeArgmax<<<1,128>>>(numbers,indices,tokens,count,cycles);});
    for(int blocks:{2,4,8,16}) {
      int past=blocks*64-1;
      record("attention_merge",blocks*4*64*4,blocks*4*64*4,
        [&]{TimeMerge<<<1,128>>>(numbers,lse,output,1088,64,past,cycles);});
    }
    // Sweep three legal tile shapes as well as reduction length. A fit based
    // only on K at one shape cannot separate bytes from MMA work.
    for(auto [name,op]:std::vector<std::pair<std::string,ServingEpilogueOp>>{
        {"gemm_store",ServingEpilogueOp::kStore},
        {"gemm_residual",ServingEpilogueOp::kResidual},
        {"gemm_swiglu",ServingEpilogueOp::kSwiGLU},
        {"gemm_argmax_partial",ServingEpilogueOp::kArgmaxPartial}})
      for(int shape=0;shape<3;++shape)for(int k:{512,1024,2048}) {
        int m=shape==2?32:16,n=shape==1?32:128;
        ServingGemmOperands p{};p.a=element;p.b=element;p.residual=element;
        p.output=output;p.partial=numbers;p.argmax_value=numbers;
        p.argmax_index=indices;p.m=m;p.n=n;p.k_total=k;p.k_count=k;
        p.a_row_stride=k;p.b_row_stride=k;p.output_stride=n;p.epilogue=op;
        record(name,2.*(m+n)*k+2*m*n,2.*m*n*k,[&]{
          if(shape==0)TimeGemm<Gemm><<<1,128,Gemm::kSharedBytes>>>(p,cycles);
          else if(shape==1)TimeGemm<NarrowGemm><<<1,128,NarrowGemm::kSharedBytes>>>(p,cycles);
          else TimeGemm<TallerGemm><<<1,128,TallerGemm::kSharedBytes>>>(p,cycles);
        });
        // GEMV: rows 1 and 4, two column widths.
        if(shape<2)for(int rows:{1,4}) {
          int gn=shape==0?64:16;auto q=p;q.m=rows;q.n=gn;q.output_stride=gn;
          record("gemv_"+name,2.*(rows+gn)*k+2*rows*gn,2.*rows*gn*k,[&]{
            if(shape==0)TimeGemm<Gemv><<<1,128,Gemv::kSharedBytes>>>(q,cycles);
            else TimeGemm<NarrowGemv><<<1,128,NarrowGemv::kSharedBytes>>>(q,cycles);
          });
        }
      }
    // Q and normalization select distinct measured bodies; a D-only fit
    // cannot represent the Q=3, Q=4 and Q=8 model paths.
    auto attention=[&](auto d_tag,auto q_tag,auto norm_tag) {
      constexpr int D=decltype(d_tag)::value,Q=decltype(q_tag)::value;
      constexpr bool Norm=decltype(norm_tag)::value;
      using Decode=AttentionBody<D,Q,1,Norm>;
      using Prefill=AttentionBody<D,Q,64,Norm>;
      Check(cudaFuncSetAttribute(TimeAttention<D,Q,1,Norm>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Decode::SharedStorage)),
          "opt in decode attention shared memory");
      Check(cudaFuncSetAttribute(TimeDecodeMma<D,Q,Norm>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(typename DecodeMmaBody<D,Q,Norm>::SharedStorage)),
          "opt in mma16 decode attention shared memory");
      Check(cudaFuncSetAttribute(TimeAttention<D,Q,64,Norm>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Prefill::SharedStorage)),
          "opt in prefill attention shared memory");
      for(int past:{1,64,128})for(int phase:{1,64}) {
        int qb=phase==1?0:(past==1?0:past==64?1:3);
        int active=phase==1?past+1:((qb+1)*16+Q-1)/Q;
        int queries=phase==1?Q:16;
        ServingAttentionOperands p{element,element,element,element,element,
            Norm?element:nullptr,Norm?element:nullptr,output,numbers,lse,
            1,1,1088,phase==1?past:0,256,1e-6f};
        double bytes=4.*D*active+2.*queries*D+(phase==1?4.*D:0.);
        double flops=4.*queries*D*(active+(phase==1?1:0));
        std::string key=std::string("fused_attention_")+(phase==1?"decode":"prefill")+
            "_d"+std::to_string(D)+"_q"+std::to_string(Q)+"_norm"+std::to_string(Norm);
        record(key,bytes,flops,[&] {
          if(phase==1)TimeAttention<D,Q,1,Norm><<<1,128,sizeof(typename Decode::SharedStorage)>>>(p,cycles,qb);
          else TimeAttention<D,Q,64,Norm><<<1,128,sizeof(typename Prefill::SharedStorage)>>>(p,cycles,qb);
        });
        if(phase==1) {
          using Mma=DecodeMmaBody<D,Q,Norm>;
          record("fused_attention_decode_mma16_d"+std::to_string(D)+"_q"+std::to_string(Q)+
                 "_norm"+std::to_string(Norm),bytes,flops,[&] {
            TimeDecodeMma<D,Q,Norm><<<1,128,sizeof(typename Mma::SharedStorage)>>>(p,cycles);
          });
        }
      }
    };
    attention(std::integral_constant<int,64>{},std::integral_constant<int,4>{},std::false_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,2>{},std::true_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,3>{},std::false_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,4>{},std::false_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,4>{},std::true_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,8>{},std::true_type{});
    auto& serving=target.CalibrationFor("bf16").task_body.serving;
    serving.clear();for(auto const& [name,samples]:observations) {
      if(samples.size()<2)throw std::runtime_error("serving kind has fewer than two samples");
      serving[name]=Fit(samples);
    }
  } catch(...) {
    cudaFree(cycles);cudaFree(element);cudaFree(output);cudaFree(numbers);
    cudaFree(indices);cudaFree(tokens);cudaFree(lse);throw;
  }
  Check(cudaFree(cycles),"free cycles");Check(cudaFree(element),"free input");
  Check(cudaFree(output),"free output");Check(cudaFree(numbers),"free floats");
  Check(cudaFree(indices),"free indices");Check(cudaFree(tokens),"free tokens");
  Check(cudaFree(lse),"free LSE");
}
} // namespace tilemega::calib
