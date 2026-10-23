// SPDX-License-Identifier: BSD-3-Clause
// In-body serving calibration: device timer around the actual body, in place
// at a plan's residency R (MX4 prompt §5.6): 104*R CTAs run the body at once,
// each on its own weight / KV slice, and the fit of R is over the CTAs'
// median body time. Keys carry R (_r<R>) and, for a GEMM, its depth (_d<S>).
// xcore1000: the 100 MHz realtime counter (P-11) replaces clock64, whose rate
// follows DVFS; bodies run for microseconds, so its 10 ns step is below 1 %.
// Decode attention on xcore1000 is the simt body (attention_impl=simt).
#ifndef TILEMEGA_ATTENTION_IMPL
#define TILEMEGA_ATTENTION_IMPL 3
#endif
#include <tilemega/Target/Calibration.h>
#include <tilemega/Solver/BackendCostQuery.h>
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
#include <regex>
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
  __syncthreads();if(threadIdx.x==0)ns[blockIdx.x]=maca::GlobalTimerNs()-start;
}
template<int R>
__global__ void __launch_bounds__(128,R) TimeEmbed(int const* tokens,Element const* table,Element* output,
    int width,unsigned long long* cycles) {
  auto start=Begin(cycles);
  ServingEmbeddingTaskBody::RunRow(tokens,table,output,0,1,0,1,width,2);
  End(start,cycles);
}
template<int R>
__global__ void __launch_bounds__(128,R) TimeNorm(Element const* input,Element const* weight,
    Element* output,int width,unsigned long long* cycles) {
  __shared__ float warp_sums[4];auto start=Begin(cycles);
  ServingRMSNormTaskBody::RunRow(input,weight,output,0,1,0,width,1e-6f,warp_sums);
  End(start,cycles);
}
template<int R>
__global__ void __launch_bounds__(128,R) TimeArgmax(float const* values,int const* indices,
    int* tokens,int count,unsigned long long* cycles) {
  __shared__ ServingArgmaxReduceTaskBody::SharedStorage storage;
  auto start=Begin(cycles);
  ServingArgmaxReduceTaskBody::RunRow(values,indices,tokens,0,count,1,0,&storage);
  End(start,cycles);
}
template<int R>
__global__ void __launch_bounds__(128,R) TimeMerge(float const* partial,float const* lse,
    Element* context,int cap,int block,int past,unsigned long long* cycles) {
  auto start=Begin(cycles);
  AttentionMergeTaskBody<64,4,1>::Run(partial,lse,context,0,0,1,cap,block,past);
  End(start,cycles);
}
using X=arch::Xcore1000;
// GEMM bodies at every legal depth (ServingBF16ShapeLegal, the solver's
// domain rule) for five tile shapes. Stages is a real pipeline depth since
// MX-3; the fit of one depth is that depth's body, and the cost model takes
// it per depth (TaskPriceParts: "_d<S>"). Shapes: 16x128, 16x32, 32x128,
// 64x64, 64x128.
template<int M,int N,int S> using GemmAt=ServingGemmTaskBody<X,M,N,64,S>;
// GEMV (M <= 4) is a distinct body; its fits carry the "gemv_" prefix.
using Gemv=ServingGemvTaskBody<X,64,128>;
using NarrowGemv=ServingGemvTaskBody<X,16,128>;
/// Each CTA reads its own slice of A and B and writes its own output tile.
// Each timing kernel is compiled for R resident CTAs per AP, as the plan's
// kernels are (__launch_bounds__(threads, TILEMEGA_MIN_BLOCKS_PER_SM)).
template<class Body,int R>
__global__ void __launch_bounds__(128,R) TimeGemm(ServingGemmOperands operands,unsigned long long* cycles,
    std::size_t a_slice,std::size_t b_slice,std::size_t out_slice,int slices) {
  extern __shared__ char storage[];
  std::size_t const slice=blockIdx.x%unsigned(slices);
  operands.a+=slice*a_slice;operands.b+=slice*b_slice;operands.output+=slice*out_slice;
  auto start=Begin(cycles);
  Body::Run(operands,0,0,storage);End(start,cycles);
}
// Decode runs the SIMT body (attention_impl=simt), prefill the MMA body.
// Prefill at the KV tile a plan instantiates (ServingAttentionKvTile: 32 or
// 64); its fits carry "_kv<tile>".
template<int D,int Q,int S,bool Norm,int KV=64>
using AttentionBody=std::conditional_t<S==1,IndependentAttentionTaskBody<X,D,Q,Norm>,
    FusedAttentionTaskBody<X,D,Q,S,16,KV,Norm>>;
template<int D,int Q,int S,bool Norm,int R,int KV=64>
__global__ void __launch_bounds__(128,R) TimeAttention(ServingAttentionOperands operands,
    unsigned long long* cycles,int query_block,std::size_t kv_slice,int slices) {
  using Body=AttentionBody<D,Q,S,Norm,KV>;
  extern __shared__ __align__(16) unsigned char bytes[];
  std::size_t const slice=blockIdx.x%unsigned(slices);
  operands.key_cache+=slice*kv_slice;operands.value_cache+=slice*kv_slice;
  auto& storage=*reinterpret_cast<typename Body::SharedStorage*>(bytes);
  auto start=Begin(cycles);Body::Run(operands,storage,0,0,query_block,0);End(start,cycles);
}

// Decode attention through the MMA body (attention_impl=mma16, MX3-3):
// PrefillAttentionMma at Tokens=1, one 16-row query tile, one KV block.
template<int D,int Q,bool Norm>
using DecodeMmaBody=mx::PrefillAttentionMma<D,Q,1,16,(Q>=8?64:32),Norm>;
template<int D,int Q,bool Norm,int R>
__global__ void __launch_bounds__(128,R) TimeDecodeMma(ServingAttentionOperands operands,unsigned long long* cycles,
    std::size_t kv_slice,int slices) {
  using Body=DecodeMmaBody<D,Q,Norm>;
  extern __shared__ __align__(16) unsigned char bytes[];
  std::size_t const slice=blockIdx.x%unsigned(slices);
  operands.key_cache+=slice*kv_slice;operands.value_cache+=slice*kv_slice;
  auto& storage=*reinterpret_cast<typename Body::SharedStorage*>(bytes);
  auto start=Begin(cycles);Body::Run(operands,storage,0,0,0,0);End(start,cycles);
}

/// The median over repeats of the median CTA body time of one launch.
template<class Launch>
double Measure(Launch launch,unsigned long long* ns,int ctas,int repeats) {
  launch();Check(cudaDeviceSynchronize(),"warm serving body");
  std::vector<double> values;
  for(int i=0;i<std::max(3,repeats);++i) {
    launch();Check(cudaDeviceSynchronize(),"time serving body");
    std::vector<double> per(ns,ns+ctas);
    std::nth_element(per.begin(),per.begin()+ctas/2,per.end());
    values.push_back(per[ctas/2]);
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
  cudaDeviceProp properties{};Check(cudaGetDeviceProperties(&properties,options.device),"device properties");
  int const aps=properties.multiProcessorCount;
  int const most=4*aps;                                  // R <= 4
  auto* cycles=Managed<unsigned long long>(most);
  auto* element=Device<Element>(std::size_t(1088)*128*6);
  auto* output=Device<Element>(std::size_t(most)*64*128);  // one output tile per CTA
  auto* numbers=Device<float>(std::size_t(1088)*128*6);
  auto* indices=Device<int>(std::size_t(1088)*128*6);
  auto* tokens=Device<int>(1088);
  auto* lse=Device<float>(1088);
  // Weights and KV: 256 MiB of slices, so the CTAs of one launch read DRAM.
  std::size_t const pool=std::size_t(128)<<20;
  auto* weights=Device<Element>(pool);
  std::map<std::string,std::vector<Sample>> observations;
  int residency=1,grid=aps;
  auto suffix=[&]{return "_r"+std::to_string(residency);};
  auto record=[&](std::string const& name,double bytes,double flops,auto launch) {
    double ns=Measure(launch,cycles,grid,std::min(9,std::max(3,options.repeats)));
    observations[name+suffix()].push_back({bytes,flops,ns});
    log<<name+suffix()<<'\t'<<bytes<<'\t'<<flops<<'\t'<<ns<<'\n';
  };
  // R CTAs of this body must be resident on every AP at once.
  std::string why;
  auto fits=[&](auto kernel,std::size_t smem) {
    int blocks=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,smem),"occupancy");
    cudaFuncAttributes attributes{};Check(cudaFuncGetAttributes(&attributes,kernel),"kernel attributes");
    why="resident="+std::to_string(blocks)+" regs="+std::to_string(attributes.numRegs)+" static="+
        std::to_string(attributes.sharedSizeBytes)+" dynamic="+std::to_string(smem)+" local="+
        std::to_string(attributes.localSizeBytes);
    return blocks>=residency;
  };
  auto skip=[&](std::string const& name){log<<name+suffix()<<"\tSKIP not resident "<<why<<"\n";};
  log<<"kind\tbytes\tflops\tbody_ns\n";
  try {
   auto pass=[&](auto r_tag) {
    constexpr int R=decltype(r_tag)::value;
    residency=R;grid=aps*R;
    for(int width:{512,1024,2048,4096}) {
      record("embedding",4*width,0,[&]{TimeEmbed<R><<<grid,128>>>(tokens,element,output,width,cycles);});
      record("rmsnorm",6*width,5*width,[&]{TimeNorm<R><<<grid,128>>>(element,element,output,width,cycles);});
    }
    for(int count:{32,64,128,256,512,1024})
      record("argmax_reduce",8*count,2*count,[&]{TimeArgmax<R><<<grid,128>>>(numbers,indices,tokens,count,cycles);});
    for(int blocks:{2,4,8,16}) {
      int past=blocks*64-1;
      record("attention_merge",blocks*4*64*4,blocks*4*64*4,
        [&]{TimeMerge<R><<<grid,128>>>(numbers,lse,output,1088,64,past,cycles);});
    }
    // Every legal (shape, depth) and three reduction lengths: a fit on K at
    // one shape cannot separate bytes from MMA work.
    for(auto [name,op]:std::vector<std::pair<std::string,ServingEpilogueOp>>{
        {"gemm_store",ServingEpilogueOp::kStore},
        {"gemm_residual",ServingEpilogueOp::kResidual},
        {"gemm_swiglu",ServingEpilogueOp::kSwiGLU},
        {"gemm_argmax_partial",ServingEpilogueOp::kArgmaxPartial}}) {
      auto shape_depth=[&](auto m_tag,auto n_tag,auto s_tag) {
        constexpr int M=decltype(m_tag)::value,N=decltype(n_tag)::value,S=decltype(s_tag)::value;
        if constexpr (solver::ServingBF16ShapeLegal(M,N,64,S)) {
          using Body=GemmAt<M,N,S>;
          std::string const key=name+"_d"+std::to_string(S);
          Check(cudaFuncSetAttribute(TimeGemm<Body,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,
              Body::kSharedBytes),"opt in GEMM shared memory");
          if(!fits(TimeGemm<Body,R>,Body::kSharedBytes)){skip(key);return;}
          for(int k:{512,1024,2048}) {
            ServingGemmOperands p{};p.a=element;p.b=weights;p.residual=element;
            p.output=output;p.partial=numbers;p.argmax_value=numbers;
            p.argmax_index=indices;p.m=M;p.n=N;p.k_total=k;p.k_count=k;
            p.a_row_stride=k;p.b_row_stride=k;p.output_stride=N;p.epilogue=op;
            std::size_t const b_slice=std::size_t(N)*k;
            int const slices=int(std::min<std::size_t>(grid,pool/b_slice));
            record(key,2.*(M+N)*k+2*M*N,2.*M*N*k,[&]{
              TimeGemm<Body,R><<<grid,128,Body::kSharedBytes>>>(p,cycles,0,b_slice,std::size_t(M)*N,slices);
            });
          }
        }
      };
      auto depths=[&](auto m_tag,auto n_tag) {
        shape_depth(m_tag,n_tag,std::integral_constant<int,1>{});
        shape_depth(m_tag,n_tag,std::integral_constant<int,2>{});
        shape_depth(m_tag,n_tag,std::integral_constant<int,3>{});
        shape_depth(m_tag,n_tag,std::integral_constant<int,4>{});
        static_assert(solver::kServingGemmMaxDepth==4,"depths below cover the domain");
      };
      depths(std::integral_constant<int,16>{},std::integral_constant<int,128>{});
      depths(std::integral_constant<int,16>{},std::integral_constant<int,32>{});
      depths(std::integral_constant<int,32>{},std::integral_constant<int,128>{});
      depths(std::integral_constant<int,64>{},std::integral_constant<int,64>{});
      depths(std::integral_constant<int,64>{},std::integral_constant<int,128>{});
      // GEMV: rows 1 and 4, two column widths.
      for(int shape=0;shape<2;++shape)for(int k:{512,1024,2048})for(int rows:{1,4}) {
        int gn=shape==0?64:16;
        ServingGemmOperands q{};q.a=element;q.b=weights;q.residual=element;q.output=output;
        q.partial=numbers;q.argmax_value=numbers;q.argmax_index=indices;q.m=rows;q.n=gn;
        q.k_total=k;q.k_count=k;q.a_row_stride=k;q.b_row_stride=k;q.output_stride=gn;q.epilogue=op;
        std::size_t const b_slice=std::size_t(gn)*k;
        int const slices=int(std::min<std::size_t>(grid,pool/b_slice));
        record("gemv_"+name,2.*(rows+gn)*k+2*rows*gn,2.*rows*gn*k,[&]{
          if(shape==0)TimeGemm<Gemv,R><<<grid,128,Gemv::kSharedBytes>>>(q,cycles,0,b_slice,std::size_t(rows)*gn,slices);
          else TimeGemm<NarrowGemv,R><<<grid,128,NarrowGemv::kSharedBytes>>>(q,cycles,0,b_slice,std::size_t(rows)*gn,slices);
        });
      }
    }
    // Q and normalization select distinct measured bodies; a D-only fit
    // cannot represent the Q=3, Q=4 and Q=8 model paths.
    auto attention=[&](auto d_tag,auto q_tag,auto norm_tag) {
      constexpr int D=decltype(d_tag)::value,Q=decltype(q_tag)::value;
      constexpr bool Norm=decltype(norm_tag)::value;
      using Decode=AttentionBody<D,Q,1,Norm>;
      using Mma=DecodeMmaBody<D,Q,Norm>;
      std::string const tag="_d"+std::to_string(D)+"_q"+std::to_string(Q)+"_norm"+std::to_string(Norm);
      std::size_t const kv_slice=std::size_t(1088)*D;
      int const slices=int(std::min<std::size_t>(grid,pool/kv_slice));
      auto operands=[&](int phase,int past) {
        return ServingAttentionOperands{element,weights,weights,element,element,
            Norm?element:nullptr,Norm?element:nullptr,output,numbers,lse,
            1,1,1088,phase==1?past:0,256,1e-6f};
      };
      // Decode: the SIMT body and the mma16 body.
      Check(cudaFuncSetAttribute(TimeAttention<D,Q,1,Norm,R>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Decode::SharedStorage)),
          "opt in decode attention shared memory");
      Check(cudaFuncSetAttribute(TimeDecodeMma<D,Q,Norm,R>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Mma::SharedStorage)),
          "opt in mma16 decode attention shared memory");
      bool const decode_fits=fits(TimeAttention<D,Q,1,Norm,R>,sizeof(typename Decode::SharedStorage));
      if(!decode_fits)skip("fused_attention_decode"+tag);
      bool const mma_fits=fits(TimeDecodeMma<D,Q,Norm,R>,sizeof(typename Mma::SharedStorage));
      if(!mma_fits)skip("fused_attention_decode_mma16"+tag);
      for(int past:{1,64,128}) {
        auto p=operands(1,past);
        double bytes=4.*D*(past+1)+2.*Q*D+4.*D, flops=4.*Q*D*(past+2);
        if(decode_fits)
          record("fused_attention_decode"+tag,bytes,flops,[&] {
            TimeAttention<D,Q,1,Norm,R><<<grid,128,sizeof(typename Decode::SharedStorage)>>>(p,cycles,0,kv_slice,slices);
          });
        if(mma_fits)
          record("fused_attention_decode_mma16"+tag,bytes,flops,[&] {
            TimeDecodeMma<D,Q,Norm,R><<<grid,128,sizeof(typename Mma::SharedStorage)>>>(p,cycles,kv_slice,slices);
          });
      }
      // Prefill: both KV tiles.
      auto prefill=[&](auto kv_tag) {
        constexpr int KV=decltype(kv_tag)::value;
        using Prefill=AttentionBody<D,Q,64,Norm,KV>;
        std::string const key="fused_attention_prefill"+tag+"_kv"+std::to_string(KV);
        Check(cudaFuncSetAttribute(TimeAttention<D,Q,64,Norm,R,KV>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Prefill::SharedStorage)),
            "opt in prefill attention shared memory");
        if(!fits(TimeAttention<D,Q,64,Norm,R,KV>,sizeof(typename Prefill::SharedStorage))){skip(key);return;}
        for(int past:{1,64,128}) {
          int qb=past==1?0:past==64?1:3;
          int active=((qb+1)*16+Q-1)/Q;
          auto p=operands(64,past);
          record(key,4.*D*active+2.*16*D,4.*16*D*active,[&] {
            TimeAttention<D,Q,64,Norm,R,KV><<<grid,128,sizeof(typename Prefill::SharedStorage)>>>(p,cycles,qb,kv_slice,slices);
          });
        }
      };
      prefill(std::integral_constant<int,32>{});
      prefill(std::integral_constant<int,64>{});
    };
    attention(std::integral_constant<int,64>{},std::integral_constant<int,4>{},std::false_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,2>{},std::true_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,3>{},std::false_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,4>{},std::false_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,4>{},std::true_type{});
    attention(std::integral_constant<int,128>{},std::integral_constant<int,8>{},std::true_type{});
   };
   pass(std::integral_constant<int,1>{});
   pass(std::integral_constant<int,2>{});
   pass(std::integral_constant<int,3>{});
   pass(std::integral_constant<int,4>{});
    // Replace what this tool measures (its keys, old unsuffixed ones too);
    // fits measured elsewhere (gemm_combine, MX-3 in situ) stay.
    auto& serving=target.CalibrationFor("bf16").task_body.serving;
    for(auto it=serving.begin();it!=serving.end();) {
      bool ours=false;
      for(auto const& [name,samples]:observations) {
        std::string const base=std::regex_replace(name,std::regex("(_d[0-9]+)?(_kv[0-9]+)?_r[0-9]+$"),"");
        ours=ours || it->first==base || it->first.rfind(base+"_",0)==0;
      }
      it=ours?serving.erase(it):std::next(it);
    }
    for(auto const& [name,samples]:observations) {
      if(samples.size()<2)throw std::runtime_error("serving kind has fewer than two samples");
      serving[name]=Fit(samples);
    }
  } catch(...) {
    cudaFree(cycles);cudaFree(element);cudaFree(output);cudaFree(numbers);
    cudaFree(indices);cudaFree(tokens);cudaFree(lse);cudaFree(weights);throw;
  }
  Check(cudaFree(cycles),"free cycles");Check(cudaFree(element),"free input");
  Check(cudaFree(output),"free output");Check(cudaFree(numbers),"free floats");
  Check(cudaFree(indices),"free indices");Check(cudaFree(tokens),"free tokens");
  Check(cudaFree(lse),"free LSE");Check(cudaFree(weights),"free weights");
}
} // namespace tilemega::calib
