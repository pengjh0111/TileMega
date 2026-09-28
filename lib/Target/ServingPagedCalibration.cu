// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#include <tilemega/Target/Calibration.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <tilemega/Codegen/tasks/PagedAttentionTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::calib {
namespace {
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using DeviceArch=std::conditional_t<std::is_void_v<arch::CurrentArch>,
    arch::Sm80,arch::CurrentArch>;
template<int N,int K>struct Shape {static constexpr int n=N,k=K;};
void Check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
constexpr int kTaskRepeats=32;
template<int N,int K>
using Body=PagedGemmTaskBody<DeviceArch,16,N,K,16384,5>;

template<int N,int K>
__global__ void TimePaged(ServingGemmOperands input) {
  using G=Body<N,K>;using Ring=typename G::Ring;
  extern __shared__ __align__(1024) char storage[];
  constexpr int work=1024;
  constexpr int scratch=G::kActivationBytes>G::kScratchBytes?
      G::kActivationBytes:G::kScratchBytes;
  constexpr int pool=(work+scratch+1023)/1024*1024;
  Ring ring{reinterpret_cast<typename Ring::Slot*>(storage),storage+pool};
  ring.Initialize();
  ServingGemmOperands p=input;
  p.output+=std::size_t(blockIdx.x)*16*N;
  if(p.residual)p.residual+=std::size_t(blockIdx.x)*16*N;
  if(p.argmax_value)p.argmax_value+=std::size_t(blockIdx.x)*16;
  if(p.argmax_index)p.argmax_index+=std::size_t(blockIdx.x)*16;
  std::uint64_t sequence=0;
  for(int repeat=0;repeat<kTaskRepeats;++repeat) {
    if(executor::IsCompute()) {
      G::Run(p,0,0,ring,sequence,storage+work);
      executor::ComputeSync();
    }else G::Load(p,0,ring,sequence);
  }
}

template<int N,int K>
double MeasurePaged(TargetSpec const& target,int iterations,int repeats,
                    backend::ServingEpilogueOp op) {
  using G=Body<N,K>;
  constexpr int work=1024;
  constexpr int scratch=G::kActivationBytes>G::kScratchBytes?
      G::kActivationBytes:G::kScratchBytes;
  constexpr int pool=(work+scratch+1023)/1024*1024;
  constexpr int shared=pool+5*16384;
  int sms=target.res.num_sms;
  Element *a=nullptr,*b=nullptr,*out=nullptr;
  float* argmax_values=nullptr;int* argmax_indices=nullptr;
  std::size_t k=std::size_t(iterations)*K;
  Check(cudaMalloc(&a,16*k*2));
  Check(cudaMalloc(&b,N*k*2));
  Check(cudaMalloc(&out,std::size_t(sms)*16*N*2));
  Check(cudaMalloc(&argmax_values,std::size_t(sms)*16*sizeof(float)));
  Check(cudaMalloc(&argmax_indices,std::size_t(sms)*16*sizeof(int)));
  Check(cudaMemset(a,0,16*k*2));Check(cudaMemset(b,0,N*k*2));
  Check(cudaMemset(out,0,std::size_t(sms)*16*N*2));
  ServingGemmOperands p;p.a=a;p.weight_base=b;p.b=b;p.output=out;
  p.m=16;p.n=N;p.k_total=int(k);p.k_total_full=int(k);
  p.k_count=int(k);p.output_stride=N;p.a_row_stride=int(k);
  p.partial_stride=N;p.epilogue=op;
  p.residual=op==backend::ServingEpilogueOp::kResidual?out:nullptr;
  p.argmax_value=op==backend::ServingEpilogueOp::kArgmaxPartial?
      argmax_values:nullptr;
  p.argmax_index=op==backend::ServingEpilogueOp::kArgmaxPartial?
      argmax_indices:nullptr;
  if(op==backend::ServingEpilogueOp::kSwiGLU)p.output_stride=N/2;
  Check(cudaFuncSetAttribute(TimePaged<N,K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
  std::vector<double> samples;
  for(int rep=-1;rep<repeats;++rep) {
    Check(cudaEventRecord(start));
    TimePaged<N,K><<<sms,160,shared>>>(p);
    Check(cudaEventRecord(stop));Check(cudaEventSynchronize(stop));
    if(rep>=0) {
      float ms=0;Check(cudaEventElapsedTime(&ms,start,stop));
      samples.push_back(double(ms)*1e6/kTaskRepeats);
    }
  }
  Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));
  std::sort(samples.begin(),samples.end());
  Check(cudaFree(a));Check(cudaFree(b));Check(cudaFree(out));
  Check(cudaFree(argmax_values));Check(cudaFree(argmax_indices));
  return samples[samples.size()/2];
}

template<int N,int K>
TargetSpec::TaskBodyCalibration::PagedFit FitPaged(TargetSpec const& target,
    int repeats,backend::ServingEpilogueOp op,std::ostream& log,
    std::string const& name) {
  std::array<double,3> x{2,8,16},y{};
  for(int i=0;i<3;++i) {
    y[i]=MeasurePaged<N,K>(target,int(x[i]),repeats,op);
    log<<name<<'\t'<<x[i]<<'\t'<<y[i]<<'\n'<<std::flush;
  }
  double sx=0,sy=0,sxx=0,sxy=0;
  for(int i=0;i<3;++i){sx+=x[i];sy+=y[i];sxx+=x[i]*x[i];sxy+=x[i]*y[i];}
  double slope=std::max(0.0,(3*sxy-sx*sy)/(3*sxx-sx*sx));
  double intercept=std::max(0.0,(sy-slope*sx)/3);
  std::array<double,3> errors{};
  for(int i=0;i<3;++i)
    errors[i]=std::abs(intercept+slope*x[i]-y[i])/std::max(1.,y[i]);
  std::sort(errors.begin(),errors.end());
  return {intercept,slope,errors[1]};
}

template<bool RowMajor>
__global__ void TimePagedLoader(char const* source,int pages_per_cta) {
  using Ring=executor::PageRing<16384,5,DeviceArch,false>;
  extern __shared__ __align__(1024) char storage[];
  Ring ring{reinterpret_cast<Ring::Slot*>(storage),storage+1024};
  ring.Initialize();std::uint64_t sequence=0;
  for(int page=0;page<pages_per_cta;++page,++sequence) {
    if(executor::IsCompute()) {
      ring.AwaitFull(sequence);ring.Release(sequence);
    }else {
      ring.AcquireEmpty(sequence);
      std::size_t const tile=std::size_t(blockIdx.x)*pages_per_cta+page;
      auto* dst=ring.Page(sequence);
      for(int v=executor::LoaderLane();v<16384/16;v+=32) {
        std::size_t offset=tile*16384+v*16;
        if constexpr(RowMajor) {
          // Same 512 MiB and CTA count as the contiguous control. Each stage
          // consists of 128 row segments of 128 B, spaced by full K=8192.
          std::size_t n_tile=tile/128,k_tile=tile%128;
          std::size_t row=v/8,k=v%8;
          offset=((n_tile*128+row)*8192+k_tile*64+k*8)*2;
        }
        Ring::Copy::Copy16(dst+v*16,source+offset);
      }
      ring.PublishCopies(sequence);
    }
  }
}

double MeasureLoader(TargetSpec const& target,int ctas,int repeats,bool row,
                     std::ostream& log) {
  int pages=512*1024*1024/(ctas*16384);
  std::size_t bytes=std::size_t(ctas)*pages*16384;
  char* source=nullptr;Check(cudaMalloc(&source,bytes));
  Check(cudaMemset(source,1,bytes));
  cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
  Check(cudaFuncSetAttribute(TimePagedLoader<false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,1024+5*16384));
  Check(cudaFuncSetAttribute(TimePagedLoader<true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,1024+5*16384));
  std::vector<double> rates;
  for(int rep=-1;rep<repeats;++rep) {
    Check(cudaEventRecord(start));
    if(row)TimePagedLoader<true><<<ctas,160,1024+5*16384>>>(source,pages);
    else TimePagedLoader<false><<<ctas,160,1024+5*16384>>>(source,pages);
    Check(cudaEventRecord(stop));Check(cudaEventSynchronize(stop));
    float ms=0;Check(cudaEventElapsedTime(&ms,start,stop));
    if(rep>=0)rates.push_back(double(bytes)/ms/1e6);
  }
  std::sort(rates.begin(),rates.end());
  Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));Check(cudaFree(source));
  double rate=rates[rates.size()/2];
  log<<"loader_"<<(row?"row":"tile")<<'\t'<<ctas<<'\t'<<rate<<'\n'<<std::flush;
  return rate;
}

template<int D,int Q>
using AttentionBody=PagedAttentionTaskBody<DeviceArch,D,Q,false,16384,5>;
template<int D,int Q>
__global__ void TimePagedAttention(ServingAttentionOperands input) {
  using Body=AttentionBody<D,Q>;using Ring=typename Body::Ring;
  extern __shared__ __align__(1024) char storage[];
  // The first KiB contains PageRing mbarriers. Attention scratch must begin
  // after it; overlapping the two makes the calibration kernel deadlock.
  constexpr int workspace=1024+(sizeof(typename Body::SharedStorage)+1023)/1024*1024;
  Ring ring{reinterpret_cast<typename Ring::Slot*>(storage),storage+workspace};
  ring.Initialize();std::uint64_t sequence=0;
  for(int repeat=0;repeat<kTaskRepeats;++repeat) {
    if(executor::IsCompute()) {
      Body::Run(input,blockIdx.x,0,0,ring,sequence,
          *reinterpret_cast<typename Body::SharedStorage*>(storage+1024));
      executor::ComputeSync();
    }else Body::Load(input,blockIdx.x,0,0,ring,sequence);
  }
}
template<int D,int Q>
double MeasureAttention(TargetSpec const& target,int extent,int past,int repeats) {
  using Body=AttentionBody<D,Q>;
  constexpr int cap=1088;
  constexpr int shared=1024+(sizeof(typename Body::SharedStorage)+1023)/1024*1024+5*16384;
  int sms=target.res.num_sms,cmax=(cap+extent-1)/extent;
  Element *qkv=nullptr,*key=nullptr,*value=nullptr,*cos=nullptr,*sin=nullptr,*context=nullptr;
  float *partial=nullptr,*lse=nullptr;
  auto alloc=[&](Element*& ptr,std::size_t elements) {
    Check(cudaMalloc(&ptr,elements*sizeof(Element)));
    Check(cudaMemset(ptr,0,elements*sizeof(Element)));
  };
  alloc(qkv,std::size_t(sms)*(Q+2)*D);
  alloc(key,std::size_t(sms)*cap*D);alloc(value,std::size_t(sms)*cap*D);
  alloc(cos,std::size_t(cap)*D);alloc(sin,std::size_t(cap)*D);
  alloc(context,std::size_t(sms)*Q*D);
  Check(cudaMalloc(&partial,std::size_t(sms)*cmax*Q*D*sizeof(float)));
  Check(cudaMalloc(&lse,std::size_t(sms)*cmax*Q*sizeof(float)));
  ServingAttentionOperands operands{qkv,key,value,cos,sin,nullptr,nullptr,
      context,partial,lse,sms,1,cap,past,extent,1e-6f};
  Check(cudaFuncSetAttribute(TimePagedAttention<D,Q>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
  std::vector<double> samples;
  for(int rep=-1;rep<repeats;++rep) {
    Check(cudaEventRecord(start));
    TimePagedAttention<D,Q><<<sms,160,shared>>>(operands);
    Check(cudaEventRecord(stop));Check(cudaEventSynchronize(stop));
    if(rep>=0) {
      float ms=0;Check(cudaEventElapsedTime(&ms,start,stop));
      samples.push_back(double(ms)*1e6/kTaskRepeats);
    }
  }
  Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));
  std::sort(samples.begin(),samples.end());
  Check(cudaFree(qkv));Check(cudaFree(key));Check(cudaFree(value));
  Check(cudaFree(cos));Check(cudaFree(sin));Check(cudaFree(context));
  Check(cudaFree(partial));Check(cudaFree(lse));
  return samples[samples.size()/2];
}
template<int D,int Q>
TargetSpec::TaskBodyCalibration::PagedFit FitAttention(TargetSpec const& target,
    int repeats,std::ostream& log) {
  // Two histories at each fixed block extent identify launch cost and the
  // marginal page cost without charging the empty blocks of a static plan.
  std::array<int,2> extents{64,256};
  double sx=0,sy=0,sxx=0,sxy=0;int points=0;
  for(int extent:extents)for(int past:{extent/2,extent-1}) {
    double ns=MeasureAttention<D,Q>(target,extent,past,repeats);
    int rows_per_page=16384/(4*D),warp_extent=((past+1+63)/64)*16;
    int warps_per_page=warp_extent<rows_per_page?
        std::min(4,rows_per_page/warp_extent):1;
    double pages=((warp_extent+rows_per_page-1)/rows_per_page)*
        (4/warps_per_page);
    log<<"attention_decode_d"<<D<<'\t'<<extent<<'\t'<<past<<'\t'
       <<pages<<'\t'<<ns<<'\n'<<std::flush;
    sx+=pages;sy+=ns;sxx+=pages*pages;sxy+=pages*ns;++points;
  }
  double denominator=points*sxx-sx*sx;
  double slope=denominator>0?std::max(0.0,(points*sxy-sx*sy)/denominator):0;
  double fixed=std::max(0.0,(sy-slope*sx)/points);
  return {fixed,slope,0};
}
}

void MeasureServingPagedBodies(TargetSpec& target,Options const& options,
                               std::ostream& log) {
  Check(cudaSetDevice(options.device));
  auto& fit=target.calib_bf16.task_body;
  auto measure=[&](auto tag,auto op,char const* name){
    constexpr int N=decltype(tag)::n,K=decltype(tag)::k;
    std::string key=std::string(name)+"_n"+std::to_string(N)+"_k"+std::to_string(K);
    fit.serving_paged[key]=FitPaged<N,K>(target,std::max(3,options.repeats/10),
        op,log,key);
  };
  for(auto op:{backend::ServingEpilogueOp::kStore,
               backend::ServingEpilogueOp::kResidual,
               backend::ServingEpilogueOp::kSwiGLU,
               backend::ServingEpilogueOp::kArgmaxPartial}) {
    char const* name=op==backend::ServingEpilogueOp::kStore?"gemm_store":
        op==backend::ServingEpilogueOp::kResidual?"gemm_residual":
        op==backend::ServingEpilogueOp::kSwiGLU?"gemm_swiglu":
        "gemm_argmax_partial";
    measure(Shape<32,64>{},op,name);
    measure(Shape<64,64>{},op,name);
    measure(Shape<128,64>{},op,name);
    measure(Shape<64,128>{},op,name);
  }
  fit.serving_paged_loader_gbps_per_sm=MeasureLoader(target,1,3,false,log);
  fit.serving_paged_loader_gbps=MeasureLoader(target,target.res.num_sms,3,false,log);
  MeasureLoader(target,target.res.num_sms,3,true,log);
  fit.serving_paged["attention_decode_d64"]=FitAttention<64,4>(target,3,log);
  fit.serving_paged["attention_decode_d128"]=FitAttention<128,2>(target,3,log);
}
} // namespace tilemega::calib
