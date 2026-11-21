// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef TILEMEGA_FUSED_SERVICE_TILE_STRIDE
#define TILEMEGA_FUSED_SERVICE_TILE_STRIDE 8
#endif
#if TILEMEGA_FUSED_SERVICE_TILE_STRIDE != 8 && !TILEMEGA_FUSED_SERVICE_SPLIT_PHASES
#error "alternate fused sample strides require split phases"
#endif

#if TILEMEGA_FUSED_SERVICE_SPLIT_PHASES
namespace tilemega::backend {
template<bool Producer>
struct Sm80MoeSinglePhaseMarks {
  // kGuEmpty is unused by the service probe. The selected phase is uniform
  // for the sampled CTA and other CTAs still use the uninstrumented body.
  template<Sm90DenseMarks::MoeCounter Phase>
  __device__ static __forceinline__ unsigned long long Begin(int tile=-1) {
    if(threadIdx.x!=(Producer?128:0) || (tile>=0 && unsigned(tile)%TILEMEGA_FUSED_SERVICE_TILE_STRIDE!=blockIdx.x%TILEMEGA_FUSED_SERVICE_TILE_STRIDE))return 0;
    auto* marks=Sm90DenseMarks::Base();
    return marks && marks[Sm90DenseMarks::kGuEmpty]==Phase?Sm90DenseMarks::Now():0;
  }
  template<Sm90DenseMarks::MoeCounter Phase>
  __device__ static __forceinline__ void End(unsigned long long begin) {
    if(begin)Sm90DenseMarks::Base()[Phase]+=Sm90DenseMarks::Now()-begin;
  }
  __device__ static __forceinline__ void RecordTask(bool,int) {}
};
template<>
struct Sm80MoeStageMarks<true,true,kMoeProducerSample>:Sm80MoeSinglePhaseMarks<true> {};
template<>
struct Sm80MoeStageMarks<true,false,kMoeConsumerSample>:Sm80MoeSinglePhaseMarks<false> {};
}  // namespace tilemega::backend
#endif

using namespace tilemega::backend;
using Elt=cutlass::bfloat16_t;
using Marks=Sm90DenseMarks;
using Tick=unsigned long long;

static void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
static double Median(std::vector<double> values) {
  if(values.empty())throw std::runtime_error("empty fused service sample");
  std::sort(values.begin(),values.end());
  return values[values.size()/2];
}
template<class T> struct Device {
  T* data=nullptr;
  std::size_t size=0;
  explicit Device(std::size_t n):size(n) {Check(cudaMalloc(&data,n*sizeof(T)));}
  Device(Device const&)=delete;
  ~Device() {cudaFree(data);}
  std::vector<T> Read() const {
    std::vector<T> result(size);
    Check(cudaMemcpy(result.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));
    return result;
  }
  void Zero() const {Check(cudaMemset(data,0,size*sizeof(T)));}
};

__global__ void Fill(Elt* values,std::size_t n) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
    values[i]=Elt(float((i*13u+7u)%59u)/1024.0f-0.025f);
}
__global__ void FillLogits(float* logits,int n,int experts) {
  for(int i=threadIdx.x;i<n;i+=blockDim.x) {
    unsigned key=(17u+i/experts)*0x9e3779b9u ^ (unsigned(i%experts)+3)*0x85ebca6bu;
    key=(key^(key>>16))*0x7feb352du;
    key=(key^(key>>15))*0x846ca68bu;
    logits[i]=float(Elt(float((key^(key>>16))%1024)/128.0f-4.0f));
  }
}

struct ResidencySpan {Tick begin,end;unsigned sm;};

int OverlappingSms(std::vector<ResidencySpan> spans,int working) {
  spans.resize(working);
  std::sort(spans.begin(),spans.end(),[](auto const& a,auto const& b){return a.sm<b.sm;});
  int total=0;
  for(std::size_t i=0,j=0;i<spans.size();i=j) {
    j=i+1;
    while(j<spans.size() && spans[j].sm==spans[i].sm)++j;
    if(j-i==2)
      total+=std::max(spans[i].begin,spans[i+1].begin)<std::min(spans[i].end,spans[i+1].end);
  }
  return total;
}

template<int B,int NGu,int NDn,int H,int TNGu,int TKGu,int TNDn,int TKDn,int Slots,
         int WNGu,int WNDn,int Experts,bool FloatWeights,int Page,int R,bool SoftmaxAll,
         int Shared,int Residency=1>
struct Case {
  static constexpr int kB=B,kNGu=NGu,kNDn=NDn,kH=H,kE=Experts,kTop=8;
  static constexpr int kTNGu=TNGu,kTKGu=TKGu,kTKDn=TKDn,kTNDn=TNDn,kSlots=Slots;
  static constexpr bool kFloatWeights=FloatWeights;
  static constexpr int kShared=Shared,kResidency=Residency;
  template<int Role> using Body=Sm90DenseMoeFusedGemm<B,NGu,NDn,H,TNGu,TKGu,TNDn,TKDn,
      Slots,WNGu,4/WNGu,WNDn,4/WNDn,16,Experts,kTop,FloatWeights,Page,R,SoftmaxAll,true,false,true,Role>;
};

template<class Case,int Role>
__global__ __launch_bounds__(160,Case::kResidency)
void Run(Sm90DenseMoeFusedOperands args,Tick* samples,int sample_cta,int phase,ResidencySpan* spans) {
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);
  ring.Initialize<true>();
  auto* data=samples+std::size_t(blockIdx.x)*(Marks::kCount+1);
  auto* marks=data;
  bool const sampled=Role && blockIdx.x==unsigned(sample_cta);
#if TILEMEGA_FUSED_SERVICE_SHARED_MARKS
  // Nothing beyond the production stage footprint belongs to this body.
  // Keep timing writes in that spare arena space until the ring drains.
  constexpr int begin_marks=Case::kShared-Marks::kCount*sizeof(Tick);
  static_assert(begin_marks>=Case::template Body<false>::kSharedBytes && begin_marks%alignof(Tick)==0);
  marks=reinterpret_cast<Tick*>(shared+begin_marks);
  if(sampled && threadIdx.x<Marks::kCount)
    marks[threadIdx.x]=threadIdx.x==Marks::kGuEmpty?phase:0;
#endif
  if(threadIdx.x==0)
    *reinterpret_cast<Tick**>(shared+Sm90DenseOperandHeader::kTraceSlot)=
        sampled?marks:nullptr;
  __syncthreads();
  Tick begin=0;
  if(threadIdx.x==0)begin=Marks::Now();
  // Other CTAs still run the production task bodies during sampling.
  if(sampled)
    Case::template Body<Role>::Run(args,shared,&ring);
  else
    Case::template Body<false>::Run(args,shared,&ring);
  if(threadIdx.x==0) {
    auto end=Marks::Now();
    data[Marks::kCount]=end-begin;
    if constexpr(Case::kResidency==2) {
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      spans[blockIdx.x]={begin,end,sm};
    }
  }
#if TILEMEGA_FUSED_SERVICE_SHARED_MARKS
  if(sampled && threadIdx.x<Marks::kCount)data[threadIdx.x]=marks[threadIdx.x];
#endif
  ring.Invalidate<true>();
}

template<class C>
int Profile(cudaDeviceProp const& device) {
  int grid=device.multiProcessorCount*C::kResidency;
  constexpr int B=C::kB,E=C::kE,Top=C::kTop,NG=C::kNGu,ND=C::kNDn,H=C::kH;
  constexpr int stage_shared=C::template Body<false>::kSharedBytes;
  constexpr int shared=C::kShared?C::kShared:stage_shared;
  static_assert(shared>=stage_shared);
  if(shared>device.sharedMemPerBlockOptin) {
    std::printf("{\"status\":\"resource_rejected\",\"dynamic_shared\":%d}\n",shared);
    return 0;
  }
  auto plain=Run<C,kMoeNoSample>;
  auto producer=Run<C,kMoeProducerSample>,consumer=Run<C,kMoeConsumerSample>;
  Check(cudaFuncSetAttribute(plain,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Check(cudaFuncSetAttribute(producer,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Check(cudaFuncSetAttribute(consumer,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaFuncAttributes plain_attr{},producer_attr{},consumer_attr{};
  Check(cudaFuncGetAttributes(&plain_attr,plain));
  Check(cudaFuncGetAttributes(&producer_attr,producer));
  Check(cudaFuncGetAttributes(&consumer_attr,consumer));
  int plain_slots,producer_slots,consumer_slots;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plain_slots,plain,160,shared));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&producer_slots,producer,160,shared));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&consumer_slots,consumer,160,shared));
  if(std::min({plain_slots,producer_slots,consumer_slots})<C::kResidency) {
    std::printf("{\"status\":\"resource_rejected\",\"dynamic_shared\":%d,"
        "\"requested\":%d,\"plain_residency\":%d,\"producer_residency\":%d,\"consumer_residency\":%d}\n",
        shared,C::kResidency,plain_slots,producer_slots,consumer_slots);
    return 0;
  }
  Device<Elt> wg(std::size_t(E)*NG*H),wd(std::size_t(E)*ND*(NG/2)),input(B*H),x(B*ND),
      hidden(E*16*(NG/2));
  Device<float> output(B*Top*ND),logits(B*E);
  Device<unsigned char> routes(B*Top*(C::kFloatWeights?sizeof(float):sizeof(Elt)));
  Device<int> index(B*Top),schedule(Sm90DenseMoeSchedule<E>::kInts);
  Device<unsigned> done(E);
  Device<std::uint64_t> combine(ND/C::kTNDn);
  Device<Tick> trace(grid*(Marks::kCount+1));
  Device<ResidencySpan> spans(C::kResidency==2?grid:1);
  struct Event {
    cudaEvent_t event;
    Event() {Check(cudaEventCreate(&event));}
    ~Event() {cudaEventDestroy(event);}
  } begin,end;
  Fill<<<256,256>>>(wg.data,wg.size);Fill<<<256,256>>>(wd.data,wd.size);
  Fill<<<1,256>>>(input.data,input.size);FillLogits<<<1,256>>>(logits.data,B*E,E);
  Sm90DenseMoeFusedOperands args{};
  args.weight_gate_up=wg.data;args.weight_down=wd.data;args.input=input.data;
  args.hidden=hidden.data;args.output=output.data;args.schedule=schedule.data;
  args.route_weights=routes.data;args.route_index=index.data;args.x=x.data;
  args.counter=combine.data;args.gate_up_done=done.data;args.logits=logits.data;args.epsilon=1e-6f;
  std::vector<double> plain_spans,producer_spans,consumer_spans,
      plain_kernel,producer_kernel,consumer_kernel,producer_off,consumer_off;
  std::vector<double> gu_issue,dn_issue,gu_mma,dn_mma,gu_epilogue,dn_epilogue,preissue,activation;
  std::vector<Elt> expected;
  std::vector<int> routes_expected;
  auto reset=[&] {
    trace.Zero();done.Zero();combine.Zero();
    Fill<<<1,256>>>(x.data,x.size);
    Check(cudaGetLastError());
  };
  auto sampled_tiles=[](int begin,int end,int lane) {
    int n=0;
    for(int tile=begin;tile<end;++tile)if(tile%TILEMEGA_FUSED_SERVICE_TILE_STRIDE==lane)++n;
    return n;
  };
  auto read=[&](auto kernel,int cta,std::vector<double>& elapsed,int repeat,int phase=-1) {
    void* params[]={&args,&trace.data,&cta,&phase,&spans.data};
    Check(cudaEventRecord(begin.event));
    Check(cudaLaunchKernel(reinterpret_cast<void const*>(kernel),grid,160,params,shared));
    Check(cudaEventRecord(end.event));
    auto raw=trace.Read();
    float ms=0;
    Check(cudaEventElapsedTime(&ms,begin.event,end.event));
    if(repeat>=0)elapsed.push_back(double(ms)*1e6);
    return raw;
  };
  constexpr int samples=32;
#if TILEMEGA_FUSED_SERVICE_SPLIT_PHASES
  constexpr std::array<int,4> producer_phases{Marks::kGuIssue,Marks::kDnIssue,Marks::kDnPreissueIssue,Marks::kDnActivation};
  constexpr std::array<int,4> consumer_phases{Marks::kGuMma,Marks::kGuEpilogue,Marks::kDnMma,Marks::kDnEpilogue};
  std::array<Tick,Marks::kCount> prod_marks{},cons_marks{};
  std::array<std::vector<double>,4> plain_phase,producer_phase,consumer_phase,producer_off_phase,consumer_off_phase;
  constexpr int phases=4;
#else
  constexpr int phases=1;
#endif
  int active=0,overlap=device.multiProcessorCount;
  for(int repeat=-1;repeat<samples*phases;++repeat) {
    // A stride coprime to 108 covers the task waves and K tile residues.
    int block=(std::max(0,repeat)/phases*37)%grid;
    reset();
    auto raw=read(plain,block,plain_kernel,repeat);
    auto span=[&](auto const& rows) {
      Tick max_time=0;
      for(int block=0;block<grid;++block)
        max_time=std::max(max_time,rows[block*(Marks::kCount+1)+Marks::kCount]);
      return double(max_time);
    };
    if(repeat>=0) {
      plain_spans.push_back(span(raw));
      if constexpr(C::kResidency==2)
        overlap=std::min(overlap,OverlappingSms(spans.Read(),
            std::min(grid,active*(NG/C::kTNGu+ND/C::kTNDn))));
    }
    expected=x.Read();routes_expected=index.Read();
    if(repeat<0)active=schedule.Read().front();
    auto check=[&](auto kernel,int cta,std::vector<double>& elapsed,int phase=-1) {
      reset();
      auto result=read(kernel,cta,elapsed,repeat,phase);
      auto actual=x.Read();
      if(std::memcmp(expected.data(),actual.data(),x.size*sizeof(Elt)) || routes_expected!=index.Read())
        throw std::runtime_error("sampling changed fused stage output or route order");
      return result;
    };
    check(producer,-1,producer_off);
    check(consumer,-1,consumer_off);
    int producer_mark=-1,consumer_mark=-1;
#if TILEMEGA_FUSED_SERVICE_SPLIT_PHASES
    auto phase=std::max(0,repeat)%phases;
    producer_mark=producer_phases[phase];
    consumer_mark=consumer_phases[phase];
#endif
    auto prod=check(producer,block,producer_kernel,producer_mark);
    auto cons=check(consumer,block,consumer_kernel,consumer_mark);
    if(repeat<0)continue;
    producer_spans.push_back(span(prod));
    consumer_spans.push_back(span(cons));
    auto* row=&prod[block*(Marks::kCount+1)];
    auto* compute=&cons[block*(Marks::kCount+1)];
#if TILEMEGA_FUSED_SERVICE_SPLIT_PHASES
    plain_phase[phase].push_back(plain_kernel.back());
    producer_phase[phase].push_back(producer_kernel.back());
    consumer_phase[phase].push_back(consumer_kernel.back());
    producer_off_phase[phase].push_back(producer_off.back());
    consumer_off_phase[phase].push_back(consumer_off.back());
    prod_marks[producer_mark]=row[producer_mark];
    cons_marks[consumer_mark]=compute[consumer_mark];
    if(phase+1<phases)continue;
    row=prod_marks.data();compute=cons_marks.data();
#endif
    // Fused Run enumerates Gu followed by Dn in the same grid-strided track.
    int split=active*(NG/C::kTNGu);
    int gu=(split+grid-1-block)/grid;
    int dn=(split+active*(ND/C::kTNDn)+grid-1-block)/grid-gu;
    if(gu>0) {
      double tiles=gu*sampled_tiles(0,H/C::kTKGu,block%TILEMEGA_FUSED_SERVICE_TILE_STRIDE);
      if(tiles>0) {
        gu_issue.push_back(row[Marks::kGuIssue]/tiles);
        gu_mma.push_back(compute[Marks::kGuMma]/tiles);
      }
      gu_epilogue.push_back(compute[Marks::kGuEpilogue]/gu);
    }
    if(dn>0) {
      int steps=(NG/2)/C::kTKDn,ahead=std::min(steps,C::kSlots);
      double all=dn*sampled_tiles(0,steps,block%TILEMEGA_FUSED_SERVICE_TILE_STRIDE),steady=dn*sampled_tiles(ahead,steps,block%TILEMEGA_FUSED_SERVICE_TILE_STRIDE);
      if(steady>0) {
        dn_issue.push_back(row[Marks::kDnIssue]/steady);
      }
      if(all>0) {
        dn_mma.push_back(compute[Marks::kDnMma]/all);
      }
      dn_epilogue.push_back(compute[Marks::kDnEpilogue]/dn);
      preissue.push_back(row[Marks::kDnPreissueIssue]/dn);
      activation.push_back(row[Marks::kDnActivation]/dn);
    }
  }
  std::printf("{\"status\":\"measured\",\"grid\":%d,\"batch\":%d,\"active_experts\":%d,"
      "\"dynamic_shared\":%d,\"plain_registers\":%d,\"producer_registers\":%d,\"consumer_registers\":%d,"
      "\"plain_local_bytes\":%zu,\"producer_local_bytes\":%zu,\"consumer_local_bytes\":%zu,"
      "\"plain_residency\":%d,\"producer_residency\":%d,\"consumer_residency\":%d,\"bitwise_equal\":true,"
      "\"sampled_ctas_per_launch\":1,\"sampled_tile_stride\":%d,\"launches\":%d,"
      "\"plain_stage_ns\":%.3f,\"producer_stage_ns\":%.3f,\"consumer_stage_ns\":%.3f,"
      "\"plain_kernel_ns\":%.3f,\"producer_kernel_ns\":%.3f,\"consumer_kernel_ns\":%.3f,"
      "\"producer_off_kernel_ns\":%.3f,\"consumer_off_kernel_ns\":%.3f,"
      "\"gu_issue_ns_per_tile\":%.3f,\"gu_mma_ns_per_tile\":%.3f,\"gu_epilogue_ns_per_task\":%.3f,"
      "\"dn_issue_ns_per_tile\":%.3f,\"dn_mma_ns_per_tile\":%.3f,\"dn_epilogue_ns_per_task\":%.3f,"
      "\"dn_preissue_ns_per_task\":%.3f,\"dn_activation_ns_per_task\":%.3f",
      grid,B,active,shared,plain_attr.numRegs,producer_attr.numRegs,consumer_attr.numRegs,
      plain_attr.localSizeBytes,producer_attr.localSizeBytes,consumer_attr.localSizeBytes,
      plain_slots,producer_slots,consumer_slots,TILEMEGA_FUSED_SERVICE_TILE_STRIDE,samples*phases,
      Median(plain_spans),Median(producer_spans),Median(consumer_spans),
      Median(plain_kernel),Median(producer_kernel),Median(consumer_kernel),
      Median(producer_off),Median(consumer_off),
      Median(gu_issue),Median(gu_mma),Median(gu_epilogue),
      dn_issue.empty()?0.0:Median(dn_issue),Median(dn_mma),Median(dn_epilogue),
      Median(preissue),Median(activation));
  if constexpr(C::kResidency==2)std::printf(",\"overlapping_sms\":%d",overlap);
#if TILEMEGA_FUSED_SERVICE_SPLIT_PHASES
  std::printf(",\"phase_timing_ns\":[");
  for(int i=0;i<phases;++i) {
    if(i)std::printf(",");
    std::printf("{\"plain\":%.3f,\"producer_off\":%.3f,\"producer\":%.3f,"
        "\"consumer_off\":%.3f,\"consumer\":%.3f}",
        Median(plain_phase[i]),Median(producer_off_phase[i]),Median(producer_phase[i]),
        Median(consumer_off_phase[i]),Median(consumer_phase[i]));
  }
  std::printf("]");
#endif
  std::printf("}\n");
  return 0;
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("specify one compiled fused stage");
    cudaDeviceProp device{};
    Check(cudaGetDeviceProperties(&device,0));
    if(device.major!=8 || device.minor || device.multiProcessorCount!=108)
      throw std::invalid_argument("requires the A800 hardware class");
#define CASE(label,...) if(!std::strcmp(argv[1],label)) { \
  int result=Profile<Case<__VA_ARGS__>>(device); \
  Check(cudaDeviceReset());std::this_thread::sleep_for(std::chrono::seconds(2));return result; \
}
#include TILEMEGA_FUSED_SERVICE_CASES
#undef CASE
    throw std::invalid_argument("unknown fused service case");
  }catch(std::exception const& error) {
    std::fprintf(stderr,"%s\n",error.what());return 1;
  }
}
