// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

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

template<int B,int NGu,int NDn,int H,int TNGu,int TKGu,int TNDn,int TKDn,int Slots,
         int WNGu,int WNDn,int Experts,bool FloatWeights,int Page,int R,bool SoftmaxAll,
         int Shared>
struct Case {
  static constexpr int kB=B,kNGu=NGu,kNDn=NDn,kH=H,kE=Experts,kTop=8;
  static constexpr int kTNGu=TNGu,kTKGu=TKGu,kTKDn=TKDn,kTNDn=TNDn,kSlots=Slots;
  static constexpr bool kFloatWeights=FloatWeights;
  static constexpr int kShared=Shared;
  template<int Role> using Body=Sm90DenseMoeFusedGemm<B,NGu,NDn,H,TNGu,TKGu,TNDn,TKDn,
      Slots,WNGu,4/WNGu,WNDn,4/WNDn,16,Experts,kTop,FloatWeights,Page,R,SoftmaxAll,true,false,true,Role>;
};

template<class Case,int Role>
__global__ __launch_bounds__(160,1)
void Run(Sm90DenseMoeFusedOperands args,Tick* samples,int sample_cta) {
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
  if(sampled && threadIdx.x<Marks::kCount)marks[threadIdx.x]=0;
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
  if(threadIdx.x==0)data[Marks::kCount]=Marks::Now()-begin;
#if TILEMEGA_FUSED_SERVICE_SHARED_MARKS
  if(sampled && threadIdx.x<Marks::kCount)data[threadIdx.x]=marks[threadIdx.x];
#endif
  ring.Invalidate<true>();
}

template<class C>
int Profile(cudaDeviceProp const& device) {
  int grid=device.multiProcessorCount;
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
  if(std::min({plain_slots,producer_slots,consumer_slots})<1)
    throw std::runtime_error("fused service cannot be resident");
  Device<Elt> wg(std::size_t(E)*NG*H),wd(std::size_t(E)*ND*(NG/2)),input(B*H),x(B*ND),
      hidden(E*16*(NG/2));
  Device<float> output(B*Top*ND),logits(B*E);
  Device<unsigned char> routes(B*Top*(C::kFloatWeights?sizeof(float):sizeof(Elt)));
  Device<int> index(B*Top),schedule(Sm90DenseMoeSchedule<E>::kInts);
  Device<unsigned> done(E);
  Device<std::uint64_t> combine(ND/C::kTNDn);
  Device<Tick> trace(grid*(Marks::kCount+1));
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
    for(int tile=begin;tile<end;++tile)if(tile%8==lane)++n;
    return n;
  };
  auto read=[&](auto kernel,int cta,std::vector<double>& elapsed,int repeat) {
    void* params[]={&args,&trace.data,&cta};
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
  int active=0;
  for(int repeat=-1;repeat<samples;++repeat) {
    // A stride coprime to 108 covers the task waves and K tile residues.
    int block=(std::max(0,repeat)*37)%grid;
    reset();
    auto raw=read(plain,block,plain_kernel,repeat);
    auto span=[&](auto const& rows) {
      Tick max_time=0;
      for(int block=0;block<grid;++block)
        max_time=std::max(max_time,rows[block*(Marks::kCount+1)+Marks::kCount]);
      return double(max_time);
    };
    if(repeat>=0)plain_spans.push_back(span(raw));
    expected=x.Read();routes_expected=index.Read();
    if(repeat<0)active=schedule.Read().front();
    auto check=[&](auto kernel,int cta,std::vector<double>& elapsed) {
      reset();
      auto result=read(kernel,cta,elapsed,repeat);
      auto actual=x.Read();
      if(std::memcmp(expected.data(),actual.data(),x.size*sizeof(Elt)) || routes_expected!=index.Read())
        throw std::runtime_error("sampling changed fused stage output or route order");
      return result;
    };
    check(producer,-1,producer_off);
    check(consumer,-1,consumer_off);
    auto prod=check(producer,block,producer_kernel);
    auto cons=check(consumer,block,consumer_kernel);
    if(repeat<0)continue;
    producer_spans.push_back(span(prod));
    consumer_spans.push_back(span(cons));
    auto* row=&prod[block*(Marks::kCount+1)];
    auto* compute=&cons[block*(Marks::kCount+1)];
    // Fused Run enumerates Gu followed by Dn in the same grid-strided track.
    int split=active*(NG/C::kTNGu);
    int gu=(split+grid-1-block)/grid;
    int dn=(split+active*(ND/C::kTNDn)+grid-1-block)/grid-gu;
    if(gu>0) {
      double tiles=gu*sampled_tiles(0,H/C::kTKGu,block%8);
      if(tiles>0) {
        gu_issue.push_back(row[Marks::kGuIssue]/tiles);
        gu_mma.push_back(compute[Marks::kGuMma]/tiles);
      }
      gu_epilogue.push_back(compute[Marks::kGuEpilogue]/gu);
    }
    if(dn>0) {
      int steps=(NG/2)/C::kTKDn,ahead=std::min(steps,C::kSlots);
      double all=dn*sampled_tiles(0,steps,block%8),steady=dn*sampled_tiles(ahead,steps,block%8);
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
      "\"sampled_ctas_per_launch\":1,\"sampled_tile_stride\":8,\"launches\":%d,"
      "\"plain_stage_ns\":%.3f,\"producer_stage_ns\":%.3f,\"consumer_stage_ns\":%.3f,"
      "\"plain_kernel_ns\":%.3f,\"producer_kernel_ns\":%.3f,\"consumer_kernel_ns\":%.3f,"
      "\"producer_off_kernel_ns\":%.3f,\"consumer_off_kernel_ns\":%.3f,"
      "\"gu_issue_ns_per_tile\":%.3f,\"gu_mma_ns_per_tile\":%.3f,\"gu_epilogue_ns_per_task\":%.3f,"
      "\"dn_issue_ns_per_tile\":%.3f,\"dn_mma_ns_per_tile\":%.3f,\"dn_epilogue_ns_per_task\":%.3f,"
      "\"dn_preissue_ns_per_task\":%.3f,\"dn_activation_ns_per_task\":%.3f}\n",
      grid,B,active,shared,plain_attr.numRegs,producer_attr.numRegs,consumer_attr.numRegs,
      plain_attr.localSizeBytes,producer_attr.localSizeBytes,consumer_attr.localSizeBytes,
      plain_slots,producer_slots,consumer_slots,samples,
      Median(plain_spans),Median(producer_spans),Median(consumer_spans),
      Median(plain_kernel),Median(producer_kernel),Median(consumer_kernel),
      Median(producer_off),Median(consumer_off),
      Median(gu_issue),Median(gu_mma),Median(gu_epilogue),
      dn_issue.empty()?0.0:Median(dn_issue),Median(dn_mma),Median(dn_epilogue),
      Median(preissue),Median(activation));
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
