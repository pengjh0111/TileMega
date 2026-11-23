// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace tilemega::backend;
using Element=cutlass::bfloat16_t;
using Tick=Sm80DenseService::Tick;

struct BodySpan {
  Tick begin,end;
  unsigned sm;
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
  Tick begin_cycles,end_cycles;
#endif
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
  Tick body_begin_cycles,body_end_cycles,barrier_cycles,release;
#endif
};

#ifndef TILEMEGA_DENSE_SERVICE_SAMPLE_STRIDE
#define TILEMEGA_DENSE_SERVICE_SAMPLE_STRIDE 1
#endif

static __host__ __device__ constexpr bool SampleBlock(unsigned block) {
  return block%TILEMEGA_DENSE_SERVICE_SAMPLE_STRIDE==0;
}

#ifndef TILEMEGA_DENSE_SERVICE_RESERVE_SHARED
#define TILEMEGA_DENSE_SERVICE_RESERVE_SHARED 0
#endif
#ifndef TILEMEGA_DENSE_SERVICE_PRODUCERS
#define TILEMEGA_DENSE_SERVICE_PRODUCERS 32
#endif
#ifndef TILEMEGA_DENSE_SERVICE_SPLIT_K
#define TILEMEGA_DENSE_SERVICE_SPLIT_K 1
#endif
#ifndef TILEMEGA_SM80_MULTISTAGE
#define TILEMEGA_SM80_MULTISTAGE 0
#endif

static void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}

__device__ static Tick WallTime() {
  Tick value;
  asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");
  return value;
}

__global__ void Fill(Element* buffer,int size) {
  for(int i=int(blockIdx.x*blockDim.x+threadIdx.x);i<size;i+=int(blockDim.x*gridDim.x))
    buffer[i]=Element(float((i*7u+2u)%31u)/256.0f-0.0625f);
}

template<int Batch,int N,int K,int TN,int TK,int Slots,int R,int Residency,int WN=4,
         ServingEpilogueOp Op=ServingEpilogueOp::kSwiGLU,int U=16,bool InputNorm=true>
struct Case {
  static constexpr int kBatch=Batch,kN=N,kK=K,kTN=TN,kTK=TK,kSlots=Slots,kR=R,kResidency=Residency;
  static constexpr int kWN=WN,kWK=4/WN,kU=U;
  static constexpr bool kNorm=InputNorm,kGate=Op==ServingEpilogueOp::kSwiGLU;
  static constexpr bool kResidual=Op==ServingEpilogueOp::kResidual;
  static constexpr bool kHead=Op==ServingEpilogueOp::kArgmaxPartial;
  static constexpr int kProducers=TILEMEGA_SM80_MULTISTAGE?0:TILEMEGA_DENSE_SERVICE_PRODUCERS;
  static constexpr int kThreads=128+kProducers,kSlices=TILEMEGA_DENSE_SERVICE_SPLIT_K;
  static constexpr int kPaddedN=(N+TN-1)/TN*TN;
  static constexpr int kOutputTiles=kPaddedN/TN,kTasks=kOutputTiles*kSlices;
  using Padded=Case<Batch,kPaddedN,K,TN,TK,Slots,R,Residency,WN,Op,U,InputNorm>;
  static constexpr int CopyGroup() {
#ifdef TILEMEGA_SM80_RESIDUAL_COPY_GROUP
    if(kResidual)return TILEMEGA_SM80_COPY_ISSUE_GROUP;
#endif
    return kGate?TILEMEGA_SM80_COPY_ISSUE_GROUP:0;
  }
  static constexpr char const* Epilogue() {
    return kGate?"swiglu":kResidual?"residual":"store";
  }
  static constexpr int kSide=kResidual?Sm90DenseLayout::ResidualTiles(kTasks,108*Residency):0;
  template<bool Probe> using Body=Sm90DenseGemm<Batch,N,K,TN,TK,Slots,WN,kWK,Op,U,InputNorm,
      R,false,kSide,true,Probe
#if TILEMEGA_DENSE_SERVICE_PRODUCERS == 64
      ,kProducers
#endif
#if TILEMEGA_SM80_SPLIT_K
      ,kSlices
#endif
      >;
};

template<class C,bool Probe>
__global__ __launch_bounds__(C::kThreads,C::kResidency)
void Run(Element const* a,Element const* b,Element const* residual,Element* output,
         Sm80DenseService* service,BodySpan* span,int* indices=nullptr,
         unsigned long long* arrivals=nullptr
#if TILEMEGA_SM80_SPLIT_K
         ,float* partials=nullptr,std::uint64_t* split_counters=nullptr
#endif
         ) {
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);
#if TILEMEGA_DENSE_SERVICE_PRODUCERS == 64
  ring.Initialize<true,C::kProducers>();
#else
  ring.Initialize<true>();
#endif
  if constexpr(Probe)if(SampleBlock(blockIdx.x)) {
    if(threadIdx.x==0)
      *reinterpret_cast<Sm80DenseService**>(shared+Sm90DenseOperandHeader::kTraceSlot)=service+blockIdx.x;
    __syncthreads();
  }
  Sm90DenseOperands args{};
  args.activation_address=a;args.weight=b;args.residual=residual;args.output=output;args.epsilon=1e-6f;
  args.argmax_index=indices;
  BodySpan observed{};
  if(threadIdx.x==0) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(observed.sm));
    observed.begin=WallTime();
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
    observed.begin_cycles=Sm80DenseService::Cycles();
#endif
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
    observed.body_begin_cycles=Sm80DenseService::Cycles();
#endif
  }
  auto gemm=[&](auto instrumented) {
    C::template Body<decltype(instrumented)::value>::Run(args,shared,&ring,Sm90DenseNoTail{},true,
        Sm90DenseNoFinish{},nullptr
#if TILEMEGA_SM80_SPLIT_K
        ,partials,split_counters
#endif
        );
  };
  if constexpr(Probe) {
    if(SampleBlock(blockIdx.x))gemm(std::true_type{});
    else gemm(std::false_type{});
  }else if constexpr(C::kHead)
    C::template Body<false>::Run(args,shared,&ring,Sm90DenseNoTail{},false,
        Sm90DenseHeadPartialFinish<C::kBatch>{});
  else gemm(std::false_type{});
  if(threadIdx.x==0) {
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
    observed.end_cycles=Sm80DenseService::Cycles();
#endif
    observed.end=WallTime();
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
    observed.body_end_cycles=Sm80DenseService::Cycles();
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(arrivals),"l"(1ull):"memory");
    unsigned long long seen;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(arrivals):"memory");}
    while(seen<gridDim.x);
    if constexpr(Probe)if(SampleBlock(blockIdx.x))
      observed.barrier_cycles=Sm80DenseService::Cycles()-observed.body_end_cycles;
    observed.release=WallTime();
#endif
    span[blockIdx.x]=observed;
  }
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
  __syncthreads();
#endif
  ring.Invalidate<true>();
}

template<class T>
struct Device {
  T* address=nullptr;
  explicit Device(std::size_t count) {Check(cudaMalloc(&address,count*sizeof(T)));}
  Device(Device const&)=delete;
  Device& operator=(Device const&)=delete;
  ~Device() {cudaFree(address);}
  void Read(std::vector<T>& host) const {
    Check(cudaMemcpy(host.data(),address,host.size()*sizeof(T),cudaMemcpyDeviceToHost));
  }
};

template<class T>
static double Median(std::vector<T> values) {
  std::sort(values.begin(),values.end());
  if(values.empty())throw std::runtime_error("empty service measurement");
  return double(values[values.size()/2]);
}

static Tick Longest(std::vector<BodySpan> const& spans) {
  Tick longest=0;
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
  Tick begin=spans[0].begin,end=spans[0].release;
  for(auto const& span:spans) {
    begin=std::min(begin,span.begin);
    end=std::max(end,span.release);
  }
  longest=end-begin;
#else
  for(auto const& span:spans)longest=std::max(longest,span.end-span.begin);
#endif
  return longest;
}

static int OverlappingSms(std::vector<BodySpan> const& spans,int active) {
  std::vector<unsigned> sms;
  for(int block=0;block<active;++block)
    for(int other=0;other<block;++other) {
      auto const& a=spans[block];auto const& b=spans[other];
      if(a.sm==b.sm && a.begin<b.end && b.begin<a.end &&
         std::find(sms.begin(),sms.end(),a.sm)==sms.end())
        sms.push_back(a.sm);
    }
  return int(sms.size());
}

#ifdef TILEMEGA_DENSE_SERVICE_WIDTH_SCAN
static Tick StageSpan(std::vector<BodySpan> const& spans,int active) {
  Tick begin=spans[0].begin,end=spans[0].end;
  for(int block=1;block<active;++block) {
    begin=std::min(begin,spans[block].begin);
    end=std::max(end,spans[block].end);
  }
  return end-begin;
}
#endif

template<class C>
__global__ void ZeroPaddedWeights(Element* weight) {
  using Layout=typename C::template Body<false>::LayoutB;
  constexpr int TN=C::kTN,TK=C::kTK,valid=C::kTN-(C::kPaddedN-C::kN);
  constexpr int kTiles=C::kK/TK,last=C::kPaddedN/TN-1;
  auto* tile=weight+(last*kTiles+blockIdx.x)*TN*TK;
  for(int i=valid*TK+threadIdx.x;i<TN*TK;i+=blockDim.x)
    tile[Layout{}(i/TK,i%TK)]=Element(0.0f);
}

template<class C>
int Profile(cudaDeviceProp const& p) {
  constexpr int M=C::kBatch,N=C::kN,K=C::kK,TN=C::kTN,TK=C::kTK,Residency=C::kResidency;
  constexpr int shared=std::max(C::template Body<false>::kSharedBytes,TILEMEGA_DENSE_SERVICE_RESERVE_SHARED);
  constexpr int output_elements=M*N/(C::kGate?2:1);
  auto plain=Run<C,false>;
#ifndef TILEMEGA_DENSE_SERVICE_WIDTH_SCAN
  auto traced=Run<C,true>;
#endif
  if(shared>p.sharedMemPerBlockOptin) {
    std::printf("{\"status\":\"resource_rejected\",\"dynamic_shared\":%d}\n",shared);
    return 0;
  }
  Check(cudaFuncSetAttribute(plain,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int plain_capacity;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plain_capacity,plain,C::kThreads,shared));
#ifdef TILEMEGA_DENSE_SERVICE_WIDTH_SCAN
  if(plain_capacity<Residency) {
    std::printf("{\"status\":\"resource_rejected\",\"requested\":%d,\"plain_residency\":%d,"
                "\"dynamic_shared\":%d}\n",Residency,plain_capacity,shared);
    return 0;
  }
#else
  Check(cudaFuncSetAttribute(traced,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int traced_capacity;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&traced_capacity,traced,C::kThreads,shared));
  cudaFuncAttributes plain_attr,traced_attr;
  Check(cudaFuncGetAttributes(&plain_attr,plain));Check(cudaFuncGetAttributes(&traced_attr,traced));
  if(std::min(plain_capacity,traced_capacity)<Residency) {
    std::printf("{\"status\":\"resource_rejected\",\"requested\":%d,\"plain_residency\":%d,"
                "\"traced_residency\":%d,\"dynamic_shared\":%d}\n",
                Residency,plain_capacity,traced_capacity,shared);
    return 0;
  }
#endif
  int grid=p.multiProcessorCount*Residency;
  if constexpr(C::kHead)if(C::kPaddedN/TN<grid)
    throw std::runtime_error("the head probe needs one tile per CTA");
  Device<Element> a(M*K),b(C::kPaddedN*K),residual(C::kResidual?M*N:1);
  Device<Element> out(C::kHead?2*M*grid:output_elements);
  Device<int> indices(C::kHead?M*grid:1);
#ifndef TILEMEGA_DENSE_SERVICE_WIDTH_SCAN
  Device<Element> trace_out(output_elements);
#endif
  Device<Sm80DenseService> service(grid);
  Device<BodySpan> span(grid);
#if TILEMEGA_SM80_SPLIT_K
  using Reduction=tilemega::codegen::Sm80SplitKTaskBody<C::kOutputTiles,C::kSlices,16,TN,C::kNorm>;
  Device<float> partials(Reduction::kValues);
  Device<std::uint64_t> split_counters(C::kOutputTiles);
  Check(cudaMemset(split_counters.address,0,C::kOutputTiles*sizeof(std::uint64_t)));
#endif
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
  Device<unsigned long long> arrived(1);
#endif
  auto launch=[&](auto kernel,Element* residual_address,Element* output) {
    unsigned long long* arrivals=nullptr;
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
    arrivals=arrived.address;
    Check(cudaMemset(arrivals,0,sizeof(unsigned long long)));
#endif
    void* args[]={&a.address,&b.address,&residual_address,&output,&service.address,
                  &span.address,&indices.address,&arrivals
#if TILEMEGA_SM80_SPLIT_K
                  ,&partials.address,&split_counters.address
#endif
                  };
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
    Check(cudaLaunchCooperativeKernel(reinterpret_cast<void const*>(kernel),grid,C::kThreads,args,shared));
#else
    Check(cudaLaunchKernel(reinterpret_cast<void const*>(kernel),grid,C::kThreads,args,shared));
#endif
  };
  Fill<<<128,256>>>(a.address,M*K);Fill<<<256,256>>>(b.address,C::kPaddedN*K);
  if constexpr(N!=C::kPaddedN)ZeroPaddedWeights<C><<<K/TK,128>>>(b.address);
  if constexpr(C::kResidual)Fill<<<128,256>>>(residual.address,M*N);
  launch(plain,residual.address,out.address);
  std::vector<Element> reference(C::kHead?2*M*grid:output_elements);
  out.Read(reference);
#ifndef TILEMEGA_DENSE_SERVICE_WIDTH_SCAN
  launch(traced,residual.address,trace_out.address);
  Check(cudaDeviceSynchronize());
  std::vector<Element> actual(output_elements);
  trace_out.Read(actual);
  if(std::memcmp(reference.data(),actual.data(),reference.size()*sizeof(Element)))
    throw std::runtime_error("instrumented production operator changed the output");
#endif
  if constexpr(N!=C::kPaddedN && !C::kHead) {
    using Padded=typename C::Padded;
    constexpr int columns=N/(C::kGate?2:1),padded_cols=C::kPaddedN/(C::kGate?2:1);
    auto padded=Run<Padded,false>;
    Check(cudaFuncSetAttribute(padded,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    Device<Element> padded_output(M*padded_cols),padded_residual(C::kResidual?M*C::kPaddedN:1);
    if constexpr(C::kResidual) {
      Check(cudaMemset(padded_residual.address,0,M*C::kPaddedN*sizeof(Element)));
      Check(cudaMemcpy2D(padded_residual.address,C::kPaddedN*sizeof(Element),
                         residual.address,N*sizeof(Element),N*sizeof(Element),M,cudaMemcpyDeviceToDevice));
    }
    launch(padded,padded_residual.address,padded_output.address);
    std::vector<Element> padded_rows(M*padded_cols);
    padded_output.Read(padded_rows);
    for(int row=0;row<M;++row)
      if(std::memcmp(reference.data()+row*columns,padded_rows.data()+row*padded_cols,
                     columns*sizeof(Element)))
        throw std::runtime_error("padded tile changed valid output columns");
  }
  std::vector<BodySpan> times(grid);
#ifdef TILEMEGA_DENSE_SERVICE_WIDTH_SCAN
  std::vector<Tick> spans;
  constexpr int tiles=C::kTasks;
  int active=std::min(grid,tiles),min_overlapping=p.multiProcessorCount;
  int evict_elements=3*p.l2CacheSize/int(sizeof(Element));
  Device<Element> evict(evict_elements);
  for(int trial=0;trial<7;++trial) {
    Fill<<<p.multiProcessorCount*4,256>>>(evict.address,evict_elements);
    launch(plain,residual.address,out.address);
    span.Read(times);
    spans.push_back(StageSpan(times,active));
    min_overlapping=std::min(min_overlapping,OverlappingSms(times,active));
  }
  double elapsed=Median(spans);
  if(elapsed<=0)throw std::runtime_error("invalid SM80 stage timer");
  std::uint64_t fingerprint=14695981039346656037ull;
  auto const* bytes=reinterpret_cast<unsigned char const*>(reference.data());
  for(std::size_t i=0;i<reference.size()*sizeof(Element);++i)
    fingerprint=(fingerprint^bytes[i])*1099511628211ull;
  std::printf("{\"status\":\"measured\",\"n\":%d,\"k\":%d,\"batch\":%d,\"tn\":%d,\"tk\":%d,"
              "\"slots\":%d,\"grid\":%d,\"requested_residency\":%d,\"plain_residency\":%d,"
              "\"dynamic_shared\":%d,\"active_ctas\":%d,\"overlapping_sms\":%d,"
              "\"logical_weight_bytes\":%lld,\"l2_eviction_bytes\":%d,\"stage_span_ns\":%.3f,"
              "\"effective_gbps\":%.3f,\"producers\":%d,\"split_k\":%d,\"output_fingerprint\":\"%016llx\","
              "\"padded_n\":%d,\"clipped_output_bitwise_equal\":%s}\n",
              N,K,M,TN,TK,C::kSlots,grid,Residency,plain_capacity,shared,active,min_overlapping,
              2ll*N*K,evict_elements*int(sizeof(Element)),elapsed,2.0*N*K/elapsed,
              C::kProducers,C::kSlices,static_cast<unsigned long long>(fingerprint),
              C::kPaddedN,N==C::kPaddedN || C::kHead?"null":"true");
#else
  std::vector<Sm80DenseService> rows(grid);
  std::vector<double> producer_wait,producer_issue,consumer_wait,consumer_mma,epilogue;
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
  std::vector<double> producer_share,issue_share,consumer_share,epilogue_share,barrier_share;
  int evict_elements=3*p.l2CacheSize/int(sizeof(Element));
  Device<Element> evict(evict_elements);
#endif
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
  std::vector<double> epilogue_cycles,sm_ghz,residual_compute_ns;
#endif
  std::vector<Tick> plain_ns,traced_ns;
  int min_overlapping=p.multiProcessorCount;
  constexpr int tiles=C::kTasks;
  for(int trial=0;trial<7;++trial) {
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
    Fill<<<p.multiProcessorCount*4,256>>>(evict.address,evict_elements);
#endif
    launch(plain,residual.address,out.address);
    span.Read(times);
    plain_ns.push_back(Longest(times));
    min_overlapping=std::min(min_overlapping,OverlappingSms(times,std::min(grid,tiles)));
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
    Fill<<<p.multiProcessorCount*4,256>>>(evict.address,evict_elements);
#endif
    launch(traced,residual.address,trace_out.address);
    span.Read(times);service.Read(rows);
    traced_ns.push_back(Longest(times));
    for(int block=0;block<grid;++block) {
      int tasks=tiles/grid+(block<tiles%grid);
      if(!tasks || !SampleBlock(block))continue;
      double steps=tasks*(K/TK);
      auto const& row=rows[block];
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
      auto const& wall=times[block];
      if(wall.end<=wall.begin || wall.body_end_cycles<=wall.body_begin_cycles)
        throw std::runtime_error("missing role clock calibration");
      double rate=double(wall.body_end_cycles-wall.body_begin_cycles)/(wall.end-wall.begin);
      double span_cycles=rate*traced_ns.back();
      producer_share.push_back(row.producer_wait/span_cycles);
      issue_share.push_back(row.producer_issue/span_cycles);
      consumer_share.push_back(row.consumer_wait/span_cycles);
      epilogue_share.push_back(row.consumer_epilogue/span_cycles);
      barrier_share.push_back(wall.barrier_cycles/span_cycles);
#else
      double rate=1.0;
#endif
      producer_wait.push_back(row.producer_wait/(steps*rate));
      producer_issue.push_back(row.producer_issue/(steps*rate));
      consumer_wait.push_back(row.consumer_wait/(steps*rate));
      consumer_mma.push_back(row.consumer_mma/(steps*rate));
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
      // The global timer has microsecond resolution on A800. Use a separate
      // per-CTA clock for the short epilogue and scale it by the full span.
      auto const& wall=times[block];
      if(wall.end<=wall.begin || wall.end_cycles<=wall.begin_cycles || !row.consumer_epilogue)
        throw std::runtime_error("invalid residual epilogue clock sample");
      rate=double(wall.end_cycles-wall.begin_cycles)/(wall.end-wall.begin);
      double cycles=double(row.consumer_epilogue)/tasks;
      sm_ghz.push_back(rate);
      epilogue_cycles.push_back(cycles);
      epilogue.push_back(cycles/rate);
      residual_compute_ns.push_back(double(row.consumer_residual_compute)/(tasks*rate));
#else
      epilogue.push_back(double(row.consumer_epilogue)/(tasks*rate));
#endif
    }
  }
  std::printf("{\"status\":\"measured\",\"epilogue\":\"%s\","
              "\"input_norm\":%s,\"weight_layout\":\"contiguous\",\"u\":%d,"
              "\"live_rows\":%s,\"copy_issue_group\":%d,\"residual_tiles\":%d,"
              "\"batch\":%d,\"n\":%d,\"k\":%d,\"tn\":%d,\"tk\":%d,\"slots\":%d,"
              "\"register_stages\":%d,\"warps_n\":%d,\"warps_k\":%d,\"grid\":%d,\"requested_residency\":%d,"
              "\"plain_residency\":%d,\"traced_residency\":%d,\"dynamic_shared\":%d,"
              "\"overlapping_sms\":%d,"
              "\"static_shared\":%zu,\"registers\":%d,\"traced_registers\":%d,"
              "\"local_bytes\":%zu,\"traced_local_bytes\":%zu,"
              "\"untraced_span_ns\":%.3f,\"traced_span_ns\":%.3f,\"producer_wait_ns_per_tile\":%.3f,"
              "\"producer_issue_ns_per_tile\":%.3f,\"consumer_wait_ns_per_tile\":%.3f,"
              "\"consumer_mma_ns_per_tile\":%.3f,\"consumer_epilogue_ns_per_task\":%.3f,\"output_bitwise_equal\":true",
              C::Epilogue(),C::kNorm?"true":"false",C::kU,
#if defined(TILEMEGA_SM80_ACTIVE_EPILOGUE) && TILEMEGA_SM80_ACTIVE_EPILOGUE
              M<16 && (C::kGate || C::kWK>1)?"true":"false",
#else
              "false",
#endif
              C::CopyGroup(),C::kSide,
              M,N,K,TN,TK,C::kSlots,C::kR,C::kWN,C::kWK,grid,Residency,plain_capacity,traced_capacity,shared,min_overlapping,plain_attr.sharedSizeBytes,
              plain_attr.numRegs,traced_attr.numRegs,plain_attr.localSizeBytes,traced_attr.localSizeBytes,
              Median(plain_ns),Median(traced_ns),Median(producer_wait),Median(producer_issue),Median(consumer_wait),
              Median(consumer_mma),Median(epilogue));
#ifdef TILEMEGA_SM80_DENSE_ROLE_CLOCK
  std::printf(",\"producer_stall_fraction\":%.6f,\"producer_issue_fraction\":%.6f,"
              "\"consumer_wait_fraction\":%.6f,\"epilogue_fraction\":%.6f,"
              "\"grid_barrier_fraction\":%.6f,\"trace_perturbation\":%.6f",
              Median(producer_share),Median(issue_share),Median(consumer_share),Median(epilogue_share),
              Median(barrier_share),std::abs(Median(traced_ns)/Median(plain_ns)-1));
#endif
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
  std::printf(",\"consumer_epilogue_cycles_per_task\":%.3f,\"epilogue_calibration_sm_ghz\":%.6f,"
              "\"consumer_residual_compute_ns_per_task\":%.3f,\"epilogue_only\":true",
      Median(epilogue_cycles),Median(sm_ghz),Median(residual_compute_ns));
#endif
  if constexpr(N!=C::kPaddedN)
    std::printf(",\"padded_n\":%d,\"clipped_output_bitwise_equal\":true",C::kPaddedN);
  if constexpr(TILEMEGA_DENSE_SERVICE_SAMPLE_STRIDE>1)
    std::printf(",\"sample_stride\":%d",TILEMEGA_DENSE_SERVICE_SAMPLE_STRIDE);
  std::printf("}\n");
#endif
  return 0;
}

// NVML may report a CUDA PID for one sample after the context retires.
static int RetireContext(int status) {
  Check(cudaDeviceReset());
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return status;
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("specify one compiled service case");
    cudaDeviceProp device;
    Check(cudaGetDeviceProperties(&device,0));
    if(device.major!=8 || device.minor || device.multiProcessorCount!=108)
      throw std::invalid_argument("requires the calibrated A800 hardware class");
#define CASE(label,...) if(!std::strcmp(argv[1],label))return RetireContext(Profile<Case<__VA_ARGS__>>(device))
#ifdef TILEMEGA_SERVICE_CASES
#include TILEMEGA_SERVICE_CASES
#else
    CASE("llama8-gate-224s2",1,28672,4096,224,64,2,1,1);
    CASE("llama1-b16-64s2",16,16384,2048,64,64,2,1,1);
    CASE("llama1-b16-64s3",16,16384,2048,64,64,3,1,1);
    CASE("llama1-b16-64s4",16,16384,2048,64,64,4,1,1);
    CASE("llama1-b16-64s8",16,16384,2048,64,64,8,1,1);
    CASE("llama1-b16-64s4-r2",16,16384,2048,64,64,4,2,1);
    CASE("llama1-b16-64s4-r4",16,16384,2048,64,64,4,4,1);
    CASE("llama1-b16-64s2-res2",16,16384,2048,64,64,2,1,2);
    CASE("llama1-b16-64s4-res2",16,16384,2048,64,64,4,1,2);
#endif
#undef CASE
    throw std::invalid_argument("unknown service case");
  }catch(std::exception const& error) {
    std::fprintf(stderr,"%s\n",error.what());return 1;
  }
}
