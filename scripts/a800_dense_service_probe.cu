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

static void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
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
  static constexpr char const* Epilogue() {
    return kGate?"swiglu":kResidual?"residual":"store";
  }
  static constexpr int kSide=kResidual?Sm90DenseLayout::ResidualTiles(N/TN,108*Residency):0;
  template<bool Probe> using Body=Sm90DenseGemm<Batch,N,K,TN,TK,Slots,WN,kWK,Op,U,InputNorm,
      R,false,kSide,true,Probe>;
};

template<class C,bool Probe>
__global__ __launch_bounds__(160,C::kResidency)
void Run(Element const* a,Element const* b,Element const* residual,Element* output,
         Sm80DenseService* service,Tick* span) {
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);
  ring.Initialize<true>();
  if constexpr(Probe) {
    if(threadIdx.x==0)
      *reinterpret_cast<Sm80DenseService**>(shared+Sm90DenseOperandHeader::kTraceSlot)=service+blockIdx.x;
    __syncthreads();
  }
  Sm90DenseOperands args{};
  args.activation_address=a;args.weight=b;args.residual=residual;args.output=output;args.epsilon=1e-6f;
  Tick start=0;
  if(threadIdx.x==0)start=Sm80DenseService::Now();
  C::template Body<Probe>::Run(args,shared,&ring);
  if(threadIdx.x==0)span[blockIdx.x]=Sm80DenseService::Now()-start;
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

template<class C>
int Profile(cudaDeviceProp const& p) {
  constexpr int M=C::kBatch,N=C::kN,K=C::kK,TN=C::kTN,TK=C::kTK,Residency=C::kResidency;
  constexpr int shared=C::template Body<false>::kSharedBytes;
  constexpr int output_elements=M*N/(C::kGate?2:1);
  auto plain=Run<C,false>;
  auto traced=Run<C,true>;
  if(shared>p.sharedMemPerBlockOptin) {
    std::printf("{\"status\":\"resource_rejected\",\"dynamic_shared\":%d}\n",shared);
    return 0;
  }
  Check(cudaFuncSetAttribute(plain,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Check(cudaFuncSetAttribute(traced,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int plain_capacity,traced_capacity;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plain_capacity,plain,160,shared));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&traced_capacity,traced,160,shared));
  cudaFuncAttributes plain_attr,traced_attr;
  Check(cudaFuncGetAttributes(&plain_attr,plain));Check(cudaFuncGetAttributes(&traced_attr,traced));
  if(std::min(plain_capacity,traced_capacity)<Residency) {
    std::printf("{\"status\":\"resource_rejected\",\"requested\":%d,\"plain_residency\":%d,"
                "\"traced_residency\":%d,\"dynamic_shared\":%d}\n",
                Residency,plain_capacity,traced_capacity,shared);
    return 0;
  }
  int grid=p.multiProcessorCount*Residency;
  Device<Element> a(M*K),b(N*K),residual(C::kResidual?M*N:1),out(output_elements),trace_out(output_elements);
  Device<Sm80DenseService> service(grid);
  Device<Tick> span(grid);
  Fill<<<128,256>>>(a.address,M*K);Fill<<<256,256>>>(b.address,N*K);
  if constexpr(C::kResidual)Fill<<<128,256>>>(residual.address,M*N);
  Run<C,false><<<grid,160,shared>>>(a.address,b.address,residual.address,out.address,service.address,span.address);
  Run<C,true><<<grid,160,shared>>>(a.address,b.address,residual.address,trace_out.address,service.address,span.address);
  Check(cudaDeviceSynchronize());
  std::vector<Element> reference(output_elements),actual(output_elements);
  out.Read(reference);trace_out.Read(actual);
  if(std::memcmp(reference.data(),actual.data(),reference.size()*sizeof(Element)))
    throw std::runtime_error("instrumented production operator changed the output");
  std::vector<Tick> times(grid);
  std::vector<Sm80DenseService> rows(grid);
  std::vector<double> producer_wait,producer_issue,consumer_wait,consumer_mma,epilogue;
  std::vector<Tick> plain_ns,traced_ns;
  for(int trial=0;trial<7;++trial) {
    Run<C,false><<<grid,160,shared>>>(a.address,b.address,residual.address,out.address,service.address,span.address);
    span.Read(times);
    plain_ns.push_back(*std::max_element(times.begin(),times.end()));
    Run<C,true><<<grid,160,shared>>>(a.address,b.address,residual.address,trace_out.address,service.address,span.address);
    span.Read(times);service.Read(rows);
    traced_ns.push_back(*std::max_element(times.begin(),times.end()));
    for(int block=0;block<grid;++block) {
      int tasks=(N/TN)/grid+(block<(N/TN)%grid);
      if(!tasks)continue;
      double steps=tasks*(K/TK);
      auto const& row=rows[block];
      producer_wait.push_back(row.producer_wait/steps);
      producer_issue.push_back(row.producer_issue/steps);
      consumer_wait.push_back(row.consumer_wait/steps);
      consumer_mma.push_back(row.consumer_mma/steps);
      epilogue.push_back(double(row.consumer_epilogue)/tasks);
    }
  }
  std::printf("{\"status\":\"measured\",\"epilogue\":\"%s\","
              "\"input_norm\":%s,\"weight_layout\":\"contiguous\",\"u\":%d,"
              "\"live_rows\":%s,\"copy_issue_group\":%d,\"residual_tiles\":%d,"
              "\"batch\":%d,\"n\":%d,\"k\":%d,\"tn\":%d,\"tk\":%d,\"slots\":%d,"
              "\"register_stages\":%d,\"warps_n\":%d,\"warps_k\":%d,\"grid\":%d,\"requested_residency\":%d,"
              "\"plain_residency\":%d,\"traced_residency\":%d,\"dynamic_shared\":%d,"
              "\"static_shared\":%zu,\"registers\":%d,\"traced_registers\":%d,"
              "\"local_bytes\":%zu,\"traced_local_bytes\":%zu,"
              "\"untraced_span_ns\":%.3f,\"traced_span_ns\":%.3f,\"producer_wait_ns_per_tile\":%.3f,"
              "\"producer_issue_ns_per_tile\":%.3f,\"consumer_wait_ns_per_tile\":%.3f,"
              "\"consumer_mma_ns_per_tile\":%.3f,\"consumer_epilogue_ns_per_task\":%.3f,\"output_bitwise_equal\":true}\n",
              C::Epilogue(),C::kNorm?"true":"false",C::kU,
#if defined(TILEMEGA_SM80_ACTIVE_EPILOGUE) && TILEMEGA_SM80_ACTIVE_EPILOGUE
              M<16 && (C::kGate || C::kWK>1)?"true":"false",
#else
              "false",
#endif
              C::kGate?TILEMEGA_SM80_COPY_ISSUE_GROUP:0,C::kSide,
              M,N,K,TN,TK,C::kSlots,C::kR,C::kWN,C::kWK,grid,Residency,plain_capacity,traced_capacity,shared,plain_attr.sharedSizeBytes,
              plain_attr.numRegs,traced_attr.numRegs,plain_attr.localSizeBytes,traced_attr.localSizeBytes,
              Median(plain_ns),Median(traced_ns),Median(producer_wait),Median(producer_issue),Median(consumer_wait),
              Median(consumer_mma),Median(epilogue));
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
