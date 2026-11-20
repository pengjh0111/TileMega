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

template<int M,int N,int K,int TN,int TK,int Slots,int R,int Residency,bool Probe>
using Body=Sm90DenseGemm<M,N,K,TN,TK,Slots,4,1,ServingEpilogueOp::kSwiGLU,16,true,R,false,0,true,Probe>;

template<int M,int N,int K,int TN,int TK,int Slots,int R,int Residency,bool Probe>
__global__ __launch_bounds__(160,Residency)
void Run(Element const* a,Element const* b,Element* output,Sm80DenseService* service,Tick* span) {
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);
  ring.Initialize<true>();
  if constexpr(Probe) {
    if(threadIdx.x==0)
      *reinterpret_cast<Sm80DenseService**>(shared+Sm90DenseOperandHeader::kTraceSlot)=service+blockIdx.x;
    __syncthreads();
  }
  Sm90DenseOperands args{};
  args.activation_address=a;args.weight=b;args.output=output;args.epsilon=1e-6f;
  Tick start=0;
  if(threadIdx.x==0)start=Sm80DenseService::Now();
  Body<M,N,K,TN,TK,Slots,R,Residency,Probe>::Run(args,shared,&ring);
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

template<int M,int N,int K,int TN,int TK,int S,int R,int Residency>
int Profile(cudaDeviceProp const& p) {
  constexpr int shared=Body<M,N,K,TN,TK,S,R,Residency,false>::kSharedBytes;
  auto plain=Run<M,N,K,TN,TK,S,R,Residency,false>;
  auto traced=Run<M,N,K,TN,TK,S,R,Residency,true>;
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
  Device<Element> a(M*K),b(N*K),out(M*N/2),trace_out(M*N/2);
  Device<Sm80DenseService> service(grid);
  Device<Tick> span(grid);
  Fill<<<128,256>>>(a.address,M*K);Fill<<<256,256>>>(b.address,N*K);
  Run<M,N,K,TN,TK,S,R,Residency,false><<<grid,160,shared>>>(a.address,b.address,out.address,service.address,span.address);
  Run<M,N,K,TN,TK,S,R,Residency,true><<<grid,160,shared>>>(a.address,b.address,trace_out.address,service.address,span.address);
  Check(cudaDeviceSynchronize());
  std::vector<Element> reference(M*N/2),actual(M*N/2);
  out.Read(reference);trace_out.Read(actual);
  if(std::memcmp(reference.data(),actual.data(),reference.size()*sizeof(Element)))
    throw std::runtime_error("instrumented production operator changed the output");
  std::vector<Tick> times(grid);
  std::vector<Sm80DenseService> rows(grid);
  std::vector<double> producer_wait,producer_issue,consumer_wait,consumer_mma,epilogue;
  std::vector<Tick> plain_ns,traced_ns;
  for(int trial=0;trial<7;++trial) {
    Run<M,N,K,TN,TK,S,R,Residency,false><<<grid,160,shared>>>(a.address,b.address,out.address,service.address,span.address);
    span.Read(times);
    plain_ns.push_back(*std::max_element(times.begin(),times.end()));
    Run<M,N,K,TN,TK,S,R,Residency,true><<<grid,160,shared>>>(a.address,b.address,trace_out.address,service.address,span.address);
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
  std::printf("{\"status\":\"measured\",\"epilogue\":\"swiglu-u16\","
              "\"input_norm\":true,\"weight_layout\":\"contiguous\","
              "\"batch\":%d,\"n\":%d,\"k\":%d,\"tn\":%d,\"tk\":%d,\"slots\":%d,"
              "\"register_stages\":%d,\"warps_n\":4,\"warps_k\":1,\"grid\":%d,\"requested_residency\":%d,"
              "\"plain_residency\":%d,\"traced_residency\":%d,\"dynamic_shared\":%d,"
              "\"static_shared\":%zu,\"registers\":%d,\"traced_registers\":%d,"
              "\"local_bytes\":%zu,\"traced_local_bytes\":%zu,"
              "\"untraced_span_ns\":%.3f,\"traced_span_ns\":%.3f,\"producer_wait_ns_per_tile\":%.3f,"
              "\"producer_issue_ns_per_tile\":%.3f,\"consumer_wait_ns_per_tile\":%.3f,"
              "\"consumer_mma_ns_per_tile\":%.3f,\"consumer_epilogue_ns_per_task\":%.3f,\"output_bitwise_equal\":true}\n",
              M,N,K,TN,TK,S,R,grid,Residency,plain_capacity,traced_capacity,shared,plain_attr.sharedSizeBytes,
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
#define CASE(label,M,N,K,TN,TK,S,R,O) if(!std::strcmp(argv[1],label))return RetireContext(Profile<M,N,K,TN,TK,S,R,O>(device))
    CASE("llama8-gate-224s2",1,28672,4096,224,64,2,1,1);
    CASE("llama1-b16-64s2",16,16384,2048,64,64,2,1,1);
    CASE("llama1-b16-64s3",16,16384,2048,64,64,3,1,1);
    CASE("llama1-b16-64s4",16,16384,2048,64,64,4,1,1);
    CASE("llama1-b16-64s8",16,16384,2048,64,64,8,1,1);
    CASE("llama1-b16-64s4-r2",16,16384,2048,64,64,4,2,1);
    CASE("llama1-b16-64s4-r4",16,16384,2048,64,64,4,4,1);
    CASE("llama1-b16-64s2-res2",16,16384,2048,64,64,2,1,2);
    CASE("llama1-b16-64s4-res2",16,16384,2048,64,64,4,1,2);
#undef CASE
    throw std::invalid_argument("unknown service case");
  }catch(std::exception const& error) {
    std::fprintf(stderr,"%s\n",error.what());return 1;
  }
}
