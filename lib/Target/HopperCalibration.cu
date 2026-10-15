// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/HopperCalibration.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::calib {
namespace {
using Copy = codegen::executor::Async<>;
constexpr int kThreads=128,kPageBytes=16384,kSlots=4,kVectors=kPageBytes/16;
constexpr int kShared=kPageBytes*kSlots+kSlots*sizeof(std::uint64_t);
void Check(cudaError_t e) {
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct Buffer {
  T* data=nullptr;
  explicit Buffer(std::size_t n) {Check(cudaMalloc(&data,n*sizeof(T)));}
  ~Buffer() {if(data)cudaFree(data);}
  Buffer(Buffer const&)=delete;
  std::vector<T> Host(std::size_t n) const {
    std::vector<T> out(n);Check(cudaMemcpy(out.data(),data,n*sizeof(T),cudaMemcpyDeviceToHost));return out;
  }
};
__device__ unsigned long long Now() {
  unsigned long long out;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(out));return out;
}
__global__ void Fill(uint4* input,std::size_t count,unsigned long long* expected) {
  unsigned long long sum=0;
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
    unsigned h=static_cast<unsigned>(i)*2654435761u;
    unsigned v[4];
#pragma unroll
    for(int j=0;j<4;++j){h=h*1664525u+1013904223u;v[j]=h;sum+=h;}
    input[i]=make_uint4(v[0],v[1],v[2],v[3]);
  }
  atomicAdd(expected,sum);
}
// Every transport consumes the same bytes and computes the same integer sum.
// Four pages bound the pipeline; capacity/SM count comes from the probed target.
template<int Method>
__global__ void Stream(uint4 const* input,void const* map,std::size_t pages,
                       unsigned long long* output) {
  extern __shared__ __align__(1024) unsigned char shared[];
  auto* barriers=reinterpret_cast<std::uint64_t*>(shared+kPageBytes*kSlots);
  if constexpr(Method>0) {
    if(threadIdx.x==0) {
      for(int s=0;s<kSlots;++s)Copy::Init(barriers+s,kThreads);
      Copy::InitFence();
    }
    __syncthreads();
  }
  auto issue=[&](std::size_t step) {
    std::size_t page=blockIdx.x+step*gridDim.x;
    if(page>=pages)return;
    int slot=step%kSlots;
    auto* destination=shared+slot*kPageBytes;
    if constexpr(Method==1) {
      for(int v=threadIdx.x;v<kVectors;v+=kThreads)
        Copy::Copy16(destination+v*16,input+page*kVectors+v);
      Copy::CompleteCopies(barriers+slot);
    }else if constexpr(Method==2 || Method==3) {
      if(threadIdx.x==0) {
        Copy::ExpectTx(barriers+slot,kPageBytes);
        if constexpr(Method==2)Copy::Bulk(destination,input+page*kVectors,kPageBytes,barriers+slot);
        else Copy::Tensor2D(destination,map,0,page*(kPageBytes/128),barriers+slot);
      }else Copy::Arrive(barriers+slot);
    }
  };
  if constexpr(Method>0)for(int s=0;s<kSlots;++s)issue(s);
  unsigned long long sum=0;
  for(std::size_t step=0,page=blockIdx.x;page<pages;++step,page+=gridDim.x) {
    int slot=step%kSlots;
    if constexpr(Method>0)Copy::Wait(barriers+slot,(step/kSlots)&1);
    for(int v=threadIdx.x;v<kVectors;v+=kThreads) {
      uint4 x;
      if constexpr(Method==0)
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(input+page*kVectors+v):"memory");
      else x=reinterpret_cast<uint4*>(shared+slot*kPageBytes)[v];
      sum+=static_cast<unsigned long long>(x.x)+x.y+x.z+x.w;
    }
    if constexpr(Method>0) {
      __syncthreads();issue(step+kSlots);
    }
  }
  output[blockIdx.x*kThreads+threadIdx.x]=sum;
}
#if defined(_CG_HAS_CLUSTER_GROUP)
// DSMEM is a ticket round trip between rank zero and a selected peer, not
// bulk-payload bandwidth. All other ranks remain alive through the final sync.
__global__ void ClusterTrial(unsigned long long* elapsed,unsigned* errors,
                             int rounds,int kind,unsigned peer_rank) {
  __shared__ unsigned ticket;
  auto cluster=cooperative_groups::this_cluster();
  unsigned rank=cluster.block_rank();
  if(threadIdx.x==0)ticket=0;
  cluster.sync();
  unsigned long long begin=Now();
  if(kind==1) {
    for(int i=0;i<rounds;++i)cluster.sync();
  }else if(kind==2 && (rank==0 || rank==peer_rank) && threadIdx.x==0) {
    auto* peer=cluster.map_shared_rank(&ticket,rank==0?peer_rank:0);
    auto local_addr=Copy::Shared(&ticket),peer_addr=Copy::Shared(peer);
    for(unsigned i=1;i<=static_cast<unsigned>(rounds);++i) {
      if(rank==0)asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;"::"r"(peer_addr),"r"(i):"memory");
      unsigned seen=0,spins=0;
      do {
        asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];":"=r"(seen):"r"(local_addr):"memory");
        if((++spins&1023u)==0 && Now()-begin>2000000000ull){atomicAdd(errors,1u);break;}
      }while(seen<i);
      if(seen<i)break;
      if(rank!=0)asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;"::"r"(peer_addr),"r"(i):"memory");
    }
    if(ticket!=static_cast<unsigned>(rounds))atomicAdd(errors,1u);
  }else if(kind==0) {
    unsigned live=threadIdx.x;
    for(int i=0;i<rounds;++i)asm volatile("add.u32 %0, %0, 1;":"+r"(live));
    if(live==0xffffffffu)atomicAdd(errors,1u);
  }
  if(threadIdx.x==0)elapsed[blockIdx.x]=Now()-begin;
  cluster.sync();
}
#endif
struct Summary {
  double median=0,p90=0,relative_stddev=0;
};
Summary Stats(std::vector<double> a) {
  if(a.empty())throw std::runtime_error("empty calibration sample");
  std::sort(a.begin(),a.end());double mean=std::accumulate(a.begin(),a.end(),0.)/a.size(),var=0;
  for(double x:a)var+=(x-mean)*(x-mean);
  return {a[a.size()/2],a[std::min(a.size()-1,static_cast<std::size_t>(std::ceil(.9*a.size())-1))],mean?std::sqrt(var/a.size())/std::abs(mean):0};
}
void JsonRecord(std::ostream& out,char const* name,int cluster,int peer,int grid,
                std::size_t bytes,std::vector<double> const& samples,char const* unit,bool& first) {
  auto s=Stats(samples);if(!first)out<<",\n";first=false;
  out<<"{\"name\":\""<<name<<"\",\"cluster_dim\":"<<cluster<<",\"peer_rank\":"<<peer
     <<",\"grid_ctas\":"<<grid<<",\"bytes\":"<<bytes<<",\"samples\":"<<samples.size()
     <<",\"median\":"<<s.median<<",\"p90\":"<<s.p90<<",\"relative_stddev\":"<<s.relative_stddev
     <<",\"unit\":\""<<unit<<"\",\"evidence\":\"verified\"}";
}
template<int Method>
void Transport(TargetSpec const& target,Options const& options,uint4 const* input,
               void const* map,std::size_t pages,unsigned long long expected,
               std::ostream& raw,std::ostream& fragment,bool& first) {
  char const* name=Method==0?"ldg128":Method==1?"cp_async":Method==2?"bulk":"tma";
  int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Stream<Method>,kThreads,kShared));
  if(resident<1)throw std::runtime_error("transport has no resident CTA");
  int grid=resident*target.res.num_sms;
  Buffer<unsigned long long> checksum(grid*kThreads);
  cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
  auto launch=[&] {Stream<Method><<<grid,kThreads,kShared>>>(input,map,pages,checksum.data);Check(cudaGetLastError());};
  for(int i=0;i<2;++i)launch();Check(cudaDeviceSynchronize());
  std::vector<double> bandwidth;
  for(int repeat=0;repeat<options.repeats;++repeat) {
    Check(cudaEventRecord(start));launch();Check(cudaEventRecord(stop));Check(cudaEventSynchronize(stop));
    float ms;Check(cudaEventElapsedTime(&ms,start,stop));
    auto sums=checksum.Host(grid*kThreads);unsigned long long actual=std::accumulate(sums.begin(),sums.end(),0ull);
    raw<<"transport\t"<<name<<"\t1\t-1\t"<<repeat<<'\t'<<grid<<'\t'<<pages*kPageBytes<<'\t'<<ms<<'\t'<<(actual==expected)<<'\n';raw.flush();
    if(actual!=expected)throw std::runtime_error(std::string(name)+" read checksum mismatch");
    if(!(ms>0))throw std::runtime_error("nonpositive stream duration");
    bandwidth.push_back(pages*kPageBytes/(double(ms)*1e6));
  }
  Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));
  JsonRecord(fragment,name,1,-1,grid,pages*kPageBytes,bandwidth,"GB/s",first);
}
}
void MeasureHopperDiagnostics(TargetSpec const& target,Options const& options,
                              std::ostream& raw,std::ostream& fragment) {
  if(options.repeats<1)throw std::runtime_error("positive repeats required");
  if(!target.caps.tma || !target.caps.bulk_copy || !target.caps.cluster)
    throw std::runtime_error("Hopper transport/cluster capabilities required");
  Check(cudaSetDevice(options.device));
  std::size_t bytes=std::max<std::size_t>(256ull<<20,16ull*target.res.l2_bytes);
  std::size_t pages=(bytes+kPageBytes-1)/kPageBytes;bytes=pages*kPageBytes;
  Buffer<uint4> input(bytes/16);Buffer<unsigned long long> expected(1);Check(cudaMemset(expected.data,0,8));
  Fill<<<target.res.num_sms*4,256>>>(input.data,bytes/16,expected.data);Check(cudaDeviceSynchronize());
  auto expected_sum=expected.Host(1)[0];
  codegen::executor::TensorMap descriptor;
  auto status=codegen::executor::EncodeTensorMap(descriptor,input.data,{64,bytes/128,128,64,kPageBytes/128});
  if(status!=CUDA_SUCCESS)throw std::runtime_error("TMA tensor map encode failed: "+std::to_string(status));
  Buffer<codegen::executor::TensorMap> map(1);Check(cudaMemcpy(map.data,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice));
  Check(cudaFuncSetAttribute(Stream<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  Check(cudaFuncSetAttribute(Stream<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  Check(cudaFuncSetAttribute(Stream<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  Check(cudaFuncSetAttribute(Stream<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  int runtime=0,driver=0;Check(cudaRuntimeGetVersion(&runtime));Check(cudaDriverGetVersion(&driver));
  raw<<"kind\tmethod\tcluster_dim\tpeer_rank\trepeat\tgrid_ctas\tbytes\telapsed_ms_or_ns_per_op\tchecksum_pass\n";
  fragment<<std::setprecision(12)<<"{\"schema\":\"tilemega.hopper_diagnostics.v1\",\"arch\":\""<<target.arch_tag
      <<"\",\"runtime_version\":"<<runtime<<",\"driver_version\":"<<driver<<",\"repeats\":"<<options.repeats
      <<",\"status\":\"diagnostic_only\",\"method\":\"same incompressible input; 4x16KiB page ring; uint64 checksum; CUDA events; cluster globaltimer 4096 iterations with empty-loop subtraction\",\"records\":[\n";
  bool first=true;
  Transport<0>(target,options,input.data,map.data,pages,expected_sum,raw,fragment,first);
  Transport<1>(target,options,input.data,map.data,pages,expected_sum,raw,fragment,first);
  Transport<2>(target,options,input.data,map.data,pages,expected_sum,raw,fragment,first);
  Transport<3>(target,options,input.data,map.data,pages,expected_sum,raw,fragment,first);
#if defined(_CG_HAS_CLUSTER_GROUP)
  constexpr int rounds=4096;
  for(unsigned size:{2,4,8}) {
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={size,1,1};
    cudaLaunchConfig_t launch{};launch.blockDim=dim3(kThreads);launch.gridDim=dim3(size);launch.attrs=&attr;launch.numAttrs=1;
    int active=0;Check(cudaOccupancyMaxActiveClusters(&active,ClusterTrial,&launch));
    if(active<1)throw std::runtime_error("no resident cluster for requested size");
    int grid=active*size;launch.gridDim=dim3(grid);
    Buffer<unsigned long long> elapsed(grid);Buffer<unsigned> errors(1);
    std::vector<double> baseline;
    auto sample=[&](int kind,unsigned peer,int repeat) {
      Check(cudaMemset(errors.data,0,sizeof(unsigned)));
      Check(cudaLaunchKernelEx(&launch,ClusterTrial,elapsed.data,errors.data,rounds,kind,peer));Check(cudaDeviceSynchronize());
      if(errors.Host(1)[0])throw std::runtime_error("cluster ticket round trip timed out or returned wrong epoch");
      auto host=elapsed.Host(grid);std::vector<double> per_cluster;
      for(int block=0;block<grid;block+=size) {
        unsigned long long duration=host[block];
        if(kind==1)for(unsigned rank=1;rank<size;++rank)duration=std::max(duration,host[block+rank]);
        per_cluster.push_back(double(duration)/rounds);
      }
      double value=Stats(per_cluster).median;
      raw<<"cluster\t"<<(kind==0?"empty":kind==1?"barrier":"dsmem_rtt")<<'\t'<<size<<'\t'<<peer<<'\t'<<repeat<<'\t'<<grid<<"\t0\t"<<value<<"\t1\n";raw.flush();return value;
    };
    for(int repeat=-2;repeat<options.repeats;++repeat) {auto value=sample(0,1,repeat);if(repeat>=0)baseline.push_back(value);}
    for(unsigned peer=0;peer<size;++peer) {
      int kind=peer==0?1:2;std::vector<double> values;
      for(int repeat=-2;repeat<options.repeats;++repeat) {
        auto value=sample(kind,peer==0?1:peer,repeat);
        if(repeat>=0)values.push_back(value-baseline[repeat]);
      }
      // Signed subtraction is retained, including noise or pathological fits.
      JsonRecord(fragment,peer==0?"cluster_barrier":"dsmem_ticket_rtt",size,peer,grid,0,values,"ns/op",first);
    }
  }
#else
  throw std::runtime_error("this CUDA device pass has no cluster_group; no fallback");
#endif
  fragment<<"\n]}\n";
}
} // namespace tilemega::calib
