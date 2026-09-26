// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/Calibration.h>
#include <tilemega/Target/ArchDispatch.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::calib {
namespace {
void Check(cudaError_t code,char const* what) {
  if(code!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(code));
}

// Every CTA owns a disjoint slice of a buffer larger than four L2 knees.
// Each committed group copies q bytes as 16-byte cp.async.cg transactions;
// S-1 groups are outstanding before the first wait. The shared-memory read
// makes the copy observable. No per-iteration global atomics enter the timing.
__global__ void InflightRead(char const* source,std::size_t segment_bytes,
    int bytes_per_stage,int stages,int rounds,unsigned* sink) {
  extern __shared__ __align__(16) unsigned char shared[];
  unsigned checksum=0;
  auto issue=[&](int step) {
    std::size_t limit=segment_bytes-std::size_t(bytes_per_stage)+16;
    std::size_t start=(std::size_t(step)*bytes_per_stage)%limit;
    char const* global=source+std::size_t(blockIdx.x)*segment_bytes+start;
    unsigned char* slot=shared+std::size_t(step%stages)*bytes_per_stage;
    for(int offset=threadIdx.x*16;offset<bytes_per_stage;offset+=blockDim.x*16) {
      unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(slot+offset));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
          "r"(address),"l"(global+offset));
    }
    asm volatile("cp.async.commit_group;");
  };
  for(int i=0;i<stages-1 && i<rounds;++i)issue(i);
  for(int i=0;i<rounds;++i) {
    if(i+stages-1<rounds)issue(i+stages-1);
    // At the tail there may be fewer groups than the steady-state wait
    // threshold. Drain them before consuming the last shared-memory slots.
    if(i+stages-1>=rounds || stages==2)asm volatile("cp.async.wait_group 0;");
    else if(stages==3)asm volatile("cp.async.wait_group 1;");
    else asm volatile("cp.async.wait_group 2;");
    __syncthreads();
    auto* word=reinterpret_cast<volatile unsigned*>(shared+
        std::size_t(i%stages)*bytes_per_stage);
    checksum+=word[threadIdx.x];
    __syncthreads();
  }
  if(threadIdx.x==0)atomicAdd(sink,checksum);
}

struct Point {double x=0,y=0;};
std::pair<std::vector<double>,std::vector<double>> Isotonic(
    std::vector<Point> points,double ceiling) {
  std::sort(points.begin(),points.end(),[](auto const& a,auto const& b){return a.x<b.x;});
  struct Block {double weighted=0,weight=0,x=0;int begin=0,end=0;};
  std::vector<Block> blocks;
  for(int i=0;i<int(points.size());++i) {
    Block current{std::min(ceiling,points[i].y),1,points[i].x,i,i};
    blocks.push_back(current);
    while(blocks.size()>1) {
      auto& right=blocks.back();auto& left=blocks[blocks.size()-2];
      if(left.weighted/left.weight<=right.weighted/right.weight)break;
      left.weighted+=right.weighted;left.weight+=right.weight;
      left.end=right.end;blocks.pop_back();
    }
  }
  std::map<double,std::pair<double,int>> grouped;
  for(auto const& block:blocks)for(int i=block.begin;i<=block.end;++i) {
    auto& item=grouped[points[i].x];item.first+=block.weighted/block.weight;++item.second;
  }
  std::vector<double> x,y;
  for(auto const& [bytes,values]:grouped) {
    x.push_back(bytes);y.push_back(values.first/values.second);
  }
  return {x,y};
}

double TimePoint(char const* data,std::size_t segment,int groups,int q,int stages,
    unsigned* sink,int repeats) {
  // The disjoint slices together cover the >4-L2 source once per launch.
  int rounds=std::max(2,int(std::ceil(double(segment)/q)));
  Check(cudaFuncSetAttribute(InflightRead,cudaFuncAttributeMaxDynamicSharedMemorySize,
      q*stages),"opt in dynamic shared memory");
  cudaEvent_t begin{},end{};Check(cudaEventCreate(&begin),"create begin event");
  Check(cudaEventCreate(&end),"create end event");
  auto launch=[&](){InflightRead<<<groups,128,q*stages>>>(data,segment,q,stages,rounds,sink);
    Check(cudaGetLastError(),"launch in-flight sweep");};
  launch();Check(cudaDeviceSynchronize(),"warm in-flight sweep");
  std::vector<double> samples;
  for(int i=0;i<std::max(1,repeats);++i) {
    Check(cudaEventRecord(begin),"record begin");launch();
    Check(cudaEventRecord(end),"record end");Check(cudaEventSynchronize(end),"synchronize end");
    float elapsed=0;Check(cudaEventElapsedTime(&elapsed,begin,end),"elapsed time");
    samples.push_back(double(groups)*q*rounds/(double(elapsed)*1e6));
  }
  Check(cudaEventDestroy(begin),"destroy begin");Check(cudaEventDestroy(end),"destroy end");
  std::sort(samples.begin(),samples.end());return samples[samples.size()/2];
}
}

void MeasureInflight(TargetSpec& target,Options const& options,std::ostream& log) {
  auto const& caps=arch::RuntimeCapsForTag(target.arch_tag);
  if(!caps.cp_async)throw std::invalid_argument("in-flight calibration needs cp.async");
  Check(cudaSetDevice(options.device),"select device");
  auto& calib=target.CalibrationFor("bf16");
  std::size_t buffer=std::size_t(std::ceil(4*calib.l2_knee_bytes));
  buffer=(buffer+2047u)&~std::size_t(2047u);
  char* data=nullptr;unsigned* sink=nullptr;
  Check(cudaMalloc(&data,buffer),"allocate streaming source");
  Check(cudaMalloc(&sink,sizeof(unsigned)),"allocate checksum");
  Check(cudaMemset(data,1,buffer),"initialize streaming source");
  Check(cudaMemset(sink,0,sizeof(unsigned)),"initialize checksum");
  std::vector<Point> device,cta;
  int const maximum=target.res.num_sms*std::max(1,target.res.max_threads_per_sm/128);
  log<<"groups\tq_bytes\tstages\tdevice_inflight_bytes\tcta_inflight_bytes\tgbps\n";
  try {
    for(int groups=1;groups<=maximum;groups*=2)for(int kib:{2,4,8,16,32,64})
      for(int stages:{2,3,4}) {
        int q=kib*1024;
        if(q*stages>target.res.max_dynamic_smem_per_cta ||
           q*stages>target.res.max_smem_per_sm ||
           groups>target.res.num_sms*std::max(1,target.res.max_smem_per_sm/(q*stages)))continue;
        std::size_t segment=(buffer/groups/2048)*2048;
        if(segment<std::size_t(q*2))continue;
        double rate=TimePoint(data,segment,groups,q,stages,sink,
            std::min(3,std::max(1,options.repeats)));
        double outstanding=double(groups)*(stages-1)*q;
        log<<groups<<'\t'<<q<<'\t'<<stages<<'\t'<<outstanding<<'\t'
           <<(stages-1)*q<<'\t'<<rate<<'\n';
        device.push_back({outstanding,rate});
        if(groups<=target.res.num_sms)cta.push_back({double((stages-1)*q),rate/groups});
      }
    if(device.empty() || cta.empty())throw std::runtime_error("no legal in-flight measurements");
    auto [dx,dy]=Isotonic(device,calib.dram_gbps);
    auto [cx,cy]=Isotonic(cta,calib.dram_gbps);
    calib.inflight_curve_bytes=std::move(dx);calib.inflight_curve_gbps=std::move(dy);
    calib.cta_stream_curve_bytes=std::move(cx);calib.cta_stream_curve_gbps=std::move(cy);
  }catch(...) {cudaFree(data);cudaFree(sink);throw;}
  Check(cudaFree(data),"free source");Check(cudaFree(sink),"free checksum");
}
} // namespace tilemega::calib
