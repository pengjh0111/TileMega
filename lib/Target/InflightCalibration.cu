// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/Calibration.h>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Solver/BackendCostQuery.h>
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
// xcore1000: the serving bodies stream weights with register-direct 16-byte
// loads (P-10, MX1_probe §2.7), not asynchronous copies, so the curve is
// measured that way. Every CTA owns a disjoint slice of a buffer larger than
// four L2 knees and 512 MiB; each of 128 threads issues `Loads` independent 16-byte loads
// before consuming them, so a CTA has Loads * 2 KiB in flight. The xor into a
// checksum keeps every load observable; no per-iteration atomics are timed.
template<int Loads>
__global__ void __launch_bounds__(128) InflightRead(uint4 const* source,std::size_t segment_vectors,
    int rounds,unsigned* sink) {
  uint4 const* slice=source+std::size_t(blockIdx.x)*segment_vectors;
  unsigned checksum=0;
  std::size_t constexpr step=std::size_t(Loads)*128;
  std::size_t const limit=segment_vectors-step+1;
  for(int i=0;i<rounds;++i) {
    std::size_t const first=(std::size_t(i)*step)%limit+threadIdx.x;
    uint4 v[Loads];
#pragma unroll
    for(int l=0;l<Loads;++l)v[l]=__ldcg(slice+first+std::size_t(l)*128);
#pragma unroll
    for(int l=0;l<Loads;++l)checksum^=v[l].x^v[l].y^v[l].z^v[l].w;
  }
  if(checksum==0x9e3779b9u)atomicAdd(sink,checksum);
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

template<int Loads>
double TimePoint(uint4 const* data,std::size_t segment_vectors,int groups,unsigned* sink,int repeats) {
  // The disjoint slices together cover the >4-L2 source once per launch.
  int rounds=std::max(2,int(segment_vectors/(std::size_t(Loads)*128)));
  cudaEvent_t begin{},end{};Check(cudaEventCreate(&begin),"create begin event");
  Check(cudaEventCreate(&end),"create end event");
  auto launch=[&](){InflightRead<Loads><<<groups,128>>>(data,segment_vectors,rounds,sink);
    Check(cudaGetLastError(),"launch in-flight sweep");};
  launch();Check(cudaDeviceSynchronize(),"warm in-flight sweep");
  std::vector<double> samples;
  for(int i=0;i<std::max(1,repeats);++i) {
    Check(cudaEventRecord(begin),"record begin");launch();
    Check(cudaEventRecord(end),"record end");Check(cudaEventSynchronize(end),"synchronize end");
    float elapsed=0;Check(cudaEventElapsedTime(&elapsed,begin,end),"elapsed time");
    samples.push_back(double(groups)*Loads*2048.0*rounds/(double(elapsed)*1e6));
  }
  Check(cudaEventDestroy(begin),"destroy begin");Check(cudaEventDestroy(end),"destroy end");
  std::sort(samples.begin(),samples.end());return samples[samples.size()/2];
}
}

void MeasureInflight(TargetSpec& target,Options const& options,std::ostream& log) {
  if(target.arch_tag!="xcore1000")throw std::invalid_argument("in-flight calibration is built for xcore1000");
  Check(cudaSetDevice(options.device),"select device");
  auto& calib=target.CalibrationFor("bf16");
  // At least 512 MiB: launches on xcore1000 cost ~10-20 us, and a 4-knee
  // (32 MiB) source read in ~20 us capped the sweep at 1 TB/s (measured).
  std::size_t buffer=std::max(std::size_t(std::ceil(4*calib.l2_knee_bytes)),std::size_t(512)<<20);
  buffer=(buffer+2047u)&~std::size_t(2047u);
  uint4* data=nullptr;unsigned* sink=nullptr;
  Check(cudaMalloc(&data,buffer),"allocate streaming source");
  Check(cudaMalloc(&sink,sizeof(unsigned)),"allocate checksum");
  Check(cudaMemset(data,1,buffer),"initialize streaming source");
  Check(cudaMemset(sink,0,sizeof(unsigned)),"initialize checksum");
  std::vector<Point> device,cta;
  int const maximum=target.res.num_sms*std::max(1,target.res.max_threads_per_sm/128);
  log<<"groups\tloads_per_thread\tdevice_inflight_bytes\tcta_inflight_bytes\tgbps\n";
  try {
    for(int groups=1;groups<=maximum;groups*=2)for(int loads:{1,2,4,8,16}) {
      std::size_t segment=(buffer/groups/2048)*2048/16;
      if(segment<std::size_t(loads)*128*2)continue;
      int const repeats=std::min(3,std::max(1,options.repeats));
      double rate=loads==1?TimePoint<1>(data,segment,groups,sink,repeats):
                  loads==2?TimePoint<2>(data,segment,groups,sink,repeats):
                  loads==4?TimePoint<4>(data,segment,groups,sink,repeats):
                  loads==8?TimePoint<8>(data,segment,groups,sink,repeats):
                           TimePoint<16>(data,segment,groups,sink,repeats);
      double const per_cta=loads*2048.0,outstanding=groups*per_cta;
      log<<groups<<'\t'<<loads<<'\t'<<outstanding<<'\t'<<per_cta<<'\t'<<rate<<'\n';
      device.push_back({outstanding,rate});
      if(groups<=target.res.num_sms)cta.push_back({per_cta,rate/groups});
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
