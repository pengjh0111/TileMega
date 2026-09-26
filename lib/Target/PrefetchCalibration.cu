// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/Calibration.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace tilemega::calib {
namespace {
void Check(cudaError_t value) {
  if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));
}
__global__ void Evict(unsigned const* source,std::size_t words,unsigned* sink) {
  unsigned sum=0;
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=std::size_t(gridDim.x)*blockDim.x)
    sum+=source[i];
  if(threadIdx.x==0)atomicAdd(sink,sum);
}
__global__ void Probe(unsigned const* line,int offset,bool prefetch,unsigned long long* cycles,unsigned* sink) {
  if(threadIdx.x)return;
  if(prefetch)codegen::executor::Async<arch::Sm80>::Prefetch(line,32);
  // Let the asynchronous hint arrive before timing the demand load.
  auto settle=clock64();while(clock64()-settle<20000){}
  unsigned value;unsigned long long begin,end;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin) :: "memory");
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(line+offset/4) : "memory");
  // The dependent sink store prevents timing only instruction issue.
  asm volatile("st.global.u32 [%0], %1;" :: "l"(sink),"r"(value) : "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(end) :: "memory");
  *cycles=end-begin;
}
}
void MeasureL2Prefetch(TargetSpec& target,Options const& options,std::ostream& log) {
  Check(cudaSetDevice(options.device));
  if(target.res.l2_bytes<=0)throw std::invalid_argument("L2 prefetch calibration requires probed cache size");
  unsigned *buffer=nullptr,*line=nullptr,*sink=nullptr;unsigned long long* cycles=nullptr;
  std::size_t bytes=std::size_t(target.res.l2_bytes)*4;
  Check(cudaMalloc(&buffer,bytes));Check(cudaMalloc(&line,4096));Check(cudaMalloc(&sink,4));Check(cudaMalloc(&cycles,8));
  Check(cudaMemset(buffer,1,bytes));Check(cudaMemset(line,1,4096));
  std::array<double,5> cold{},warm{};
  log<<"offset_bytes\tprefetch\tmedian_cycles\tsamples\n";
  try {
    for(int col=0;col<5;++col)for(int hint=0;hint<2;++hint) {
      std::vector<unsigned long long> samples;
      for(int i=0;i<std::max(21,options.repeats);++i) {
        Evict<<<target.res.num_sms,128>>>(buffer,bytes/4,sink);
        Probe<<<1,32>>>(line,col*32,hint,cycles,sink);
        Check(cudaGetLastError());unsigned long long value;
        Check(cudaMemcpy(&value,cycles,8,cudaMemcpyDeviceToHost));samples.push_back(value);
      }
      std::sort(samples.begin(),samples.end());double median=samples[samples.size()/2];
      (hint?warm:cold)[col]=median;
      log<<col*32<<'\t'<<hint<<'\t'<<median<<'\t'<<samples.size()<<'\n';
    }
    // A neighbouring sector counts as brought in only with at least half
    // the latency reduction observed at the hinted address. Offset 128 is
    // the negative control; ambiguous measurements reject the calibration.
    double gain=cold[0]-warm[0];
    if(gain<0.15*cold[0] || cold[4]-warm[4]>0.5*gain)
      throw std::runtime_error("L2 prefetch sector experiment has no identifiable hit/miss separation");
    int covered=32;
    for(int col=1;col<4 && cold[col]-warm[col]>0.5*gain;++col)covered+=32;
    if(covered!=32 && covered!=64 && covered!=128)
      throw std::runtime_error("L2 prefetch experiment found a nonuniform sector footprint");
    target.CalibrationFor("bf16").l2_prefetch_bytes=covered;
  }catch(...) {cudaFree(buffer);cudaFree(line);cudaFree(sink);cudaFree(cycles);throw;}
  Check(cudaFree(buffer));Check(cudaFree(line));Check(cudaFree(sink));Check(cudaFree(cycles));
}
} // namespace tilemega::calib
