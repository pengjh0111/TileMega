// SPDX-License-Identifier: BSD-3-Clause
// Dithered cross-SM timestamps retain the SIMULATOR contention protocol.
#include <tilemega/Target/Calibration.h>
#include <tilemega/Codegen/tasks/EventSync.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
namespace tilemega::calib {
namespace {

__device__ inline unsigned long long Now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}

/// One event row, padded exactly as EventCounter is, so a row's traffic does
/// not land on a neighbour's cache line and R is really the number of lines.
struct alignas(128) Row {
  unsigned long long epoch;
  unsigned char padding[120];
};

/// CTAs [0, rows) publish, CTAs [rows, rows + consumers) poll.  Consumer c
/// polls row c % rows, so `consumers / rows` CTAs share a line.
template<int Spin, int Backoff>
__global__ void Contend(Row* rows_data, unsigned long long* publish_ns,
                        unsigned long long* observe_ns, unsigned int* count,
                        unsigned int* sense, unsigned int participants,
                        int rows, int consumers, int rounds, int guard_cycles,
                        int unused) {
  if (threadIdx.x != 0) return;
  unsigned int local_sense = 0;
  bool const publisher = blockIdx.x < static_cast<unsigned int>(rows);
  int const row = publisher ? static_cast<int>(blockIdx.x)
                            : (static_cast<int>(blockIdx.x) - rows) % rows;
  int const consumer = publisher ? -1 : static_cast<int>(blockIdx.x) - rows;

  for (int r = 0; r < rounds; ++r) {
    // Sense-reversing barrier over the resident participants: every consumer
    // must be inside its poll loop before any publish, or the measurement is
    // of arrival rather than propagation.  Same shape as TRACE_V2/calibrate.cu.
    local_sense ^= 1u;
    unsigned int const arrived = atomicAdd(count, 1u) + 1u;
    if (arrived == participants) {
      atomicExch(count, 0u);
      __threadfence();
      atomicExch(sense, local_sense);
    } else {
      while (atomicAdd(sense, 0u) != local_sense) { /* resident spin */ }
    }

    unsigned long long const need = static_cast<unsigned long long>(r) + 1ull;
    if (publisher) {
      // The guard covers barrier release skew; it is outside the stamp, so it
      // costs wall clock and not accuracy.  The dither is what makes averaging
      // work: without it the loop is periodic, the publish lands at a fixed
      // phase inside the 1024 ns tick, and every round reports exactly one tick
      // (measured -- that was this benchmark's first result).  The range spans
      // more than a tick at any plausible clock, so the phase is uniform and
      // the mean of the quantized differences converges on the true delay.
      // The dither depends on the round alone, so all R publishers release
      // together: R rows contending at the same instant is the regime mode 5
      // puts the device in, and staggering them would measure a gentler one.
      unsigned int const dither = (static_cast<unsigned int>(r) * 2654435761u) % 4096u;
      long long const until = clock64() + guard_cycles + dither;
      while (clock64() < until) { /* let the consumers settle into the poll */ }
      publish_ns[static_cast<std::size_t>(r) * rows + row] = Now();
      __threadfence();
      atomicExch(&rows_data[row].epoch, need);
    } else {
      // backoff_ns = 64 is the generated wait verbatim; 0 is the same loop with
      // the backoff removed, which is the only way to tell "contention is small"
      // apart from "the backoff hides it".  Neither arm changes the protocol.
      int spun=0;
      while(::tilemega::codegen::EventPoll(&rows_data[row].epoch)<need) {
        if(spun<Spin)++spun;
        else if constexpr(Backoff>0)__nanosleep(Backoff);
      }
      observe_ns[static_cast<std::size_t>(r) * consumers + consumer] = Now();
    }
  }
}

void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> T* Allocate(std::size_t n) {
  T* p=nullptr;Check(cudaMalloc(&p,n*sizeof(T)));Check(cudaMemset(p,0,n*sizeof(T)));return p;
}
struct HopPoint {double consumers,rows,ns;};
template<int Spin,int Backoff>
std::vector<HopPoint> Sweep(TargetSpec const& target,std::ostream& out,bool full) {
  int const rounds=4096,warmup=64,guard=20000;
  int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Contend<Spin,Backoff>,32,0));
  resident*=target.res.num_sms;
  std::vector<HopPoint> result;
  out<<"spin\tbackoff\tconsumers\trows\tmean_ns\ttrim_mean_ns\tinversions\n";
  for(int rows=1;rows<=target.res.num_sms;rows*=4)
    for(int consumers=rows;consumers<=2*target.res.num_sms;consumers*=2) {
      int participants=rows+consumers;
      if(participants>resident || (!full && (rows!=1||consumers!=1)))continue;
      std::size_t total=rounds+warmup;
      auto* data=Allocate<Row>(rows);auto* pub=Allocate<unsigned long long>(total*rows);
      auto* obs=Allocate<unsigned long long>(total*consumers);
      auto* count=Allocate<unsigned>(1);auto* sense=Allocate<unsigned>(1);
      Contend<Spin,Backoff><<<participants,32>>>(data,pub,obs,count,sense,participants,
          rows,consumers,rounds+warmup,guard,0);
      Check(cudaDeviceSynchronize());Check(cudaGetLastError());
      std::vector<unsigned long long> published(total*rows),observed(total*consumers);
      Check(cudaMemcpy(published.data(),pub,published.size()*8,cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(observed.data(),obs,observed.size()*8,cudaMemcpyDeviceToHost));
      Check(cudaFree(data));Check(cudaFree(pub));Check(cudaFree(obs));Check(cudaFree(count));Check(cudaFree(sense));
      std::vector<double> samples;int inversions=0;
      for(int r=warmup;r<rounds+warmup;++r)for(int c=0;c<consumers;++c) {
        double value=double(observed[r*consumers+c])-double(published[r*rows+c%rows]);
        inversions+=value<0;samples.push_back(value);
      }
      double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
      std::sort(samples.begin(),samples.end());
      auto keep=samples.size()-std::size_t(samples.size()*.001);
      double trimmed=std::accumulate(samples.begin(),samples.begin()+keep,0.0)/keep;
      result.push_back({double(consumers),double(rows),trimmed});
      out<<Spin<<'\t'<<Backoff<<'\t'<<consumers<<'\t'<<rows<<'\t'<<mean<<'\t'<<trimmed<<'\t'<<inversions<<'\n';
    }
  return result;
}
std::vector<HopPoint> SelectedSweep(TargetSpec const& target,std::ostream& out,bool full) {
  auto const& c=target.CalibrationFor("bf16");
  if(c.wait_backoff_grow!=1 || c.wait_backoff_cap_ns!=c.wait_backoff_ns)
    throw std::runtime_error("serving wait sweep requires a fixed backoff policy");
  if(c.wait_spin_iters==0 && c.wait_backoff_ns==0)return Sweep<0,0>(target,out,full);
  if(c.wait_spin_iters==0 && c.wait_backoff_ns==64)return Sweep<0,64>(target,out,full);
  if(c.wait_spin_iters==64 && c.wait_backoff_ns==64)return Sweep<64,64>(target,out,full);
  throw std::runtime_error("wait policy is outside the measured suite domain");
}

// Successful waits and publications are separate from the cross-worker hop.
// The barriers/fences match the default task protocol; EX-E3 variants are
// calibrated separately before being enabled in serving builds.
__global__ void TaskEvents(unsigned long long* event,unsigned long long* samples,
                           int kind,int rounds) {
  __shared__ unsigned long long start;
  for(int r=0;r<rounds;++r) {
    if(threadIdx.x==0){event[0]=1;event[1]=0;start=Now();}
    __syncthreads();
    if(kind==1) {
      if(threadIdx.x==0) {
        while(::tilemega::codegen::EventPoll(event)<1ull) {}
      }
      __syncthreads();__threadfence();
    } else if(kind==2) {
      __threadfence();__syncthreads();
      if(threadIdx.x==0) {
        auto ticket=atomicAdd(event+1,1ull);
        if(ticket+1==1ull){__threadfence();atomicExch(event,1ull);}
      }
    } else if(kind==3 && threadIdx.x==0) atomicExch(event,1ull);
    else if(kind==4 && threadIdx.x==0) (void)::tilemega::codegen::EventPoll(event);
    else if(kind==5) __threadfence();
    __syncthreads();
    if(threadIdx.x==0)samples[r]=Now()-start;
    __syncthreads();
  }
}
} // namespace

void MeasureServingWaitPolicy(TargetSpec& target,Options const&,std::ostream& raw) {
  double best=1e300;int spin=0,backoff=64;
  for(int candidate=0;candidate<3;++candidate) {
    auto points=candidate==0?Sweep<0,64>(target,raw,false):
        candidate==1?Sweep<64,64>(target,raw,false):Sweep<0,0>(target,raw,false);
    if(points.empty())throw std::runtime_error("no resident hop calibration cell");
    if(points.front().ns<best){best=points.front().ns;spin=candidate==1?64:0;backoff=candidate==2?0:64;}
  }
  target.calib_bf16.wait_spin_iters=spin;target.calib_bf16.wait_backoff_ns=backoff;
  target.calib_bf16.wait_backoff_grow=1;target.calib_bf16.wait_backoff_cap_ns=backoff;
}
void MeasureServingHop(TargetSpec& target,Options const&,std::ostream& raw) {
  auto fit=[](auto const& points) {
  double a[3][4]{};
  for(auto const& point:points) {
    double x[]={1,std::log2(1+point.consumers/point.rows),std::log2(point.rows)};
    for(int i=0;i<3;++i){for(int j=0;j<3;++j)a[i][j]+=x[i]*x[j];a[i][3]+=x[i]*point.ns;}
  }
  for(int i=0;i<3;++i) {
    int pivot=i;for(int j=i+1;j<3;++j)if(std::abs(a[j][i])>std::abs(a[pivot][i]))pivot=j;
    for(int j=0;j<4;++j)std::swap(a[i][j],a[pivot][j]);
    if(std::abs(a[i][i])<1e-12)throw std::runtime_error("hop fit is not identifiable on this device");
    double d=a[i][i];for(int j=i;j<4;++j)a[i][j]/=d;
    for(int k=0;k<3;++k)if(k!=i){double f=a[k][i];for(int j=i;j<4;++j)a[k][j]-=f*a[i][j];}
  }
  return std::vector<double>{a[0][3],a[1][3],a[2][3]};
  };
  target.serving_hop_coefficients=fit(SelectedSweep(target,raw,true));
  target.serving_legacy_hop_coefficients=fit(Sweep<0,64>(target,raw,true));
}
void MeasureServingEvents(TargetSpec& target,Options const& options,std::ostream& raw) {
  int const rounds=4096;auto* event=Allocate<unsigned long long>(2);
  auto* samples=Allocate<unsigned long long>(rounds);std::array<double,6> means{};
  raw<<"kind\trepeat\tround\tns\n";
  for(int kind=0;kind<6;++kind) {
    std::vector<double> repeats;
    for(int repeat=0;repeat<options.repeats;++repeat) {
      TaskEvents<<<1,128>>>(event,samples,kind,rounds);Check(cudaDeviceSynchronize());
      std::vector<unsigned long long> host(rounds);Check(cudaMemcpy(host.data(),samples,rounds*8,cudaMemcpyDeviceToHost));
      double sum=0;for(int r=64;r<rounds;++r){sum+=host[r];raw<<kind<<'\t'<<repeat<<'\t'<<r<<'\t'<<host[r]<<'\n';}
      repeats.push_back(sum/(rounds-64));
    }
    std::sort(repeats.begin(),repeats.end());means[kind]=repeats[repeats.size()/2];
  }
  Check(cudaFree(event));Check(cudaFree(samples));
  auto rate=[&](int kind,char const* unit){return TargetSpec::EventRate{std::max(0.0,means[kind]-means[0]),"measured",unit};};
  auto& e=target.event_bf16;e={};
  e.notify=rate(3,"ns/notify");e.poll=rate(4,"ns/poll");e.fence=rate(5,"ns/fence");
  e.task_publication=rate(2,"ns/publishing_runtime_task");e.task_wait=rate(1,"ns/waiting_runtime_task");
  e.method="per-task barrier/fence protocol, 4096 globaltimer samples; empty-loop subtraction; median across repeats";
}
} // namespace tilemega::calib
