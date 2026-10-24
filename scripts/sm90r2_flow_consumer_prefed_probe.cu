// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DN_VECTOR_SUMS 1
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <cuda_runtime.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

using U64=unsigned long long;
using BF=cutlass::bfloat16_t;
using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
using G=tilemega::backend::Sm90DenseGemm<16,4096,1024,32,512,2,1,4,
    tilemega::backend::ServingEpilogueOp::kStore,8>;
constexpr int Grid=132,Shared=tilemega::backend::Sm90DenseLayout::Gemm(8,2048,2,4).end;
void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ U64 Now() {U64 time;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(time)::"memory");return time;}
__global__ void Frequency(U64* out) {
  if(threadIdx.x)return;U64 t=Now(),c=clock64();while(Now()-t<2000000) {}
  out[0]=clock64()-c;out[1]=Now()-t;
}

static_assert(Shared==199808 && G::kIterations==2);
static_assert(tilemega::backend::Sm90DenseOperandHeader::kOffset==448);
struct alignas(64) GridCounter {U64 arrivals;};
struct CtaRecord {
  U64 smid,full_ready_mask,full0_confirmed,full1_confirmed;
  U64 setup_acquired,v3_arrive,v3_acquired;
};

// Native Program::Boundary's release/acquire protocol, after the Run CTA join.
// The acquire timestamp precedes the final CTA join; it is not a release trace.
__device__ __forceinline__ void NativeV3(GridCounter* event,U64* arrive,U64* acquired) {
  __syncthreads();
  if(threadIdx.x==0) {
    if(arrive)*arrive=Now();
    auto* counter=&event->arrivals;
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
    U64 seen;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
    while(seen<gridDim.x);
    if(acquired)*acquired=Now();
  }
  __syncthreads();
}

__device__ __forceinline__ void SignalSlot(tilemega::backend::Sm90DenseRingState& ring,int slot) {
  Async::Wait(ring.empty+slot,0);
  if(threadIdx.x==128)Async::Arrive(ring.full+slot);
  Async::CompleteCopies(ring.full+slot);
}

__global__ __launch_bounds__(160,1)
void Consumer(tilemega::backend::Sm90DenseOperands p,int active,bool prefeed,bool observe,
    U64* stamps,U64* empty_observations,CtaRecord* ctas,GridCounter* events) {
  using namespace tilemega::backend;
  extern __shared__ __align__(1024) char arena[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(arena);ring.Initialize();
  auto& record=Sm90DenseOperandHeader::Bank(arena,0).gemm;
  auto* cta=ctas+blockIdx.x;
  if(threadIdx.x==0) {
    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    cta->smid=smid;record=p;
  }
  auto* a=reinterpret_cast<BF*>(arena+1024);
  auto* b=a+2*16*512;
  auto* scratch=reinterpret_cast<float*>(arena+1024+G::kOperandBytes);
  for(int i=threadIdx.x;i<2*512*(16+32);i+=160)a[i]=BF(1.f);
  __syncthreads();
  bool work=blockIdx.x<unsigned(active);
  if(work && prefeed && threadIdx.x>=128)
    for(int slot=0;slot<2;++slot)SignalSlot(ring,slot);
  __syncthreads();
  if(work && prefeed && threadIdx.x==0) {
    // A wait is non-consuming. No later producer can change these full phases.
    Async::Wait(ring.full,0);cta->full0_confirmed=Now();
    Async::Wait(ring.full+1,0);cta->full1_confirmed=Now();
    cta->full_ready_mask=unsigned(Async::Ready(ring.full,0)) |
        (unsigned(Async::Ready(ring.full+1,0))<<1);
  }
  // Begin every context after a real native V3 boundary. Idle CTAs then wait
  // at the next V3 boundary while active CTAs run Consume and its observer.
  NativeV3(events,nullptr,&cta->setup_acquired);
  if(work) {
    if(threadIdx.x<128) {
      U64 begin=Now();
      G::template Consume<true,false>(record,ring,a,b,scratch,1,0);
      U64 end=Now();
      if((threadIdx.x&31)==0) {
        auto* row=stamps+blockIdx.x*8+threadIdx.x/32*2;row[0]=begin;row[1]=end;
      }
    }else {
      if(!prefeed)for(int slot=0;slot<2;++slot)SignalSlot(ring,slot);
      if(observe)for(int slot=0;slot<2;++slot) {
        U64 begin=Now();Async::Wait(ring.empty+slot,1);U64 ready=Now();
        if(threadIdx.x==128) {
          empty_observations[blockIdx.x*4+slot*2]=begin;
          empty_observations[blockIdx.x*4+slot*2+1]=ready;
        }
      }
    }
  }
  NativeV3(events+1,&cta->v3_arrive,&cta->v3_acquired);
  ring.Invalidate();
}

void RequireGuard() {
  auto visible=std::getenv("CUDA_VISIBLE_DEVICES"),index=std::getenv("TILEMEGA_DEVICE_INDEX");
  auto held=std::getenv("TILEMEGA_GPU_LOCK_HELD"),lock=std::getenv("TILEMEGA_GPU_LOCK");
  if(!visible || !index || std::strcmp(visible,index) || !held || std::strcmp(held,"1") || !lock)
    throw std::runtime_error("explicit physical guard required");
  int gpu=std::stoi(index);if(!((gpu>=0 && gpu<=4)||gpu==7))throw std::runtime_error("excluded timing GPU");
  if(std::string(lock)!="/tmp/tilemega_gpu"+std::to_string(gpu)+".lock")throw std::runtime_error("wrong GPU lock");
  int fd=open(lock,O_CREAT|O_RDWR,0600);if(fd<0)throw std::runtime_error("lock inspection failed");
  int status=flock(fd,LOCK_EX|LOCK_NB),saved=errno;if(status==0)flock(fd,LOCK_UN);close(fd);
  if(status==0 || (saved!=EAGAIN && saved!=EWOULDBLOCK))throw std::runtime_error("outer lock not held");
}

int main(int argc,char** argv) {try {
  if(argc!=5 || std::strcmp(argv[1],"--fresh") || std::strcmp(argv[3],"--out"))
    throw std::runtime_error("usage: --fresh 0..4 --out NEW_DIRECTORY");
  int fresh=std::stoi(argv[2]);if(fresh<0 || fresh>4)throw std::runtime_error("fresh outside protocol");
  RequireGuard();std::filesystem::path out=argv[4];
  if(std::filesystem::exists(out))throw std::runtime_error("preserve prior output");std::filesystem::create_directories(out);
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0 || prop.multiProcessorCount!=Grid)throw std::runtime_error("expected H800");
  Check(cudaFuncSetAttribute(Consumer,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
  int occupancy=0;cudaFuncAttributes attr{};Check(cudaFuncGetAttributes(&attr,Consumer));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,Consumer,160,Shared));
  if(occupancy!=1)throw std::runtime_error("consumer probe occupancy mismatch");
  BF* result;float* norm;U64 *device,*empty;CtaRecord* ctas;GridCounter* events;
  Check(cudaMalloc(&result,16*4096*sizeof(BF)));Check(cudaMalloc(&norm,16*1024/8*sizeof(float)));
  Check(cudaMalloc(&device,Grid*8*sizeof(U64)));Check(cudaMalloc(&empty,Grid*4*sizeof(U64)));
  Check(cudaMalloc(&ctas,Grid*sizeof(CtaRecord)));Check(cudaMalloc(&events,2*sizeof(GridCounter)));
  std::vector<float> sums(16*1024/8,8.f);Check(cudaMemcpy(norm,sums.data(),sums.size()*sizeof(float),cudaMemcpyHostToDevice));
  tilemega::backend::Sm90DenseOperands p{};p.output=result;p.norm_sums=norm;p.epsilon=0;
  U64 before[2],after[2];Frequency<<<1,1>>>(device);Check(cudaMemcpy(before,device,16,cudaMemcpyDeviceToHost));
  std::ofstream raw(out/"raw.csv"),cta_raw(out/"ctas.csv"),meta(out/"metadata.json");
  if(!raw || !cta_raw || !meta)throw std::runtime_error("output failed");
  raw<<"fresh,active_ctas,feed_mode,observe_empty,trial,cta,smid,warp,entry,returned,full_ready_mask,full0_confirmed,full1_confirmed,setup_acquired,v3_arrive,v3_acquired,empty0_wait_begin,empty0_observed_ready,empty1_wait_begin,empty1_observed_ready\n";
  cta_raw<<"fresh,active_ctas,feed_mode,observe_empty,trial,cta,smid,has_work,full_ready_mask,full0_confirmed,full1_confirmed,setup_acquired,v3_arrive,v3_acquired\n";
  std::vector<U64> host(Grid*8),empty_host(Grid*4);std::vector<BF> output(16*4096);
  std::vector<CtaRecord> cta_host(Grid);std::vector<GridCounter> event_host(2);
  for(int active:{1,32,64,128})for(int trial=0;trial<17;++trial)
    for(int feed_index=0;feed_index<2;++feed_index)for(int observer_index=0;observer_index<2;++observer_index) {
    bool prefeed=feed_index^(trial&1),observe=observer_index^((trial>>1)&1);
    Check(cudaMemset(result,0xff,output.size()*sizeof(BF)));Check(cudaMemset(device,0,host.size()*sizeof(U64)));
    Check(cudaMemset(empty,0,empty_host.size()*sizeof(U64)));Check(cudaMemset(ctas,0,Grid*sizeof(CtaRecord)));
    Check(cudaMemset(events,0,2*sizeof(GridCounter)));
    Consumer<<<Grid,160,Shared>>>(p,active,prefeed,observe,device,empty,ctas,events);
    Check(cudaGetLastError());Check(cudaMemcpy(host.data(),device,host.size()*sizeof(U64),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(output.data(),result,output.size()*sizeof(BF),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(empty_host.data(),empty,empty_host.size()*sizeof(U64),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(cta_host.data(),ctas,cta_host.size()*sizeof(CtaRecord),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(event_host.data(),events,event_host.size()*sizeof(GridCounter),cudaMemcpyDeviceToHost));
    for(auto const& event:event_host)if(event.arrivals!=Grid)throw std::runtime_error("V3 arrival count mismatch");
    for(int row=0;row<16;++row)for(int col=0;col<active*32;++col)
      if(output[row*4096+col].raw()!=BF(1024.f).raw())throw std::runtime_error("consumer output mismatch");
    std::vector<U64> smids;
    for(int cta=0;cta<Grid;++cta) {
      auto const& info=cta_host[cta];smids.push_back(info.smid);
      if(!info.setup_acquired || info.v3_arrive<info.setup_acquired || info.v3_acquired<info.v3_arrive)
        throw std::runtime_error("V3 timestamp order failed");
      if(cta<active && prefeed && (info.full_ready_mask!=3 || !info.full0_confirmed ||
          info.full1_confirmed<info.full0_confirmed || info.setup_acquired<info.full1_confirmed))
        throw std::runtime_error("both full slots were not confirmed before Consume");
      if((!prefeed || cta>=active) && (info.full_ready_mask || info.full0_confirmed || info.full1_confirmed))
        throw std::runtime_error("unexpected prefeed confirmation");
      if(!trial)continue;
      cta_raw<<fresh<<','<<active<<','<<(prefeed?"prefed":"concurrent")<<','<<observe<<','<<trial<<','<<cta<<','<<info.smid<<','<<(cta<active)<<','
          <<info.full_ready_mask<<','<<info.full0_confirmed<<','<<info.full1_confirmed<<','<<info.setup_acquired<<','<<info.v3_arrive<<','<<info.v3_acquired<<'\n';
    }
    std::sort(smids.begin(),smids.end());
    if(std::unique(smids.begin(),smids.end())-smids.begin()!=Grid)throw std::runtime_error("132 resident CTAs did not map to distinct SMIDs");
    for(int cta=0;cta<active;++cta)for(int warp=0;warp<4;++warp) {
      auto* record=host.data()+cta*8+warp*2;auto const& info=cta_host[cta];
      if(record[0]<info.setup_acquired || record[1]<record[0] || info.v3_arrive<record[1])
        throw std::runtime_error("Consume timestamp order failed");
      if(prefeed && (info.full_ready_mask!=3 || record[0]<info.full1_confirmed))
        throw std::runtime_error("Consume entered before full-ready confirmation");
      for(int slot=0;slot<2;++slot) {
        auto* observation=empty_host.data()+cta*4+slot*2;
        if(observe ? (!observation[0] || observation[1]<observation[0] || info.v3_arrive<observation[1]) : (observation[0] || observation[1]))
          throw std::runtime_error("empty observer control failed");
      }
      if(!trial)continue;
      raw<<fresh<<','<<active<<','<<(prefeed?"prefed":"concurrent")<<','<<observe<<','<<trial<<','<<cta<<','<<info.smid<<','<<warp<<','<<record[0]<<','<<record[1]
          <<','<<info.full_ready_mask<<','<<info.full0_confirmed<<','<<info.full1_confirmed<<','<<info.setup_acquired<<','<<info.v3_arrive<<','<<info.v3_acquired;
      for(int word=0;word<4;++word)raw<<','<<empty_host[cta*4+word];raw<<'\n';
    }
  }
  Frequency<<<1,1>>>(device);Check(cudaMemcpy(after,device,16,cudaMemcpyDeviceToHost));
  if(!before[0] || !before[1] || !after[0] || !after[1])throw std::runtime_error("invalid frequency bracket");
  double pre=double(before[0])/before[1],post=double(after[0])/after[1],drift=std::abs(post/pre-1)*100;
  meta<<"{\"schema\":2,\"fresh\":"<<fresh<<",\"numerical_failures\":0,\"frequency_drift_pct\":"<<drift
      <<",\"grid\":132,\"contexts\":16,\"active_ctas\":[1,32,64,128],\"feed_modes\":[\"concurrent\",\"prefed\"],\"observer_modes\":[false,true],\"measured_trials\":16,\"threads\":160,\"shared_bytes\":"<<Shared
      <<",\"record_offset\":448,\"slots\":2,\"iterations\":2,\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes
      <<",\"observed_residency\":"<<occupancy
      <<",\"smids\":\"actual_per_CTA_in_raw_and_ctas_csv\",\"idle_CTAs\":\"resident_at_native_V3_release_acquire_until_all_active_CTAs_finish\",\"prefeed_confirmation\":\"both_full_phases_waited_and_ready_mask_3_before_setup_V3_and_timed_Consume_entry\",\"observation\":\"actual_Consume_entry_to_return_with_static_operands_and_dummy_producer\",\"includes\":[\"MMA\",\"full_empty_protocol\",\"finalize\",\"epilogue\",\"generation_publication\"],\"excludes\":[\"memory_transfer\",\"grid_barriers_outside_timed_Consume\"],\"prefed_is_pure_compute\":false,\"production_binary_equivalence\":\"requires_separate_function_byte_audit\",\"synchronization_acceptance\":\"requires_5_fresh_processes\",\"performance_eligible\":false}\n";
  raw.close();cta_raw.close();meta.close();if(drift>2)throw std::runtime_error("frequency bracket exceeds 2 percent");return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
