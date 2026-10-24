// SPDX-License-Identifier: BSD-3-Clause
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
#include <array>
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
constexpr int Grid=132,Slots=2,WeightBytes=32768,ActivationBytes=16384;
// The retained Qwen B16 program reserves its largest (o) arena at every stage.
constexpr int Shared=tilemega::backend::Sm90DenseLayout::Gemm(8,2048,2,4).end;
constexpr int Words=7,Trials=17;

void Check(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
__device__ U64 Now() {U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;}
__host__ __device__ unsigned Pattern(std::size_t i) {return unsigned(i*747796405u+2891336453u);}
__global__ void Initialize(unsigned* data,std::size_t count) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)data[i]=Pattern(i);
}
__global__ void Touch(unsigned const* data,std::size_t count,unsigned* sums) {
  unsigned sum=0;for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)sum^=data[i];
  if(sum)atomicXor(sums,sum);
}
__global__ void Frequency(U64* out) {
  if(threadIdx.x)return;auto time=Now(),cycle=clock64();while(Now()-time<2000000) {}
  out[0]=clock64()-cycle;out[1]=Now()-time;
}

// This isolates the literal production transfer packet, not the GEMM body.
// A full request includes 32 cp.async arrivals even in the weight-only arm.
template<bool Activation>
__global__ __launch_bounds__(160,1)
void Transfer(unsigned const* weights,unsigned const* activation,int operations,int active,
    U64* output,unsigned* errors) {
  extern __shared__ __align__(1024) char arena[];
  auto& ring=*reinterpret_cast<tilemega::backend::Sm90DenseRingState*>(arena);
  auto* a=reinterpret_cast<BF*>(arena+1024);
  auto* b=a+Slots*16*512;
  __shared__ U64 stamps[8][Words];
  __shared__ U64 warp_ready[8][4],warp_retired[8][4];
  ring.Initialize();
  if(blockIdx.x<unsigned(active)) {
    if(threadIdx.x>=128) {
      int lane=threadIdx.x-128;
      for(int sequence=0;sequence<operations;++sequence) {
        int slot=sequence%Slots,phase=(sequence/Slots)&1;
        if(!lane)stamps[sequence][0]=Now();
        Async::Wait(ring.empty+slot,phase);
        if(!lane) {
          stamps[sequence][1]=Now();
          Async::ExpectTx(ring.full+slot,WeightBytes);
          Async::Bulk(b+slot*32*512,weights+(std::size_t(blockIdx.x)*operations+sequence)*WeightBytes/4,
              WeightBytes,ring.full+slot);
        }
        if constexpr(Activation)for(int v=lane*8;v<16*512;v+=32*8) {
          int row=v/512,col=v%512,tile=sequence%2;
          Async::Copy16Bytes(a+slot*16*512+G::LayoutA{}(row,col),
              reinterpret_cast<BF const*>(activation)+row*1024+tile*512+col,16);
        }
        Async::CompleteCopies(ring.full+slot);
        if(!lane)stamps[sequence][2]=Now();
      }
    }else {
      int warp=threadIdx.x/32,lane=threadIdx.x%32;
      for(int sequence=0;sequence<operations;++sequence) {
        int slot=sequence%Slots,phase=(sequence/Slots)&1;
        Async::Wait(ring.full+slot,phase);
        if(!lane)warp_ready[sequence][warp]=Now();
        if(threadIdx.x==0) {
          auto word=*reinterpret_cast<unsigned const*>(b+slot*32*512);
          if(word!=Pattern((std::size_t(blockIdx.x)*operations+sequence)*WeightBytes/4))atomicAdd(errors,1);
          if constexpr(Activation) {
            auto copy=*reinterpret_cast<unsigned const*>(a+slot*16*512+G::LayoutA{}(0,0));
            if(copy!=Pattern((sequence%2)*512/2))atomicAdd(errors,1);
          }
        }
        // Every reader retires before allowing the next slot generation.
        __syncwarp();Async::Arrive(ring.empty+slot);
        if(!lane)warp_retired[sequence][warp]=Now();
      }
    }
  }
  __syncthreads();
  if(threadIdx.x==0 && blockIdx.x<unsigned(active))for(int i=0;i<operations;++i) {
    U64 ready_min=~0ull,ready_max=0,retired=0;
    for(int warp=0;warp<4;++warp) {
      ready_min=min(ready_min,warp_ready[i][warp]);ready_max=max(ready_max,warp_ready[i][warp]);
      retired=max(retired,warp_retired[i][warp]);
    }
    auto* row=output+(blockIdx.x*8+i)*Words;
    row[0]=stamps[i][0];row[1]=stamps[i][1];row[2]=stamps[i][2];
    row[3]=ready_min;row[4]=ready_max;row[5]=retired;row[6]=Now();
  }
  ring.Invalidate();
}

void RequireGuard() {
  auto visible=std::getenv("CUDA_VISIBLE_DEVICES"),index=std::getenv("TILEMEGA_DEVICE_INDEX");
  auto held=std::getenv("TILEMEGA_GPU_LOCK_HELD"),lock=std::getenv("TILEMEGA_GPU_LOCK");
  if(!visible || !index || std::strcmp(visible,index) || !held || std::strcmp(held,"1") || !lock)
    throw std::runtime_error("explicit guarded physical device required");
  int gpu=std::stoi(index);if(!((gpu>=0 && gpu<=4)||gpu==7))throw std::runtime_error("excluded timing GPU");
  if(std::string(lock)!="/tmp/tilemega_gpu"+std::to_string(gpu)+".lock")throw std::runtime_error("wrong GPU lock");
  int fd=open(lock,O_CREAT|O_RDWR,0600);if(fd<0)throw std::runtime_error("cannot inspect lock");
  int status=flock(fd,LOCK_EX|LOCK_NB),saved=errno;if(status==0)flock(fd,LOCK_UN);close(fd);
  if(status==0 || (saved!=EAGAIN && saved!=EWOULDBLOCK))throw std::runtime_error("outer GPU lock not held");
}

int main(int argc,char** argv) {try {
  if(argc!=5 || std::strcmp(argv[1],"--fresh") || std::strcmp(argv[3],"--out"))
    throw std::runtime_error("usage: --fresh 0..4 --out NEW_DIRECTORY");
  int fresh=std::stoi(argv[2]);if(fresh<0 || fresh>4)throw std::runtime_error("fresh outside protocol");
  RequireGuard();std::filesystem::path out=argv[4];
  if(std::filesystem::exists(out))throw std::runtime_error("preserve prior results");std::filesystem::create_directories(out);
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0 || prop.multiProcessorCount!=Grid)throw std::runtime_error("expected H800 sm90a");
  for(auto kernel:{Transfer<false>,Transfer<true>}) {
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
    int occupancy=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,160,Shared));
    if(occupancy!=1)throw std::runtime_error("transfer probe occupancy mismatch");
  }
  constexpr std::size_t Bytes=std::size_t(128)*8*WeightBytes,ScrubBytes=256ull<<20;
  unsigned *weights,*activation,*scrub,*errors,*sums;U64* device;
  Check(cudaMalloc(&weights,Bytes));Check(cudaMalloc(&activation,16*1024*2));Check(cudaMalloc(&scrub,ScrubBytes));
  Check(cudaMalloc(&errors,4));Check(cudaMalloc(&sums,4));Check(cudaMalloc(&device,Grid*8*Words*8));
  Check(cudaMemset(errors,0,4));Check(cudaMemset(sums,0,4));
  Initialize<<<Grid*4,256>>>(weights,Bytes/4);Initialize<<<Grid*4,256>>>(activation,16*1024/2);
  Initialize<<<Grid*4,256>>>(scrub,ScrubBytes/4);Check(cudaGetLastError());
  U64 before[2],after[2];Frequency<<<1,1>>>(device);Check(cudaMemcpy(before,device,16,cudaMemcpyDeviceToHost));
  std::ofstream raw(out/"raw.csv"),meta(out/"metadata.json");if(!raw || !meta)throw std::runtime_error("output failed");
  raw<<"fresh,cache_preparation,activation,active_ctas,operations,trial,cta,sequence,wait_begin,issue_begin,packet_end,first_warp_ready,all_warps_ready,all_warps_retired,joined\n";
  std::vector<U64> host(Grid*8*Words);
  for(int active:{1,32,64,128})for(int operations:{1,2,8})for(bool with_a:{false,true})for(bool warm:{false,true}) {
    for(int trial=0;trial<Trials;++trial) {
      if(warm)Touch<<<Grid*4,256>>>(weights,std::size_t(active)*operations*WeightBytes/4,sums);
      else Touch<<<Grid*4,256>>>(scrub,ScrubBytes/4,sums);
      Touch<<<Grid,256>>>(activation,16*1024/2,sums);
      if(with_a)Transfer<true><<<Grid,160,Shared>>>(weights,activation,operations,active,device,errors);
      else Transfer<false><<<Grid,160,Shared>>>(weights,activation,operations,active,device,errors);
      Check(cudaGetLastError());Check(cudaMemcpy(host.data(),device,host.size()*8,cudaMemcpyDeviceToHost));
      if(!trial)continue;
      for(int cta=0;cta<active;++cta)for(int seq=0;seq<operations;++seq) {
        auto* row=host.data()+(cta*8+seq)*Words;
        if(row[1]<row[0] || row[2]<row[1] || row[3]<row[1] || row[4]<row[3] || row[5]<row[4])
          throw std::runtime_error("invalid timestamp order");
        raw<<fresh<<','<<(warm?"warm_touch":"scrubbed_surrogate")<<','<<with_a<<','<<active<<','<<operations<<','<<trial<<','<<cta<<','<<seq;
        for(int word=0;word<Words;++word)raw<<','<<row[word];raw<<'\n';
      }
    }
  }
  Frequency<<<1,1>>>(device);Check(cudaMemcpy(after,device,16,cudaMemcpyDeviceToHost));
  unsigned failures=0;Check(cudaMemcpy(&failures,errors,4,cudaMemcpyDeviceToHost));
  if(!before[0] || !before[1] || !after[0] || !after[1])throw std::runtime_error("invalid frequency bracket");
  double pre=double(before[0])/before[1],post=double(after[0])/after[1],drift=std::abs(post/pre-1)*100;
  meta<<"{\"schema\":1,\"fresh\":"<<fresh<<",\"numerical_failures\":"<<failures
      <<",\"frequency_before\":"<<pre<<",\"frequency_after\":"<<post<<",\"frequency_drift_pct\":"<<drift
      <<",\"grid\":132,\"threads\":160,\"slots\":2,\"weight_bytes\":32768,\"activation_bytes\":16384,\"shared_bytes\":"<<Shared
      <<",\"production_binary_equivalent\":false,\"literal_transfer_operations\":true,\"cache_hit_ratio_measured\":false,\"hardware_admission_time_observed\":false,\"first_byte_observed\":false,\"performance_eligible\":false}\n";
  raw.close();meta.close();if(failures)throw std::runtime_error("transfer sentinel mismatch");
  if(drift>2)throw std::runtime_error("frequency bracket exceeds 2 percent");
  return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
