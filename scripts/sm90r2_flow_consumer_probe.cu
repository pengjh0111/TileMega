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

__global__ __launch_bounds__(160,1)
void Consumer(tilemega::backend::Sm90DenseOperands p,int active,bool observe,
    U64* stamps,U64* empty_observations) {
  using namespace tilemega::backend;
  extern __shared__ __align__(1024) char arena[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(arena);ring.Initialize();
  auto& record=Sm90DenseOperandHeader::Bank(arena,0).gemm;
  if(threadIdx.x==0)record=p;
  auto* a=reinterpret_cast<BF*>(arena+1024);
  auto* b=a+2*16*512;
  auto* scratch=reinterpret_cast<float*>(arena+1024+G::kOperandBytes);
  for(int i=threadIdx.x;i<2*512*(16+32);i+=160)a[i]=BF(1.f);
  __syncthreads();
  if(blockIdx.x<unsigned(active)) {
    if(threadIdx.x<128) {
      U64 begin=Now();
      // Same outlined production Consume, operand-bank location and flags.
      G::template Consume<true,false>(record,ring,a,b,scratch,1,0);
      U64 end=Now();
      if((threadIdx.x&31)==0) {
        auto* row=stamps+blockIdx.x*8+threadIdx.x/32*2;row[0]=begin;row[1]=end;
      }
    }else {
      for(int iteration=0;iteration<G::kIterations;++iteration) {
        int slot=iteration%2,phase=(iteration/2)&1;
        Async::Wait(ring.empty+slot,phase);
        // Static operands are never overwritten. Supply the same 33 arrivals
        // without a memory transfer; this is a consumer-service surrogate.
        if(threadIdx.x==128)Async::Arrive(ring.full+slot);
        Async::CompleteCopies(ring.full+slot);
      }
      if(observe)for(int slot=0;slot<2;++slot) {
        U64 begin=Now();Async::Wait(ring.empty+slot,1);U64 ready=Now();
        if(threadIdx.x==128) {
          empty_observations[blockIdx.x*4+slot*2]=begin;
          empty_observations[blockIdx.x*4+slot*2+1]=ready;
        }
      }
    }
  }
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
  BF* result;float* norm;U64 *device,*empty;
  Check(cudaMalloc(&result,16*4096*2));Check(cudaMalloc(&norm,16*1024/8*4));Check(cudaMalloc(&device,Grid*8*8));
  Check(cudaMalloc(&empty,Grid*4*8));
  std::vector<float> sums(16*1024/8,8.f);Check(cudaMemcpy(norm,sums.data(),sums.size()*4,cudaMemcpyHostToDevice));
  tilemega::backend::Sm90DenseOperands p{};p.output=result;p.norm_sums=norm;p.epsilon=0;
  U64 before[2],after[2];Frequency<<<1,1>>>(device);Check(cudaMemcpy(before,device,16,cudaMemcpyDeviceToHost));
  std::ofstream raw(out/"raw.csv"),meta(out/"metadata.json");if(!raw || !meta)throw std::runtime_error("output failed");
  raw<<"fresh,active_ctas,observe_empty,trial,cta,warp,entry,returned,empty0_wait_begin,empty0_observed_ready,empty1_wait_begin,empty1_observed_ready\n";
  std::vector<U64> host(Grid*8),empty_host(Grid*4);std::vector<BF> output(16*4096);
  for(int active:{1,32,64,128})for(bool observe:{false,true})for(int trial=0;trial<17;++trial) {
    Check(cudaMemset(result,0xff,output.size()*2));
    Check(cudaMemset(empty,0,Grid*4*8));Consumer<<<Grid,160,Shared>>>(p,active,observe,device,empty);
    Check(cudaGetLastError());Check(cudaMemcpy(host.data(),device,host.size()*8,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(output.data(),result,output.size()*2,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(empty_host.data(),empty,empty_host.size()*8,cudaMemcpyDeviceToHost));
    for(int row=0;row<16;++row)for(int col=0;col<active*32;++col)
      if(output[row*4096+col].raw()!=BF(1024.f).raw())throw std::runtime_error("consumer output mismatch");
    if(!trial)continue;
    for(int cta=0;cta<active;++cta)for(int warp=0;warp<4;++warp) {
      auto* record=host.data()+cta*8+warp*2;if(record[1]<record[0])throw std::runtime_error("timestamp order failed");
      raw<<fresh<<','<<active<<','<<observe<<','<<trial<<','<<cta<<','<<warp<<','<<record[0]<<','<<record[1];
      for(int word=0;word<4;++word)raw<<','<<empty_host[cta*4+word];raw<<'\n';
    }
  }
  Frequency<<<1,1>>>(device);Check(cudaMemcpy(after,device,16,cudaMemcpyDeviceToHost));
  if(!before[0] || !before[1] || !after[0] || !after[1])throw std::runtime_error("invalid frequency bracket");
  double pre=double(before[0])/before[1],post=double(after[0])/after[1],drift=std::abs(post/pre-1)*100;
  meta<<"{\"schema\":1,\"fresh\":"<<fresh<<",\"numerical_failures\":0,\"frequency_drift_pct\":"<<drift
      <<",\"grid\":132,\"active_ctas_max\":128,\"threads\":160,\"shared_bytes\":"<<Shared
      <<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes
      <<",\"observed_residency\":"<<occupancy
      <<",\"observation\":\"actual_Consume_entry_to_return_with_static_operands_and_dummy_producer\",\"includes\":[\"MMA\",\"full_empty_protocol\",\"finalize\",\"epilogue\",\"generation_publication\"],\"excludes\":[\"memory_transfer\",\"grid_barrier\"],\"production_binary_equivalence\":\"requires_separate_function_byte_audit\",\"performance_eligible\":false}\n";
  raw.close();meta.close();if(drift>2)throw std::runtime_error("frequency bracket exceeds 2 percent");return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
