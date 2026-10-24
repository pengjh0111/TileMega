// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

using U64=unsigned long long;
using Op=tilemega::backend::ServingEpilogueOp;
template<int N_,int K_,int TN_,int TK_,int S_,int WN_,int WK_,Op O_>
struct Shape {
  static constexpr int N=N_,K=K_,TN=TN_,TK=TK_,Slots=S_,WN=WN_,WK=WK_;
  using G=tilemega::backend::Sm90DenseGemm<1,N,K,TN,TK,Slots,WN,WK,O_,16,1>;
};
using Gate0=Shape<28672,4096,224,64,2,1,4,Op::kSwiGLU>;
using Gate3=Shape<28672,4096,224,128,3,4,1,Op::kSwiGLU>;
using Down0=Shape<4096,14336,32,256,3,1,4,Op::kResidual>;
using Down3=Shape<4096,14336,32,256,3,4,1,Op::kResidual>;
constexpr int Active=128,Trials=8;
char const* Modes[] = {"mma_only","mma_finalize","finalize_only"};
char const* Labels[] = {"gate_C0","gate_C3","down_C0","down_C3"};
constexpr int Reserved[] = {136320,230528,136320,230528};

void Check(cudaError_t value) {if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
__device__ __forceinline__ U64 Ticks() {
  U64 value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ U64 Global() {
  U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
}
__global__ void Frequency(U64* output) {
  if(threadIdx.x)return;
  U64 begin=Global(),clock=Ticks();while(Global()-begin<2000000ull) {}
  U64 end_clock=Ticks(),end=Global();output[0]=end_clock-clock;output[1]=end-begin;
}

template<class S>
__global__ __launch_bounds__(160,1)
void FinalizeProbe(U64* stamps,unsigned* errors,int mode) {
  using namespace cute;using G=typename S::G;using Element=typename G::Element;
  if(blockIdx.x>=Active)return;
  extern __shared__ __align__(1024) char shared[];
  auto* a=reinterpret_cast<Element*>(shared+1024);auto* b=a+S::Slots*16*S::TK;
  auto* scratch=reinterpret_cast<float*>(shared+1024+G::kOperandBytes);
  for(int i=threadIdx.x;i<S::Slots*16*S::TK;i+=160) {
    int slot=i/(16*S::TK),row=i/S::TK%16,col=i%S::TK;
    a[slot*16*S::TK+typename G::LayoutA{}(row,col)]=Element(row==0?1:0);
  }
  for(int i=threadIdx.x;i<S::Slots*S::TN*S::TK;i+=160)b[i]=Element(1);
  __syncthreads();if(threadIdx.x>=128)return;
  int tid=threadIdx.x,lane=tid%(32*S::WN),split=tid/(32*S::WN);
  typename G::Mma mma;auto thread=mma.get_slice(lane);
  auto owned=thread.partition_C(make_identity_tensor(cute::Shape<_16,Int<S::TN>>{}));
  auto accum=partition_fragment_C(mma,cute::Shape<_16,Int<S::TN>>{});clear(accum);
  if(mode==2)for_each(make_int_sequence<size(accum)>{},[&](auto i) {
    accum(i)=get<0>(owned(i))==0?float(S::K/S::WK):0.f;
  });
  U64 begin=Ticks(),end=0;
  if(mode!=2) {
    #pragma unroll 1
    for(int iteration=0;iteration<G::kIterations;++iteration) {
      int slot=iteration%S::Slots;auto* sa=a+slot*16*S::TK;auto* sb=b+slot*S::TN*S::TK;
      // The production wait clobbers shared memory before each MmaSlice.
      asm volatile("" ::: "memory");
      if constexpr(S::WK==1)G::template MmaSlice<0>(sa,sb,accum);
      else if constexpr(S::WK==2) {
        if(split==0)G::template MmaSlice<0>(sa,sb,accum);else G::template MmaSlice<1>(sa,sb,accum);
      }else {
        if(split==0)G::template MmaSlice<0>(sa,sb,accum);
        else if(split==1)G::template MmaSlice<1>(sa,sb,accum);
        else if(split==2)G::template MmaSlice<2>(sa,sb,accum);
        else G::template MmaSlice<3>(sa,sb,accum);
      }
    }
  }
  if(mode==0)end=Ticks();
  // Keep the frozen Consume finalize sequence, including fixed rank order.
  for(int i=0;i<size(accum);++i)
    scratch[split*16*S::TN+get<0>(owned(i))*S::TN+get<1>(owned(i))]=accum(i);
  tilemega::codegen::executor::ComputeSync();
  if constexpr(S::WK>1) {
    for(int i=tid;i<16*S::TN;i+=128) {
      float sum=scratch[i];
      #pragma unroll
      for(int rank=1;rank<S::WK;++rank)sum+=scratch[rank*16*S::TN+i];
      scratch[i]=sum;
    }
    tilemega::codegen::executor::ComputeSync();
  }
  if(mode!=0)end=Ticks();
  if((tid&31)==0) {
    stamps[blockIdx.x*8+(tid/32)*2]=begin;
    stamps[blockIdx.x*8+(tid/32)*2+1]=end;
  }
  // Validate after timing and keep the finalized tile observable in every mode.
  for(int i=tid;i<16*S::TN;i+=128)
    if(scratch[i]!=(i/S::TN==0?float(S::K):0.f))atomicAdd(errors,1);
}

struct Facts {int tn,tk,wn,wk,slots,k,registers,local,resident,shared;};
template<class S> Facts Configure(int shared) {
  Check(cudaFuncSetAttribute(FinalizeProbe<S>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  cudaFuncAttributes attributes{};Check(cudaFuncGetAttributes(&attributes,FinalizeProbe<S>));
  int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,FinalizeProbe<S>,160,shared));
  if(resident!=1 || attributes.localSizeBytes)throw std::runtime_error("probe occupancy/local-memory mismatch");
  return {S::TN,S::TK,S::WN,S::WK,S::Slots,S::K,attributes.numRegs,int(attributes.localSizeBytes),resident,shared};
}
void Launch(int shape,int mode,int grid,U64* stamps,unsigned* errors) {
  switch(shape) {
    case 0:FinalizeProbe<Gate0><<<grid,160,Reserved[0]>>>(stamps,errors,mode);break;
    case 1:FinalizeProbe<Gate3><<<grid,160,Reserved[1]>>>(stamps,errors,mode);break;
    case 2:FinalizeProbe<Down0><<<grid,160,Reserved[2]>>>(stamps,errors,mode);break;
    case 3:FinalizeProbe<Down3><<<grid,160,Reserved[3]>>>(stamps,errors,mode);break;
    default:throw std::runtime_error("unknown shape");
  }
  Check(cudaGetLastError());
}
void RequireGuard() {
  auto visible=std::getenv("CUDA_VISIBLE_DEVICES"),index=std::getenv("TILEMEGA_DEVICE_INDEX");
  auto held=std::getenv("TILEMEGA_GPU_LOCK_HELD"),lock=std::getenv("TILEMEGA_GPU_LOCK");
  if(!index || !visible || std::strcmp(index,visible) || !held || std::strcmp(held,"1") || !lock)
    throw std::runtime_error("guard and explicit matching device variables required");
  int gpu=std::stoi(index);if(!((gpu>=0 && gpu<=4) || gpu==7))throw std::runtime_error("timing device excluded");
  if(std::string(lock)!="/tmp/tilemega_gpu"+std::to_string(gpu)+".lock")throw std::runtime_error("wrong per-card lock");
  int descriptor=open(lock,O_CREAT|O_RDWR,0600);if(descriptor<0)throw std::runtime_error("cannot inspect per-card lock");
  int status=flock(descriptor,LOCK_EX|LOCK_NB),saved=errno;
  if(status==0)flock(descriptor,LOCK_UN);close(descriptor);
  if(status==0 || (saved!=EWOULDBLOCK && saved!=EAGAIN))throw std::runtime_error("outer guard lock is not held");
}
int main(int argc,char** argv) {try {
  if(argc==2 && std::strcmp(argv[1],"--help")==0) {
    std::puts("F117 diagnostic surrogate: --fresh 0..4 --out DIRECTORY; requires guarded scheduler");return 0;
  }
  if(argc!=5 || std::strcmp(argv[1],"--fresh") || std::strcmp(argv[3],"--out"))
    throw std::runtime_error("supply --fresh 0..4 --out DIRECTORY");
  int fresh=std::stoi(argv[2]);if(fresh<0 || fresh>=5)throw std::runtime_error("fresh index outside frozen protocol");
  RequireGuard();std::filesystem::path folder=argv[4];
  if(std::filesystem::exists(folder))throw std::runtime_error("preserve prior probe output");
  std::filesystem::create_directories(folder);
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0 || prop.multiProcessorCount!=132 || prop.sharedMemPerBlockOptin<Reserved[1])
    throw std::runtime_error("probe requires the declared 132-SM Hopper profile");
  U64* device_stamps;unsigned* errors;Check(cudaMalloc(&device_stamps,prop.multiProcessorCount*8*sizeof(U64)));
  Check(cudaMalloc(&errors,sizeof(unsigned)));Check(cudaMemset(errors,0,sizeof(unsigned)));
  Frequency<<<1,1>>>(device_stamps);Check(cudaGetLastError());U64 frequency[2];
  Check(cudaMemcpy(frequency,device_stamps,sizeof(frequency),cudaMemcpyDeviceToHost));
  if(!frequency[0] || !frequency[1])throw std::runtime_error("invalid frequency interval");
  double ns_per_cycle=double(frequency[1])/double(frequency[0]);
  std::array<Facts,4> facts={Configure<Gate0>(Reserved[0]),Configure<Gate3>(Reserved[1]),Configure<Down0>(Reserved[2]),Configure<Down3>(Reserved[3])};
  std::array<std::array<std::vector<double>,3>,4> values;
  std::ofstream raw(folder/"raw.tsv"),metrics(folder/"metrics.csv"),meta(folder/"metadata.json");
  if(!raw || !metrics || !meta)throw std::runtime_error("cannot open raw output");
  raw<<"shape\tmode\ttrial\tcta\twarp\tclock_begin\tclock_end\tinteger_delta_cycles\n";
  for(int shape=0;shape<4;++shape)for(int mode=0;mode<3;++mode)Launch(shape,mode,prop.multiProcessorCount,device_stamps,errors);
  Check(cudaDeviceSynchronize());std::vector<U64> host(prop.multiProcessorCount*8);
  for(int trial=0;trial<Trials;++trial)for(int ordinal=0;ordinal<4;++ordinal) {
    int shape=(fresh%2?3-ordinal:ordinal);
    for(int turn=0;turn<3;++turn) {
      int mode=(turn+fresh+trial)%3;Launch(shape,mode,prop.multiProcessorCount,device_stamps,errors);
      Check(cudaMemcpy(host.data(),device_stamps,host.size()*sizeof(U64),cudaMemcpyDeviceToHost));
      for(int cta=0;cta<Active;++cta) {
        U64 first=~0ull,last=0;
        for(int warp=0;warp<4;++warp) {
          U64 begin=host[cta*8+warp*2],end=host[cta*8+warp*2+1];
          if(end<begin)throw std::runtime_error("clock64 interval wrapped or reversed");
          first=std::min(first,begin);last=std::max(last,end);
          raw<<Labels[shape]<<'\t'<<Modes[mode]<<'\t'<<trial<<'\t'<<cta<<'\t'<<warp<<'\t'<<begin<<'\t'<<end<<'\t'<<(end-begin)<<'\n';
        }
        U64 delta=last-first;values[shape][mode].push_back(double(delta)*ns_per_cycle);
      }
    }
  }
  unsigned failures;Check(cudaMemcpy(&failures,errors,sizeof(failures),cudaMemcpyDeviceToHost));
  char const* header="metric,shape,mode,grid,active_ctas,batch,tile_n,tile_k,warps_n,warps_k,register_stages,slots,input_k,samples,median,p99,unit";
  metrics<<header<<'\n';std::puts(header);
  for(int shape=0;shape<4;++shape)for(int mode=0;mode<3;++mode) {
    auto& data=values[shape][mode];std::sort(data.begin(),data.end());auto const& f=facts[shape];char row[1024];
    std::snprintf(row,sizeof(row),"mma_finalize_surrogate,%s,%s,%d,%d,1,%d,%d,%d,%d,1,%d,%d,%zu,%.12g,%.12g,ns",Labels[shape],Modes[mode],prop.multiProcessorCount,Active,f.tn,f.tk,f.wn,f.wk,f.slots,f.k,data.size(),data[data.size()/2],data[std::min(data.size()-1,data.size()*99/100)]);
    metrics<<row<<'\n';std::puts(row);
  }
  meta<<"{\n  \"schema\":1,\n  \"fresh\":"<<fresh<<",\n  \"frequency_clock_ticks\":"<<frequency[0]<<",\n  \"frequency_elapsed_ns\":"<<frequency[1]<<",\n  \"timestamp_arithmetic\":\"integer_clock64_delta_v1\",\n  \"numerical_failures\":"<<failures<<",\n  \"performance_eligible\":false,\n  \"surrogate\":\"shared operands; no async producer, epilogue or grid barrier\",\n  \"same_kernel_for_all_modes\":true,\n  \"shapes\":[\n";
  for(int shape=0;shape<4;++shape) {auto const& f=facts[shape];
    meta<<"    {\"shape\":\""<<Labels[shape]<<"\",\"registers\":"<<f.registers<<",\"local_bytes\":"<<f.local<<",\"resident_ctas_per_sm\":"<<f.resident<<",\"dynamic_shared_bytes\":"<<f.shared<<"}"<<(shape==3?"\n":",\n");
  }
  meta<<"  ]\n}\n";raw.close();metrics.close();meta.close();
  Check(cudaFree(errors));Check(cudaFree(device_stamps));
  if(failures)throw std::runtime_error("finalized accumulator numerical sum check failed");return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
