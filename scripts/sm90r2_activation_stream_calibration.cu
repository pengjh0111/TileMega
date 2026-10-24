// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
using U64=unsigned long long;
using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
constexpr int Batch=16,K=4096,Active=128,Samples=64,WarmIndex=8,Trials=8,Shared=232448,Words=50;
char const* Labels[]={"tk256_s2","tk256_s3","tk512_s2","tk2048_s2"};
char const* Metrics[]={"first_full_ready","first_release","steady_full_ready_period","steady_release_period","stream64_span"};
void Check(cudaError_t value) {if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
__host__ __device__ unsigned Mix(unsigned x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__device__ __forceinline__ U64 Global() {U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;}
__device__ __forceinline__ U64 Clock() {U64 value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;}
__global__ void Frequency(U64* output) {
  if(threadIdx.x)return;U64 begin=Global(),clock=Clock();while(Global()-begin<2000000ull) {}
  U64 end_clock=Clock(),end=Global();output[0]=end_clock-clock;output[1]=end-begin;
}
__global__ void Initialize(uint4* input,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    input[i]=make_uint4(Mix(i*4),Mix(i*4+1),Mix(i*4+2),Mix(i*4+3));
}
__global__ void Warm(uint4 const* input,int count,unsigned* sink) {
  unsigned sum=0;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
    auto value=input[i];sum^=value.x^value.y^value.z^value.w;
  }
  atomicXor(sink,sum);
}
template<int TK,int Slots>
__global__ __launch_bounds__(160,1)
void ActivationStream(char const* input,U64* output,unsigned* checksums,unsigned* errors) {
  using namespace cute;
  using Atom=decltype(composition(Swizzle<3,3,3>{},Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
  using A=decltype(tile_to_shape(Atom{},Shape<_16,Int<TK>>{}));
  if(blockIdx.x>=Active)return;
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<tilemega::backend::Sm90DenseRingState*>(shared);
  auto* data=shared+1024;ring.Initialize();
  for(int i=threadIdx.x*16;i<Slots*32*TK;i+=160*16)*reinterpret_cast<uint4*>(data+i)=make_uint4(0,0,0,0);
  // All roles join before the producer first-issue anchor; initialization is untimed.
  __syncthreads();unsigned sum=0;U64 issue_g=0,issue_c=0,stamps[12];
  if(threadIdx.x>=128) {
    int lane=threadIdx.x-128;
    #pragma unroll 1
    for(int sequence=0;sequence<Samples;++sequence) {
      int slot=sequence%Slots;unsigned phase=(sequence/Slots)&1;
      if(sequence==0 && lane==0) {issue_g=Global();issue_c=Clock();}
      Async::Wait(ring.empty+slot,phase);
      if(lane==0)Async::Arrive(ring.full+slot);
      for(int i=lane*8;i<Batch*TK;i+=32*8) {
        int row=i/TK,col=i%TK,tile=sequence*TK%K;
        Async::Copy16Bytes(data+slot*32*TK+2*A{}(row,col),input+2*(row*K+tile+col),16);
      }
      Async::CompleteCopies(ring.full+slot);
    }
  }else {
    #pragma unroll 1
    for(int sequence=0;sequence<Samples;++sequence) {
      int slot=sequence%Slots;unsigned phase=(sequence/Slots)&1;
      Async::Wait(ring.full+slot,phase);
      if((threadIdx.x&31)==0) {
        if(sequence==0) {stamps[0]=Global();stamps[1]=Clock();}
        if(sequence==WarmIndex) {stamps[4]=Global();stamps[5]=Clock();}
        if(sequence==Samples-1) {stamps[8]=Global();stamps[9]=Clock();}
      }
      for(int i=threadIdx.x*16;i<32*TK;i+=128*16) {
        auto value=*reinterpret_cast<uint4*>(data+slot*32*TK+i);sum^=value.x^value.y^value.z^value.w;
      }
      Async::Arrive(ring.empty+slot);
      if((threadIdx.x&31)==0) {
        if(sequence==0) {stamps[2]=Global();stamps[3]=Clock();}
        if(sequence==WarmIndex) {stamps[6]=Global();stamps[7]=Clock();}
        if(sequence==Samples-1) {stamps[10]=Global();stamps[11]=Clock();}
      }
    }
  }
  // Publish observations only after every measured copy/read/release is over.
  __syncthreads();
  if(threadIdx.x==128) {output[blockIdx.x*Words]=issue_g;output[blockIdx.x*Words+1]=issue_c;}
  if(threadIdx.x<128 && (threadIdx.x&31)==0) {
    #pragma unroll
    for(int i=0;i<12;++i)output[blockIdx.x*Words+2+(threadIdx.x/32)*12+i]=stamps[i];
  }
  if(threadIdx.x<128)checksums[blockIdx.x*128+threadIdx.x]=sum;
  for(int i=threadIdx.x;i<Slots*16*TK/8;i+=160) {
    int slot=i/(16*TK/8),within=i%(16*TK/8),row=within/(TK/8),col=within%(TK/8)*8;
    int last=Samples-1-((Samples-1-slot)%Slots),tile=last*TK%K;
    auto value=*reinterpret_cast<uint4*>(data+slot*32*TK+2*A{}(row,col));int word=(row*K+tile+col)/2;
    if(value.x!=Mix(word) || value.y!=Mix(word+1) || value.z!=Mix(word+2) || value.w!=Mix(word+3))atomicAdd(errors,1);
  }
  ring.Invalidate();
}
// The extra comparison instructions change this untimed consumer schedule.
// This check cannot establish race freedom for the original timed reader.
template<int TK,int Slots>
__global__ __launch_bounds__(160,1)
void ValidationStream(char const* input,unsigned* errors,unsigned* counts) {
  using namespace cute;
  using Atom=decltype(composition(Swizzle<3,3,3>{},Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
  using A=decltype(tile_to_shape(Atom{},Shape<_16,Int<TK>>{}));
  if(blockIdx.x>=Active)return;
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<tilemega::backend::Sm90DenseRingState*>(shared);
  auto* data=shared+1024;ring.Initialize();
  for(int i=threadIdx.x*16;i<Slots*32*TK;i+=160*16)*reinterpret_cast<uint4*>(data+i)=make_uint4(0,0,0,0);
  __syncthreads();unsigned failures=0,slot_counts[Slots]={};
  if(threadIdx.x>=128) {
    int lane=threadIdx.x-128;
    #pragma unroll 1
    for(int sequence=0;sequence<Samples;++sequence) {
      int slot=sequence%Slots;unsigned phase=(sequence/Slots)&1;
      Async::Wait(ring.empty+slot,phase);
      if(lane==0)Async::Arrive(ring.full+slot);
      for(int i=lane*8;i<Batch*TK;i+=32*8) {
        int row=i/TK,col=i%TK;
        Async::Copy16Bytes(data+slot*32*TK+2*A{}(row,col),input+2*(row*Samples*TK+sequence*TK+col),16);
      }
      Async::CompleteCopies(ring.full+slot);
    }
  }else {
    #pragma unroll 1
    for(int sequence=0;sequence<Samples;++sequence) {
      int slot=sequence%Slots;unsigned phase=(sequence/Slots)&1;
      Async::Wait(ring.full+slot,phase);
      for(int i=threadIdx.x*8;i<Batch*TK;i+=128*8) {
        int row=i/TK,col=i%TK,word=(row*Samples*TK+sequence*TK+col)/2;
        auto value=*reinterpret_cast<uint4*>(data+slot*32*TK+2*A{}(row,col));
        failures+=(value.x!=Mix(word))+(value.y!=Mix(word+1))+(value.z!=Mix(word+2))+(value.w!=Mix(word+3));
      }
      Async::Arrive(ring.empty+slot);
      if(threadIdx.x==0)++slot_counts[slot];
    }
  }
  __syncthreads();
  if(failures)atomicAdd(errors,failures);
  if(threadIdx.x==0)for(int slot=0;slot<Slots;++slot)counts[blockIdx.x*8+slot]=slot_counts[slot];
  ring.Invalidate();
}
struct ValidationFact {int shape;unsigned failures,generations,word_checks;};
template<int TK,int Slots> ValidationFact Validate(int shape) {
  Check(cudaFuncSetAttribute(ValidationStream<TK,Slots>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
  char* input;unsigned *errors,*counts;Check(cudaMalloc(&input,Batch*Samples*TK*2));Check(cudaMalloc(&errors,4));Check(cudaMalloc(&counts,132*8*4));
  Check(cudaMemset(errors,0,4));Check(cudaMemset(counts,0,132*8*4));
  Initialize<<<132,256>>>(reinterpret_cast<uint4*>(input),Batch*Samples*TK*2/16);Check(cudaGetLastError());
  ValidationStream<TK,Slots><<<132,160,Shared>>>(input,errors,counts);Check(cudaGetLastError());
  unsigned failures;std::array<unsigned,132*8> actual{};Check(cudaMemcpy(&failures,errors,4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(actual.data(),counts,actual.size()*4,cudaMemcpyDeviceToHost));
  unsigned generations=0;
  for(int cta=0;cta<132;++cta)for(int slot=0;slot<8;++slot) {
    unsigned expected=cta<Active && slot<Slots?(Samples+Slots-1-slot)/Slots:0;
    if(actual[cta*8+slot]!=expected)throw std::runtime_error("validation generation/slot coverage differs");
    generations+=actual[cta*8+slot];
  }
  Check(cudaFree(input));Check(cudaFree(errors));Check(cudaFree(counts));
  if(failures)throw std::runtime_error("unique-sequence per-word validation failed");
  return {shape,failures,generations,generations*unsigned(Batch*TK/2)};
}
struct Facts {int tk,slots,registers,local,resident;};
template<int TK,int Slots> Facts Configure() {
  Check(cudaFuncSetAttribute(ActivationStream<TK,Slots>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
  cudaFuncAttributes a{};Check(cudaFuncGetAttributes(&a,ActivationStream<TK,Slots>));int resident=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ActivationStream<TK,Slots>,160,Shared));
  if(resident!=1 || a.localSizeBytes)throw std::runtime_error("occupancy/local-memory mismatch");
  return {TK,Slots,a.numRegs,int(a.localSizeBytes),resident};
}
void Launch(int shape,int grid,char const* input,U64* stamps,unsigned* checksum,unsigned* errors) {
  switch(shape) {
    case 0:ActivationStream<256,2><<<grid,160,Shared>>>(input,stamps,checksum,errors);break;
    case 1:ActivationStream<256,3><<<grid,160,Shared>>>(input,stamps,checksum,errors);break;
    case 2:ActivationStream<512,2><<<grid,160,Shared>>>(input,stamps,checksum,errors);break;
    case 3:ActivationStream<2048,2><<<grid,160,Shared>>>(input,stamps,checksum,errors);break;
    default:throw std::runtime_error("unknown shape");
  }
  Check(cudaGetLastError());
}
void RequireGuard() {
  auto visible=std::getenv("CUDA_VISIBLE_DEVICES"),index=std::getenv("TILEMEGA_DEVICE_INDEX");
  auto held=std::getenv("TILEMEGA_GPU_LOCK_HELD"),lock=std::getenv("TILEMEGA_GPU_LOCK");
  if(!visible || !index || std::strcmp(visible,index) || !held || std::strcmp(held,"1") || !lock)throw std::runtime_error("matching guarded device required");
  int gpu=std::stoi(index);if(!((gpu>=0 && gpu<=4) || gpu==7))throw std::runtime_error("excluded timing GPU");
  if(std::string(lock)!="/tmp/tilemega_gpu"+std::to_string(gpu)+".lock")throw std::runtime_error("wrong per-card lock");
  int descriptor=open(lock,O_CREAT|O_RDWR,0600);if(descriptor<0)throw std::runtime_error("lock inspection failed");
  int status=flock(descriptor,LOCK_EX|LOCK_NB),saved=errno;if(status==0)flock(descriptor,LOCK_UN);close(descriptor);
  if(status==0 || (saved!=EAGAIN && saved!=EWOULDBLOCK))throw std::runtime_error("outer per-card lock not held");
}
int main(int argc,char** argv) {try {
  if(argc==2 && std::strcmp(argv[1],"--help")==0) {std::puts("F118 activation stream: --fresh 0..4 --out DIRECTORY; guarded scheduler only");return 0;}
  if(argc!=5 || std::strcmp(argv[1],"--fresh") || std::strcmp(argv[3],"--out"))throw std::runtime_error("supply --fresh 0..4 --out DIRECTORY");
  int fresh=std::stoi(argv[2]);if(fresh<0 || fresh>=5)throw std::runtime_error("fresh outside protocol");RequireGuard();
  std::filesystem::path folder=argv[4];if(std::filesystem::exists(folder))throw std::runtime_error("preserve prior output");std::filesystem::create_directories(folder);
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0 || prop.multiProcessorCount!=132 || prop.sharedMemPerBlockOptin<Shared)throw std::runtime_error("expected 132-SM Hopper profile");
  char* input;U64* device;unsigned *checksum,*errors;
  Check(cudaMalloc(&input,Batch*K*2));Check(cudaMalloc(&device,132*Words*sizeof(U64)));
  Check(cudaMalloc(&checksum,132*128*sizeof(unsigned)));Check(cudaMalloc(&errors,sizeof(unsigned)));Check(cudaMemset(errors,0,4));Check(cudaMemset(checksum,0,132*128*4));
  Initialize<<<132,256>>>(reinterpret_cast<uint4*>(input),Batch*K*2/16);Check(cudaGetLastError());
  std::array<ValidationFact,4> validation={Validate<256,2>(0),Validate<256,3>(1),Validate<512,2>(2),Validate<2048,2>(3)};
  Frequency<<<1,1>>>(device);Check(cudaGetLastError());U64 pre[2],post[2];Check(cudaMemcpy(pre,device,16,cudaMemcpyDeviceToHost));
  std::array<Facts,4> facts={Configure<256,2>(),Configure<256,3>(),Configure<512,2>(),Configure<2048,2>()};
  for(int shape=0;shape<4;++shape) {Warm<<<132,256>>>(reinterpret_cast<uint4*>(input),Batch*K*2/16,checksum);Launch(shape,132,input,device,checksum,errors);}Check(cudaDeviceSynchronize());
  std::ofstream raw(folder/"raw.tsv"),metrics(folder/"metrics.csv"),meta(folder/"metadata.json");
  if(!raw || !metrics || !meta)throw std::runtime_error("output open failed");
  raw<<"shape\ttrial\tcta\trole\tmilestone\tglobal_ns\tclock64\n";
  std::array<std::array<std::vector<double>,5>,4> values;std::vector<U64> host(132*Words);std::vector<unsigned> sums(132*128);unsigned checksum_failures=0;
  for(int trial=0;trial<Trials;++trial)for(int ordinal=0;ordinal<4;++ordinal) {
    int shape=fresh%2?3-ordinal:ordinal;Warm<<<132,256>>>(reinterpret_cast<uint4*>(input),Batch*K*2/16,checksum);
    Launch(shape,132,input,device,checksum,errors);Check(cudaMemcpy(host.data(),device,host.size()*8,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(sums.data(),checksum,sums.size()*4,cudaMemcpyDeviceToHost));
    for(int cta=0;cta<Active;++cta) {
      U64* row=host.data()+cta*Words;std::array<U64,6> maxima{};
      raw<<Labels[shape]<<'\t'<<trial<<'\t'<<cta<<"\tproducer\tissue\t"<<row[0]<<'\t'<<row[1]<<'\n';
      for(int warp=0;warp<4;++warp)for(int milestone=0;milestone<6;++milestone) {
        U64 g=row[2+warp*12+2*milestone],c=row[3+warp*12+2*milestone];maxima[milestone]=std::max(maxima[milestone],g);
        if(g<row[0] || c<row[1])throw std::runtime_error("endpoint precedes common issue anchor");
        raw<<Labels[shape]<<'\t'<<trial<<'\t'<<cta<<'\t'<<warp<<'\t'<<milestone<<'\t'<<g<<'\t'<<c<<'\n';
      }
      if(!std::is_sorted(maxima.begin(),maxima.end()) || maxima[4]<=maxima[2] || maxima[5]<=maxima[3])throw std::runtime_error("nonmonotone stream endpoints");
      values[shape][0].push_back(double(maxima[0]-row[0]));values[shape][1].push_back(double(maxima[1]-row[0]));
      values[shape][2].push_back(double(maxima[4]-maxima[2])/(Samples-1-WarmIndex));
      values[shape][3].push_back(double(maxima[5]-maxima[3])/(Samples-1-WarmIndex));values[shape][4].push_back(double(maxima[5]-row[0]));
      for(int tid=0;tid<128;++tid)checksum_failures+=sums[cta*128+tid]!=0;
    }
  }
  Frequency<<<1,1>>>(device);Check(cudaGetLastError());Check(cudaMemcpy(post,device,16,cudaMemcpyDeviceToHost));
  if(!pre[0] || !pre[1] || !post[0] || !post[1])throw std::runtime_error("invalid frequency bracket");
  double before=double(pre[0])/pre[1],after=double(post[0])/post[1],drift=std::abs(after/before-1)*100;
  unsigned failures;Check(cudaMemcpy(&failures,errors,4,cudaMemcpyDeviceToHost));
  char const* header="metric,shape,grid,active_ctas,batch,tile_k,slots,input_k,stream_tiles,period_denominator,samples,median,p99,unit";metrics<<header<<'\n';std::puts(header);
  for(int shape=0;shape<4;++shape)for(int metric=0;metric<5;++metric) {
    auto& data=values[shape][metric];std::sort(data.begin(),data.end());char line[1024];auto const& f=facts[shape];
    std::snprintf(line,sizeof(line),"%s,%s,132,128,16,%d,%d,4096,64,55,%zu,%.12g,%.12g,ns",Metrics[metric],Labels[shape],f.tk,f.slots,data.size(),data[data.size()/2],data[data.size()*99/100]);metrics<<line<<'\n';std::puts(line);
  }
  meta<<"{\n\"schema\":1,\n\"fresh\":"<<fresh<<",\n\"performance_eligible\":false,\n\"timestamp_arithmetic\":\"integer_globaltimer_delta_v1\",\n\"frequency_before_clock_ticks\":"<<pre[0]<<",\n\"frequency_before_elapsed_ns\":"<<pre[1]<<",\n\"frequency_after_clock_ticks\":"<<post[0]<<",\n\"frequency_after_elapsed_ns\":"<<post[1]<<",\n\"frequency_drift_pct\":"<<drift<<",\n\"frequency_reject_above_pct\":2,\n\"numerical_failures\":"<<failures<<",\n\"checksum_failures\":"<<checksum_failures<<",\n\"reader\":\"16xTK BF16 uint4 scan and XOR; no production MMA\",\n\"first_completion\":\"includes issuing-warp dispatch/wait/copy and consumer observation latency\",\n\"weight_transaction\":\"immediate lane0 arrival proxy; no weight TMA\",\n\"shapes\":[\n";
  for(int shape=0;shape<4;++shape) {auto const& f=facts[shape];meta<<"{\"shape\":\""<<Labels[shape]<<"\",\"registers\":"<<f.registers<<",\"local_bytes\":"<<f.local<<",\"resident_ctas_per_sm\":"<<f.resident<<",\"dynamic_shared_bytes\":"<<Shared<<"}"<<(shape==3?"\n":",\n");}
  meta<<"],\n\"untimed_validation\":{\"unique_sequence_input\":true,\"consumer_duration_differs\":true,\"timed_schedule_race_proof\":false,\"before_suite_frequency\":true,\"shapes\":[\n";
  for(int shape=0;shape<4;++shape) {auto const& v=validation[shape];meta<<"{\"shape\":\""<<Labels[v.shape]<<"\",\"failures\":"<<v.failures<<",\"generations\":"<<v.generations<<",\"word_checks\":"<<v.word_checks<<",\"ctas\":128,\"tiles_per_cta\":64,\"slot_coverage_checked_per_cta\":true}"<<(shape==3?"\n":",\n");}
  meta<<"]}\n}\n";raw.close();metrics.close();meta.close();Check(cudaFree(input));Check(cudaFree(device));Check(cudaFree(checksum));Check(cudaFree(errors));
  if(failures || checksum_failures)throw std::runtime_error("slot/read checksum numerical check failed");
  if(drift>2)throw std::runtime_error("suite frequency drift exceeds predeclared 2 percent quality limit");return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
