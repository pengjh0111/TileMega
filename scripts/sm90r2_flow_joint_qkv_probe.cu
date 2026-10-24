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
constexpr std::size_t ScrubBytes=128ull*1024*1024;
constexpr int Rows=16,Columns=4096,K=1024,TileN=32,TileK=512,Slots=2,Redzone=64;
constexpr unsigned short Sentinel=0xffff;
struct CtaRecord {
  U64 smid,setup_acquired,producer_entry,producer_returned,roles_join_observed;
  U64 producer_phase0,producer_phase1,consumer_phase0,consumer_phase1;
  U64 v3_arrive,v3_acquired;
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


__global__ void WarmTouch(tilemega::backend::Sm90DenseOperands p,int active,unsigned* sink) {
  unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x,sum=0;
  auto* weight=reinterpret_cast<volatile unsigned const*>(p.weight);
  auto* activation=reinterpret_cast<volatile unsigned const*>(p.activation);
  auto* norm=reinterpret_cast<volatile unsigned const*>(p.norm_sums);
  for(unsigned i=tid;i<unsigned(active*TileN*K/2);i+=stride)sum+=weight[i];
  for(unsigned i=tid;i<Rows*K/2;i+=stride)sum+=activation[i];
  for(unsigned i=tid;i<Rows*K/8;i+=stride)sum+=norm[i];
  sink[tid]=sum;
}
__global__ void Scrub(unsigned* data,std::size_t words,unsigned* sink) {
  unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x,sum=0;
  volatile unsigned* buffer=data;
  for(std::size_t i=tid;i<words;i+=stride) {
    unsigned value=buffer[i]+unsigned(i)+0x9e3779b9u;buffer[i]=value;sum+=value;
  }
  sink[tid]=sum;
}

__global__ __launch_bounds__(160,1)
void Joint(tilemega::backend::Sm90DenseOperands p,int active,U64* stamps,
    CtaRecord* ctas,GridCounter* events) {
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
  auto* b=a+Slots*Rows*TileK;
  auto* scratch=reinterpret_cast<float*>(arena+1024+G::kOperandBytes);
  NativeV3(events,nullptr,&cta->setup_acquired);
  if(blockIdx.x<unsigned(active)) {
    if(threadIdx.x<128) {
      U64 begin=Now();
      G::template Consume<true,false>(record,ring,a,b,scratch,1,0);
      U64 end=Now();
      if((threadIdx.x&31)==0) {
        auto* row=stamps+blockIdx.x*8+threadIdx.x/32*2;row[0]=begin;row[1]=end;
      }
    }else {
      U64 begin=Now();
      G::template Produce<0>(record,ring,a,b,1,0);
      U64 end=Now();
      if(threadIdx.x==128) {cta->producer_entry=begin;cta->producer_returned=end;}
    }
  }
  __syncthreads();
  if(threadIdx.x==0) {
    cta->roles_join_observed=Now();
    cta->producer_phase0=ring.producer_generations[0];
    cta->producer_phase1=ring.producer_generations[1];
    cta->consumer_phase0=ring.consumer_generations[0];
    cta->consumer_phase1=ring.consumer_generations[1];
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

template<class T> std::vector<T> Load(std::filesystem::path path,std::size_t count) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file || file.tellg()!=std::streamoff(count*sizeof(T)))throw std::runtime_error("wrong input size: "+path.string());
  file.seekg(0);std::vector<T> result(count);
  if(!file.read(reinterpret_cast<char*>(result.data()),count*sizeof(T)))throw std::runtime_error("input read failed");
  return result;
}
std::vector<BF> Pack(std::vector<BF> const& weight) {
  std::vector<int> layout(TileN*TileK);std::vector<bool> used(TileN*TileK);
  for(int n=0;n<TileN;++n)for(int k=0;k<TileK;++k) {
    int address=int(G::LayoutB{}(n,k));
    if(address<0 || address>=TileN*TileK || used[address])throw std::runtime_error("G::LayoutB is not bijective");
    used[address]=true;layout[n*TileK+k]=address;
  }
  std::vector<BF> packed(Columns*K);
  for(int task=0;task<Columns/TileN;++task)for(int tile=0;tile<K/TileK;++tile)
    for(int n=0;n<TileN;++n)for(int k=0;k<TileK;++k)
      packed[((task*G::kIterations+tile)*TileN*TileK)+layout[n*TileK+k]]=weight[(task*TileN+n)*K+tile*TileK+k];
  return packed;
}
void Reference(std::vector<BF> const& a,std::vector<BF> const& weight,std::vector<float> const& norm,
    std::vector<float> const& expected_float,std::vector<BF> const& expected) {
  for(auto value:a)if(!std::isfinite(float(value)) || std::abs(float(value))>8 || float(value)!=int(float(value)))
    throw std::runtime_error("activation is not a small finite integer");
  for(auto value:weight)if(float(value)!=-1 && float(value)!=0 && float(value)!=1)
    throw std::runtime_error("weight is not signed ternary");
  for(int row=0;row<Rows;++row) {
    float total=0;
    for(int bin=0;bin<K/8;++bin) {
      float sum=0;for(int d=0;d<8;++d) {float v=float(a[row*K+bin*8+d]);sum+=v*v;}
      if(norm[row*(K/8)+bin]!=sum)throw std::runtime_error("eight-element norm input mismatch");
      total+=sum;
    }
    float inverse=1.f/std::sqrt(total/K);
    if(!(inverse==1 || inverse==0.5f || inverse==0.25f))throw std::runtime_error("reference norm not exact power of two");
    for(int n=0;n<Columns;++n) {
      long long dot=0;for(int k=0;k<K;++k)dot+=int(float(a[row*K+k]))*int(float(weight[n*K+k]));
      if(std::abs(dot)>=16777216)throw std::runtime_error("dot is not exact FP32 integer");
      float value=float(dot)*inverse;
      if(value!=expected_float[row*Columns+n] || BF(value).raw()!=expected[row*Columns+n].raw())
        throw std::runtime_error("prepared reference disagrees with independent host dense dot/norm");
    }
  }
}

int main(int argc,char** argv) {try {
  if(argc!=7 || std::strcmp(argv[1],"--fresh") || std::strcmp(argv[3],"--inputs") || std::strcmp(argv[5],"--out"))
    throw std::runtime_error("usage: --fresh 0..4 --inputs FROZEN_DIRECTORY --out NEW_DIRECTORY");
  int fresh=std::stoi(argv[2]);if(fresh<0 || fresh>4)throw std::runtime_error("fresh outside protocol");
  RequireGuard();std::filesystem::path input=argv[4],out=argv[6];
  if(std::filesystem::exists(out))throw std::runtime_error("preserve prior output");std::filesystem::create_directories(out);
  auto activation=Load<BF>(input/"activation.bf16",Rows*K);
  auto weight=Load<BF>(input/"weight-rowmajor.bf16",Columns*K);
  auto norm=Load<float>(input/"norm8.f32",Rows*K/8);
  auto expected_float=Load<float>(input/"expected.f32",Rows*Columns);
  auto expected=Load<BF>(input/"expected.bf16",Rows*Columns);
  Reference(activation,weight,norm,expected_float,expected);auto packed=Pack(weight);
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0 || prop.multiProcessorCount!=Grid)throw std::runtime_error("expected H800");
  Check(cudaFuncSetAttribute(Joint,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
  int occupancy=0;cudaFuncAttributes attr{};Check(cudaFuncGetAttributes(&attr,Joint));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,Joint,160,Shared));
  if(occupancy!=1)throw std::runtime_error("joint probe occupancy mismatch");
  BF *d_a,*d_weight,*d_output;float* d_norm;U64* device;CtaRecord* ctas;GridCounter* events;
  unsigned *scrub,*sink;
  Check(cudaMalloc(&d_a,activation.size()*sizeof(BF)));Check(cudaMalloc(&d_weight,packed.size()*sizeof(BF)));
  Check(cudaMalloc(&d_norm,norm.size()*sizeof(float)));Check(cudaMalloc(&d_output,(Rows*Columns+2*Redzone)*sizeof(BF)));
  Check(cudaMalloc(&device,Grid*8*sizeof(U64)));Check(cudaMalloc(&ctas,Grid*sizeof(CtaRecord)));
  Check(cudaMalloc(&events,2*sizeof(GridCounter)));Check(cudaMalloc(&scrub,ScrubBytes));Check(cudaMalloc(&sink,Grid*160*sizeof(unsigned)));
  Check(cudaMemcpy(d_a,activation.data(),activation.size()*sizeof(BF),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_weight,packed.data(),packed.size()*sizeof(BF),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_norm,norm.data(),norm.size()*sizeof(float),cudaMemcpyHostToDevice));Check(cudaMemset(scrub,0,ScrubBytes));
  tilemega::backend::Sm90DenseOperands p{};p.activation=d_a;p.weight=d_weight;p.output=d_output+Redzone;p.norm_sums=d_norm;p.epsilon=0;
  U64 before[2],after[2];Frequency<<<1,1>>>(device);Check(cudaMemcpy(before,device,16,cudaMemcpyDeviceToHost));
  std::ofstream raw(out/"raw.csv"),cta_raw(out/"ctas.csv"),contexts(out/"contexts.csv"),meta(out/"metadata.json");
  if(!raw || !cta_raw || !contexts || !meta)throw std::runtime_error("output failed");
  raw<<"fresh,active_ctas,cache_mode,trial,cta,smid,warp,consumer_entry,consumer_returned,producer_entry,producer_returned,setup_acquired,roles_join_observed,v3_arrive,v3_acquired\n";
  cta_raw<<"fresh,active_ctas,cache_mode,trial,cta,smid,has_work,setup_acquired,producer_entry,producer_returned,roles_join_observed,producer_phase0,producer_phase1,consumer_phase0,consumer_phase1,v3_arrive,v3_acquired\n";
  contexts<<"fresh,active_ctas,cache_mode,trial,setup_arrivals,final_arrivals,cache_checksum,checked_outputs,checked_sentinels\n";
  std::vector<U64> stamps(Grid*8);std::vector<BF> output(Rows*Columns+2*Redzone);
  std::vector<CtaRecord> cta_host(Grid);std::vector<GridCounter> event_host(2);std::vector<unsigned> sink_host(Grid*160);
  for(int active:{1,32,64,128})for(int trial=0;trial<17;++trial)for(int cache_index=0;cache_index<2;++cache_index) {
    bool scrub_mode=cache_index^(trial&1);char const* mode=scrub_mode?"scrub_surrogate":"warm_touch";
    Check(cudaMemset(d_output,0xff,output.size()*sizeof(BF)));Check(cudaMemset(device,0,stamps.size()*sizeof(U64)));
    Check(cudaMemset(ctas,0,cta_host.size()*sizeof(CtaRecord)));Check(cudaMemset(events,0,event_host.size()*sizeof(GridCounter)));
    if(scrub_mode)Scrub<<<Grid,160>>>(scrub,ScrubBytes/sizeof(unsigned),sink);
    else WarmTouch<<<Grid,160>>>(p,active,sink);
    Check(cudaGetLastError());Check(cudaMemcpy(sink_host.data(),sink,sink_host.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
    U64 checksum=0;for(unsigned value:sink_host)checksum+=value;
    Joint<<<Grid,160,Shared>>>(p,active,device,ctas,events);
    Check(cudaGetLastError());Check(cudaMemcpy(stamps.data(),device,stamps.size()*sizeof(U64),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(output.data(),d_output,output.size()*sizeof(BF),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(cta_host.data(),ctas,cta_host.size()*sizeof(CtaRecord),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(event_host.data(),events,event_host.size()*sizeof(GridCounter),cudaMemcpyDeviceToHost));
    for(auto const& event:event_host)if(event.arrivals!=Grid)throw std::runtime_error("V3 arrival count mismatch");
    for(int row=0;row<Rows;++row)for(int n=0;n<Columns;++n) {
      unsigned short wanted=n<active*TileN?expected[row*Columns+n].raw():Sentinel;
      if(output[Redzone+row*Columns+n].raw()!=wanted)throw std::runtime_error("joint output/inactive sentinel mismatch");
    }
    for(int i=0;i<Redzone;++i)if(output[i].raw()!=Sentinel || output[Redzone+Rows*Columns+i].raw()!=Sentinel)
      throw std::runtime_error("output redzone changed");
    std::vector<U64> smids;
    for(int cta=0;cta<Grid;++cta) {
      auto const& info=cta_host[cta];smids.push_back(info.smid);bool work=cta<active;
      if(!info.setup_acquired || info.roles_join_observed<info.setup_acquired || info.v3_arrive<info.roles_join_observed || info.v3_acquired<info.v3_arrive)
        throw std::runtime_error("CTA join/V3 timestamp order failed");
      if((work && (info.producer_entry<info.setup_acquired || info.producer_returned<info.producer_entry || info.roles_join_observed<info.producer_returned)) ||
          (!work && (info.producer_entry || info.producer_returned)))throw std::runtime_error("producer paired timestamp order failed");
      if(info.producer_phase0!=unsigned(work) || info.producer_phase1!=unsigned(work) ||
          info.consumer_phase0!=unsigned(work) || info.consumer_phase1!=unsigned(work))throw std::runtime_error("joint role generation mismatch");
      if(!trial)continue;
      cta_raw<<fresh<<','<<active<<','<<mode<<','<<trial<<','<<cta<<','<<info.smid<<','<<work<<','<<info.setup_acquired<<','
          <<info.producer_entry<<','<<info.producer_returned<<','<<info.roles_join_observed<<','<<info.producer_phase0<<','<<info.producer_phase1<<','
          <<info.consumer_phase0<<','<<info.consumer_phase1<<','<<info.v3_arrive<<','<<info.v3_acquired<<'\n';
    }
    std::sort(smids.begin(),smids.end());if(std::unique(smids.begin(),smids.end())-smids.begin()!=Grid)
      throw std::runtime_error("132 resident CTAs did not map to distinct SMIDs");
    for(int cta=0;cta<active;++cta)for(int warp=0;warp<4;++warp) {
      auto* stamp=stamps.data()+cta*8+warp*2;auto const& info=cta_host[cta];
      if(stamp[0]<info.setup_acquired || stamp[1]<stamp[0] || info.roles_join_observed<stamp[1])
        throw std::runtime_error("consumer paired timestamp order failed");
      if(!trial)continue;
      raw<<fresh<<','<<active<<','<<mode<<','<<trial<<','<<cta<<','<<info.smid<<','<<warp<<','<<stamp[0]<<','<<stamp[1]<<','
          <<info.producer_entry<<','<<info.producer_returned<<','<<info.setup_acquired<<','<<info.roles_join_observed<<','<<info.v3_arrive<<','<<info.v3_acquired<<'\n';
    }
    if(trial)contexts<<fresh<<','<<active<<','<<mode<<','<<trial<<','<<event_host[0].arrivals<<','<<event_host[1].arrivals<<','
        <<checksum<<','<<Rows*active*TileN<<','<<Rows*(Columns-active*TileN)+2*Redzone<<'\n';
  }
  Frequency<<<1,1>>>(device);Check(cudaMemcpy(after,device,16,cudaMemcpyDeviceToHost));
  if(!before[0] || !before[1] || !after[0] || !after[1])throw std::runtime_error("invalid frequency bracket");
  double pre=double(before[0])/before[1],post=double(after[0])/after[1],drift=std::abs(post/pre-1)*100;
  meta<<"{\"schema\":1,\"fresh\":"<<fresh<<",\"numerical_failures\":0,\"reference\":\"independent_exact_integer_dot_power_of_two_norm_BF16\",\"frequency_drift_pct\":"<<drift
      <<",\"grid\":132,\"contexts\":8,\"active_ctas\":[1,32,64,128],\"cache_modes\":[\"warm_touch\",\"scrub_surrogate\"],\"measured_trials\":16,\"threads\":160,\"shared_bytes\":"<<Shared
      <<",\"record_offset\":448,\"slots\":2,\"iterations\":2,\"tasks_per_active_cta\":1,\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"observed_residency\":"<<occupancy
      <<",\"scrub_bytes\":"<<ScrubBytes<<",\"epsilon\":0,\"input_norm\":true,\"output_sums\":false,\"prefix_bytes\":0,\"real_transfers\":true,\"dummy_producer\":false,\"output_and_sentinel_checks\":\"every_launch_including_warmup\",\"counter_values\":\"serialized_per_context\",\"roles_join_observed\":\"thread_zero_timestamp_after_all160_threads_join\",\"v3_acquired\":\"thread_zero_after_global_acquire_before_final_CTA_join\",\"observation\":\"joint_actual_Produce_Consume_role_entry_return_under_native_V3\",\"verified_cold_cache\":false,\"first_byte_observed\":false,\"hardware_admission_observed\":false,\"pure_compute_observed\":false,\"E2E_offset_fit\":false,\"production_binary_equivalence\":\"requires_exact_Produce_and_Consume_function_byte_resource_audits\",\"synchronization_acceptance\":\"requires_5_fresh_processes\",\"performance_eligible\":false}\n";
  std::filesystem::copy_file(input/"manifest.json",out/"input-manifest.json");
  raw.close();cta_raw.close();contexts.close();meta.close();if(drift>2)throw std::runtime_error("frequency bracket exceeds2percent");return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
