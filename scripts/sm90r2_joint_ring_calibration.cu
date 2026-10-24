// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseGemm.h>
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
using Op=tilemega::backend::ServingEpilogueOp;
using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
template<int N_,int K_,int TN_,int TK_,int S_,int WN_,int WK_,Op O_>
struct Shape {
  static constexpr int N=N_,K=K_,TN=TN_,TK=TK_,Slots=S_,WN=WN_,WK=WK_;
  using G=tilemega::backend::Sm90DenseGemm<1,N,K,TN,TK,Slots,WN,WK,O_,16,1>;
};
using Gate0=Shape<28672,4096,224,64,2,1,4,Op::kSwiGLU>;
using Gate3=Shape<28672,4096,224,128,3,4,1,Op::kSwiGLU>;
using Down0=Shape<4096,14336,32,256,3,1,4,Op::kResidual>;
using Down3=Shape<4096,14336,32,256,3,4,1,Op::kResidual>;
constexpr int Active=128,Grid=132,Trials=8,WarmIndex=8,MaxMarks=13,Words=2+4*MaxMarks*2;
char const* Labels[]={"gate_C0","gate_C3","down_C0","down_C3"};
constexpr int Reserved[]={136320,230528,136320,230528};
void Check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
__host__ __device__ unsigned Mix(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__host__ __device__ int AValue(int k){return int(Mix(unsigned(k)^0x6a09e667u)%3)-1;}
__host__ __device__ int BValue(int n,int k,int K){return int(Mix(unsigned(n)*unsigned(K)+unsigned(k)^0xbb67ae85u)%3)-1;}
__device__ __forceinline__ U64 Global(){U64 x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ U64 Clock(){U64 x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
__global__ void Frequency(U64* out){if(threadIdx.x)return;U64 g=Global(),c=Clock();while(Global()-g<2000000ull){}U64 endc=Clock(),endg=Global();out[0]=endc-c;out[1]=endg-g;}
template<class S> __global__ void Initialize(typename S::G::Element* a,typename S::G::Element* b){
  using E=typename S::G::Element;
  for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<S::K;k+=gridDim.x*blockDim.x)a[k]=E(AValue(k));
  for(std::size_t q=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;q<std::size_t(S::N)*S::K;q+=gridDim.x*blockDim.x){
    int task=q/(S::TN*S::K),within=q%(S::TN*S::K),tile=within/(S::TN*S::TK),n=within/S::TK%S::TN,k=within%S::TK;
    b[(std::size_t(task)*(S::K/S::TK)+tile)*S::TN*S::TK+typename S::G::LayoutB{}(n,k)]=E(BValue(task*S::TN+n,tile*S::TK+k,S::K));
  }
}
template<class S>
__global__ __launch_bounds__(160,1)
void JointRing(tilemega::backend::Sm90DenseOperands p,int const* expected,U64* output,unsigned* errors){
  using namespace cute;using G=typename S::G;using E=typename G::Element;
  static_assert(S::N/S::TN==Active && G::kIterations>WarmIndex+1);
  if(blockIdx.x>=Active)return;
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<tilemega::backend::Sm90DenseRingState*>(shared);
  auto* a=reinterpret_cast<E*>(shared+1024);auto* b=a+S::Slots*16*S::TK;
  auto* scratch=reinterpret_cast<float*>(shared+1024+G::kOperandBytes);
  state.Initialize();G::InitializeInactive(a);__syncthreads();
  U64 issue_g=0,issue_c=0,stamp[2*MaxMarks];
  constexpr int middle=2*(S::Slots+1),last=middle+2,final=last+2,marks=final+1;
  if(threadIdx.x>=128){
    if(threadIdx.x==128){issue_g=Global();issue_c=Clock();}
    // Exact frozen producer: real bulk weights plus cp.async activations.
    G::template Produce<0>(p,state,a,b,1,0);
  }else{
    int tid=threadIdx.x,lane=tid%(32*S::WN),split=tid/(32*S::WN);unsigned phase=0;
    typename G::Mma mma;auto thread=mma.get_slice(lane);
    auto owned=thread.partition_C(make_identity_tensor(cute::Shape<_16,Int<S::TN>>{}));
    auto accum=partition_fragment_C(mma,cute::Shape<_16,Int<S::TN>>{});clear(accum);
    // NumericalConsumer is the frozen Consume MMA/finalize body without epilogue.
    #pragma unroll 1
    for(int iteration=0;iteration<G::kIterations;++iteration){
      int slot=iteration%S::Slots;
      Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
      if((tid&31)==0){
        for_each(make_int_sequence<S::Slots+1>{},[&](auto q){if(iteration==int(q)){stamp[4*int(q)]=Global();stamp[4*int(q)+1]=Clock();}});
        if(iteration==WarmIndex){stamp[2*middle]=Global();stamp[2*middle+1]=Clock();}
        if(iteration==G::kIterations-1){stamp[2*last]=Global();stamp[2*last+1]=Clock();}
      }
      auto* sa=a+slot*16*S::TK;auto* sb=b+slot*S::TN*S::TK;
      if constexpr(S::WK==1)G::template MmaSlice<0>(sa,sb,accum);
      else if constexpr(S::WK==2){if(split==0)G::template MmaSlice<0>(sa,sb,accum);else G::template MmaSlice<1>(sa,sb,accum);}
      else{if(split==0)G::template MmaSlice<0>(sa,sb,accum);else if(split==1)G::template MmaSlice<1>(sa,sb,accum);else if(split==2)G::template MmaSlice<2>(sa,sb,accum);else G::template MmaSlice<3>(sa,sb,accum);}
      Async::Arrive(state.empty+slot);
      if((tid&31)==0){
        for_each(make_int_sequence<S::Slots+1>{},[&](auto q){if(iteration==int(q)){stamp[4*int(q)+2]=Global();stamp[4*int(q)+3]=Clock();}});
        if(iteration==WarmIndex){stamp[2*(middle+1)]=Global();stamp[2*(middle+1)+1]=Clock();}
        if(iteration==G::kIterations-1){stamp[2*(last+1)]=Global();stamp[2*(last+1)+1]=Clock();}
      }
    }
    for(int i=0;i<size(accum);++i)scratch[split*16*S::TN+get<0>(owned(i))*S::TN+get<1>(owned(i))]=accum(i);
    tilemega::codegen::executor::ComputeSync();
    if constexpr(S::WK>1){
      for(int i=tid;i<16*S::TN;i+=128){float sum=scratch[i];
        #pragma unroll
        for(int rank=1;rank<S::WK;++rank)sum+=scratch[rank*16*S::TN+i];scratch[i]=sum;}
      tilemega::codegen::executor::ComputeSync();
    }
    if((tid&31)==0){stamp[2*final]=Global();stamp[2*final+1]=Clock();}
  }
  // No endpoint global stores until all producer/consumer work has joined.
  __syncthreads();
  if(threadIdx.x==128){output[blockIdx.x*Words]=issue_g;output[blockIdx.x*Words+1]=issue_c;}
  if(threadIdx.x<128 && (threadIdx.x&31)==0){
    for_each(make_int_sequence<2*marks>{},[&](auto i){output[blockIdx.x*Words+2+(threadIdx.x/32)*2*MaxMarks+int(i)]=stamp[int(i)];});
  }
  if(threadIdx.x<128)for(int i=threadIdx.x;i<16*S::TN;i+=128){
    float want=i/S::TN==0?float(expected[blockIdx.x*S::TN+i%S::TN]):0.f;
    if(scratch[i]!=want)atomicAdd(errors,1);
  }
  state.Invalidate();
}
struct Facts{int n,k,tn,tk,slots,wn,wk,registers,local,resident,shared;};
struct Data{tilemega::backend::Sm90DenseOperands p{};int* expected;};
template<class S> Facts Configure(int shared){
  Check(cudaFuncSetAttribute(JointRing<S>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));cudaFuncAttributes a{};Check(cudaFuncGetAttributes(&a,JointRing<S>));int resident=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,JointRing<S>,160,shared));
  if(a.localSizeBytes || resident!=1)throw std::runtime_error("observer local memory/occupancy rejected");
  return {S::N,S::K,S::TN,S::TK,S::Slots,S::WN,S::WK,a.numRegs,int(a.localSizeBytes),resident,shared};
}
template<class S> Data PrepareData(){
  using E=typename S::G::Element;Data d;E *a,*b;Check(cudaMalloc(&a,S::K*2));Check(cudaMalloc(&b,std::size_t(S::N)*S::K*2));d.p.activation=a;d.p.weight=b;
  Check(cudaMalloc(&d.expected,S::N*sizeof(int)));std::vector<int> oracle(S::N),activation(S::K);
  for(int k=0;k<S::K;++k)activation[k]=AValue(k);
  // Independent CPU integer dot uses logical coordinates, not GPU packed offsets.
  for(int n=0;n<S::N;++n){int sum=0;for(int k=0;k<S::K;++k)sum+=activation[k]*BValue(n,k,S::K);oracle[n]=sum;}
  Check(cudaMemcpy(d.expected,oracle.data(),oracle.size()*sizeof(int),cudaMemcpyHostToDevice));
  Initialize<S><<<132,256>>>(a,b);Check(cudaGetLastError());return d;
}
void Launch(int shape,Data const& d,U64* stamps,unsigned* errors){
  switch(shape){
    case 0:JointRing<Gate0><<<Grid,160,Reserved[0]>>>(d.p,d.expected,stamps,errors);break;
    case 1:JointRing<Gate3><<<Grid,160,Reserved[1]>>>(d.p,d.expected,stamps,errors);break;
    case 2:JointRing<Down0><<<Grid,160,Reserved[2]>>>(d.p,d.expected,stamps,errors);break;
    case 3:JointRing<Down3><<<Grid,160,Reserved[3]>>>(d.p,d.expected,stamps,errors);break;
    default:throw std::runtime_error("unknown shape");
  }Check(cudaGetLastError());
}
void RequireGuard(){
  auto visible=std::getenv("CUDA_VISIBLE_DEVICES"),index=std::getenv("TILEMEGA_DEVICE_INDEX"),held=std::getenv("TILEMEGA_GPU_LOCK_HELD"),lock=std::getenv("TILEMEGA_GPU_LOCK");
  if(!visible || !index || std::strcmp(visible,index) || !held || std::strcmp(held,"1") || !lock)throw std::runtime_error("matching guarded device required");
  int gpu=std::stoi(index);if(!((gpu>=0 && gpu<=4)||gpu==7))throw std::runtime_error("excluded timing GPU");
  if(std::string(lock)!="/tmp/tilemega_gpu"+std::to_string(gpu)+".lock")throw std::runtime_error("wrong per-card lock");
  int fd=open(lock,O_CREAT|O_RDWR,0600);if(fd<0)throw std::runtime_error("lock inspection failed");int status=flock(fd,LOCK_EX|LOCK_NB),saved=errno;if(status==0)flock(fd,LOCK_UN);close(fd);
  if(status==0 || (saved!=EAGAIN && saved!=EWOULDBLOCK))throw std::runtime_error("outer per-card lock not held");
}
std::vector<std::string> MetricNames(int slots){
  std::vector<std::string> names={"first_all_warps_observed_full","first_all_warps_observed_release","steady_observed_release_period","first_full_to_last_release","issue_to_last_release","last_release_to_finalize_envelope","issue_to_finalize"};
  for(int q=1;q<=slots;++q){names.push_back("prefix_seq"+std::to_string(q)+"_all_warps_observed_full");names.push_back("prefix_seq"+std::to_string(q)+"_all_warps_observed_release");}return names;
}
int main(int argc,char** argv){try{
  if(argc==2 && !std::strcmp(argv[1],"--help")){std::puts("F119 real joint-ring diagnostic: --fresh 0..4 --out DIRECTORY; guarded scheduler only");return 0;}
  if(argc!=5 || std::strcmp(argv[1],"--fresh") || std::strcmp(argv[3],"--out"))throw std::runtime_error("supply --fresh 0..4 --out DIRECTORY");
  int fresh=std::stoi(argv[2]);if(fresh<0||fresh>=5)throw std::runtime_error("fresh outside protocol");RequireGuard();
  std::filesystem::path folder=argv[4];if(std::filesystem::exists(folder))throw std::runtime_error("preserve prior output");std::filesystem::create_directories(folder);
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));if(prop.major!=9 || prop.minor!=0 || prop.multiProcessorCount!=Grid || prop.sharedMemPerBlockOptin<230528)throw std::runtime_error("expected 132-SM Hopper");
  std::array<Facts,4> facts={Configure<Gate0>(Reserved[0]),Configure<Gate3>(Reserved[1]),Configure<Down0>(Reserved[2]),Configure<Down3>(Reserved[3])};
  std::array<Data,4> data={PrepareData<Gate0>(),PrepareData<Gate3>(),PrepareData<Down0>(),PrepareData<Down3>()};
  U64* device;unsigned* errors;Check(cudaMalloc(&device,Grid*Words*sizeof(U64)));Check(cudaMalloc(&errors,4));Check(cudaMemset(errors,0,4));
  Frequency<<<1,1>>>(device);Check(cudaGetLastError());U64 pre[2],post[2];Check(cudaMemcpy(pre,device,16,cudaMemcpyDeviceToHost));
  for(int shape=0;shape<4;++shape)Launch(shape,data[shape],device,errors);Check(cudaDeviceSynchronize());
  std::ofstream raw(folder/"raw.tsv"),csv(folder/"metrics.csv"),meta(folder/"metadata.json");if(!raw||!csv||!meta)throw std::runtime_error("output open failed");
  raw<<"shape\ttrial\tcta\trole\tsequence\tevent\tglobal_ns\tclock64\n";
  std::array<std::vector<std::vector<double>>,4> values;for(int s=0;s<4;++s)values[s].resize(MetricNames(facts[s].slots).size());
  std::vector<U64> host(Grid*Words);U64 closure_max=0;
  for(int trial=0;trial<Trials;++trial)for(int ordinal=0;ordinal<4;++ordinal){
    int shape=fresh%2?3-ordinal:ordinal;auto const& f=facts[shape];int iterations=f.k/f.tk,middle=2*(f.slots+1),last=middle+2,final=last+2,marks=final+1;
    Launch(shape,data[shape],device,errors);Check(cudaMemcpy(host.data(),device,host.size()*8,cudaMemcpyDeviceToHost));
    for(int cta=0;cta<Active;++cta){
      auto* row=host.data()+cta*Words;std::vector<U64> maxima(marks);
      raw<<Labels[shape]<<'\t'<<trial<<'\t'<<cta<<"\tproducer\t-1\tissue_call_anchor\t"<<row[0]<<'\t'<<row[1]<<'\n';
      for(int warp=0;warp<4;++warp){U64 prior_g=row[0],prior_c=row[1];for(int mark=0;mark<marks;++mark){
        U64 g=row[2+warp*2*MaxMarks+2*mark],c=row[3+warp*2*MaxMarks+2*mark];if(g<prior_g||c<prior_c)throw std::runtime_error("reversed endpoint/issue anchor");prior_g=g;prior_c=c;maxima[mark]=std::max(maxima[mark],g);
        int sequence=mark<middle?mark/2:mark<last?WarmIndex:iterations-1;char const* event=mark==final?"finalize_complete":(mark%2?"release":"full_ready");
        raw<<Labels[shape]<<'\t'<<trial<<'\t'<<cta<<'\t'<<warp<<'\t'<<sequence<<'\t'<<event<<'\t'<<g<<'\t'<<c<<'\n';
      }}
      if(maxima[last+1]<=maxima[middle+1] || maxima[final]<maxima[last+1])throw std::runtime_error("invalid release/finalize span");
      U64 first=maxima[0]-row[0],body=maxima[last+1]-maxima[0],tail=maxima[final]-maxima[last+1],total=maxima[final]-row[0];
      U64 residual=first+body+tail>total?first+body+tail-total:total-(first+body+tail);closure_max=std::max(closure_max,residual);
      std::vector<double> nums={double(first),double(maxima[1]-row[0]),double(maxima[last+1]-maxima[middle+1])/(iterations-1-WarmIndex),double(body),double(maxima[last+1]-row[0]),double(tail),double(total)};
      for(int q=1;q<=f.slots;++q){nums.push_back(double(maxima[2*q]-row[0]));nums.push_back(double(maxima[2*q+1]-row[0]));}
      for(std::size_t m=0;m<nums.size();++m)values[shape][m].push_back(nums[m]);
    }
  }
  Frequency<<<1,1>>>(device);Check(cudaGetLastError());Check(cudaMemcpy(post,device,16,cudaMemcpyDeviceToHost));if(!pre[0]||!pre[1]||!post[0]||!post[1])throw std::runtime_error("invalid frequency bracket");
  double before=double(pre[0])/pre[1],after=double(post[0])/post[1],drift=std::abs(after/before-1)*100;unsigned failures;Check(cudaMemcpy(&failures,errors,4,cudaMemcpyDeviceToHost));
  char const* header="metric,shape,grid,active_ctas,batch,tile_n,tile_k,warps_n,warps_k,slots,input_k,iterations,period_denominator,samples,median,p99,unit";csv<<header<<'\n';std::puts(header);
  for(int s=0;s<4;++s){auto names=MetricNames(facts[s].slots);auto const& f=facts[s];for(std::size_t m=0;m<names.size();++m){auto& v=values[s][m];std::sort(v.begin(),v.end());char line[1024];std::snprintf(line,sizeof(line),"%s,%s,132,128,1,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%.12g,%.12g,ns",names[m].c_str(),Labels[s],f.tn,f.tk,f.wn,f.wk,f.slots,f.k,f.k/f.tk,f.k/f.tk-1-WarmIndex,v.size(),v[v.size()/2],v[v.size()*99/100]);csv<<line<<'\n';std::puts(line);}}
  meta<<"{\n\"schema\":1,\n\"fresh\":"<<fresh<<",\n\"performance_eligible\":false,\n\"timestamp_arithmetic\":\"integer_globaltimer_delta_v1\",\n\"frequency_before_clock_ticks\":"<<pre[0]<<",\n\"frequency_before_elapsed_ns\":"<<pre[1]<<",\n\"frequency_after_clock_ticks\":"<<post[0]<<",\n\"frequency_after_elapsed_ns\":"<<post[1]<<",\n\"frequency_drift_pct\":"<<drift<<",\n\"frequency_reject_above_pct\":2,\n\"numerical_failures\":"<<failures<<",\n\"closure_max_abs_ns\":"<<closure_max<<",\n\"real_frozen_producer\":true,\n\"epilogue\":false,\n\"grid\":132,\n\"active_ctas\":128,\n\"tasks_per_cta\":1,\n\"continuous_task_drain\":false,\n\"production_binary_equivalent\":false,\n\"endpoint_semantics\":\"warp-observed endpoints; issue is pre-Produce call anchor; tail is observation envelope; joint body already includes MMA\",\n\"shapes\":[\n";
  for(int s=0;s<4;++s){auto const& f=facts[s];meta<<"{\"shape\":\""<<Labels[s]<<"\",\"registers\":"<<f.registers<<",\"local_bytes\":"<<f.local<<",\"resident_ctas_per_sm\":"<<f.resident<<",\"dynamic_shared_bytes\":"<<f.shared<<"}"<<(s==3?"\n":",\n");}meta<<"]\n}\n";raw.close();csv.close();meta.close();
  if(failures||closure_max)throw std::runtime_error("numeric/span closure failure");if(drift>2)throw std::runtime_error("suite frequency drift exceeds predeclared 2 percent quality limit");return 0;
}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
