// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseAttentionStream.cuh>
#include <tilemega/Codegen/tasks/Sm90DenseStage.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
#ifndef TILEMEGA_NATIVE_CAL_RESIDENCY
#define TILEMEGA_NATIVE_CAL_RESIDENCY 1
#endif
static_assert(TILEMEGA_NATIVE_CAL_RESIDENCY==1 || TILEMEGA_NATIVE_CAL_RESIDENCY==2);
constexpr int residency=TILEMEGA_NATIVE_CAL_RESIDENCY;
bool audit_only=false;
void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* Device(std::size_t n) {T* p;Check(cudaMalloc(&p,n*sizeof(T)));return p;}
__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ void Grid(U64* counter,int repeat) {
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    U64 v;do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<U64(repeat+1)*gridDim.x);
  }
  __syncthreads();
}
__global__ void Initialize(BF* out,std::size_t size) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
    unsigned x=unsigned(i);x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    out[i]=BF((int(x%255)-127)/128.f);
  }
}

template<int B,int D,int Q,bool Norm,int S,int Keys=64,int Slots=3,int NewRowWork=0>
using Attention=std::conditional_t<(B<=4),Sm90DenseAttention<B,D,Q,Norm,S,NewRowWork>,
    Sm90DenseAttentionStream<B,D,Q,Norm,S,Keys,Slots,NewRowWork>>;

template<int B,int D,int Q,bool Norm,int S,int Keys,int Slots,int NewRowWork>
__global__ __launch_bounds__(160,TILEMEGA_NATIVE_CAL_RESIDENCY)
void Run(Sm90DenseAttentionOperands p,U64* grid,U64* stamps,int repeats) {
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);ring.Initialize();
  for(int repeat=0;repeat<repeats;++repeat) {
    auto* out=stamps+(repeat*gridDim.x+blockIdx.x)*3;
    if(threadIdx.x==0)out[0]=Now();
    using Entry=tilemega::codegen::native_dense::StageEntry<
        Attention<B,D,Q,Norm,S,Keys,Slots,NewRowWork>>;
    Entry::Attention(p,shared,&ring,Sm90DenseNoTail{});
    if(threadIdx.x==0)out[1]=Now();
    Grid(grid,repeat);
    if(threadIdx.x==0)out[2]=Now();
  }
  ring.Invalidate();
}

void Print(char const* metric,int b,int d,int q,bool norm,int s,int keys,int slots,
    int new_row_work,int past,int grid,std::vector<double> x) {
  std::sort(x.begin(),x.end());
  std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%.12g,%.12g,ns,integer_delta_v1\n",metric,
      b,d,q,int(norm),s,keys,slots,new_row_work,past,grid,x.size(),x[x.size()/2],x[std::min(x.size()-1,x.size()*99/100)]);
}

template<int B,int D,int Q,bool Norm,int S,int Keys=64,int Slots=3,int NewRowWork=0>
void Case(cudaDeviceProp const& prop) {
  using A=Attention<B,D,Q,Norm,S,Keys,Slots,NewRowWork>;
  constexpr int groups=B*8,capacity=1088,repeats=64;
  int shared=int(std::min(prop.sharedMemPerBlockOptin,
      prop.sharedMemPerMultiprocessor/residency-prop.reservedSharedMemPerBlock)/256)*256;
  int grid=prop.multiProcessorCount*residency;
  Check(cudaFuncSetAttribute(Run<B,D,Q,Norm,S,Keys,Slots,NewRowWork>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int resident;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
      Run<B,D,Q,Norm,S,Keys,Slots,NewRowWork>,160,shared));
  if(audit_only) {
    cudaFuncAttributes attributes{};
    Check(cudaFuncGetAttributes(&attributes,Run<B,D,Q,Norm,S,Keys,Slots,NewRowWork>));
    std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d\n",B,D,Q,int(Norm),
        S,Keys,Slots,NewRowWork,grid,shared,attributes.numRegs,attributes.localSizeBytes,
        resident,int(resident==residency && A::kSharedBytes<=shared));
    return;
  }
  if(A::kSharedBytes>shared || resident!=residency) {
    // An excluded surrogate contributes no timing row; its resource facts
    // remain in the audit, without asserting production-kernel legality.
    std::fflush(stdout);
    std::fprintf(stderr,"excluded calibration B%d D%d Q%d S%d keys%d slots%d row%d grid%d shared%d required%d resident%d\n",
        B,D,Q,S,Keys,Slots,NewRowWork,grid,shared,A::kSharedBytes,resident);
    return;
  }
  auto* qkv=Device<BF>(groups*(Q+2)*D);auto* key=Device<BF>(groups*capacity*D);
  auto* value=Device<BF>(groups*capacity*D);auto* output=Device<BF>(groups*Q*D);
  auto* cosine=Device<BF>(capacity*D);auto* sine=Device<BF>(capacity*D);auto* norm=Device<BF>(D);
  auto* partial=Device<float>(groups*S*Q*D);auto* lse=Device<float>(groups*S*Q);
  auto* arrivals=Device<std::uint64_t>(groups);auto* counter=Device<U64>(1);
  auto* stamps=Device<U64>(repeats*grid*3);
  Initialize<<<grid*4,256>>>(qkv,groups*(Q+2)*D);
  Initialize<<<grid*4,256>>>(key,groups*capacity*D);
  Initialize<<<grid*4,256>>>(value,groups*capacity*D);
  std::vector<BF> one(capacity*D,BF(1));
  Check(cudaMemcpy(cosine,one.data(),one.size()*2,cudaMemcpyHostToDevice));
  Check(cudaMemcpy(norm,one.data(),D*2,cudaMemcpyHostToDevice));Check(cudaMemset(sine,0,one.size()*2));
  for(int past:{64,575,1086}) {
    Sm90DenseAttentionOperands p{qkv,key,value,output,cosine,sine,norm,norm,
        partial,lse,arrivals,past,1e-6f};
    Check(cudaMemset(arrivals,0,groups*8));Check(cudaMemset(counter,0,8));
    Run<B,D,Q,Norm,S,Keys,Slots,NewRowWork><<<grid,160,shared>>>(p,counter,stamps,4);
    Check(cudaDeviceSynchronize());
    Check(cudaMemset(arrivals,0,groups*8));Check(cudaMemset(counter,0,8));
    Run<B,D,Q,Norm,S,Keys,Slots,NewRowWork><<<grid,160,shared>>>(p,counter,stamps,repeats);
    std::vector<U64> host(repeats*grid*3);
    Check(cudaMemcpy(host.data(),stamps,host.size()*8,cudaMemcpyDeviceToHost));
    std::vector<double> active,critical,envelope,release;
    for(int repeat=1;repeat<repeats;++repeat) {
      U64 previous=0,end=0,last=0;
      for(int cta=0;cta<grid;++cta) {
        previous=std::max(previous,host[((repeat-1)*grid+cta)*3+2]);
        end=std::max(end,host[(repeat*grid+cta)*3+2]);
        last=std::max(last,host[(repeat*grid+cta)*3+1]);
        // Absolute globaltimer epochs can exceed double's 32 ns resolution.
        if(cta<A::kTasks)active.push_back(double(host[(repeat*grid+cta)*3+1]-host[(repeat*grid+cta)*3]));
      }
      critical.push_back(double(last-previous));envelope.push_back(double(end-previous));
      release.push_back(double(end-last));
    }
    Print("attention_active_cta_body",B,D,Q,Norm,S,Keys,Slots,NewRowWork,past,grid,active);
    Print("attention_critical_before_barrier",B,D,Q,Norm,S,Keys,Slots,NewRowWork,past,grid,critical);
    Print("attention_stage_envelope",B,D,Q,Norm,S,Keys,Slots,NewRowWork,past,grid,envelope);
    Print("attention_barrier_tail",B,D,Q,Norm,S,Keys,Slots,NewRowWork,past,grid,release);
    std::vector<BF> result(groups*Q*D);
    Check(cudaMemcpy(result.data(),output,result.size()*2,cudaMemcpyDeviceToHost));
    for(auto v:result)if(!std::isfinite(float(v)))throw std::runtime_error("nonfinite attention calibration output");
  }
  for(auto* p:{qkv,key,value,output,cosine,sine,norm})Check(cudaFree(p));
  for(auto* p:{partial,lse})Check(cudaFree(p));
  Check(cudaFree(arrivals));Check(cudaFree(counter));Check(cudaFree(stamps));
}

template<int D,int Q,bool Norm>
void Shape(cudaDeviceProp const& p) {
  Case<1,D,Q,Norm,8>(p);Case<1,D,Q,Norm,16>(p);Case<1,D,Q,Norm,32>(p);
  Case<2,D,Q,Norm,4>(p);Case<2,D,Q,Norm,8>(p);Case<2,D,Q,Norm,16>(p);
  Case<4,D,Q,Norm,2>(p);Case<4,D,Q,Norm,4>(p);Case<4,D,Q,Norm,8>(p);
  Case<8,D,Q,Norm,1>(p);Case<8,D,Q,Norm,2>(p);Case<8,D,Q,Norm,4>(p);
  Case<16,D,Q,Norm,1>(p);Case<16,D,Q,Norm,2>(p);Case<16,D,Q,Norm,4>(p);
  Case<16,D,Q,Norm,1,32,2>(p);Case<16,D,Q,Norm,1,128,2>(p);
  Case<16,D,Q,Norm,1,64,4>(p);
}
template<int B,int D,int Q,bool Norm,int S>
void Balance(cudaDeviceProp const& p) {
  Case<B,D,Q,Norm,S,64,3,0>(p);
  Case<B,D,Q,Norm,S,64,3,8>(p);
  Case<B,D,Q,Norm,S,64,3,16>(p);
}
template<int B,int D,int Q,bool Norm>
void ChunkShape(cudaDeviceProp const& p) {
  Case<B,D,Q,Norm,1>(p);Case<B,D,Q,Norm,2>(p);Case<B,D,Q,Norm,4>(p);
  Case<B,D,Q,Norm,8>(p);Case<B,D,Q,Norm,16>(p);Case<B,D,Q,Norm,32>(p);
  Case<B,D,Q,Norm,64>(p);
}
template<int D,int Q,bool Norm>
void AllChunks(cudaDeviceProp const& p) {
  ChunkShape<1,D,Q,Norm>(p);ChunkShape<2,D,Q,Norm>(p);ChunkShape<4,D,Q,Norm>(p);
  ChunkShape<8,D,Q,Norm>(p);ChunkShape<16,D,Q,Norm>(p);
}
int main(int argc,char** argv) {try {
  for(int i=1;i<argc;++i) {
    if(std::strcmp(argv[i],"--audit-only")==0)audit_only=true;
    else if(std::strcmp(argv[i],"--residency")==0 && i+1<argc) {
      char const* expected=residency==1?"1":"2";
      if(std::strcmp(argv[++i],expected)!=0)
        throw std::runtime_error("requested residency differs from compiled launch bounds");
    }else throw std::runtime_error("use [--audit-only] [--residency compiled-value]");
  }
  Check(cudaSetDevice(0));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.minor!=0)throw std::runtime_error("Hopper calibration requires sm90");
  std::printf(audit_only?
      "batch,head_dim,queries,qk_norm,chunks,block_keys,slots,new_row_work,grid,shared_bytes,registers,local_bytes,resident,compatible\n":
      "metric,batch,head_dim,queries,qk_norm,chunks,block_keys,slots,new_row_work,past,grid,samples,median,p99,unit,timestamp_arithmetic\n");
#ifdef TILEMEGA_CALIBRATION_ALL_CHUNKS
  AllChunks<128,2,true>(p);AllChunks<64,4,false>(p);
  AllChunks<128,3,false>(p);AllChunks<128,4,false>(p);
#elif defined(TILEMEGA_CALIBRATION_BALANCE_ONLY)
  // These four shape/batch classes cover the low-batch development cells.
  Balance<1,128,2,true,8>(p);Balance<1,128,2,true,16>(p);Balance<1,128,2,true,32>(p);
  Balance<1,64,4,false,8>(p);Balance<1,64,4,false,16>(p);Balance<1,64,4,false,32>(p);
  Balance<4,128,3,false,2>(p);Balance<4,128,3,false,4>(p);Balance<4,128,3,false,8>(p);
  Balance<1,128,4,false,8>(p);Balance<1,128,4,false,16>(p);Balance<1,128,4,false,32>(p);
#else
  Shape<128,2,true>(p);Shape<64,4,false>(p);Shape<128,3,false>(p);Shape<128,4,false>(p);
#endif
  return 0;
} catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
