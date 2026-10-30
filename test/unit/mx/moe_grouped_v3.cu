// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/GroupedGemmV2.h>
#include <tilemega/Backend/Mx/GroupedGemmV3.h>
#include <cuda_runtime.h>
#include <cstddef>

using namespace tilemega;
using namespace tilemega::backend::mx;
using namespace tilemega::codegen;

struct Mx6GroupedArgs {
  Bf16 const *x, *w;
  Bf16 *out, *residual, *combined;
  unsigned *tickets, *active, *count, *rows, *calls;
  Bf16 const* table;
  unsigned units;
};
using Rms = DmEpilogueStep<DmEpilogueKind::kDeferredRMSNorm,DmActivation::kRelu,
    DmGatePair::kSimpleGate,1,DmRounding::kFP32,DmRounding::kBF16,DmWriteKind::kDense,1>;
using Gate = DmEpilogueStep<DmEpilogueKind::kGatePair,DmActivation::kRelu,
    DmGatePair::kSwiGLU,16,DmRounding::kBF16,DmRounding::kBF16,DmWriteKind::kDense,1>;

template<int B, int N, int K>
struct Mx6GroupedShape : MxStaticShapeV2<N,K,B,512> {
  using Sync = mx::StageSyncV3<512>;
};

template<int B,int TN,int WN,int Depth,bool Gu,int Version>
__global__ void __launch_bounds__(512) Mx6GroupedKernel(Mx6GroupedArgs a) {
  extern __shared__ __align__(16) char storage[];
  constexpr int N = 512, K = Gu ? 512 : 256;
  auto next = [&] {
    if (threadIdx.x % 64 == 0) atomicAdd(a.calls + blockIdx.x * 8 + threadIdx.x / 64,1u);
    return MxNoPrefetchedV2{};
  };
  if constexpr (Version == 2) {
    using Shape = MxStaticShapeV2<N,K,B,512>;
    using Completion = std::conditional_t<Gu,MxNoUnitCompletionV2,MxMoeDownCompletionV2<false>>;
    using Body = MxGroupedGemm<TN,WN,Depth,Gu?MxOpV2::kSwiGLU:MxOpV2::kStore,
        Gu,MxMoeGateChainV2<Rms,Gate>,Completion,Gu?0:Depth>;
    auto weights = a.table + offsetof(MxRouteTableV3,weight) / 2;
    MxMoeCombineOperandsV2 combine{a.out,weights,a.residual,a.combined,
        (MxMoeGlobalTicketV2*)a.tickets};
    MxGroupedGemmOperandsV2 p{a.x,a.w,a.out,a.active,a.count,reinterpret_cast<uint2 const*>(a.rows),a.units,
        N*K,1e-6f,{},combine};
    auto pf = Body::Prefetch(p,Shape{});
    Body::Run(pf,p,Shape{},storage,next);
  } else {
    using Shape = Mx6GroupedShape<B,N,K>;
    using Body = MxGroupedGemmV3<TN,WN,Depth,Gu?MxOpV2::kSwiGLU:MxOpV2::kStore,
        Gu,MxMoeGateChainV3<Rms,Gate>,Gu?0:Depth>;
    constexpr int body = (MxGemmSmemV2(B,TN,WN).bytes + 15) / 16 * 16;
    constexpr int offset = body > 32784 ? body : 32784;
    auto* table = (MxSharedRouteTableV3<false>*)(storage + offset);
    using Vector = unsigned __attribute__((ext_vector_type(4)));
    using SharedVector = __attribute__((address_space(3))) Vector;
    if (threadIdx.x < sizeof(MxRouteTableV3)/16) {
      uint4 v = tilemega::backend::LoadGlobal16Gvm(a.table + threadIdx.x*8);
      __builtin_mxc_arrive_gvmcnt(0);
      ((SharedVector*)table)[threadIdx.x] = Vector{v.x,v.y,v.z,v.w};
    }
    Shape::Sync::LdsSync();
    MxMoeCombineOperandsV3<false> combine{a.out,table,a.residual,a.combined,
        (MxMoeGlobalTicketV2*)a.tickets};
    MxGroupedGemmOperandsV3<false> p{a.x,a.w,a.out,table,N*K,1e-6f,{},combine,nullptr};
    auto pf = Body::Prefetch(p,Shape{});
    Body::Run(pf,p,Shape{},storage,next);
  }
}

template<int B,int TN,int WN,int Depth,bool Gu,int Version>
int Launch(Mx6GroupedArgs a,cudaStream_t stream,int* resource,bool query) {
  constexpr int core = (MxGemmSmemV2(B,TN,WN).bytes + 15)/16*16;
  constexpr int bytes = (core > 32784 ? core : 32784) + sizeof(MxRouteTableV3);
  auto kernel = Mx6GroupedKernel<B,TN,WN,Depth,Gu,Version>;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,kernel)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,bytes)!=cudaSuccess)return 7;
  if(resource){resource[0]=p.numRegs;resource[1]=p.localSizeBytes;resource[2]=p.sharedSizeBytes;resource[3]=resident;}
  if(p.localSizeBytes||p.sharedSizeBytes||p.numRegs>255||resident!=1)return 8;
  if(query)return 0;
  kernel<<<104,512,bytes,stream>>>(a);
  return int(cudaGetLastError());
}

template<int B,int TN,int WN,int Depth>
int Select(Mx6GroupedArgs a,int version,int gu,cudaStream_t stream,int* resource,bool query) {
  if(version==2&&gu)return Launch<B,TN,WN,Depth,true,2>(a,stream,resource,query);
  if(version==2)return Launch<B,TN,WN,Depth,false,2>(a,stream,resource,query);
  if(version==3&&gu)return Launch<B,TN,WN,Depth,true,3>(a,stream,resource,query);
  if(version==3)return Launch<B,TN,WN,Depth,false,3>(a,stream,resource,query);
  return 4;
}
extern "C" int mx6_grouped(Mx6GroupedArgs a,int batch,int tn,int wn,int depth,int version,int gu,
    cudaStream_t stream,int* resource,int query) {
  if((gu!=0&&gu!=1)||(query!=0&&query!=1))return 4;
#define CASE(B,T,W,D) if(batch==B&&tn==T&&wn==W&&depth==D)return Select<B,T,W,D>(a,version,gu,stream,resource,bool(query))
  CASE(1,32,1,1);CASE(1,64,1,2);CASE(1,32,2,2);
  CASE(16,32,1,1);CASE(16,64,1,2);CASE(16,32,2,2);CASE(16,32,8,2);
#undef CASE
  return 4;
}
extern "C" unsigned mx6_grouped_args_size(){return sizeof(Mx6GroupedArgs);}
