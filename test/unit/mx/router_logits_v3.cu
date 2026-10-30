// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/RouterGemmV2.h>
#include <tilemega/Backend/Mx/RouterLogitsV3.h>
#include <cuda_runtime.h>
using namespace tilemega::backend::mx;

template <int B,int E>
struct ShapeV3 : MxStaticShapeV2<E,2048,B,512> {
  using Sync=tilemega::codegen::mx::StageSyncV3<512>;
};

template <int B,int E,int D,int Version>
__global__ void __launch_bounds__(512) RouterLogitsTest(MxMoeRouterOperandsV2 p) {
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char smem[];
  if constexpr(Version==2) {
    using Shape=MxStaticShapeV2<E,2048,B,512>;
    using Dispatch=MxMoeDispatchV2<B,2048,1024,64,32,false,512,E,E==128>;
    using Body=MxRouterGemmV2<16,D,MxPart::kTileRange,Dispatch,D>;
    auto pf=Body::Prefetch(p,Shape{});
    Body::Run(pf,p,Shape{},smem,[]{return MxNoPrefetchedV2{};});
  } else {
    using Body=MxRouterLogitsV3<16,D,D>;
    MxRouterLogitsOperandsV3 ops{p.a,p.w,p.residual,p.out,p.a_tokens,p.residual_tokens,
                               p.a_stride,p.residual_stride,p.epsilon};
    auto pf=Body::Prefetch(ops,ShapeV3<B,E>{});
    Body::Run(pf,ops,ShapeV3<B,E>{},smem,[]{return MxNoPrefetchedV2{};});
  }
}

template<int B,int E,int D>
int Launch(MxMoeRouterOperandsV2 p,int version,cudaStream_t stream,int* resources,bool query) {
  constexpr int bytes=32784;
  void const* kernel=version==2?(void const*)RouterLogitsTest<B,E,D,2>:(void const*)RouterLogitsTest<B,E,D,3>;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes))return 5;
  cudaFuncAttributes attr{};
  if(cudaFuncGetAttributes(&attr,kernel))return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,bytes)||resident!=1)return 7;
  if(attr.localSizeBytes||attr.sharedSizeBytes||attr.numRegs>255)return 8;
  if(resources){resources[0]=attr.numRegs;resources[1]=attr.localSizeBytes;resources[2]=attr.sharedSizeBytes;resources[3]=resident;}
  if(query)return 0;
  if(version==2)RouterLogitsTest<B,E,D,2><<<104,512,bytes,stream>>>(p);
  else RouterLogitsTest<B,E,D,3><<<104,512,bytes,stream>>>(p);
  return int(cudaGetLastError());
}

extern "C" unsigned mx6_router_operands_size(){return sizeof(MxMoeRouterOperandsV2);}
extern "C" int mx6_router_logits(MxMoeRouterOperandsV2 p,int batch,int experts,int depth,int version,
                                 cudaStream_t stream,int* resources,int query) {
  if((version!=2&&version!=3)||(query!=0&&query!=1))return 4;
#define CASE(B,E,D) if(batch==B&&experts==E&&depth==D)return Launch<B,E,D>(p,version,stream,resources,query)
#define DEPTHS(B,E) CASE(B,E,1);CASE(B,E,2);CASE(B,E,3);CASE(B,E,4)
  DEPTHS(1,64);DEPTHS(1,128);DEPTHS(16,64);DEPTHS(16,128);
#undef DEPTHS
#undef CASE
  return 4;
}
