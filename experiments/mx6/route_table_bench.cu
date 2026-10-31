// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/RouteTableV3.h>
#include <cuda_runtime.h>

#ifndef MX6_ROUTE_WEIGHTS
#define MX6_ROUTE_WEIGHTS 1
#endif
#ifndef MX6_ROUTE_ORDINAL
#define MX6_ROUTE_ORDINAL 1
#endif
#ifndef MX6_OLD_TABLE
#define MX6_OLD_TABLE 0
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
template<int B,int E>
__global__ void __launch_bounds__(512) RouteTableBench(Bf16 const* logits,unsigned* sink,int iterations) {
  extern __shared__ __align__(16) char storage[];
#if MX6_OLD_TABLE
  using Build=MxBuildRouteTableV3<B,2048,1024,64,32,E,(E==128)>;
#else
  using Build=MxBuildRouteTableV3<B,2048,1024,64,32,E,(E==128),false,
      bool(MX6_ROUTE_WEIGHTS),bool(MX6_ROUTE_ORDINAL)>;
#endif
  static_assert(Build::kSharedBytes<=32784);
  auto* table=(MxSharedRouteTableV3<false>*)(storage+32784);
  for(int i=0;i<iterations;++i) {
    Build::Run(logits,table,storage);
    if(threadIdx.x==0) {
      using Global=__attribute__((address_space(1))) volatile unsigned;
      *((Global*)sink+blockIdx.x)=table->n_active+table->weight[0][i&7];
    }
  }
}
template<int B,int E>
int Launch(Bf16 const* logits,unsigned* sink,int iterations,cudaStream_t stream,int* resource,bool query) {
  constexpr int bytes=32784+sizeof(MxRouteTableV3);auto kernel=RouteTableBench<B,E>;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes))return 5;
  cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,kernel))return 6;
  int resident=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,bytes))return 7;
  if(resource){resource[0]=p.numRegs;resource[1]=p.localSizeBytes;resource[2]=p.sharedSizeBytes;resource[3]=resident;}
  if(p.numRegs>255||p.localSizeBytes||p.sharedSizeBytes||resident!=1)return 8;
  if(query)return 0;
  kernel<<<104,512,bytes,stream>>>(logits,sink,iterations);return int(cudaGetLastError());
}
extern "C" int mx6_route_bench(Bf16 const* logits,unsigned* sink,int b,int e,int iterations,
    cudaStream_t stream,int* resource,int query) {
  if(iterations<1||(query!=0&&query!=1))return 4;
#define CASE(B,E) if(b==B&&e==E)return Launch<B,E>(logits,sink,iterations,stream,resource,bool(query))
  CASE(1,64);CASE(1,128);CASE(16,64);CASE(16,128);
#undef CASE
  return 4;
}
