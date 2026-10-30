// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/MoeDispatchV2.h>
#include <tilemega/Backend/Mx/RouteTableV3.h>
#include <cuda_runtime.h>

using namespace tilemega;
using namespace tilemega::backend::mx;

template <int B, int E, bool Fp32>
__device__ __forceinline__ MxMoeDispatchOperands Case(MxMoeDispatchOperands p, unsigned index) {
  p.logits += index * B * E;
  p.n_active += index; p.u_gate_up += index; p.u_down += index;
  p.active += index * E; p.row_count += index * E; p.rows += index * E * B;
  p.weights = (char*)p.weights + index * B * 8 * (Fp32 ? 4 : 2);
  return p;
}

template <int B, int E, bool Normalize, bool Fp32, int Version>
__global__ void __launch_bounds__(512) RouteTableTest(MxMoeDispatchOperands output) {
  extern __shared__ __align__(16) char smem[];
  auto p = Case<B,E,Fp32>(output, blockIdx.x);
  if constexpr (Version == 2) {
    MxMoeDispatchV2<B,2048,1024,64,32,Fp32,512,E,Normalize>::Run(p,smem);
  } else {
    using Build = MxBuildRouteTableV3<B,2048,1024,64,32,E,Normalize,Fp32>;
    auto* table = (MxSharedRouteTableV3<Fp32>*)(smem + Build::kSharedBytes);
    Build::Run(p.logits,table,smem);
    unsigned const t = threadIdx.x;
    if (t < table->n_active) {
      MxMoeStoreGlobal(p.active+t,unsigned(table->expert[t]));
      MxMoeStoreGlobal(p.row_count+t,unsigned(table->count[t]));
      for (unsigned row = 0; row < table->count[t]; ++row) {
        unsigned const packed = table->row[t][row];
        MxMoeStoreGlobal(p.rows+t*B+row,uint2{packed>>3,packed&7});
      }
    }
    if (t < B*8) {
      if constexpr (Fp32) MxMoeStoreGlobal((float*)p.weights+t,table->weight[t/8][t%8]);
      else MxMoeStoreGlobal((unsigned short*)p.weights+t,table->weight[t/8][t%8]);
    }
    if (t == 0) {
      MxMoeStoreGlobal(p.n_active,table->n_active);
      MxMoeStoreGlobal(p.u_gate_up,table->units_gate_up);
      MxMoeStoreGlobal(p.u_down,table->units_down);
    }
  }
}

template <int B, int E, bool Normalize, bool Fp32>
int Launch(MxMoeDispatchOperands p,int version,int cases,cudaStream_t stream,int* resources,bool query) {
  using Build = MxBuildRouteTableV3<B,2048,1024,64,32,E,Normalize,Fp32>;
  constexpr int need = Build::kSharedBytes + sizeof(MxRouteTableStorageV3<Fp32>);
  constexpr int bytes = need > 32784 ? (need+15)/16*16 : 32784;
  void const* kernel = version == 2 ? (void const*)RouteTableTest<B,E,Normalize,Fp32,2>
                                  : (void const*)RouteTableTest<B,E,Normalize,Fp32,3>;
  if (cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)) return 5;
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr,kernel)) return 6;
  int resident = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,bytes) || resident != 1) return 7;
  if (attr.numRegs > 255 || attr.localSizeBytes || attr.sharedSizeBytes) return 8;
  if (resources) {
    resources[0]=attr.numRegs;resources[1]=attr.localSizeBytes;
    resources[2]=attr.sharedSizeBytes;resources[3]=resident;
  }
  if (query) return 0;
  if (version == 2) RouteTableTest<B,E,Normalize,Fp32,2><<<cases,512,bytes,stream>>>(p);
  else RouteTableTest<B,E,Normalize,Fp32,3><<<cases,512,bytes,stream>>>(p);
  return int(cudaGetLastError());
}

template <int B, int E>
int NormalizeDispatch(MxMoeDispatchOperands p,int norm,int fp32,int version,int cases,
                      cudaStream_t stream,int* resources,bool query) {
  if (norm) {
    if (fp32) return Launch<B,E,true,true>(p,version,cases,stream,resources,query);
    return Launch<B,E,true,false>(p,version,cases,stream,resources,query);
  }
  if (fp32) return Launch<B,E,false,true>(p,version,cases,stream,resources,query);
  return Launch<B,E,false,false>(p,version,cases,stream,resources,query);
}

extern "C" int mx6_route_table(MxMoeDispatchOperands p,int batch,int experts,int norm,int fp32,
                               int version,int cases,cudaStream_t stream,int* resources,int query) {
  if ((experts!=64&&experts!=128)||(norm!=0&&norm!=1)||(fp32!=0&&fp32!=1)||
      (version!=2&&version!=3)||cases<200||(query!=0&&query!=1)) return 4;
#define CASE(B) case B: if(experts==64)return NormalizeDispatch<B,64>(p,norm,fp32,version,cases,stream,resources,query); return NormalizeDispatch<B,128>(p,norm,fp32,version,cases,stream,resources,query);
  switch(batch) {CASE(1);CASE(2);CASE(4);CASE(8);CASE(16);default:return 4;}
#undef CASE
}
