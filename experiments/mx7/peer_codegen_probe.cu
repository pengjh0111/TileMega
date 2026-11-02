// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
using U64=unsigned long long;
__global__ void VolatileStore(U64* p,U64 v) {
 using Global=__attribute__((address_space(1))) volatile U64;
 #pragma unroll
 for(int i=0;i<4;++i)*((Global*)p+threadIdx.x*4+i)=v;
}
__global__ void AtomicStore(U64* p,U64 v) {
 #pragma unroll
 for(int i=0;i<4;++i)__maca_atomic_store(p+threadIdx.x*4+i,v,__ATOMIC_RELAXED,memory_scope_system);
}
__global__ void ExchangeStore(U64* p,U64 v) {
 #pragma unroll
 for(int i=0;i<4;++i)atomicExch_system(p+threadIdx.x*4+i,v);
}
__global__ void CachedRead(U64 const* p,U64* out) {
 U64 a[4];
 #pragma unroll
 for(int i=0;i<4;++i)a[i]=__ldcg(p+threadIdx.x*4+i);
 #pragma unroll
 for(int i=0;i<4;++i)out[threadIdx.x*4+i]=a[i];
}
__global__ void AtomicRead(U64 const* p,U64* out) {
 U64 a[4];
 #pragma unroll
 for(int i=0;i<4;++i)a[i]=__maca_atomic_load(p+threadIdx.x*4+i,__ATOMIC_RELAXED,memory_scope_system);
 #pragma unroll
 for(int i=0;i<4;++i)out[threadIdx.x*4+i]=a[i];
}
int main(){return 0;}
