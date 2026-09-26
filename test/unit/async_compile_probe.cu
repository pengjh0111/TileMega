// Compile/SASS probe only: deliberately not an executable protocol test.
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
using A=tilemega::codegen::executor::Async<>;
extern "C" __global__ void tilemega_l1_kernel(void const* src,void const* map,int* output) {
 __shared__ alignas(1024) unsigned char page[8192];
 __shared__ unsigned long long barrier;
 auto* b=reinterpret_cast<std::uint64_t*>(&barrier);
 A::Init(b,1);A::InitFence();
 if constexpr(A::Caps::kBulkCopy) {A::ExpectTx(b,128);A::Bulk(page,src,128,b);}
 else {A::Copy16(page,src);A::CompleteCopies(b);}
 A::Wait(b,0);A::Arrive(b);
 if constexpr(A::Caps::kTma){A::ExpectTx(b,1024);A::Tensor2D(page,map,0,0,b);A::Wait(b,1);}
 A::Prefetch(src,128);A::WaitPreviousGrid();output[threadIdx.x]=page[threadIdx.x];A::LaunchDependents();
}
