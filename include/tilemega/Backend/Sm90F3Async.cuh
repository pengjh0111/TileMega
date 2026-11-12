// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/TensorIm2col.h>
#include <cstdint>

// PTX the F3 forward skeleton needs beyond `Codegen/executor/Async.cuh`
// (DNN3 prompt §1.6): named barriers with an explicit thread count, the
// epoch-counted stage gate, and the wide global accesses the register
// epilogue and split-K use. Everything here is a thin wrapper: no policy.
#ifndef TILEMEGA_F3_H800_GRID
#define TILEMEGA_F3_H800_GRID 0
#endif
#if TILEMEGA_F3_H800_GRID != 0 && TILEMEGA_F3_H800_GRID != 1
#error "TILEMEGA_F3_H800_GRID must be 0 or 1"
#endif

#ifndef TILEMEGA_F3_RECOMPUTE_TASK
#define TILEMEGA_F3_RECOMPUTE_TASK 0
#endif
#if TILEMEGA_F3_RECOMPUTE_TASK != 0 && TILEMEGA_F3_RECOMPUTE_TASK != 1
#error "TILEMEGA_F3_RECOMPUTE_TASK must be 0 or 1"
#endif

namespace tilemega::backend {

struct Sm90F3Async {
  using Base=codegen::executor::Async<arch::Sm90>;

  // The opt-in runtime rejects a non-132 grid before changing the epoch.
  // Keeping the count static folds task division and prefetch partitioning.
  __device__ static __forceinline__ unsigned GridSize() {
#if TILEMEGA_F3_H800_GRID
    return 132;
#else
    return gridDim.x;
#endif
  }

  // B8's fixed-grid producer hoists shape-specific row/side offsets across
  // the repeated body and spills20B at both72/208 and72/216. Recompute the
  // CTA index at each use to bound those address lifetimes.
  __device__ static __forceinline__ unsigned BlockIndex() {
#if TILEMEGA_F3_RECOMPUTE_TASK
    unsigned value;
    asm volatile("mov.u32 %0, %%ctaid.x;":"=r"(value));
    return value;
#else
    return blockIdx.x;
#endif
  }

  __device__ static __forceinline__ void ArriveCount(std::uint64_t* barrier,unsigned count) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0], %1;"
        ::"r"(Base::Shared(barrier)),"r"(count):"memory");
  }

  __device__ static __forceinline__ void BulkMulticast(void* destination,
      void const* source,unsigned bytes,std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1], %2, [%3], %4;" ::"r"(Base::Shared(destination)),"l"(source),
        "r"(bytes),"r"(Base::Shared(barrier)),"h"(std::uint16_t(3)):"memory");
  }

  // DT-STEM: the immutable executor encoder deliberately accepts C64 only.
  // F3's stem uses a16B channel box and a nonswizzled tap-major A panel.
  static CUresult EncodeStemMap(CUtensorMap& out,void* address,
      codegen::ConvDesc const& conv,codegen::DmBufferLayout const& layout) {
    if(!address || reinterpret_cast<std::uintptr_t>(address)%16 ||
        layout.physical[3]!=8 || conv.dilation_h>128 || conv.dilation_w>128)
      throw std::invalid_argument("F3 stem im2col requires aligned C8 and supported dilation");
    cuuint64_t dimensions[]={layout.physical[3],layout.physical[2],
        layout.physical[1],layout.physical[0]};
    cuuint64_t strides[]={layout.strides[2]*2,layout.strides[1]*2,layout.strides[0]*2};
    int lower[]={int(layout.halo_left)-int(conv.pad_w),int(layout.halo_top)-int(conv.pad_h)};
    int upper[]={int(conv.w+layout.halo_left+conv.pad_w)-int(layout.physical[2])-int((conv.s-1)*conv.dilation_w),
        int(conv.h+layout.halo_top+conv.pad_h)-int(layout.physical[1])-int((conv.r-1)*conv.dilation_h)};
    cuuint32_t traversal[]={1,conv.stride_w,conv.stride_h,1};
    void* entry=nullptr;cudaDriverEntryPointQueryResult query;
    auto status=cudaGetDriverEntryPoint("cuTensorMapEncodeIm2col",&entry,cudaEnableDefault,&query);
    if(status!=cudaSuccess || query!=cudaDriverEntryPointSuccess || !entry)
      return CUDA_ERROR_NOT_SUPPORTED;
    return reinterpret_cast<decltype(&cuTensorMapEncodeIm2col)>(entry)(&out,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,address,dimensions,strides,lower,upper,8,64,
        traversal,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }

  __device__ static __forceinline__ unsigned long long GlobalTimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");
    return value;
  }

  // Positive finite F3 normalization denominators (row
  // counts or softmax sums). Reciprocal and quotient residuals stay FP32.
  // This avoids CUDA division slowpath calls that serialize WGMMA. It is
  // compensated division, not a claim of general IEEE correctly-rounded div.
  __device__ static __forceinline__ float DivideNormalization(float a,float b) {
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0, %1;":"=f"(reciprocal):"f"(b));
    reciprocal=__fmaf_rn(__fmaf_rn(-b,reciprocal,1.0f),reciprocal,reciprocal);
    float quotient=__fmul_rn(a,reciprocal);
    return __fmaf_rn(__fmaf_rn(-quotient,b,a),reciprocal,quotient);
  }

  // Recompute leader tests at their use. Caching them across a whole BERT
  // kernel makes ptxas spill a packed predicate word despite spare FP32 regs.
  __device__ static __forceinline__ unsigned ThreadIndex() {
    unsigned value;
    asm volatile("mov.u32 %0, %%tid.x;":"=r"(value));
    return value;
  }

  // Register attention operands use the SM80 warp MMA layout unchanged.
  __device__ static __forceinline__ void MatrixLoad4(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void MatrixLoad4Trans(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void Mma16x8x16(float (&c)[4],unsigned const (&a)[4],
      unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }

  template<int Registers>
  __device__ static __forceinline__ void RegDealloc() {
    static_assert(Registers>=24 && Registers<=256 && Registers%8==0);
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(Registers):"memory");
  }
  template<int Registers>
  __device__ static __forceinline__ void RegAlloc() {
    static_assert(Registers>=24 && Registers<=256 && Registers%8==0);
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(Registers):"memory");
  }

  // Named barriers (§5.1 item 4). F3 owns ids 1..5 with explicit counts, so a
  // body must never call the legacy `executor::ComputeSync`, which occupies
  // `bar 1, 128` and would release the CTA before all roles arrived.
  enum : int {kBarAll=0, kBarGate=1, kBarConsumers=2, kBarWg0=3, kBarWg1=4,
      kBarGather=5};

  __device__ static __forceinline__ void GatherCommit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  template<int Pending>
  __device__ static __forceinline__ void GatherWait() {
    static_assert(Pending>=0 && Pending<8);
    asm volatile("cp.async.wait_group %0;" :: "n"(Pending):"memory");
  }
  template<int Id,int Threads>
  __device__ static __forceinline__ void NamedBarrier() {
    static_assert(Id>=0 && Id<16,"a named barrier id is 0..15");
    static_assert(Threads>0 && Threads%32==0 && Threads<=1024);
    // Role loops meet at different PCs; bar.sync is the aligned alias.
    asm volatile("barrier.sync %0, %1;" ::"n"(Id),"n"(Threads):"memory");
  }
  // Arrive without blocking: the producer announces its own gate arrival.
  template<int Id,int Threads>
  __device__ static __forceinline__ void NamedArrive() {
    static_assert(Id>=0 && Id<16,"a named barrier id is 0..15");
    static_assert(Threads>0 && Threads%32==0 && Threads<=1024);
    asm volatile("barrier.arrive %0, %1;" ::"n"(Id),"n"(Threads):"memory");
  }

  // The stage gate (§5.5). `events[]` only ever grows: an epoch multiplies the
  // grid size, so a launch needs no memset to reset the counters (§5.7 item 3).
  __device__ static __forceinline__ void GateArrive(unsigned long long* counter) {
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(counter),"l"(1ull):"memory");
  }
  __device__ static __forceinline__ unsigned long long GateLoad(
      unsigned long long const* counter) {
    unsigned long long seen;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");
    return seen;
  }
  __device__ static __forceinline__ void GateWait(unsigned long long* counter,
      unsigned long long target) {
    while(GateLoad(counter)<target) {}
  }

  // Split-K partials cross SMs, whose L1 caches are not coherent, so both
  // sides use GPU-scope relaxed accesses published by the counter's release
  // (§6.5, the same contract the legacy body uses).
  __device__ static __forceinline__ void StorePartial2(float* address,float a,float b) {
    asm volatile("st.relaxed.gpu.global.v2.f32 [%0], {%1,%2};" ::"l"(address),
        "f"(a),"f"(b):"memory");
  }
  __device__ static __forceinline__ void LoadPartial2(float const* address,float& a,float& b) {
    asm volatile("ld.relaxed.gpu.global.v2.f32 {%0,%1}, [%2];":"=f"(a),"=f"(b)
        :"l"(address):"memory");
  }
  __device__ static __forceinline__ void StorePartial4(float* address,float a,float b,
      float c,float d) {
    asm volatile("st.relaxed.gpu.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(address),
        "f"(a),"f"(b),"f"(c),"f"(d):"memory");
  }
  __device__ static __forceinline__ void LoadPartial4(float const* address,
      float& a,float& b,float& c,float& d) {
    asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];"
        :"=f"(a),"=f"(b),"=f"(c),"=f"(d):"l"(address):"memory");
  }
  __device__ static __forceinline__ unsigned long long CounterAdd(
      unsigned long long* counter,unsigned long long value) {
    unsigned long long old;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;":"=l"(old)
        :"l"(counter),"l"(value):"memory");
    return old;
  }

  // Output stores (§6.4 step 5, and DT-ST's 16 B upgrade in §10.1).
  __device__ static __forceinline__ void Store32(void* address,unsigned word) {
    asm volatile("st.global.b32 [%0], %1;" ::"l"(address),"r"(word):"memory");
  }
  __device__ static __forceinline__ float Exp2(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;":"=f"(result):"f"(value));
    return result;
  }
  __device__ static __forceinline__ void Store128(void* address,unsigned a,unsigned b,
      unsigned c,unsigned d) {
    asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(address),
        "r"(a),"r"(b),"r"(c),"r"(d):"memory");
  }
  __device__ static __forceinline__ void StoreShared128(void* address,unsigned a,unsigned b,
      unsigned c,unsigned d) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(Base::Shared(address)),
        "r"(a),"r"(b),"r"(c),"r"(d):"memory");
  }

  __device__ static __forceinline__ void MatrixStore2(void* address,unsigned a,unsigned b) {
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1,%2};"
        ::"r"(Base::Shared(address)),"r"(a),"r"(b):"memory");
  }

  // A row-statistics or GAP side output is read by a later stage on another
  // SM, so it is published at GPU scope before the stage gate's release.
  __device__ static __forceinline__ void StoreRelaxed(float* address,float value) {
    asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" ::"l"(address),"f"(value):"memory");
  }
  __device__ static __forceinline__ float LoadRelaxed(float const* address) {
    float value;
    asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(value):"l"(address):"memory");
    return value;
  }

  // A TMA store of one epilogue tile, kept for DT-ST: the descriptor's box is
  // the tile, so the whole store is one instruction per box.
  __device__ static __forceinline__ void TensorStore2D(void const* map,int x,int y,
      void const* source) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];" ::"l"(map),"r"(x),"r"(y),
        "r"(Base::Shared(source)):"memory");
  }
};

}  // namespace tilemega::backend
