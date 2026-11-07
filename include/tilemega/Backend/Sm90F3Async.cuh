// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/Async.cuh>
#include <cstdint>

// PTX the F3 forward skeleton needs beyond `Codegen/executor/Async.cuh`
// (DNN3 prompt §1.6): named barriers with an explicit thread count, the
// epoch-counted stage gate, and the wide global accesses the register
// epilogue and split-K use. Everything here is a thin wrapper: no policy.
namespace tilemega::backend {

struct Sm90F3Async {
  using Base=codegen::executor::Async<arch::Sm90>;

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

  // Named barriers (§5.1 item 4). F3 owns ids 1..4 with explicit counts, so a
  // body must never call the legacy `executor::ComputeSync`, which occupies
  // `bar 1, 128` and would release a 288-thread CTA one warpgroup early.
  enum : int {kBarAll=0, kBarGate=1, kBarConsumers=2, kBarWg0=3, kBarWg1=4};

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
  __device__ static __forceinline__ void Store128(void* address,unsigned a,unsigned b,
      unsigned c,unsigned d) {
    asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(address),
        "r"(a),"r"(b),"r"(c),"r"(d):"memory");
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
