// P-7: cross-AP memory-ordering litmus tests.
//
// Grid = 104 blocks with 64 KiB dynamic smem, so each block owns one AP.
// Blocks pair up (2k producer, 2k+1 consumer); 52 pairs run concurrently.
//
// MP (message passing), stale-L1 variant: the consumer reads the data line
// first (to leave the old value in its L1), then waits for the flag, then
// reads the data again and checks it. Producer and consumer alternate through
// an ack word, so every iteration has a stale line available.
//
// SB (store buffering): x=1; fence; r1=y || y=1; fence; r2=x. r1=r2=0 is
// forbidden with sequentially consistent fences.
//
// Select variants with -DREL=, -DWAIT=, -DREAD= (see below).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#ifndef REL
#define REL 0  // 0: threadfence + atomicAdd   1: release fetch_add (device scope)
               // 2: mxc_fence(release,device) + relaxed atomic store (backend error)
               // 3: no fence, relaxed atomicAdd (sensitivity control)
#endif
#ifndef WAIT
#define WAIT 0  // 0: atomicAdd(f,0) poll   1: relaxed atomic load poll
                // 2: volatile poll        3: acquire atomic load poll
#endif
#ifndef READ
#define READ 0  // 0: plain load   1: volatile load   2: (inv_kl1 needs xcore2000; unused)
                // 3: ldg_b128_bsm into smem, arrive_gvmcnt(0)
#endif
#ifndef SB_FENCE
#define SB_FENCE 1
#endif
#ifndef POSTFENCE
#define POSTFENCE 1  // consumer fence after observing the flag: 0 none, 1 __threadfence
#endif

constexpr int kLanes = 64;
struct alignas(256) Pair {
  int data[kLanes];
  unsigned flag;
  unsigned pad0[63];
  unsigned ack;
  unsigned pad1[63];
};

__device__ __forceinline__ unsigned Poll(unsigned* f) {
#if WAIT == 0
  return atomicAdd(f, 0u);
#elif WAIT == 1
  return __maca_atomic_load(f, __ATOMIC_RELAXED, memory_scope_device);
#elif WAIT == 2
  return *reinterpret_cast<volatile unsigned*>(f);
#else
  return __maca_atomic_load(f, __ATOMIC_ACQUIRE, memory_scope_device);
#endif
}

__device__ __forceinline__ void Publish(unsigned* f) {
#if REL == 0
  __threadfence();
  atomicAdd(f, 1u);
#elif REL == 1
  __maca_atomic_fetch_add(f, 1u, __ATOMIC_RELEASE, memory_scope_device);
#elif REL == 3
  atomicAdd(f, 1u);  // sensitivity control: no fence at all
#else
  __builtin_mxc_fence(__ATOMIC_RELEASE, "device");
  __scoped_atomic_store_n(f, __maca_atomic_load(f, __ATOMIC_RELAXED, memory_scope_device) + 1u,
                          __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
#endif
}

__device__ __forceinline__ int ReadData(int* p, int lane, int* smem) {
#if READ == 0
  return p[lane];
#elif READ == 1
  return reinterpret_cast<volatile int*>(p)[lane];
#elif READ == 2
  __builtin_mxc_inv_kl1();
  return p[lane];
#else
  if (lane < kLanes / 4)
    __builtin_mxc_ldg_b128_bsm(smem + lane * 4, p + lane * 4, 0, -1, false, true, false, false);
  __builtin_mxc_arrive_gvmcnt(0);
  __syncthreads();
  int v = smem[lane];
  __syncthreads();
  return v;
#endif
}

__global__ void MP(Pair* pairs, int iters, unsigned long long* violations, int* sink) {
  extern __shared__ int smem[];
  int const pair = blockIdx.x / 2;
  bool const producer = (blockIdx.x % 2) == 0;
  Pair* p = pairs + pair;
  int const lane = threadIdx.x;
  __shared__ unsigned go;
  unsigned long long bad = 0;
  int acc = 0;
  for (int i = 1; i <= iters; ++i) {
    if (producer) {
      if (lane == 0) { while (Poll(&p->ack) < static_cast<unsigned>(i - 1)) {} }
      __syncthreads();
      p->data[lane] = i;
      __syncthreads();
      if (lane == 0) Publish(&p->flag);
    } else {
      acc += ReadData(p->data, lane, smem);  // pre-read: stale line candidate
      if (lane == 0) {
        while (Poll(&p->flag) < static_cast<unsigned>(i)) {}
#if POSTFENCE
        __threadfence();
#endif
        go = 1;
      }
      __syncthreads();
      int v = ReadData(p->data, lane, smem);
      if (v != i) ++bad;
      __syncthreads();
      if (lane == 0) Publish(&p->ack);
    }
  }
  if (bad) atomicAdd(violations, bad);
  if (acc == 0x7fffffff) sink[0] = acc;
}

struct alignas(256) SbPair {
  unsigned bar;
  unsigned pad[63];
};

__global__ void SB(int* x, int* y, SbPair* sync, int iters, unsigned long long* violations) {
  extern __shared__ int smem[];
  int const pair = blockIdx.x / 2;
  bool const first = (blockIdx.x % 2) == 0;
  if (threadIdx.x != 0) return;
  unsigned long long bad = 0;
  int* xs = x + static_cast<size_t>(pair) * iters;
  int* ys = y + static_cast<size_t>(pair) * iters;
  unsigned* bar = &sync[pair].bar;
  for (int i = 0; i < iters; ++i) {
    atomicAdd(bar, 1u);
    while (atomicAdd(bar, 0u) < 4u * i + 2u) {}
    int r;
    if (first) {
      reinterpret_cast<volatile int*>(xs)[i] = 1;
      if (SB_FENCE) __threadfence();
      r = reinterpret_cast<volatile int*>(ys)[i];
      smem[0] = r;
    } else {
      reinterpret_cast<volatile int*>(ys)[i] = 1;
      if (SB_FENCE) __threadfence();
      r = reinterpret_cast<volatile int*>(xs)[i];
      smem[0] = r;
    }
    // Exchange results through the slot pair: encode r into the array as 2+r.
    if (first) reinterpret_cast<volatile int*>(xs)[i] = 2 + r;
    else reinterpret_cast<volatile int*>(ys)[i] = 2 + r;
    atomicAdd(bar, 1u);
    while (atomicAdd(bar, 0u) < 4u * i + 4u) {}
    if (first) {
      int r1 = reinterpret_cast<volatile int*>(xs)[i] - 2;
      int r2 = reinterpret_cast<volatile int*>(ys)[i] - 2;
      if (r1 == 0 && r2 == 0) ++bad;
    }
  }
  if (bad) atomicAdd(violations, bad);
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 1000000;
  bool sb = argc > 2 && argv[2][0] == 's';
  int const blocks = 104;
  unsigned long long* v; cudaMalloc(&v, 8); cudaMemset(v, 0, 8);
  int* sink; cudaMalloc(&sink, 4);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  if (!sb) {
    Pair* pairs; cudaMalloc(&pairs, sizeof(Pair) * blocks / 2); cudaMemset(pairs, 0, sizeof(Pair) * blocks / 2);
    cudaFuncSetAttribute(MP, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    cudaEventRecord(a);
    MP<<<blocks, kLanes, 65536>>>(pairs, iters, v, sink);
    cudaEventRecord(b);
  } else {
    int *x, *y; size_t n = static_cast<size_t>(blocks / 2) * iters;
    cudaMalloc(&x, n * 4); cudaMalloc(&y, n * 4); cudaMemset(x, 0, n * 4); cudaMemset(y, 0, n * 4);
    SbPair* s; cudaMalloc(&s, sizeof(SbPair) * blocks / 2); cudaMemset(s, 0, sizeof(SbPair) * blocks / 2);
    cudaFuncSetAttribute(SB, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    cudaEventRecord(a);
    SB<<<blocks, 64, 65536>>>(x, y, s, iters, v);
    cudaEventRecord(b);
  }
  cudaError_t le = cudaGetLastError();
  cudaError_t e = cudaDeviceSynchronize();
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  unsigned long long h = 0; cudaMemcpy(&h, v, 8, cudaMemcpyDeviceToHost);
  std::printf("P7 %s REL=%d WAIT=%d READ=%d POSTFENCE=%d iters=%d pairs=%d launch=%s sync=%s violations=%llu ms=%.1f us/iter=%.3f\n",
              sb ? "SB" : "MP", REL, WAIT, READ, POSTFENCE, iters, blocks / 2, cudaGetErrorString(le),
              cudaGetErrorString(e), h, ms, ms * 1000.0 / iters);
  return e != cudaSuccess || le != cudaSuccess || h != 0;
}
