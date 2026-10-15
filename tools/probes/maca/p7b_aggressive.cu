// P-7b: aggressive single-thread message passing to test the litmus sensitivity.
// The producer thread writes kLines words on distinct 256 B lines (spreading
// them over L2 slices) and immediately bumps the flag; the consumer thread
// pre-reads the lines, spins on the flag, then reads every line.
// -DPFENCE=1 adds __threadfence before the flag; -DCFENCE=1 after the spin.
// -DFLAG=0 relaxed atomicAdd flag, 1 volatile store flag.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef PFENCE
#define PFENCE 0
#endif
#ifndef CFENCE
#define CFENCE 0
#endif
#ifndef VREAD
#define VREAD 0  // 1: consumer data reads are volatile (forces the loads)
#endif
#ifndef FLAG
#define FLAG 0
#endif
constexpr int kLines = 64;
constexpr int kStride = 64;  // ints = 256 B
struct Pair { int data[kLines * kStride]; unsigned flag; unsigned pad[63]; unsigned ack; unsigned pad2[63]; };

__global__ void MP(Pair* pairs, int iters, unsigned long long* viol, int* sink) {
  extern __shared__ int smem[];
  if (threadIdx.x != 0) return;
  Pair* p = pairs + blockIdx.x / 2;
  bool producer = (blockIdx.x & 1) == 0;
  volatile unsigned* vflag = &p->flag;
  volatile unsigned* vack = &p->ack;
  unsigned long long bad = 0;
  int acc = 0;
  for (int i = 1; i <= iters; ++i) {
    if (producer) {
      while (*vack < static_cast<unsigned>(i - 1)) {}
#pragma unroll 8
      for (int l = 0; l < kLines; ++l) p->data[l * kStride] = i;
#if PFENCE
      __threadfence();
#endif
#if FLAG == 0
      atomicAdd(&p->flag, 1u);
#else
      *vflag = static_cast<unsigned>(i);
#endif
    } else {
#if VREAD
      for (int l = 0; l < kLines; ++l) acc += reinterpret_cast<volatile int*>(p->data)[l * kStride];
#else
      for (int l = 0; l < kLines; ++l) acc += p->data[l * kStride];
#endif
      while (*vflag < static_cast<unsigned>(i)) {}
#if CFENCE
      __threadfence();
#endif
#if VREAD
      for (int l = 0; l < kLines; ++l) bad += reinterpret_cast<volatile int*>(p->data)[l * kStride] != i;
#else
      for (int l = 0; l < kLines; ++l) bad += p->data[l * kStride] != i;
#endif
      *vack = static_cast<unsigned>(i);
    }
  }
  if (bad) atomicAdd(viol, bad);
  if (acc == 0x7fffffff) sink[0] = acc;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 200000;
  Pair* pairs; cudaMalloc(&pairs, sizeof(Pair) * 52); cudaMemset(pairs, 0, sizeof(Pair) * 52);
  unsigned long long* v; cudaMalloc(&v, 8); cudaMemset(v, 0, 8);
  int* sink; cudaMalloc(&sink, 4);
  cudaFuncSetAttribute(MP, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  MP<<<104, 64, 65536>>>(pairs, iters, v, sink);
  cudaEventRecord(b);
  cudaError_t le = cudaGetLastError(), e = cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, a, b);
  unsigned long long h; cudaMemcpy(&h, v, 8, cudaMemcpyDeviceToHost);
  std::printf("P7b VREAD=%d PFENCE=%d CFENCE=%d FLAG=%d iters=%d launch=%s sync=%s violations=%llu checks=%llu us/iter=%.3f\n",
              VREAD, PFENCE, CFENCE, FLAG, iters, cudaGetErrorString(le), cudaGetErrorString(e), h,
              52ull * kLines * iters, ms * 1000.0 / iters);
  return 0;
}
