// P-6 / P-8 / P-11: barrier costs and partial barriers, persistent co-residency,
// timer rates against host-side event timing.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdint>

// P-6: __syncthreads cost per call at 128/256/512 threads.
__global__ void SyncCost(long long* out, int iters) {
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

// P-6: a compute-only barrier in smem: waves [0, kComputeWaves) synchronize,
// the remaining (loader) wave keeps running and must not be required.
template <int kParticipants>
__device__ void SmemBarrier(unsigned* count, unsigned* gen) {
  __threadfence_block();
  if (threadIdx.x % 64 == 0) {
    unsigned g = *reinterpret_cast<volatile unsigned*>(gen);
    if (atomicAdd(count, 1u) == kParticipants - 1) {
      *count = 0;
      __threadfence_block();
      atomicAdd(gen, 1u);
    } else {
      while (*reinterpret_cast<volatile unsigned*>(gen) == g) {}
    }
  }
  __builtin_mxc_barrier_warp();
  __threadfence_block();
}

__global__ void PartialBarrier(int iters, unsigned* errors, long long* cycles) {
  __shared__ unsigned count, gen;
  __shared__ int stage[2];
  if (threadIdx.x == 0) { count = 0; gen = 0; stage[0] = stage[1] = 0; }
  __syncthreads();
  int wave = threadIdx.x / 64;
  long long t0 = clock64();
  if (wave < 2) {
    for (int i = 1; i <= iters; ++i) {
      if (threadIdx.x == wave * 64) stage[wave] = i;
      SmemBarrier<2>(&count, &gen);
      if (stage[0] != i || stage[1] != i) atomicAdd(errors, 1u);
      SmemBarrier<2>(&count, &gen);
    }
  } else {
    // Loader wave: busy with unrelated work; never joins the compute barrier.
    volatile int x = 0;
    for (int i = 0; i < iters * 4; ++i) x = x + 1;
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cycles[blockIdx.x] = (t1 - t0) / (2 * iters);
}

// P-8: persistent kernel rendezvous with timeout; reports how many blocks arrived.
__global__ void Rendezvous(unsigned* arrived, unsigned* ok, long long timeout_cycles) {
  extern __shared__ char smem[];
  if (threadIdx.x == 0) {
    smem[0] = 1;
    atomicAdd(arrived, 1u);
    long long t0 = clock64();
    while (*reinterpret_cast<volatile unsigned*>(arrived) < gridDim.x) {
      if (clock64() - t0 > timeout_cycles) return;
    }
    atomicAdd(ok, 1u);
  }
}

__global__ void CoopSync(unsigned* out) {
  cooperative_groups::grid_group g = cooperative_groups::this_grid();
  atomicAdd(out, 1u);
  g.sync();
  if (threadIdx.x == 0 && *reinterpret_cast<volatile unsigned*>(out) != gridDim.x * blockDim.x) atomicAdd(out + 1, 1u);
}

// P-11: spin for N cycles, report clock64 and realtime deltas.
__global__ void Spin(long long cycles, unsigned long long* out) {
  unsigned long long r0 = __builtin_mxc_get_realtime();
  long long c0 = clock64();
  while (clock64() - c0 < cycles) {}
  long long c1 = clock64();
  unsigned long long r1 = __builtin_mxc_get_realtime();
  if (threadIdx.x == 0) { out[0] = c1 - c0; out[1] = r1 - r0; }
}

int main() {
  long long* d; cudaMalloc(&d, 4096 * 8);
  long long h[256];
  for (int threads : {128, 256, 512}) {
    SyncCost<<<104, threads>>>(d, 100000);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost);
    std::printf("P6 syncthreads threads=%d cycles/call=%.1f\n", threads, h[0] / 100000.0);
  }
  unsigned* u; cudaMalloc(&u, 64); cudaMemset(u, 0, 64);
  PartialBarrier<<<104, 192>>>(100000, u, d);
  cudaError_t e = cudaDeviceSynchronize();
  unsigned errs; cudaMemcpy(&errs, u, 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost);
  std::printf("P6 smem compute-only barrier (2 of 3 waves) err=%s errors=%u cycles/barrier=%lld\n",
              cudaGetErrorString(e), errs, h[0]);

  for (int k : {1, 2, 3}) {
    int smem = 65536 / k;
    cudaFuncSetAttribute(Rendezvous, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    for (int threads : {128, 192, 256}) {
      cudaMemset(u, 0, 64);
      Rendezvous<<<104 * k, threads, smem>>>(u, u + 1, 2000000000ll);
      cudaError_t le = cudaGetLastError();
      e = cudaDeviceSynchronize();
      unsigned hu[2]; cudaMemcpy(hu, u, 8, cudaMemcpyDeviceToHost);
      std::printf("P8 rendezvous grid=104x%d threads=%d smem=%d arrived=%u completed=%u launch=%s sync=%s\n", k, threads,
                  smem, hu[0], hu[1], cudaGetErrorString(le), cudaGetErrorString(e));
    }
  }
  int coop = 0; cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
  cudaMemset(u, 0, 64);
  void* args[] = {&u};
  cudaError_t ce = cudaLaunchCooperativeKernel((void*)CoopSync, dim3(104), dim3(128), args, 0, 0);
  e = cudaDeviceSynchronize();
  unsigned hu[2]; cudaMemcpy(hu, u, 8, cudaMemcpyDeviceToHost);
  std::printf("P8 cooperative attr=%d launch=%s sync=%s count=%u bad=%u\n", coop, cudaGetErrorString(ce),
              cudaGetErrorString(e), hu[0], hu[1]);

  unsigned long long* t; cudaMalloc(&t, 16);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (long long cyc : {1000000ll, 10000000ll, 100000000ll}) {
    cudaEventRecord(a);
    Spin<<<1, 64>>>(cyc, t);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    unsigned long long ht[2]; cudaMemcpy(ht, t, 16, cudaMemcpyDeviceToHost);
    std::printf("P11 spin cycles=%llu realtime_ticks=%llu event_ms=%.3f -> clock64 MHz=%.1f realtime MHz=%.2f\n", ht[0],
                ht[1], ms, ht[0] / (ms * 1e3), ht[1] / (ms * 1e3));
  }
  return 0;
}
