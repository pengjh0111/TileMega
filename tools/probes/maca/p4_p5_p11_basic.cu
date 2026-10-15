// P-4 / P-5 / P-11: device attributes and occupancy, wave64 intrinsics, timers.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

template <int kThreads>
__global__ void __launch_bounds__(kThreads) SmemUser(float* out) {
  extern __shared__ float smem[];
  smem[threadIdx.x] = threadIdx.x;
  __syncthreads();
  out[blockIdx.x * kThreads + threadIdx.x] = smem[kThreads - 1 - threadIdx.x];
}

// P-5: every lane records results of the shuffle variants.
__global__ void WaveOps(int* out) {
  int lane = threadIdx.x;  // one wave of 64
  int v = 1000 + lane;
  int* o = out + lane * 16;
  o[0] = __shfl_xor_sync(0xffffffffffffffffull, v, 1);
  o[1] = __shfl_xor_sync(0xffffffffu, v, 1);
  o[2] = __shfl_xor_sync(0xffffffffffffffffull, v, 32);
  o[3] = __shfl_sync(0xffffffffffffffffull, v, 0);           // width 64
  o[4] = __shfl_sync(0xffffffffffffffffull, v, 0, 32);       // width 32
  o[5] = __shfl_down_sync(0xffffffffffffffffull, v, 1);
  o[6] = __shfl_down_sync(0xffffffffffffffffull, v, 1, 32);
  o[7] = __shfl_up_sync(0xffffffffffffffffull, v, 1);
  o[8] = __shfl_up_sync(0xffffffffffffffffull, v, 1, 32);
  unsigned long long b = __ballot_sync(0xffffffffffffffffull, lane % 3 == 0);
  o[9] = static_cast<int>(b & 0xffffffffu);
  o[10] = static_cast<int>(b >> 32);
  o[11] = static_cast<int>(sizeof(__ballot_sync(0xffffffffffffffffull, 1)));
  unsigned long long am = 0;
  if (lane < 32) am = __activemask();
  o[12] = static_cast<int>(am & 0xffffffffu);
  o[13] = static_cast<int>(am >> 32);
  // Half-wave divergent __syncwarp: only the low half executes it with a
  // 32-bit mask. A hang here would be caught by the host timeout.
  int r = 0;
  if (lane < 32) { __syncwarp(0xffffffffu); r = 1; }
  o[14] = r;
  o[15] = warpSize;
}

// P-11: timers and AP id.
__global__ void Timers(long long* out, int sleep_ns) {
  long long c0 = clock64();
  unsigned long long t0 = __builtin_mxc_get_realtime();
  unsigned long long g0 = __builtin_mxc_get_time();
  __nanosleep(sleep_ns);
  long long c1 = clock64();
  unsigned long long t1 = __builtin_mxc_get_realtime();
  unsigned long long g1 = __builtin_mxc_get_time();
  if (threadIdx.x == 0) {
    out[blockIdx.x * 8 + 0] = c1 - c0;
    out[blockIdx.x * 8 + 1] = static_cast<long long>(t1 - t0);
    out[blockIdx.x * 8 + 2] = static_cast<long long>(g1 - g0);
    out[blockIdx.x * 8 + 3] = static_cast<long long>(t0);
    out[blockIdx.x * 8 + 4] = static_cast<long long>(g0);
  }
}

__global__ void PrintfTest() {
  if (threadIdx.x == 0 && blockIdx.x == 0) printf("device printf ok warpSize=%d\n", warpSize);
}

int main() {
  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("P4 name=%s cc=%d.%d sms=%d smem/block=%zu smem/block_optin=%zu smem/sm=%zu regs/block=%d regs/sm=%d "
              "maxthr/block=%d maxthr/sm=%d maxblk/sm=%d warp=%d l2=%d clock_khz=%d memclk_khz=%d busw=%d "
              "concurrentManaged=%d coopLaunch=%d\n",
              p.name, p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlock, p.sharedMemPerBlockOptin,
              p.sharedMemPerMultiprocessor, p.regsPerBlock, p.regsPerMultiprocessor, p.maxThreadsPerBlock,
              p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor, p.warpSize, p.l2CacheSize, p.clockRate,
              p.memoryClockRate, p.memoryBusWidth, p.concurrentManagedAccess, p.cooperativeLaunch);
  int v = 0;
  CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  std::printf("P4 attr MaxSharedMemoryPerBlockOptin=%d\n", v);
  float* buf; CHECK(cudaMalloc(&buf, 1 << 24));
  auto occ = [&](auto kernel, int threads, int smem, char const* name) {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int blocks = -1;
    cudaError_t e2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem);
    cudaFuncAttributes a{}; cudaFuncGetAttributes(&a, kernel);
    std::printf("P4 occ %s threads=%d smem=%d setattr=%s occ=%s blocks/AP=%d regs=%d\n", name, threads, smem,
                cudaGetErrorString(e), cudaGetErrorString(e2), blocks, a.numRegs);
  };
  for (int smem : {16384, 32768, 49152, 65536}) {
    occ(SmemUser<128>, 128, smem, "t128");
    occ(SmemUser<256>, 256, smem, "t256");
    occ(SmemUser<512>, 512, smem, "t512");
  }
  // Launch with 64 KiB really works?
  CHECK(cudaFuncSetAttribute(SmemUser<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
  SmemUser<128><<<104, 128, 65536>>>(buf);
  std::printf("P4 launch 64KiB: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  SmemUser<128><<<104, 128, 65536 + 1024>>>(buf);
  std::printf("P4 launch 65KiB (expected failure): %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  cudaGetLastError();

  int* wout; CHECK(cudaMalloc(&wout, 64 * 16 * sizeof(int)));
  WaveOps<<<1, 64>>>(wout);
  CHECK(cudaDeviceSynchronize());
  int h[64 * 16]; CHECK(cudaMemcpy(h, wout, sizeof(h), cudaMemcpyDeviceToHost));
  char const* names[] = {"xor1_m64", "xor1_m32", "xor32", "shfl0_w64", "shfl0_w32", "down1_w64", "down1_w32",
                         "up1_w64", "up1_w32", "ballot_lo", "ballot_hi", "ballot_sizeof", "active_lo(low half)",
                         "active_hi(low half)", "syncwarp_half", "warpSize"};
  for (int k = 0; k < 16; ++k) {
    std::printf("P5 %-20s", names[k]);
    for (int lane : {0, 1, 30, 31, 32, 33, 62, 63}) std::printf(" L%d=%d", lane, h[lane * 16 + k]);
    std::printf("\n");
  }

  long long* tout; CHECK(cudaMalloc(&tout, 8 * 8 * sizeof(long long)));
  for (int ns : {0, 1000, 10000, 100000}) {
    Timers<<<2, 64>>>(tout, ns);
    CHECK(cudaDeviceSynchronize());
    long long t[16]; CHECK(cudaMemcpy(t, tout, sizeof(t), cudaMemcpyDeviceToHost));
    std::printf("P11 sleep=%d clock64_delta=%lld realtime_delta=%lld gettime_delta=%lld realtime0=%lld gettime0=%lld | blk1 realtime0=%lld\n",
                ns, t[0], t[1], t[2], t[3], t[4], t[11]);
  }
  PrintfTest<<<1, 64>>>();
  CHECK(cudaDeviceSynchronize());
  std::printf("DONE\n");
  return 0;
}
