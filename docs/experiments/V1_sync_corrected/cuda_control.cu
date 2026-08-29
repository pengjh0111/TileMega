// Control experiment: the same producer/consumer handshake written in plain
// CUDA C++, compiled by nvcc instead of the Tile IR toolchain.
//
// Structure mirrors V1-min exactly:
//   producer block 0: fill 1024 floats, __syncthreads(), __threadfence(),
//                     atomicExch(flag, 1)   <- release
//   consumer blocks : spin on flag (volatile relaxed poll), then
//                     __threadfence() (acquire), then read and reduce
//
// If this passes where the Tile IR version fails, the defect is in the Tile IR
// toolchain rather than in the hardware or the memory model itself.
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void handshake(float *data, int *flag, float *out) {
  int bx = blockIdx.x;
  int t = threadIdx.x;
  if (bx == 0) {
    for (int i = t; i < 1024; i += blockDim.x)
      data[i] = 1.0f;
    __syncthreads();
    __threadfence();          // release fence covering all threads' stores
    if (t == 0)
      atomicExch(flag, 1);
  } else {
    if (t == 0) {
      while (atomicAdd(flag, 0) != 1) { /* relaxed poll */ }
    }
    __syncthreads();
    __threadfence();          // acquire fence
    float s = 0.0f;
    for (int i = t; i < 1024; i += blockDim.x)
      s += data[i];
    __shared__ float red[128];
    red[t] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
      if (t < k) red[t] += red[t + k];
      __syncthreads();
    }
    if (t == 0) out[bx] = red[0];
  }
}

int main(int argc, char **argv) {
  int grid = argc > 1 ? std::atoi(argv[1]) : 170;
  int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  float *data, *out;
  int *flag;
  cudaMalloc(&data, 262144 * sizeof(float));
  cudaMalloc(&flag, sizeof(int));
  cudaMalloc(&out, 524288 * sizeof(float));
  std::vector<float> poison(262144, -1.0f), host(524288);

  int fails = 0;
  for (int r = 0; r < reps; ++r) {
    cudaMemcpy(data, poison.data(), poison.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(flag, 0, sizeof(int));
    cudaMemset(out, 0, 524288 * sizeof(float));
    cudaDeviceSynchronize();
    handshake<<<grid, 128>>>(data, flag, out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("launch error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(host.data(), out, 524288 * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int b = 1; b < grid; ++b)
      if (host[b] != 1024.0f) ++bad;
    if (bad) { ++fails; if (fails == 1) std::printf("  first failure: %d bad slots, e.g. out[1]=%g\n", bad, host[1]); }
  }
  std::printf("CUDA C++ control  grid=%-4d : %d/%d runs had bad slots\n", grid, fails, reps);
  return 0;
}
