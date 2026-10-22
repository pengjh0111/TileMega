// SPDX-License-Identifier: BSD-3-Clause
// MX-4 D2: can the decode program take its binding structure by value?
//
// The structure has the shape of MX4 prompt §3.8 at L = 32 layers, with room
// for the program's internal buffers. The host fills every byte with a
// pattern; the kernel checks each byte of its parameter, then reads the layer
// pointers through a runtime layer index (as the program's layer loop does)
// and checks those too. Prints `BINDING bytes=<n> mismatches=<m> PASS|FAIL`.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
using bf16 = std::uint16_t;
constexpr int kLayers = 32;

struct MxLayerBinding {
  bf16 const *w_qkv, *w_o, *w_gu, *w_down, *q_norm, *k_norm;
  bf16 *k_cache, *vt_cache;
};
struct MxBinding {
  MxLayerBinding layer[kLayers];
  bf16 const *embed, *w_head, *rope_cos, *rope_sin;
  std::int32_t* tokens;
  // Internal buffers (activations, partials, tickets, barriers, trace).
  void* internal[16];
};

__host__ __device__ inline unsigned char Pattern(unsigned i) { return static_cast<unsigned char>(i * 131u + 7u); }

__global__ void Check(MxBinding b, int layers, unsigned* mismatches) {
  auto const* bytes = reinterpret_cast<unsigned char const*>(&b);
  unsigned bad = 0;
  for (unsigned i = threadIdx.x; i < sizeof(MxBinding); i += blockDim.x) bad += bytes[i] != Pattern(i);
  // Runtime layer index, as in the program's `#pragma unroll 1` layer loop.
#pragma unroll 1
  for (int l = threadIdx.x % 8; l < layers; l += 8) {
    MxLayerBinding expected;
    auto* e = reinterpret_cast<unsigned char*>(&expected);
    unsigned const base = unsigned(l) * sizeof(MxLayerBinding);
    for (unsigned i = 0; i < sizeof(MxLayerBinding); ++i) e[i] = Pattern(base + i);
    bad += b.layer[l].w_qkv != expected.w_qkv;
    bad += b.layer[l].w_down != expected.w_down;
    bad += b.layer[l].vt_cache != expected.vt_cache;
  }
  atomicAdd(mismatches, bad);
}
}  // namespace

int main() {
  MxBinding b;
  auto* bytes = reinterpret_cast<unsigned char*>(&b);
  for (unsigned i = 0; i < sizeof(MxBinding); ++i) bytes[i] = Pattern(i);
  unsigned* d;
  cudaMalloc(&d, 4);
  cudaMemset(d, 0, 4);
  Check<<<104, 512>>>(b, kLayers, d);
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("BINDING bytes=%zu launch error %s FAIL\n", sizeof(MxBinding), cudaGetErrorString(e));
    return 1;
  }
  unsigned bad = 0;
  cudaMemcpy(&bad, d, 4, cudaMemcpyDeviceToHost);
  std::printf("BINDING bytes=%zu mismatches=%u %s\n", sizeof(MxBinding), bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
