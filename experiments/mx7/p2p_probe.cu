// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void Check(cudaError_t error, char const* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__global__ void PeerRoundTrip(unsigned long long* remote, unsigned long long* result,
                              unsigned long long signature) {
  using Global = __attribute__((address_space(1))) volatile unsigned long long;
  *((Global*)remote) = signature;
  *result = *((Global*)remote);
}

int main() {
  try {
    int world = 0;
    Check(cudaGetDeviceCount(&world), "device_count");
    if (world != 4) throw std::runtime_error("P2P probe requires exactly four visible GPUs");
    std::vector<unsigned long long*> pointers(world);
    for (int r = 0; r < world; ++r) {
      Check(cudaSetDevice(r), "set_device");
      Check(cudaMalloc(&pointers[r], 16), "malloc");
      Check(cudaMemset(pointers[r], 0, 16), "memset");
    }
    bool pass = true;
    for (int r = 0; r < world; ++r)
      for (int q = 0; q < world; ++q) {
        if (q == r) continue;
        Check(cudaSetDevice(r), "set_device_source");
        int can = 0;
        Check(cudaDeviceCanAccessPeer(&can, r, q), "can_access_peer");
        if (!can) {
          std::printf("{\"source\":%d,\"destination\":%d,\"can_access\":0,\"pass\":false}\n", r, q);
          pass = false;
          continue;
        }
        auto enabled = cudaDeviceEnablePeerAccess(q, 0);
        if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else Check(enabled, "enable_peer");
        unsigned long long const signature = 0x5c55000000000000ull + (r << 8) + q;
        PeerRoundTrip<<<1, 1>>>(pointers[q], pointers[r] + 1, signature);
        Check(cudaGetLastError(), "launch_peer_roundtrip");
        Check(cudaDeviceSynchronize(), "synchronize_source");
        unsigned long long readback = 0, stored = 0;
        Check(cudaMemcpy(&readback, pointers[r] + 1, 8, cudaMemcpyDeviceToHost), "readback_source");
        Check(cudaSetDevice(q), "set_device_destination");
        Check(cudaMemcpy(&stored, pointers[q], 8, cudaMemcpyDeviceToHost), "readback_destination");
        bool const okay = readback == signature && stored == signature;
        pass &= okay;
        std::printf("{\"source\":%d,\"destination\":%d,\"can_access\":1,\"peer_read\":%llu,\"peer_stored\":%llu,\"pass\":%s}\n",
                    r, q, readback, stored, okay ? "true" : "false");
      }
    for (int r = 0; r < world; ++r) {
      Check(cudaSetDevice(r), "set_device_free");
      Check(cudaFree(pointers[r]), "free");
    }
    return pass ? 0 : 2;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
