// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
#include <mcr/mc_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using Word = unsigned long long;
static void Check(cudaError_t error, char const* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
__device__ __forceinline__ Word Read(Word const* p, int method) {
  using Global = __attribute__((address_space(1))) volatile Word const;
  switch (method) {
    case 1: return __ldcg(p);
    case 2: return __ldcs(p);
    case 3: return __ldlu(p);
    case 4: return __ldcv(p);
    case 5: return atomicAdd_system(const_cast<Word*>(p), 0ull);
    case 6: return __maca_atomic_load(p, __ATOMIC_RELAXED, memory_scope_system);
    default: return *((Global*)p);
  }
}
__device__ __forceinline__ void Write(Word* p, Word v, int method) {
  using Global = __attribute__((address_space(1))) volatile Word;
  if (method == 1) atomicExch_system(p, v);
  else *((Global*)p) = v;
}
struct Args {
  Word* p[4];
  unsigned* errors;
  int rank, store, load, peers;
};
__device__ __forceinline__ bool Await(Word const* p, unsigned e, Args const& a) {
  for (unsigned n = 0; n < (1u << 24); ++n)
    if (unsigned(Read(p, a.load) >> 32) == e) return true;
  atomicAdd(a.errors, 1u);
  return false;
}
__global__ void PingPong(Args a) {
  for (unsigned e = 1; e <= 1000; ++e) {
    Word const packet = (Word(e) << 32) | (0x55aa00u ^ e);
    if (a.rank == 0) {
      Write(a.p[1], packet, a.store);
      if (!Await(a.p[0], e, a)) return;
    } else {
      if (!Await(a.p[1], e, a)) return;
      Write(a.p[0], packet, a.store);
    }
  }
}
__global__ void Bandwidth(Args a) {
  unsigned const thread = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned const stride = gridDim.x * blockDim.x;
  constexpr unsigned packets = 16 * 1024 * 1024 / sizeof(Word);
  for (int q = 1; q <= a.peers; ++q)
    for (unsigned i = thread; i < packets; i += stride)
      Write(a.p[q] + i, 0x155aa55aaull, a.store);
}
static void* Allocate(std::size_t bytes, int method) {
  void* p = nullptr;
  if (method < 0) Check(cudaMalloc(&p, bytes), "cudaMalloc");
  else if (auto e = mcExtMallocWithFlags(&p, bytes, unsigned(method)); e != mcSuccess)
    throw std::runtime_error("mcExtMallocWithFlags: " + std::to_string(int(e)));
  Check(cudaMemset(p, 0, bytes), "clear");
  return p;
}
int main(int argc, char** argv) {
  try {
    if (argc != 7) throw std::runtime_error("usage: p2p_bw <ping|bandwidth> <allocation> <store> <load> <CTAs> <peers>");
    std::string const mode = argv[1];
    int const alloc = std::atoi(argv[2]), store = std::atoi(argv[3]), load = std::atoi(argv[4]);
    int const ctas = std::atoi(argv[5]), peers = std::atoi(argv[6]);
    bool const ping = mode == "ping";
    int world = 0;
    Check(cudaGetDeviceCount(&world), "device_count");
    if (world != peers + 1 || (world != 2 && world != 4) || (ping && (world != 2 || ctas != 1)) ||
        (!ping && mode != "bandwidth") || (!ping && ctas != 1 && ctas != 8 && ctas != 32 && ctas != 104))
      throw std::runtime_error("invalid mode, CTA count or visible group");
    std::size_t const bytes = ping ? 64 : 16 * 1024 * 1024;
    Args args[4]{};
    cudaStream_t streams[4]{};
    cudaEvent_t begin{}, end{};
    for (int r = 0; r < world; ++r) {
      Check(cudaSetDevice(r), "set_device");
      args[r].p[r] = static_cast<Word*>(Allocate(bytes, alloc));
      cudaPointerAttributes attributes{};
      Check(cudaPointerGetAttributes(&attributes, args[r].p[r]), "pointer_attributes");
      std::printf("{\"record_type\":\"allocation_attributes\",\"rank\":%d,\"allocation\":%d,\"device\":%d,\"aligned64\":%s,\"pass\":%s}\n",
                  r, alloc, attributes.device,
                  (reinterpret_cast<std::uintptr_t>(args[r].p[r]) & 63) == 0 ? "true" : "false",
                  attributes.device == r && (reinterpret_cast<std::uintptr_t>(args[r].p[r]) & 63) == 0 ? "true" : "false");
      if (attributes.device != r || (reinterpret_cast<std::uintptr_t>(args[r].p[r]) & 63) != 0)
        throw std::runtime_error("allocator does not preserve bind_tp device ownership and alignment");
      Check(cudaMalloc(&args[r].errors, 4), "errors_malloc");
      Check(cudaMemset(args[r].errors, 0, 4), "errors_clear");
      Check(cudaStreamCreate(&streams[r]), "stream_create");
      Check(cudaDeviceSynchronize(), "initial_sync");
    }
    for (int r = 0; r < world; ++r) {
      Check(cudaSetDevice(r), "set_device");
      for (int q = 0; q < world; ++q) {
        args[r].p[q] = args[q].p[q];
        if (q == r) continue;
        auto e = cudaDeviceEnablePeerAccess(q, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else Check(e, "enable_peer");
      }
      args[r].rank = r; args[r].store = store; args[r].load = load; args[r].peers = peers;
    }
    Check(cudaSetDevice(0), "set_device_source");
    Check(cudaEventCreate(&begin), "event_create");
    Check(cudaEventCreate(&end), "event_create");
    Check(cudaEventRecord(begin, streams[0]), "event_record");
    if (ping) PingPong<<<1, 1, 0, streams[0]>>>(args[0]);
    else Bandwidth<<<ctas, 512, 0, streams[0]>>>(args[0]);
    Check(cudaGetLastError(), "source_launch");
    Check(cudaEventRecord(end, streams[0]), "event_record");
    if (ping) {
      Check(cudaSetDevice(1), "set_device_peer");
      PingPong<<<1, 1, 0, streams[1]>>>(args[1]);
      Check(cudaGetLastError(), "peer_launch");
    }
    bool pass = true;
    for (int r = 0; r < world; ++r) {
      Check(cudaSetDevice(r), "set_device_sync");
      Check(cudaStreamSynchronize(streams[r]), "stream_sync");
      unsigned errors = 0;
      Check(cudaMemcpy(&errors, args[r].errors, 4, cudaMemcpyDeviceToHost), "errors_read");
      pass &= errors == 0;
      if (!ping && r > 0) {
        Word first = 0, last = 0;
        Check(cudaMemcpy(&first, args[r].p[r], 8, cudaMemcpyDeviceToHost), "read_first");
        Check(cudaMemcpy(&last, args[r].p[r] + bytes / 8 - 1, 8, cudaMemcpyDeviceToHost), "read_last");
        pass &= first == 0x155aa55aaull && last == first;
      }
    }
    Check(cudaSetDevice(0), "set_device_timing");
    float elapsed_ms = 0;
    Check(cudaEventElapsedTime(&elapsed_ms, begin, end), "event_elapsed");
    std::printf("{\"mode\":\"%s\",\"allocation\":%d,\"store\":%d,\"load\":%d,\"ctas\":%d,\"peers\":%d,\"bytes_per_peer\":%zu,\"elapsed_ms\":%.6f,\"single_way_ns\":%.3f,\"bytes_per_second\":%.3f,\"pass\":%s}\n",
                mode.c_str(), alloc, store, load, ctas, peers, bytes, elapsed_ms,
                ping ? elapsed_ms * 1e6 / 2000 : 0., ping ? 0. : bytes * double(peers) / (elapsed_ms / 1000),
                pass ? "true" : "false");
    Check(cudaEventDestroy(begin), "event_destroy"); Check(cudaEventDestroy(end), "event_destroy");
    for (int r = 0; r < world; ++r) {
      Check(cudaSetDevice(r), "set_device_free");
      Check(cudaFree(args[r].p[r]), "free"); Check(cudaFree(args[r].errors), "errors_free");
      Check(cudaStreamDestroy(streams[r]), "stream_destroy");
    }
    return pass ? 0 : 2;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
