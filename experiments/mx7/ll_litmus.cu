// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
#include <mcr/mc_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

static void Check(cudaError_t error, char const* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

template <class Word>
__device__ __forceinline__ Word Read(Word const* p, int method) {
  using Global = __attribute__((address_space(1))) volatile Word const;
  switch (method) {
    case 1: return __ldcg(p);
    case 2: return __ldcs(p);
    case 3: return __ldlu(p);
    case 4: return __ldcv(p);
    case 5: return atomicAdd_system(const_cast<Word*>(p), Word(0));
    case 6: return __maca_atomic_load(p, __ATOMIC_RELAXED, memory_scope_system);
    default: return *((Global*)p);
  }
}

template <class Word>
__device__ __forceinline__ void Write(Word* p, Word value, int method) {
  using Global = __attribute__((address_space(1))) volatile Word;
  if (method == 1) atomicExch_system(p, value);
  else *((Global*)p) = value;
}

template <class Word>
__device__ __forceinline__ Word Pack(unsigned epoch, unsigned data) {
  if constexpr (sizeof(Word) == 8) return (Word(epoch) << 32) | data;
  else return (Word(data) << 16) | (epoch & 65535);
}

template <class Word>
__device__ __forceinline__ unsigned Epoch(Word word) {
  if constexpr (sizeof(Word) == 8) return unsigned(word >> 32);
  else return unsigned(word & 65535);
}

template <class Word>
__device__ __forceinline__ unsigned Data(Word word) {
  if constexpr (sizeof(Word) == 8) return unsigned(word);
  else return unsigned(word >> 16);
}

__device__ __forceinline__ unsigned Hash(unsigned epoch, unsigned item, unsigned src, unsigned dst) {
  unsigned v = epoch * 0x9e3779b9u ^ item * 0x85ebca6bu ^ src * 0xc2b2ae35u ^ dst;
  v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu;
  return v ^ (v >> 16);
}

template <class Word>
struct Args {
  Word* recv[4];
  Word* ack[4];
  unsigned long long* errors;
  int rank, world, packets, epochs, store_method, load_method, all_to_all;
};

template <class Word>
__device__ __forceinline__ bool Wait(Word const* p, unsigned e, Args<Word> const& a,
                                    Word& value, bool ack = false) {
  for (unsigned n = 0; n < (1u << 24); ++n) {
    value = Read(p, a.load_method);
    unsigned const observed = Epoch(value);
    if (observed == e) return true;
    if (ack && observed == (sizeof(Word) == 8 ? ~0u : 65535u)) return false;
  }
  atomicAdd(a.errors + 1, 1ull);
  return false;
}

template <class Word>
__global__ void __launch_bounds__(512) Litmus(Args<Word> a) {
  extern __shared__ unsigned failed[];
  // Interleaving gives all 104 CTAs packets even at P=1024; an empty CTA
  // could otherwise overwrite ack=e with ack=e+1 before the writer sees e.
  unsigned const t = threadIdx.x * 104 + blockIdx.x;
  unsigned const threads = 104 * 512;
  for (unsigned e = 1; e <= unsigned(a.epochs); ++e) {
    if (threadIdx.x == 0) failed[0] = 0;
    __syncthreads();
    if (a.all_to_all || a.rank == 0) {
      for (int q = 0; q < a.world; ++q) {
        if (q == a.rank || (!a.all_to_all && q != 1)) continue;
        for (unsigned i = t; i < unsigned(a.packets); i += threads)
          Write(a.recv[q] + std::size_t(a.rank) * a.packets + i,
                Pack<Word>(e, Hash(e, i, a.rank, q)), a.store_method);
      }
    }
    if (a.all_to_all || a.rank == 1) {
      for (int q = 0; q < a.world; ++q) {
        if (q == a.rank || (!a.all_to_all && q != 0)) continue;
        for (unsigned i = t; i < unsigned(a.packets); i += threads) {
          Word got = 0;
          if (!Wait(a.recv[a.rank] + std::size_t(q) * a.packets + i, e, a, got)) {
            atomicExch(failed, 1u);
            break;
          }
          unsigned expected = Hash(e, i, q, a.rank);
          if constexpr (sizeof(Word) == 4) expected &= 65535u;
          if (Data(got) != expected) {
            atomicAdd(a.errors, 1ull);
            atomicExch(failed, 1u);
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        unsigned const ack_epoch = failed[0] ? ~0u : e;
        for (int q = 0; q < a.world; ++q)
          if (q != a.rank && (a.all_to_all || q == 0))
            Write(a.ack[q] + a.rank * 104 + blockIdx.x,
                  Pack<Word>(ack_epoch, 0), a.store_method);
      }
    }
    if ((a.all_to_all || a.rank == 0) && threadIdx.x == 0) {
      for (int q = 0; q < a.world; ++q) {
        if (q == a.rank || (!a.all_to_all && q != 1)) continue;
        Word got = 0;
        if (!Wait(a.ack[a.rank] + q * 104 + blockIdx.x, e, a, got, true)) failed[0] = 1;
      }
    }
    __syncthreads();
    if (failed[0]) return;
  }
}

static void* Allocate(std::size_t bytes, int method) {
  void* p = nullptr;
  if (method < 0) Check(cudaMalloc(&p, bytes), "cudaMalloc");
  else {
    auto error = mcExtMallocWithFlags(&p, bytes, unsigned(method));
    if (error != mcSuccess) throw std::runtime_error("mcExtMallocWithFlags failed, code=" + std::to_string(int(error)));
  }
  Check(cudaMemset(p, 0, bytes), "clear");
  return p;
}

template <class Word>
static int Run(int packets, int epochs, int alloc, int store, int load, bool all) {
  int world = 0;
  Check(cudaGetDeviceCount(&world), "device_count");
  if (world != (all ? 4 : 2)) throw std::runtime_error("litmus visible group has wrong size");
  Args<Word> args[4]{};
  cudaStream_t streams[4]{};
  cudaEvent_t begin[4]{}, end[4]{};
  for (int r = 0; r < world; ++r) {
    Check(cudaSetDevice(r), "set_device");
    args[r].recv[r] = static_cast<Word*>(Allocate(std::size_t(world) * packets * sizeof(Word), alloc));
    args[r].ack[r] = static_cast<Word*>(Allocate(std::size_t(world) * 104 * sizeof(Word), alloc));
    Check(cudaMalloc(&args[r].errors, 16), "errors_malloc");
    Check(cudaMemset(args[r].errors, 0, 16), "errors_clear");
    Check(cudaStreamCreate(&streams[r]), "stream_create");
    Check(cudaEventCreate(&begin[r]), "event_create");
    Check(cudaEventCreate(&end[r]), "event_create");
  }
  for (int r = 0; r < world; ++r) {
    Check(cudaSetDevice(r), "set_device");
    for (int q = 0; q < world; ++q) {
      args[r].recv[q] = args[q].recv[q]; args[r].ack[q] = args[q].ack[q];
      if (q == r) continue;
      auto error = cudaDeviceEnablePeerAccess(q, 0);
      if (error == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      else Check(error, "enable_peer");
    }
    args[r].rank = r; args[r].world = world; args[r].packets = packets; args[r].epochs = epochs;
    args[r].store_method = store; args[r].load_method = load; args[r].all_to_all = all;
    Check(cudaDeviceSynchronize(), "initial_sync");
  }
  for (int r = 0; r < world; ++r) {
    Check(cudaSetDevice(r), "set_device_launch");
    Check(cudaEventRecord(begin[r], streams[r]), "event_record");
    Litmus<<<104, 512, 4, streams[r]>>>(args[r]);
    Check(cudaGetLastError(), "launch");
    Check(cudaEventRecord(end[r], streams[r]), "event_record");
  }
  unsigned long long torn = 0, timeouts = 0;
  std::vector<float> elapsed(world);
  for (int r = 0; r < world; ++r) {
    Check(cudaSetDevice(r), "set_device_sync");
    Check(cudaStreamSynchronize(streams[r]), "stream_sync");
    unsigned long long errors[2]{};
    Check(cudaMemcpy(errors, args[r].errors, 16, cudaMemcpyDeviceToHost), "errors_copy");
    torn += errors[0]; timeouts += errors[1];
    Check(cudaEventElapsedTime(&elapsed[r], begin[r], end[r]), "event_elapsed");
  }
  std::sort(elapsed.begin(), elapsed.end());
  double const ns = double(elapsed[world / 2]) * 1e6 / epochs;
  std::printf("{\"width\":%zu,\"allocation\":%d,\"store\":%d,\"load\":%d,\"packets\":%d,\"epochs\":%d,\"world\":%d,\"all_to_all\":%s,\"torn\":%llu,\"timeouts\":%llu,\"roundtrip_ns\":%.3f,\"pass\":%s}\n",
              sizeof(Word), alloc, store, load, packets, epochs, world, all ? "true" : "false", torn, timeouts, ns,
              torn == 0 && timeouts == 0 ? "true" : "false");
  for (int r = 0; r < world; ++r) {
    Check(cudaSetDevice(r), "set_device_free");
    Check(cudaFree(args[r].recv[r]), "recv_free"); Check(cudaFree(args[r].ack[r]), "ack_free");
    Check(cudaFree(args[r].errors), "errors_free"); Check(cudaStreamDestroy(streams[r]), "stream_destroy");
    Check(cudaEventDestroy(begin[r]), "event_destroy"); Check(cudaEventDestroy(end[r]), "event_destroy");
  }
  return torn || timeouts ? 2 : 0;
}

int main(int argc, char** argv) {
  try {
    if (argc != 8) throw std::runtime_error("usage: ll_litmus <bytes=4|8> <P> <epochs> <alloc=-1|0..7> <store=0|1> <load=0..6> <all=0|1>");
    int const width = std::atoi(argv[1]), packets = std::atoi(argv[2]), epochs = std::atoi(argv[3]);
    int const alloc = std::atoi(argv[4]), store = std::atoi(argv[5]), load = std::atoi(argv[6]), all = std::atoi(argv[7]);
    if ((width != 4 && width != 8) || (packets != 1024 && packets != 262144) || epochs < 2000 ||
        alloc < -1 || alloc > 7 || store < 0 || store > 1 || load < 0 || load > 6 || all < 0 || all > 1)
      throw std::runtime_error("invalid litmus arguments (epochs must be >=2000)");
    return width == 8 ? Run<unsigned long long>(packets, epochs, alloc, store, load, all)
                      : Run<unsigned>(packets, epochs, alloc, store, load, all);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
