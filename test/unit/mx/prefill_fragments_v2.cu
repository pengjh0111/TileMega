// SPDX-License-Identifier: BSD-3-Clause
#if defined(TILEMEGA_PREFILL_FRAGMENTS_CPU_PROOF)
#include <tilemega/Backend/Mx/KvFragmentLayoutV2.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define __device__
struct alignas(16) uint4 { unsigned x, y, z, w; };
namespace tilemega::codegen {
struct ServingAttentionOperands {
  Bf16 const *qkv, *q_norm, *k_norm, *cosine, *sine;
  Bf16 *key_cache, *value_cache;
  int heads_kv, capacity, past, block_extent;
  float epsilon;
};
namespace mx {
inline constexpr int kComputeThreads = 128;
int cpu_thread;
inline int ComputeThread() { return cpu_thread; }
inline void ComputeSync() {}
template <int D, bool Norm, class Store>
void RotateRow(Bf16 const*, Bf16 const*, Bf16 const*, Bf16 const*, float, Store store) {
  int const d = cpu_thread % 32 * 8;
  if (d < D) store(d, uint4{});
}
}  // namespace mx
}  // namespace tilemega::codegen
namespace tilemega::maca {
[[noreturn]] inline void Trap() { throw std::runtime_error("contract trap"); }
}  // namespace tilemega::maca
namespace {
struct Capture {
  tilemega::Bf16 *key, *value;
  std::size_t elements;
  int dim, capacity, qb, chunk, query_blocks, cache_blocks;
  std::vector<unsigned char> key_count, value_count;
  std::vector<int> key_owner, value_owner;
} capture;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
}  // namespace
namespace tilemega::backend {
inline uint4 LoadGlobal16(Bf16 const* p) {
  auto const address = reinterpret_cast<std::uintptr_t>(p);
  for (auto base : {capture.key, capture.value}) {
    auto const begin = reinterpret_cast<std::uintptr_t>(base);
    Check(address < begin || address >= begin + capture.elements * sizeof(Bf16), "cache read/RMW");
  }
  return uint4{};
}
inline void StoreGlobal16(Bf16* p, uint4) {
  auto const address = reinterpret_cast<std::uintptr_t>(p);
  bool const key = address >= reinterpret_cast<std::uintptr_t>(capture.key) &&
                   address < reinterpret_cast<std::uintptr_t>(capture.key + capture.elements);
  auto const base = key ? capture.key : capture.value;
  auto const begin = reinterpret_cast<std::uintptr_t>(base);
  Check(address >= begin && address + 16 <= begin + capture.elements * sizeof(Bf16), "store outside cache");
  Check(address % 16 == 0, "unaligned vector store");
  auto const offset = (address - begin) / sizeof(Bf16);
  int const head = int(offset / (capture.capacity * capture.dim));
  int const local = int(offset % (capture.capacity * capture.dim));
  Check(local + 8 <= 64 * capture.dim, "write beyond token 63");
  int const block = local / (32 * capture.dim), owner = capture.qb * capture.cache_blocks + capture.chunk;
  auto& owners = key ? capture.key_owner : capture.value_owner;
  int& recorded = owners[head * 2 + block];
  Check(recorded == -1 || recorded == owner, "cache block has multiple owning CTAs");
  recorded = owner;
  auto& counts = key ? capture.key_count : capture.value_count;
  for (int i = 0; i < 8; ++i) Check(++counts[head * 64 * capture.dim + local + i] == 1, "duplicate element store");
}
}  // namespace tilemega::backend

// The CPU build supplies an empty AttentionCommon.h shim after defining the
// arithmetic/barrier stubs above. Ownership and store loops use the real helper.
#include <tilemega/Codegen/tasks/mx/PrefillKvFragmentsV2.h>

namespace {
template <int D, int Q, int Rows, int Tile, bool Norm>
void Prove(int batch, int hkv, int extent) {
  using namespace tilemega::codegen;
  using Layout = tilemega::backend::mx::MxKvFragmentLayoutV2<D>;
  struct Storage { alignas(16) tilemega::Bf16 key[Tile * D]{}, value_t[D * Tile]{}; } storage;
  int constexpr capacity = 1088;
  int const heads = batch * hkv, queries = (64 * Q + Rows - 1) / Rows;
  std::size_t const elements = std::size_t(heads) * capacity * D;
  std::vector<tilemega::Bf16> key(elements), value(elements), qkv(std::size_t(heads) * 64 * (Q + 2) * D), trig(64 * D);
  capture = {key.data(), value.data(), elements, D, capacity, 0, 0, queries,
             (capacity + extent - 1) / extent,
             std::vector<unsigned char>(heads * 64 * D), std::vector<unsigned char>(heads * 64 * D),
             std::vector<int>(heads * 2, -1), std::vector<int>(heads * 2, -1)};
  ServingAttentionOperands p{qkv.data(), qkv.data(), qkv.data(), trig.data(), trig.data(),
                            key.data(), value.data(), hkv, capacity, 0, extent, 1e-6f};
  for (int b = 0; b < batch; ++b) for (int g = 0; g < hkv; ++g)
    for (int qb = 0; qb < queries; ++qb)
      for (int chunk = 0; chunk < capture.cache_blocks; ++chunk) {
        if (chunk * extent >= 64 && chunk != capture.cache_blocks - 1) continue;
        capture.qb = qb; capture.chunk = chunk;
        for (mx::cpu_thread = 0; mx::cpu_thread < 128; ++mx::cpu_thread)
          mx::PrefillKvFragmentsV2<D, Q, 64, Rows, Tile, Norm>::Run(p, storage, b, g, qb, chunk);
      }
  for (int head = 0; head < heads; ++head) for (int block = 0; block < 2; ++block) {
    int const owner = (block * 32 * Q / Rows) * capture.cache_blocks + block * 32 / extent;
    Check(capture.key_owner[head * 2 + block] == owner, "wrong K owner");
    Check(capture.value_owner[head * 2 + block] == owner, "wrong V owner");
    for (int token = block * 32; token < (block + 1) * 32; ++token) for (int d = 0; d < D; ++d) {
      auto const k = Layout::K(head, token, d, capacity) - head * capacity * D;
      auto const v = Layout::V(head, token, d, capacity) - head * capacity * D;
      Check(capture.key_count[head * 64 * D + k] == 1, "missing K element");
      Check(capture.value_count[head * 64 * D + v] == 1, "missing V element");
    }
  }
  if (batch == 1 && hkv == 1 && extent == capacity) {
    for (int invalid : {0, 1, 2, 3, 4, 5}) {
      auto bad = p;
      if (invalid == 0) bad.past = 1;
      if (invalid == 1) bad.capacity = 32;
      if (invalid == 2) bad.capacity = 1087;
      if (invalid == 3) bad.block_extent = 33;
      if (invalid == 4) bad.block_extent = 0;
      if (invalid == 5) bad.block_extent = -32;
      bool trapped = false;
      try { mx::PrefillKvFragmentsV2<D, Q, 64, Rows, Tile, Norm>::Run(bad, storage, 0, 0, 0, 0); }
      catch (std::runtime_error const& error) { trapped = std::string(error.what()) == "contract trap"; }
      Check(trapped, "invalid prefill contract did not trap");
    }
  }
}
template <int D, int Q, int Rows, bool Norm>
void Tiles(int b, int h, int extent) { Prove<D, Q, Rows, 32, Norm>(b,h,extent); Prove<D,Q,Rows,64,Norm>(b,h,extent); }
template <int D, int Q, bool Norm>
void Rows(int b, int h, int extent) {
  Tiles<D,Q,16,Norm>(b,h,extent); Tiles<D,Q,32,Norm>(b,h,extent);
  Tiles<D,Q,48,Norm>(b,h,extent); Tiles<D,Q,64,Norm>(b,h,extent);
}
template <int D, bool Norm>
void Queries(int b,int h,int extent) { Rows<D,2,Norm>(b,h,extent); Rows<D,3,Norm>(b,h,extent); Rows<D,4,Norm>(b,h,extent); Rows<D,8,Norm>(b,h,extent); }
}  // namespace
int main() {
  try {
    for (int b : {1,16}) for (int h : {1,4,8}) for (int extent : {32,64,1088}) {
      Queries<64,false>(b,h,extent); Queries<128,false>(b,h,extent); Queries<128,true>(b,h,extent);
    }
    std::puts("PASS CPU actual-helper ownership/layout: 1728 pins, complete aligned unique K/V stores, no cache reads/tail writes; invalid contracts trap");
  } catch (std::exception const& error) { std::fprintf(stderr,"FAIL %s\n",error.what()); return 1; }
}
#else
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <cuda_runtime.h>
static_assert(TILEMEGA_MX_KV_FRAGMENTS_V2 == 0, "fragment writer must stay off by default");
#ifndef MX5_TEST_DIM
#define MX5_TEST_DIM 128
#endif
#ifndef MX5_TEST_NORM
#define MX5_TEST_NORM 0
#endif
using tilemega::Bf16;
using tilemega::codegen::ServingAttentionOperands;
struct PrefillFragmentsArgs {
  Bf16 const* qkv;
  Bf16 *key_cache, *value_cache;
  Bf16 const *cosine, *sine, *q_norm, *k_norm;
  Bf16* context;
  int batch, heads_kv, capacity, past, block_extent;
  float epsilon;
};
template <class Body>
__global__ void __launch_bounds__(128) PrefillFragmentsKernel(PrefillFragmentsArgs a, int query_blocks) {
  extern __shared__ __align__(16) char shared[];
  ServingAttentionOperands p{};
  p.qkv=a.qkv; p.key_cache=a.key_cache; p.value_cache=a.value_cache;
  p.cosine=a.cosine; p.sine=a.sine; p.q_norm=a.q_norm; p.k_norm=a.k_norm; p.context=a.context;
  p.batch=a.batch; p.heads_kv=a.heads_kv; p.capacity=a.capacity; p.past=a.past;
  p.block_extent=a.block_extent; p.epsilon=a.epsilon;
  int const qb=blockIdx.x%query_blocks, g=blockIdx.x/query_blocks%a.heads_kv, b=blockIdx.x/query_blocks/a.heads_kv;
  Body::Run(p,*reinterpret_cast<typename Body::SharedStorage*>(shared),b,g,qb,0);
}
template <int Q, int Rows, int Tile, bool Fragments>
int Launch(PrefillFragmentsArgs const& a,cudaStream_t stream) {
  using Body=tilemega::codegen::mx::PrefillAttentionMma<MX5_TEST_DIM,Q,64,Rows,Tile,MX5_TEST_NORM!=0,true,Fragments>;
  using Old=tilemega::codegen::mx::PrefillAttentionMma<MX5_TEST_DIM,Q,64,Rows,Tile,MX5_TEST_NORM!=0,true,false>;
  static_assert(sizeof(typename Body::SharedStorage)==sizeof(typename Old::SharedStorage));
  int const smem=sizeof(typename Body::SharedStorage), blocks=(64*Q+Rows-1)/Rows;
  if (cudaFuncSetAttribute(PrefillFragmentsKernel<Body>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)!=cudaSuccess) return 1;
  PrefillFragmentsKernel<Body><<<a.batch*a.heads_kv*blocks,128,smem,stream>>>(a,blocks);
  return int(cudaGetLastError());
}
template <int Q,int Rows>
int Tile(PrefillFragmentsArgs const& a,int tile,int fragments,cudaStream_t s) {
  if (tile==32) return fragments ? Launch<Q,Rows,32,true>(a,s) : Launch<Q,Rows,32,false>(a,s);
  if (tile==64) return fragments ? Launch<Q,Rows,64,true>(a,s) : Launch<Q,Rows,64,false>(a,s);
  return 4;
}
template <int Q>
int Rows(PrefillFragmentsArgs const& a,int rows,int tile,int fragments,cudaStream_t s) {
  if (rows==16) return Tile<Q,16>(a,tile,fragments,s);
  if (rows==32) return Tile<Q,32>(a,tile,fragments,s);
  if (rows==48) return Tile<Q,48>(a,tile,fragments,s);
  if (rows==64) return Tile<Q,64>(a,tile,fragments,s);
  return 4;
}
extern "C" int mx5_prefill_fragments_v2_args_size() { return sizeof(PrefillFragmentsArgs); }
extern "C" void mx5_prefill_fragments_v2_coordinates(int* out) { out[0]=MX5_TEST_DIM; out[1]=MX5_TEST_NORM; out[2]=64; out[3]=128; }
extern "C" int mx5_prefill_fragments_v2(PrefillFragmentsArgs const* a,int query,int rows,int tile,int fragments,void* stream) {
  if (!a || a->batch<1 || a->batch>16 || a->heads_kv<1 || a->past!=0 || a->capacity<64 || a->capacity%32 ||
      a->block_extent<64 || a->block_extent%32 || (fragments!=0 && fragments!=1)) return 4;
  auto s=static_cast<cudaStream_t>(stream);
  if (query==2) return Rows<2>(*a,rows,tile,fragments,s);
  if (query==3) return Rows<3>(*a,rows,tile,fragments,s);
  if (query==4) return Rows<4>(*a,rows,tile,fragments,s);
  if (query==8) return Rows<8>(*a,rows,tile,fragments,s);
  return 4;
}
#endif
