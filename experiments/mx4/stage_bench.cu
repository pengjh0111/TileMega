// SPDX-License-Identifier: BSD-3-Clause
// MX-4 measured stage table (MX4 prompt §4.2).
//
// A resident kernel of the program's shape (G = 104 CTAs x 512 threads,
// dynamic shared memory max(need, 32 KiB + 16 B)) runs one stage R times,
// each followed by the program's Barrier with the next stage's prefetch
// (Codegen/tasks/mx/DecodeBarrier.cuh); CTA 0 stamps the global timer after
// every barrier, so stamp[i] - stamp[i-1] is iteration i's envelope: stage
// i's work, its barrier and the prefetch under it. Weights rotate through
// copies of >= 256 MiB together (the KV cache likewise for attention), hash
// filled, so the stage streams from DRAM as in the program. Each key runs
// once to warm up and once timed; the row gives the median and p90 envelope.
//
// The candidate domain is §4.1's, for the four MX-4 models x B in
// {1,2,4,8,16}: every GEMM class (qkv, o, gate_up, down, head) x TN x D x
// Part, attention (D, Q, B, L in {65,256,512,768,1088}, Tmin in
// {32,64,128}) and Embed. N, K, B are runtime values here, constexpr in the
// program; the body is the same code.
//
// Build it as the program is built (tools/commands/mx_decode.cpp):
//   cucc -std=c++17 -O3 -Xcompiler -resource-usage -fno-strict-aliasing
//        -DTILEMEGA_SYNC_V3=1 -DTILEMEGA_WATCHDOG=0
//
// Usage: stage_bench <iterations> <out.csv> [filter-substring]
#include <tilemega/Backend/Mx/DecodeAttention.h>
#include <tilemega/Backend/Mx/DecodeGemm.h>
#include <tilemega/Codegen/tasks/mx/DecodeBarrier.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace tilemega;
using namespace tilemega::backend::mx;
using tilemega::codegen::mx::Barrier;
using tilemega::codegen::mx::NoPrefetch;

namespace {

constexpr int kGrid = 104;
constexpr int kProgramSmem = 65536 / 2 + 16;
constexpr std::size_t kRotateBytes = 256ull << 20;
constexpr int kCapacity = 1088;  // ISL 64 + OSL 1024

struct Model {
  char const* name;
  int hidden, heads, hkv, dim, inter, vocab;
  int rope;  // 0: D128 Q2 q/k-norm, 1: D64 Q4, 2: D128 Q4, 3: D128 Q3
};
Model const kModels[] = {
    {"qwen3_0_6b", 1024, 16, 8, 128, 3072, 151936, 0},
    {"qwen3_1_7b", 2048, 16, 8, 128, 6144, 151936, 0},
    {"llama3_2_1b", 2048, 32, 8, 64, 8192, 128256, 1},
    {"llama3_1_8b", 4096, 32, 8, 128, 14336, 128256, 2},
    // Not one of the four: benched alone (filter llama3_2_3b) and appended.
    {"llama3_2_3b", 3072, 24, 8, 128, 8192, 128256, 3},
};

struct Pools {
  Bf16* weights = nullptr;
  std::size_t weight_elems = 0;
  Bf16 *x, *out, *qkv, *k_cache, *vt_cache, *cosine, *sine, *q_norm, *k_norm, *embed;
  float *partial, *head_value, *attn_partial;
  int *head_index, *tokens;
  unsigned *tickets, *head_ticket;
  unsigned long long *arrivals, *stamps;
};

__global__ void Fill(unsigned* p, std::size_t n, unsigned seed) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    unsigned x = unsigned(i) * 2654435761u + seed;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    // Finite BF16 pairs in about [-2, 2): keep sign, clamp the exponent.
    unsigned lo = (x & 0x80ffu) | 0x3e00u, hi = ((x >> 16) & 0x80ffu) | 0x3f00u;
    p[i] = lo | (hi << 16);
  }
}

struct GemmLaunch {
  MxGemmOperands p;
  int n, k, b, iterations, copies;
  std::size_t copy_elems;
  unsigned long long* arrivals;
  unsigned long long* stamps;
};

template <class Body>
__global__ void __launch_bounds__(512) GemmStage(GemmLaunch g) {
  extern __shared__ __align__(16) char smem[];
  MxShape const s{g.n, g.k, g.b};
  auto at = [&](int i) {
    MxGemmOperands q = g.p;
    q.w = g.p.w + std::size_t(i % g.copies) * g.copy_elems;
    return q;
  };
  auto pf = Body::Prefetch(at(0), s);
#pragma unroll 1
  for (int i = 0; i < g.iterations; ++i) {
    Body::Run(pf, at(i), s, smem);
    pf = Barrier(g.arrivals + (i % 64) * 8, (unsigned long long)gridDim.x * (i / 64 + 1),
                 [&] { return Body::Prefetch(at(i + 1), s); });
    if (blockIdx.x == 0 && threadIdx.x == 0) g.stamps[i] = maca::GlobalTimerNs();
  }
}

struct AttnLaunch {
  MxAttentionOperands p;
  int b, hkv, iterations, copies;
  std::size_t copy_elems;
  unsigned long long* arrivals;
  unsigned long long* stamps;
};

template <class Body>
__global__ void __launch_bounds__(512) AttnStage(AttnLaunch g) {
  extern __shared__ __align__(16) char smem[];
  MxAttentionShape const s{g.b, g.hkv};
  auto at = [&](int i) {
    MxAttentionOperands q = g.p;
    q.k_cache = g.p.k_cache + std::size_t(i % g.copies) * g.copy_elems;
    q.vt_cache = g.p.vt_cache + std::size_t(i % g.copies) * g.copy_elems;
    return q;
  };
  auto pf = Body::Prefetch(at(0), s);
#pragma unroll 1
  for (int i = 0; i < g.iterations; ++i) {
    Body::Run(pf, at(i), s, smem);
    pf = Barrier(g.arrivals + (i % 64) * 8, (unsigned long long)gridDim.x * (i / 64 + 1),
                 [&] { return Body::Prefetch(at(i + 1), s); });
    if (blockIdx.x == 0 && threadIdx.x == 0) g.stamps[i] = maca::GlobalTimerNs();
  }
}

__global__ void __launch_bounds__(512) EmbedStage(Bf16 const* embed, int const* tokens, Bf16* x, int b, int hidden,
                                                  int iterations, unsigned long long* arrivals,
                                                  unsigned long long* stamps) {
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    MxDecodeEmbed(embed, tokens, x, MxShape{0, 0, b}, hidden, kCapacity, 64 + i % 512);
    Barrier(arrivals + (i % 64) * 8, (unsigned long long)gridDim.x * (i / 64 + 1), [] { return NoPrefetch{}; });
    if (blockIdx.x == 0 && threadIdx.x == 0) stamps[i] = maca::GlobalTimerNs();
  }
}

struct Stats {
  double median, p90;
};

Stats Envelope(Pools const& pools, int iterations) {
  std::vector<unsigned long long> stamps(iterations);
  cudaMemcpy(stamps.data(), pools.stamps, iterations * 8, cudaMemcpyDeviceToHost);
  std::vector<double> d;
  for (int i = 1; i < iterations; ++i) d.push_back(double(stamps[i] - stamps[i - 1]));
  std::sort(d.begin(), d.end());
  return {d[d.size() / 2], d[std::min(d.size() - 1, d.size() * 9 / 10)]};
}

int Check(char const* what) {
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "ERROR %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
  return 0;
}

template <class Kernel, class Launch>
Stats Time(Kernel kernel, Launch const& launch, int smem, Pools const& pools, int iterations) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 512, smem);
  if (blocks != 1) {
    std::fprintf(stderr, "ERROR occupancy %d\n", blocks);
    std::exit(3);
  }
  for (int run = 0; run < 2; ++run) {  // warm-up, then timed
    cudaMemset(pools.arrivals, 0, 64 * 64);
    kernel<<<kGrid, 512, smem>>>(launch);
    Check("stage");
  }
  return Envelope(pools, iterations);
}

// I1 uses one binary/kernel for both orders, keeping register allocation and
// code placement identical. Each body prefetches at entry in both modes;
// these numbers diagnose instruction order, not the stage table's envelope.
template <int C>
__device__ __forceinline__ void InstructionBody(Pools const& p, int ordinal, char* smem) {
  constexpr int n = C == 0 ? 4096 : C == 2 ? 1024 : C == 3 ? 6144 : 1024;
  constexpr int k = C == 0 ? 1024 : C == 2 ? 2048 : C == 3 ? 1024 : 3072;
  constexpr std::size_t elems = std::size_t(n) * k;
  constexpr int copies = int((kRotateBytes + elems * 2 - 1) / (elems * 2));
  constexpr std::size_t kv_elems = std::size_t(8) * kCapacity * 128;
  constexpr int kv_copies = int((kRotateBytes + kv_elems * 4 - 1) / (kv_elems * 4));
  auto const* weights = p.weights + std::size_t(ordinal % copies) * elems;
  auto* keys = p.k_cache + std::size_t(ordinal % kv_copies) * kv_elems;
  auto* values = p.vt_cache + std::size_t(ordinal % kv_copies) * kv_elems;
  if constexpr (C == 1) {
    using Body = MxDecodeAttention<128, 2, 128, 4>;
    MxAttentionOperands const operands{p.qkv, keys, values, p.out, p.attn_partial, p.tickets, kCapacity, 576};
    MxStaticAttentionShape<1, 8> const shape{};
    auto const pf = Body::Prefetch(operands, shape);
    Body::Run(pf, operands, shape, smem);
  } else {
    using Body = std::conditional_t<C == 0,
        MxDecodeGemm<128, 1, MxDecodeOp::kQkvRope, true, MxPart::kTileRange, MxRope<128, 2, 8, true>>,
        std::conditional_t<C == 2,
            MxDecodeGemm<16, 2, MxDecodeOp::kResidual, false, MxPart::kTileRange>,
            std::conditional_t<C == 3,
                MxDecodeGemm<64, 2, MxDecodeOp::kSwiGLU, true, MxPart::kTileRange>,
                MxDecodeGemm<16, 3, MxDecodeOp::kResidual, false, MxPart::kTileRange>>>>;
    auto const* a = C == 2 ? p.out : C == 4 ? p.qkv : p.x;
    auto* out = C == 0 || C == 3 ? p.qkv : p.x;
    MxGemmOperands const operands{a, weights, out, p.partial, p.tickets, p.q_norm, p.k_norm,
        p.cosine, p.sine, keys, values, p.head_value, p.head_index, p.head_ticket, p.tokens,
        kCapacity, 576, 1e-6f};
    MxStaticShape<n, k, 1> const shape{};
    auto const pf = Body::Prefetch(operands, shape);
    Body::Run(pf, operands, shape, smem);
  }
}

__global__ void __launch_bounds__(512) InstructionOrder(Pools p, int iterations, int fixed_stage) {
  extern __shared__ __align__(16) char smem[];
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    int const stage = __builtin_mxc_readfirstlane(fixed_stage < 0 ? i % 5 : fixed_stage);
    int const ordinal = __builtin_mxc_readfirstlane(fixed_stage < 0 ? i / 5 : i);
    switch (stage) {
      case 0: InstructionBody<0>(p, ordinal, smem); break;
      case 1: InstructionBody<1>(p, ordinal, smem); break;
      case 2: InstructionBody<2>(p, ordinal, smem); break;
      case 3: InstructionBody<3>(p, ordinal, smem); break;
      case 4: InstructionBody<4>(p, ordinal, smem); break;
    }
    Barrier(p.arrivals + (i % 64) * 8, (unsigned long long)gridDim.x * (i / 64 + 1),
            [] { return NoPrefetch{}; });
    if (blockIdx.x == 0 && threadIdx.x == 0) p.stamps[i] = maca::GlobalTimerNs();
  }
}

template <int C>
__global__ void __launch_bounds__(512) InstructionSize(Pools p, int ordinal) {
  extern __shared__ __align__(16) char smem[];
  InstructionBody<C>(p, ordinal, smem);
}

void RunInstructionOrder(Pools const& p, int iterations, FILE* output) {
  int resident = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, InstructionOrder, 512, kProgramSmem);
  if (resident != 1) { std::fprintf(stderr, "I1 resident=%d\n", resident); std::exit(3); }
  char const* names[] = {"qkv", "attn", "o", "gate_up", "down"};
  using SizeKernel = void (*)(Pools, int);
  SizeKernel const sizes[] = {InstructionSize<0>, InstructionSize<1>, InstructionSize<2>,
                              InstructionSize<3>, InstructionSize<4>};
  for (int c = 0; c < 5; ++c) {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, sizes[c]) != cudaSuccess) std::exit(4);
    std::printf("I1_BODY class=%s regs=%d static_smem=%zu\n", names[c], attributes.numRegs,
                 attributes.sharedSizeBytes);
  }
  for (int fixed : {0, 1, 2, 3, 4, -1}) {
    int const count = fixed < 0 ? iterations * 5 : iterations;
    for (int run = 0; run < 2; ++run) {
      cudaMemset(p.arrivals, 0, 64 * 64);
      cudaMemset(p.tickets, 0, 1 << 20);
      InstructionOrder<<<kGrid, 512, kProgramSmem>>>(p, count, fixed);
      Check("I1");
    }
    std::vector<unsigned long long> stamps(count);
    cudaMemcpy(stamps.data(), p.stamps, count * 8, cudaMemcpyDeviceToHost);
    for (int c = 0; c < 5; ++c) {
      if (fixed >= 0 && c != fixed) continue;
      std::vector<double> durations;
      for (int i = 10; i < count; ++i)
        if (fixed >= 0 || i % 5 == c) durations.push_back(double(stamps[i] - stamps[i - 1]));
      std::sort(durations.begin(), durations.end());
      std::fprintf(output, "I1,qwen3_0_6b,1,%s,%s,%.0f,%.0f,%d\n", names[c],
                   fixed < 0 ? "program_order" : "repeat", durations[durations.size() / 2],
                   durations[durations.size() * 9 / 10], iterations);
      std::fflush(output);
    }
  }
}

#ifndef TILEMEGA_MX5_I1_ONLY
// ---- GEMM dispatch -----------------------------------------------------------
template <MxDecodeOp Op, bool Norm, MxPart Part, int TN, int D, class Rope>
bool Gemm(GemmLaunch const& g, Pools const& pools, Stats* stats) {
  constexpr bool kLegal = MxDepthFits(TN, D) && (Op != MxDecodeOp::kSwiGLU || TN >= 32) &&
                          (Op != MxDecodeOp::kArgmax || Part == MxPart::kTileRange) &&
                          (Op != MxDecodeOp::kQkvRope ||
                           ((TN == Rope::kHeadDim || TN == 2 * Rope::kHeadDim) && TN <= 128 &&
                            ((Rope::kQ + 2) * Rope::kHeadDim) % TN == 0));
  if constexpr (!kLegal) {
    return false;
  } else {
    using Body = MxDecodeGemm<TN, D, Op, Norm, Part, Rope>;
    *stats = Time(GemmStage<Body>, g, std::max(MxGemmSmem(g.b, TN).bytes, kProgramSmem), pools, g.iterations);
    return true;
  }
}

template <MxDecodeOp Op, bool Norm, MxPart Part, class Rope>
bool GemmTD(int tn, int d, GemmLaunch const& g, Pools const& pools, Stats* stats) {
#define TM_CASE(TN, D) \
  if (tn == TN && d == D) return Gemm<Op, Norm, Part, TN, D, Rope>(g, pools, stats);
#define TM_TN(TN) TM_CASE(TN, 1) TM_CASE(TN, 2) TM_CASE(TN, 3) TM_CASE(TN, 4)
  if constexpr (Op == MxDecodeOp::kQkvRope) {
    if constexpr (Rope::kHeadDim == 64) {
      TM_TN(64) TM_TN(128)
    } else {
      TM_TN(128)
    }
  } else if constexpr (Op == MxDecodeOp::kSwiGLU) {
    TM_TN(32) TM_TN(64) TM_TN(128)
  } else {
    TM_TN(16) TM_TN(32) TM_TN(64) TM_TN(128)
  }
#undef TM_TN
#undef TM_CASE
  return false;
}

/// The program's classes: qkv (Qkv, Norm), o and down (Residual), gate_up
/// (SwiGLU, Norm), head (Argmax, Norm, kTileRange).
enum Class { kQkv, kO, kGateUp, kDown, kHead };
char const* const kClassNames[] = {"qkv", "o", "gate_up", "down", "head"};

bool RunGemm(Class c, int rope, int tn, int d, int part, GemmLaunch const& g, Pools const& pools, Stats* stats) {
  auto parts = [&](auto op, auto norm, auto rope_type) {
    constexpr MxDecodeOp kOp = decltype(op)::value;
    constexpr bool kNorm = decltype(norm)::value;
    using R = decltype(rope_type);
    if (part == 0) return GemmTD<kOp, kNorm, MxPart::kStreamK, R>(tn, d, g, pools, stats);
    return GemmTD<kOp, kNorm, MxPart::kTileRange, R>(tn, d, g, pools, stats);
  };
  using OpC = std::integral_constant<MxDecodeOp, MxDecodeOp::kResidual>;
  using T = std::true_type;
  using F = std::false_type;
  switch (c) {
    case kO:
    case kDown: return parts(OpC{}, F{}, MxNoRope{});
    case kGateUp: return parts(std::integral_constant<MxDecodeOp, MxDecodeOp::kSwiGLU>{}, T{}, MxNoRope{});
    case kHead:
      return part == 1 &&
             GemmTD<MxDecodeOp::kArgmax, true, MxPart::kTileRange, MxNoRope>(tn, d, g, pools, stats);
    case kQkv: {
      auto qkv = std::integral_constant<MxDecodeOp, MxDecodeOp::kQkvRope>{};
      if (rope == 0) return parts(qkv, T{}, MxRope<128, 2, 8, true>{});
      if (rope == 1) return parts(qkv, T{}, MxRope<64, 4, 8, false>{});
      if (rope == 3) return parts(qkv, T{}, MxRope<128, 3, 8, false>{});
      return parts(qkv, T{}, MxRope<128, 4, 8, false>{});
    }
  }
  return false;
}

// ---- attention dispatch --------------------------------------------------------
template <int D, int Q, int Tmin>
Stats Attn(AttnLaunch const& g, Pools const& pools) {
  using Body = MxDecodeAttention<D, Q, Tmin>;
  return Time(AttnStage<Body>, g, std::max(Body::kSharedBytes, kProgramSmem), pools, g.iterations);
}
Stats RunAttn(int dim, int q, int tmin, AttnLaunch const& g, Pools const& pools) {
#define TM_CASE(D, Q, T) \
  if (dim == D && q == Q && tmin == T) return Attn<D, Q, T>(g, pools);
#define TM_DQ(D, Q) TM_CASE(D, Q, 32) TM_CASE(D, Q, 64) TM_CASE(D, Q, 128)
  TM_DQ(128, 2) TM_DQ(64, 4) TM_DQ(128, 4) TM_DQ(128, 3)
#undef TM_DQ
#undef TM_CASE
  std::fprintf(stderr, "ERROR no attention body %d %d %d\n", dim, q, tmin);
  std::exit(2);
}
#endif

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: stage_bench <iterations> <out.csv> [filter]\n");
    return 2;
  }
  int const iterations = std::atoi(argv[1]);
  std::string const filter = argc > 3 ? argv[3] : "";
  bool const instruction_order = argc > 4 && std::string(argv[4]) == "program-order";
  FILE* out = std::fopen(argv[2], "w");
  if (instruction_order)
    std::fprintf(out, "probe,model,batch,class,mode,median_ns,p90_ns,iterations\n");
  else
    std::fprintf(out, "kind,model,class,n,k,b,op,norm,tn,d,part,dim,q,hkv,length,tmin,median_ns,p90_ns,iterations\n");

  // Pools sized for the largest key (Llama-3.1-8B head, two copies).
  Pools pools{};
  std::size_t max_weight = 0;
  for (auto const& m : kModels) {
    int const n_qkv = (m.heads + 2 * m.hkv) * m.dim;
    for (std::size_t e : {std::size_t(n_qkv) * m.hidden, std::size_t(m.hidden) * m.heads * m.dim,
                          std::size_t(2 * m.inter) * m.hidden, std::size_t(m.hidden) * m.inter,
                          std::size_t(m.vocab) * m.hidden})
      max_weight = std::max(max_weight, e);
  }
  std::size_t const pool_bytes = std::max(2 * max_weight * 2, kRotateBytes + max_weight * 2);
  cudaMalloc(&pools.weights, pool_bytes);
  pools.weight_elems = pool_bytes / 2;
  Fill<<<2048, 256>>>(reinterpret_cast<unsigned*>(pools.weights), pool_bytes / 4, 0x9e3779b9u);
  std::size_t const act = 16 * 32768;
  cudaMalloc(&pools.x, act * 2);
  cudaMalloc(&pools.out, act * 2);
  cudaMalloc(&pools.qkv, act * 2);
  Fill<<<64, 256>>>(reinterpret_cast<unsigned*>(pools.x), act / 2, 7);
  Fill<<<64, 256>>>(reinterpret_cast<unsigned*>(pools.out), act / 2, 11);
  Fill<<<64, 256>>>(reinterpret_cast<unsigned*>(pools.qkv), act / 2, 13);
  // KV for attention: copies of B=16 x hkv=8 x cap x D=128 >= 256 MiB.
  std::size_t const kv_elems = std::size_t(16) * 8 * kCapacity * 128;
  int const kv_copies = int((kRotateBytes + kv_elems * 2 - 1) / (kv_elems * 2)) + 1;
  cudaMalloc(&pools.k_cache, kv_elems * 2 * kv_copies);
  cudaMalloc(&pools.vt_cache, kv_elems * 2 * kv_copies);
  Fill<<<2048, 256>>>(reinterpret_cast<unsigned*>(pools.k_cache), kv_elems * kv_copies / 2, 17);
  Fill<<<2048, 256>>>(reinterpret_cast<unsigned*>(pools.vt_cache), kv_elems * kv_copies / 2, 19);
  cudaMalloc(&pools.cosine, kCapacity * 128 * 2);
  cudaMalloc(&pools.sine, kCapacity * 128 * 2);
  cudaMalloc(&pools.q_norm, 128 * 2);
  cudaMalloc(&pools.k_norm, 128 * 2);
  Fill<<<64, 256>>>(reinterpret_cast<unsigned*>(pools.cosine), kCapacity * 64, 23);
  Fill<<<64, 256>>>(reinterpret_cast<unsigned*>(pools.sine), kCapacity * 64, 29);
  Fill<<<1, 64>>>(reinterpret_cast<unsigned*>(pools.q_norm), 64, 31);
  Fill<<<1, 64>>>(reinterpret_cast<unsigned*>(pools.k_norm), 64, 37);
  std::size_t const partial_floats = std::size_t(64) << 20;
  cudaMalloc(&pools.partial, partial_floats * 4);
  cudaMalloc(&pools.attn_partial, partial_floats * 4);
  cudaMalloc(&pools.tickets, 1 << 20);
  cudaMemset(pools.tickets, 0, 1 << 20);
  cudaMalloc(&pools.head_value, 16 * kGrid * 4);
  cudaMalloc(&pools.head_index, 16 * kGrid * 4);
  cudaMalloc(&pools.head_ticket, 64);
  cudaMemset(pools.head_ticket, 0, 64);
  cudaMalloc(&pools.tokens, 16 * kCapacity * 4);
  cudaMemset(pools.tokens, 0, 16 * kCapacity * 4);
  cudaMalloc(&pools.embed, 4096 * 2);  // tokens are 0: only row 0 is read
  cudaMalloc(&pools.arrivals, 64 * 64);
  cudaMalloc(&pools.stamps, std::size_t(iterations) * (instruction_order ? 5 : 1) * 8 + 64);
  Check("setup");

  if (instruction_order) {
    RunInstructionOrder(pools, iterations, out);
    std::fclose(out);
    return 0;
  }
#ifdef TILEMEGA_MX5_I1_ONLY
  std::fprintf(stderr, "this binary requires: <iterations> <out.csv> qwen3_0_6b program-order\n");
  return 2;
#else
  int rows = 0;
  for (auto const& m : kModels) {
    int const q = m.heads / m.hkv;
    int const n_qkv = (q + 2) * m.hkv * m.dim;
    struct Shape { Class c; int n, k; };
    Shape const shapes[] = {{kQkv, n_qkv, m.hidden},
                            {kO, m.hidden, m.heads * m.dim},
                            {kGateUp, 2 * m.inter, m.hidden},
                            {kDown, m.hidden, m.inter},
                            {kHead, m.vocab, m.hidden}};
    for (int b : {1, 2, 4, 8, 16}) {
      for (auto const& sh : shapes) {
        std::size_t const elems = std::size_t(sh.n) * sh.k;
        int const copies = int(std::min<std::size_t>(pools.weight_elems / elems,
                                                     std::max<std::size_t>(2, (kRotateBytes / 2 + elems - 1) / elems)));
        for (int tn : {16, 32, 64, 128})
          for (int d = 1; d <= 4; ++d)
            for (int part = 0; part < 2; ++part) {
              if (sh.n % tn || !MxDepthFits(tn, d) || !MxReduceRounds(b, tn)) continue;
              if (part == 0 && (long long)(sh.n / tn) * (sh.k / 32) < kGrid) continue;  // stream-K needs U >= G
              char key[256];
              std::snprintf(key, sizeof key, "%s,%s,b%d,tn%d,d%d,p%d", m.name, kClassNames[sh.c], b, tn, d, part);
              if (!filter.empty() && std::string(key).find(filter) == std::string::npos) continue;
              GemmLaunch g{};
              g.p = MxGemmOperands{pools.x,       pools.weights,   sh.c == kQkv ? pools.qkv : pools.out,
                                   pools.partial, pools.tickets,   pools.q_norm, pools.k_norm,
                                   pools.cosine,  pools.sine,      pools.k_cache, pools.vt_cache,
                                   pools.head_value, pools.head_index, pools.head_ticket, pools.tokens,
                                   kCapacity, 576, 1e-6f};
              g.n = sh.n;
              g.k = sh.k;
              g.b = b;
              g.iterations = iterations;
              g.copies = copies;
              g.copy_elems = elems;
              g.arrivals = pools.arrivals;
              g.stamps = pools.stamps;
              Stats st;
              if (!RunGemm(sh.c, m.rope, tn, d, part, g, pools, &st)) continue;
              static char const* const ops[] = {"kQkvRope", "kResidual", "kSwiGLU", "kResidual", "kArgmax"};
              std::fprintf(out, "gemm,%s,%s,%d,%d,%d,%s,%d,%d,%d,%s,,,,,,%.0f,%.0f,%d\n", m.name,
                           kClassNames[sh.c], sh.n, sh.k, b, ops[sh.c], sh.c == kQkv || sh.c == kGateUp || sh.c == kHead,
                           tn, d, part ? "kTileRange" : "kStreamK", st.median, st.p90, iterations);
              std::fflush(out);
              ++rows;
            }
      }
      // Attention: L in {65, 256, 512, 768, 1088}, Tmin in {32, 64, 128}.
      bool const shared_config = m.rope == 0 && std::strcmp(m.name, "qwen3_0_6b") != 0;  // 1.7B = 0.6B's body
      std::vector<int> lengths{65, 256, 512, 768, 1088};
      if (char const* env = std::getenv("STAGE_BENCH_LENGTHS")) {  // diagnostics: other L
        lengths.clear();
        for (char const* p = env; *p;) {
          lengths.push_back(std::atoi(p));
          while (*p && *p != ',') ++p;
          if (*p) ++p;
        }
      }
      for (int length : lengths)
        for (int tmin : {32, 64, 128}) {
          if (shared_config) continue;
          char key[256];
          std::snprintf(key, sizeof key, "%s,attention,b%d,l%d,t%d", m.name, b, length, tmin);
          if (!filter.empty() && std::string(key).find(filter) == std::string::npos) continue;
          AttnLaunch g{};
          std::size_t const elems = std::size_t(b) * m.hkv * kCapacity * m.dim;
          g.p = MxAttentionOperands{pools.qkv, pools.k_cache, pools.vt_cache, pools.out, pools.attn_partial,
                                    pools.tickets, kCapacity, length - 1};
          g.b = b;
          g.hkv = m.hkv;
          g.iterations = iterations;
          g.copies = int(std::max<std::size_t>(2, (kRotateBytes + 2 * elems * 2 - 1) / (2 * elems * 2)));
          g.copies = std::min(g.copies, int(kv_elems * kv_copies / elems));
          g.copy_elems = elems;
          g.arrivals = pools.arrivals;
          g.stamps = pools.stamps;
          Stats const st = RunAttn(m.dim, q, tmin, g, pools);
          std::fprintf(out, "attention,%s,attention,,,%d,,,,,,%d,%d,%d,%d,%d,%.0f,%.0f,%d\n", m.name, b, m.dim, q,
                       m.hkv, length, tmin, st.median, st.p90, iterations);
          std::fflush(out);
          ++rows;
        }
      char key[256];
      std::snprintf(key, sizeof key, "%s,embed,b%d", m.name, b);
      if (filter.empty() || std::string(key).find(filter) != std::string::npos) {
        cudaMemset(pools.arrivals, 0, 64 * 64);
        for (int run = 0; run < 2; ++run) {
          cudaMemset(pools.arrivals, 0, 64 * 64);
          EmbedStage<<<kGrid, 512, kProgramSmem>>>(pools.embed, pools.tokens, pools.out, b, m.hidden, iterations,
                                                   pools.arrivals, pools.stamps);
          Check("embed");
        }
        Stats const st = Envelope(pools, iterations);
        std::fprintf(out, "embed,%s,embed,%d,,%d,,,,,,,,,,,%.0f,%.0f,%d\n", m.name, m.hidden, b, st.median, st.p90,
                     iterations);
        std::fflush(out);
        ++rows;
      }
    }
  }
  std::fclose(out);
  std::printf("rows %d\n", rows);
  return 0;
#endif
}
