// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <cuda_runtime.h>

#ifndef MX5_TEST_TN
#define MX5_TEST_TN 16
#endif
#ifndef MX5_TEST_DEPTH
#define MX5_TEST_DEPTH 1
#endif
#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif

using namespace tilemega;
using namespace tilemega::backend::mx;

struct Mx5GemmArgs {
  Bf16 const *a, *w, *residual;
  Bf16* out;
  float* partial;
  unsigned *tickets, *next_calls;
  int n, k, b;
  float epsilon;
};

template <class Body, MxPart Part>
__global__ void __launch_bounds__(MX5_TEST_THREADS) Mx5GemmKernel(Mx5GemmArgs args) {
  extern __shared__ __align__(16) char smem[];
  MxRuntimeShapeV2<MX5_TEST_THREADS> const shape{args.n, args.k, args.b};
  Bf16 const* weight = args.w;
  if constexpr (Part == MxPart::kTileRange) {
    // Independent test geometry: an initially idle group never owns a later tile.
    int const wave = int(threadIdx.x) / 64, wk = 8 / Body::kWN;
    int const tiles = args.n / Body::kTN;
    int const first = int(blockIdx.x) * tiles / 104;
    int const last = (int(blockIdx.x) + 1) * tiles / 104;
    bool const consumer = wave < 8 && first + wave / wk < last && args.k / 32 > wave % wk;
    if (!consumer) weight = nullptr;
  }
  MxGemmOperandsV2 const operands{args.a, weight, args.residual, args.out, args.partial,
                                  args.tickets, nullptr, nullptr, args.k, args.n, args.epsilon};
  MxGemmOperandsV2 prefetch_operands = operands;
  if constexpr (Part == MxPart::kStreamK) {
    // The Run pointer remains real: a later segment may have this wave's chunks.
    int const wave = int(threadIdx.x) / 64, wk = 8 / Body::kWN;
    int const kc = args.k / 32, units = (args.n / Body::kTN) * kc;
    int const begin = int(blockIdx.x) * units / 104;
    int const end = (int(blockIdx.x) + 1) * units / 104;
    int const first = (begin / kc) * kc;
    int const k0 = begin - first, k1 = min(end - first, kc);
    if (!(wave < 8 && begin < end && k0 + wave % wk < k1)) prefetch_operands.w = nullptr;
  }
  auto const pf = Body::Prefetch(prefetch_operands, shape);
  Body::Run(pf, operands, shape, smem, [&] {
    if (threadIdx.x % 64 == 0) atomicAdd(args.next_calls + blockIdx.x * 8 + threadIdx.x / 64, 1u);
    return MxNoPrefetchedV2{};
  });
}

template <int WN, MxOpV2 Op, bool Norm, MxPart Part, int Pw>
int Mx5Launch(Mx5GemmArgs const& a, void* stream) {
  constexpr int tn = MX5_TEST_TN, depth = MX5_TEST_DEPTH;
  if constexpr (!MxDepthFits(tn, depth) || (Op == MxOpV2::kSwiGLU && tn < 32) ||
                (Part == MxPart::kStreamK && WN != 1)) return 4;
  else {
    if (!MxGemmLegalV2(a.n, a.k, a.b, tn, WN, depth, Op, Part)) return 4;
    using Body = MxGemm<tn, WN, depth, Op, Norm, Part, Pw>;
    int const smem = max(32784, MxGemmSmemV2(a.b, tn, WN).bytes);
    if (smem > 65536) return 4;
    if (cudaFuncSetAttribute(Mx5GemmKernel<Body, Part>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return 5;
    Mx5GemmKernel<Body, Part><<<104, MX5_TEST_THREADS, smem, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaGetLastError() == cudaSuccess ? 0 : 6;
  }
}

template <int WN, MxOpV2 Op, bool Norm, MxPart Part>
int Mx5Width(Mx5GemmArgs const& a, int pw, void* stream) {
  if (pw == MX5_TEST_DEPTH) return Mx5Launch<WN, Op, Norm, Part, MX5_TEST_DEPTH>(a, stream);
  // The width examples exercise the derived specialization without adding
  // a search coordinate or instantiating every width for every body tuple.
  if constexpr (MX5_TEST_TN == 16 && MX5_TEST_DEPTH == 1 && WN == 1) {
    if (pw == 0) return Mx5Launch<WN, Op, Norm, Part, 0>(a, stream);
    if (pw == 8) return Mx5Launch<WN, Op, Norm, Part, 8>(a, stream);
    if (pw == 16) return Mx5Launch<WN, Op, Norm, Part, 16>(a, stream);
  }
  return 4;
}

template <int WN, MxOpV2 Op>
int Mx5NP(Mx5GemmArgs const& a, int norm, int part, int pw, void* stream) {
  if (norm) {
    if (part) return Mx5Width<WN, Op, true, MxPart::kTileRange>(a, pw, stream);
    return Mx5Width<WN, Op, true, MxPart::kStreamK>(a, pw, stream);
  }
  if (part) return Mx5Width<WN, Op, false, MxPart::kTileRange>(a, pw, stream);
  return Mx5Width<WN, Op, false, MxPart::kStreamK>(a, pw, stream);
}

template <int WN>
int Mx5Op(Mx5GemmArgs const& a, int op, int norm, int part, int pw, void* stream) {
  if (op == 0) return Mx5NP<WN, MxOpV2::kStore>(a, norm, part, pw, stream);
  if (op == 1) return Mx5NP<WN, MxOpV2::kResidual>(a, norm, part, pw, stream);
  if (op == 2) return Mx5NP<WN, MxOpV2::kSwiGLU>(a, norm, part, pw, stream);
  return 4;
}

extern "C" int mx5_gemm_args_size() { return sizeof(Mx5GemmArgs); }
extern "C" int mx5_gemm(Mx5GemmArgs const* args, int wn, int op, int norm, int part, int pw, void* stream) {
  if (wn == 1) return Mx5Op<1>(*args, op, norm, part, pw, stream);
  if (wn == 2) return Mx5Op<2>(*args, op, norm, part, pw, stream);
  if (wn == 4) return Mx5Op<4>(*args, op, norm, part, pw, stream);
  if (wn == 8) return Mx5Op<8>(*args, op, norm, part, pw, stream);
  return 4;
}
