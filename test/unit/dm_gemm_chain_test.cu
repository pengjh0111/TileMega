// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 3
#include <tilemega/Codegen/DmDescriptors.h>

namespace tilemega::codegen {
using K = DmEpilogueKind;
using R = DmRounding;
using W = DmWriteKind;
template <K Kind, R In = R::kFP32, R Out = R::kBF16,
          DmActivation Act = DmActivation::kRelu, W Map = W::kDense, unsigned Factor = 1>
using ChainStep = DmEpilogueStep<Kind, Act, DmGatePair::kSwiGLU, 16, In, Out, Map, Factor>;
template <W Map, unsigned Factor, class... Steps>
using Spec = DmEpilogueSpec<DmEpilogueProgram<Steps...>, Map, Factor, R::kBF16,
                          DmSideOutputSpec<DmSideOutputKind::kRowStats>>;
using S0 = Spec<W::kDense, 1, ChainStep<K::kBias, R::kFP32, R::kFP32>, ChainStep<K::kActivation>>;
using S1 = Spec<W::kDense, 1, ChainStep<K::kGatePair, R::kBF16>>;
using S2 = Spec<W::kPixelShuffle, 2, ChainStep<K::kBias>,
                ChainStep<K::kResidual, R::kBF16, R::kBF16, DmActivation::kRelu, W::kPixelShuffle, 2>>;
using S3 = Spec<W::kNCHW, 1, ChainStep<K::kScale, R::kFP32, R::kFP32>,
                ChainStep<K::kActivation, R::kFP32, R::kBF16, DmActivation::kRelu6>>;
template <class Runner>
__device__ void DispatchDmEpilogue(std::uint32_t id, Runner const& runner) {
  switch (id) {
    case 0: runner.template Run<S0>(); break;
    case 1: runner.template Run<S1>(); break;
    case 2: runner.template Run<S2>(); break;
    case 3: runner.template Run<S3>(); break;
    default: asm volatile("trap;");
  }
}
}  // namespace tilemega::codegen

#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/GemmCombineTaskBody.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using namespace tilemega::codegen;
using E = cutlass::bfloat16_t;
#ifndef TILEMEGA_ARCH_ID
#define TILEMEGA_ARCH_ID 800
#endif
using Arch = typename tilemega::arch::ArchFromId<TILEMEGA_ARCH_ID>::type;
struct Scratch { GemmVariantSmem gemm; };
using G = GemmStageTaskBody<Arch, Scratch, 128>;
using C = GemmCombineTaskBody<Arch, Scratch, 128>;
using P = PagedGemmTaskBody<Arch, 16, 64, 64, 8192, 3>;

__global__ void Dense(GemmInvocation const* invocations, int tiles) {
  __shared__ Scratch shared;
  int chunk = blockIdx.x / tiles, task = blockIdx.x % tiles;
  G::RunTask<0>(invocations[chunk], task, reinterpret_cast<char*>(&shared.gemm));
}
__global__ void Combine(Params p, StageDesc stage) {
  __shared__ Scratch shared;
  C::RunServingVariant<0>(p, stage, shared, blockIdx.x);
}
struct PageRunner {
  ServingGemmOperands const& operands;
  P::Ring const& ring;
  std::uint64_t& sequence;
  char* work;
  int m, n;
  template <class S> __device__ void Run() const {
    P::RunDm<S>(operands, m, n, ring, sequence, work);
  }
};
__global__ void Pages(ServingGemmOperands p, std::uint32_t id, bool partial) {
  extern __shared__ __align__(1024) char storage[];
  constexpr int pool = ((1024 + P::kDmWorkspaceBytes + 1023) / 1024) * 1024;
  P::Ring ring{reinterpret_cast<P::Ring::Slot*>(storage), storage + pool};
  ring.Initialize(); std::uint64_t sequence = 0;
  int tm = blockIdx.x / ((p.n + 63) / 64), tn = blockIdx.x % ((p.n + 63) / 64);
  if (!executor::IsCompute()) P::Load(p, tn, ring, sequence);
  else if (partial) P::Run(p, tm, tn, ring, sequence, storage + 1024);
  else DispatchDmEpilogue(id, PageRunner{p, ring, sequence, storage + 1024, tm, tn});
}

void Check(cudaError_t value) {
  if (value != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(value)); std::exit(2); }
}
struct Memory {
  std::vector<void*> allocations;
  template <class T> T* Copy(std::vector<T> const& values) {
    T* pointer; Check(cudaMalloc(&pointer, values.size() * sizeof(T)));
    Check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    allocations.push_back(pointer); return pointer;
  }
  ~Memory() { for (auto p : allocations) Check(cudaFree(p)); }
};
template <class T> T Read(std::ifstream& input) {
  T value; if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) std::exit(2);
  return value;
}
std::vector<float> Vector(std::ifstream& input) {
  auto size = Read<std::uint32_t>(input); if (size > (1u << 22)) std::exit(2);
  std::vector<float> values(size);
  if (!input.read(reinterpret_cast<char*>(values.data()), size * sizeof(float))) std::exit(2);
  return values;
}
DmBufferLayout Layout(std::initializer_list<unsigned> shape) {
  DmBufferLayout layout; layout.rank = shape.size(); unsigned axis = 0;
  for (auto value : shape) { layout.logical[axis] = layout.physical[axis] = value; ++axis; }
  std::uint64_t stride = 1;
  while (axis--) { layout.strides[axis] = stride; stride *= layout.physical[axis]; }
  return layout;
}
template <class... Steps> DmEpilogueChain MakeChain(DmEpilogueProgram<Steps...>) {
  DmEpilogueChain chain;
  auto add = [&](auto step) {
    using S = decltype(step); auto& op = chain.operations[chain.count++];
    op.kind = S::kKind; op.activation = S::kActivation; op.gate = S::kGate; op.unit = S::kUnit;
    op.input_rounding = S::kInputRounding; op.output_rounding = S::kOutputRounding;
    op.residual_map.kind = S::kResidualMap; op.residual_map.factor = S::kResidualFactor;
    if constexpr (S::kKind == K::kBias) op.parameter[0] = 0;
    if constexpr (S::kKind == K::kScale) op.parameter[0] = 1;
    if constexpr (S::kKind == K::kResidual) { op.parameter[0] = 2; op.residual_map.layout = 3; }
  };
  (add(Steps{}), ...);
  chain.side_count = 1; chain.side[0] = {DmSideOutputKind::kRowStats, 4, kDmNoIndex, 0};
  return chain;
}
template <class T> std::vector<T> Compare(T* device, std::vector<float> const& reference,
                                        unsigned mode, int chunks, char const* label) {
  std::vector<T> actual(reference.size());
  Check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (unsigned i = 0; i < actual.size(); ++i) {
    float value = float(actual[i]);
    float limit = reference[i] == -123.f ? 0 : 1.6e-2f + 1.6e-2f * std::fabs(reference[i]);
    if (!std::isfinite(value) || std::fabs(value - reference[i]) > limit) {
      std::fprintf(stderr, "mode=%u split=%d %s index=%u value=%g expected=%g\n",
                   mode, chunks, label, i, value, reference[i]); std::exit(3);
    }
  }
  return actual;
}
template <class S> void Run(std::array<unsigned, 6> h,
                            std::array<std::vector<float>, 7> const& data) {
  unsigned mode = h[0], m = h[1], n = h[2], k = h[3], stride = h[4], offset = h[5];
  Memory memory;
  E* a = memory.Copy(std::vector<E>(data[0].begin(), data[0].end()));
  E* b = memory.Copy(std::vector<E>(data[1].begin(), data[1].end()));
  E* output = memory.Copy(std::vector<E>(data[5].size(), E(-123.f)));
  float* stats = memory.Copy(std::vector<float>(data[6].size(), -123.f));
  float* partials = memory.Copy(std::vector<float>(3 * m * n, 0));
  std::vector<void*> pointers{memory.Copy(data[2]), memory.Copy(data[3]),
      memory.Copy(std::vector<E>(data[4].begin(), data[4].end())), output, stats};
  std::vector<DmBufferLayout> layouts(5);
  layouts[4] = Layout({m, (n + 63) / 64, 2});
  if (mode == 0) {
    layouts[3] = Layout({3, 3, 15, 64}); layouts[3].kind = DmLayout::kNHWC;
    layouts[3].logical[1] = 1; layouts[3].logical[2] = 11; layouts[3].logical[3] = n;
    layouts[3].halo_top = layouts[3].halo_bottom = 1;
    layouts[3].halo_left = layouts[3].halo_right = 2;
  } else if (mode == 2) {
    layouts[3] = Layout({2, 4, 6, 8}); layouts[3].kind = DmLayout::kNHWC;
    layouts[3].logical[3] = 3;
  } else if (mode == 3) layouts[3] = Layout({2, 3, 3, 5});
  DmBufferView view{memory.Copy(pointers), memory.Copy(layouts),
      memory.Copy(std::vector<std::uint32_t>{1, 1, 0, 0, 1}), unsigned(pointers.size())};
  auto chain = MakeChain(typename S::Chain{});
  chain.store_rounding = S::kStore;
  for (int chunks : {1, 2, 3}) {
    std::vector<E> dense_output;
    for (bool paged : {false, true}) {
      std::vector<E> poison(data[5].size(), E(-123.f));
      Check(cudaMemcpy(output, poison.data(), poison.size() * sizeof(E), cudaMemcpyHostToDevice));
      std::vector<GemmInvocation> invocations;
      unsigned tiles = ((m + 15) / 16) * ((n + 63) / 64);
      for (int chunk = 0; chunk < chunks; ++chunk) {
        int begin = chunk * 3 / chunks * 64, end = (chunk + 1) * 3 / chunks * 64;
        GemmInvocation inv{};
        inv.problem = {int(m), int(n), end - begin, 1};
        inv.mainloop.ptr_A = a + offset * k + begin; inv.mainloop.ptr_B = b + begin;
        inv.mainloop.dA = cute::make_stride(std::int64_t(k * stride), cute::_1{}, std::int64_t(0));
        inv.mainloop.dB = cute::make_stride(std::int64_t(k), cute::_1{}, std::int64_t(0));
        inv.epilogue.ptr_D = output; inv.tiles_m = (m + 15) / 16; inv.tiles_n = (n + 63) / 64;
        inv.tile_m = 16; inv.tile_n = 64; inv.k_total = k; inv.chunks = chunks;
        inv.serving_output_stride = mode == 1 ? n / 2 : n;
        inv.serving_partial_stride = n; inv.serving_partial = partials + chunk * m * n;
        inv.dm_gemm = mode; inv.chain = chain; inv.dm_buffers = view;
        inv.access.write = {S::kWrite, S::kFactor, mode == 1 ? kDmNoIndex : 3u, kDmNoIndex};
        inv.access.rows_per_batch = mode == 0 ? 11 : m;
        invocations.push_back(inv);
        if (paged) {
          ServingGemmOperands p;
          p.a = a + offset * k; p.b = b; p.output = output; p.partial = inv.serving_partial;
          p.m = m; p.n = n; p.k_total = k; p.k_begin = begin; p.k_count = end - begin;
          p.a_row_stride = k * stride; p.b_row_stride = k;
          p.output_stride = inv.serving_output_stride; p.partial_stride = n;
          p.access = inv.access; p.chain = chain; p.dm_buffers = view;
          p.epilogue = chunks > 1 ? tilemega::backend::ServingEpilogueOp::kPartial :
                                  tilemega::backend::ServingEpilogueOp::kStore;
          constexpr int bytes = ((1024 + P::kDmWorkspaceBytes + 1023) / 1024) * 1024 + 3 * 8192;
          Pages<<<tiles, 160, bytes>>>(p, mode, chunks > 1);
        }
      }
      auto inv = memory.Copy(invocations);
      if (!paged) Dense<<<tiles * chunks, 128>>>(inv, tiles);
      if (chunks > 1) {
        Params params{}; params.gemms = inv;
        params.buffers = memory.Copy(std::vector<E*>{reinterpret_cast<E*>(partials), output});
        StageDesc stage{}; stage.width = n; stage.operand[0] = 0; stage.operand[1] = 1;
        Combine<<<tiles, 128>>>(params, stage);
      }
      Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
      auto values = Compare(output, data[5], mode, chunks, paged ? "pages" : "dense");
      Compare(stats, data[6], mode, chunks, "row_stats");
      if (!paged) dense_output = values;
      else if (std::memcmp(values.data(), dense_output.data(), values.size() * sizeof(E))) {
        std::fprintf(stderr, "page/dense bitwise mismatch mode=%u split=%d\n", mode, chunks); std::exit(3);
      }
    }
  }
  std::printf("mode=%u: dense/page and split-K chains passed\n", mode);
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  if (Read<unsigned>(input) != 0x444D4731 || Read<unsigned>(input) != 4) return 2;
  for (unsigned mode = 0; mode < 4; ++mode) {
    std::array<unsigned, 6> h; for (auto& value : h) value = Read<unsigned>(input);
    std::array<std::vector<float>, 7> data; for (auto& value : data) value = Vector(input);
    if (h[0] != mode) return 2;
    switch (mode) {
      case 0: Run<S0>(h, data); break; case 1: Run<S1>(h, data); break;
      case 2: Run<S2>(h, data); break; case 3: Run<S3>(h, data); break;
    }
  }
  return input.peek() == EOF ? 0 : 2;
}
