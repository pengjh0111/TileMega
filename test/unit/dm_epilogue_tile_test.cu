// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace tilemega::codegen;
using namespace tilemega::backend;
using BF16 = cutlass::bfloat16_t;
using Kind = DmEpilogueKind;
using Act = DmActivation;
using Round = DmRounding;
using Write = DmWriteKind;
using Side = DmSideOutputKind;
#ifndef TILEMEGA_ARCH_ID
#define TILEMEGA_ARCH_ID 800
#endif
using Arch = typename tilemega::arch::ArchFromId<TILEMEGA_ARCH_ID>::type;

template <Kind K, Act A = Act::kRelu, Round In = Round::kFP32,
          Round Out = Round::kBF16, DmGatePair Gate = DmGatePair::kSwiGLU,
          Write Map = Write::kDense, unsigned Factor = 1>
using Step = DmEpilogueStep<K, A, Gate, 16, In, Out, Map, Factor>;
template <class... Steps> using Chain = DmEpilogueProgram<Steps...>;
template <class C, Write W = Write::kDense, unsigned Factor = 1,
          Round Store = Round::kBF16, class... Sides>
using Spec = DmEpilogueSpec<C, W, Factor, Store, Sides...>;

template <int Mode> struct Case;
template <> struct Case<0> { using Type = Spec<Chain<
    Step<Kind::kBias, Act::kRelu, Round::kFP32, Round::kFP32>,
    Step<Kind::kActivation>>, Write::kDense, 1, Round::kBF16,
    DmSideOutputSpec<Side::kRowStats>, DmSideOutputSpec<Side::kChannelPartialSums>,
    DmSideOutputSpec<Side::kTopKPartial, 8>, DmSideOutputSpec<Side::kArgmaxPartial>,
    DmSideOutputSpec<Side::kPartial>>; };
template <> struct Case<1> { using Type = Spec<Chain<Step<Kind::kBias>,
    Step<Kind::kResidual, Act::kRelu, Round::kBF16>,
    Step<Kind::kActivation, Act::kRelu, Round::kBF16>>>; };
template <> struct Case<2> { using Type = Spec<Chain<
    Step<Kind::kScale, Act::kRelu, Round::kFP32, Round::kFP32>,
    Step<Kind::kActivation, Act::kRelu6>>>; };
template <> struct Case<3> { using Type = Spec<Chain<Step<Kind::kBias>,
    Step<Kind::kActivation, Act::kGeluErf, Round::kBF16>>>; };
template <> struct Case<4> { using Type = Spec<Chain<Step<Kind::kActivation, Act::kGeluTanh, Round::kBF16>>>; };
template <> struct Case<5> { using Type = Spec<Chain<Step<Kind::kActivation, Act::kTanh, Round::kBF16>>>; };
template <> struct Case<6> { using Type = Spec<Chain<Step<Kind::kActivation, Act::kSilu, Round::kBF16>>>; };
template <> struct Case<7> { using Type = Spec<Chain<Step<Kind::kGatePair, Act::kRelu, Round::kBF16,
    Round::kBF16, DmGatePair::kSimpleGate>>>; };
template <> struct Case<8> { using Type = Spec<Chain<Step<Kind::kGatePair, Act::kRelu, Round::kBF16>>>; };
template <> struct Case<9> { using Type = Spec<Chain<Step<Kind::kDeferredLayerNorm>>>; };
template <> struct Case<10> { using Type = Spec<Chain<Step<Kind::kResidualLN, Act::kRelu, Round::kBF16>>>; };
template <> struct Case<11> { using Type = Spec<Chain<Step<Kind::kDeferredRMSNorm>>>; };
template <> struct Case<12> { using Type = Spec<Chain<Step<Kind::kBias>,
    Step<Kind::kResidual, Act::kRelu, Round::kBF16, Round::kBF16,
         DmGatePair::kSwiGLU, Write::kPixelShuffle, 2>>, Write::kPixelShuffle, 2>; };
template <> struct Case<13> { using Type = Spec<Chain<Step<Kind::kBias>,
    Step<Kind::kResidual, Act::kRelu, Round::kBF16, Round::kBF16,
         DmGatePair::kSwiGLU, Write::kNCHW>>, Write::kNCHW>; };
template <> struct Case<14> { using Type = Spec<Chain<Step<Kind::kBias>>, Write::kRowScatter>; };
template <> struct Case<15> { using Type = Spec<Chain<>, Write::kDense, 1, Round::kFP32>; };

void Check(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(2); }
}
template <class T> T Read(std::ifstream& input) {
  T value{};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) std::exit(2);
  return value;
}
std::vector<float> Vector(std::ifstream& input) {
  auto count = Read<std::uint32_t>(input);
  if (count > (1u << 22)) std::exit(2);
  std::vector<float> values(count);
  if (!input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))) std::exit(2);
  return values;
}
struct Allocations {
  std::vector<void*> pointers;
  template <class T> T* Copy(std::vector<T> const& values) {
    if (values.empty()) return nullptr;
    T* pointer;
    Check(cudaMalloc(&pointer, values.size() * sizeof(T)));
    Check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    pointers.push_back(pointer);
    return pointer;
  }
  ~Allocations() { for (auto pointer : pointers) Check(cudaFree(pointer)); }
};
DmBufferLayout Dense(std::initializer_list<std::uint32_t> dimensions) {
  DmBufferLayout layout;
  layout.rank = dimensions.size(); unsigned i = 0;
  for (auto value : dimensions) {
    layout.logical[i] = layout.physical[i] = value;
    ++i;
  }
  std::uint64_t stride = 1;
  for (unsigned axis = layout.rank; axis-- > 0;) {
    layout.strides[axis] = stride; stride *= layout.physical[axis];
  }
  return layout;
}
template <class... Steps>
DmEpilogueChain MakeChain(DmEpilogueProgram<Steps...>) {
  DmEpilogueChain chain;
  auto append = [&](auto step) {
    using S = decltype(step);
    auto& op = chain.operations[chain.count++];
    op.kind = S::kKind; op.activation = S::kActivation; op.gate = S::kGate;
    op.unit = S::kUnit; op.input_rounding = S::kInputRounding;
    op.output_rounding = S::kOutputRounding;
    op.residual_map.kind = S::kResidualMap; op.residual_map.factor = S::kResidualFactor;
    if constexpr (S::kKind == Kind::kBias) op.parameter[0] = 0;
    if constexpr (S::kKind == Kind::kScale) op.parameter[0] = 1;
    if constexpr (S::kKind == Kind::kResidual) {
      op.parameter[0] = 2;
      if constexpr (S::kResidualMap != Write::kDense) op.residual_map.layout = 9;
    }
    if constexpr (S::kKind == Kind::kDeferredLayerNorm) {
      op.parameter[0] = 3; op.parameter[1] = 6; op.parameter[2] = 7;
    }
    if constexpr (S::kKind == Kind::kDeferredRMSNorm) op.parameter[0] = 3;
    if constexpr (S::kKind == Kind::kResidualLN) {
      op.parameter[0] = 2; op.parameter[1] = 3; op.parameter[2] = 4; op.parameter[3] = 5;
    }
  };
  (append(Steps{}), ...);
  return chain;
}

template <class S> __global__ void Evaluate(float const* input, DmEpilogueArguments p) {
  using Body = ServingDmEpilogue<Arch, S, 16, 64>;
  __shared__ float tile[Body::kSharedBytes / sizeof(float)];
  for (int i = threadIdx.x; i < 16 * 64; i += 128) {
    int row = i / 64, col = i % 64;
    int global_row = blockIdx.x * 16 + row, global_col = blockIdx.y * 64 + col;
    tile[Body::Index(row, col)] = global_row < p.m && global_col < p.n ?
        input[global_row * p.n + global_col] : 0;
  }
  tilemega::codegen::executor::ComputeSync();
  Body::RunFromTile(tile, p, blockIdx.x, blockIdx.y);
}

template <class T> void Compare(T* device, std::vector<float> const& expected,
                               unsigned mode, char const* name, bool exact = false) {
  std::vector<T> actual(expected.size());
  Check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
  float maximum = 0;
  for (unsigned i = 0; i < actual.size(); ++i) {
    float value = float(actual[i]), error = std::fabs(value - expected[i]);
    maximum = std::fmax(maximum, error);
    float threshold = exact || expected[i] == -123.f ? 0 :
        1.6e-2f + 1.6e-2f * std::fabs(expected[i]);
    if (!std::isfinite(value) || error > threshold) {
      std::fprintf(stderr, "mode=%u %s element=%u actual=%g reference=%g\n",
                   mode, name, i, value, expected[i]); std::exit(3);
    }
  }
  std::printf("mode=%u %s elements=%zu max_abs_error=%g\n", mode, name, actual.size(), maximum);
}

template <int Mode> void Run(std::array<std::uint32_t, 11> const& h,
                            std::vector<std::vector<float>> const& data) {
  using S = typename Case<Mode>::Type;
  Allocations memory;
  std::array<void*, 17> addresses{};
  std::array<DmBufferLayout, 17> layouts{};
  std::array<std::uint32_t, 17> dtypes{};
  auto floats = [&](unsigned slot, std::vector<float> const& values) {
    addresses[slot] = memory.Copy(values); dtypes[slot] = 1;
  };
  auto integers = [&](unsigned slot, std::vector<float> const& values) {
    addresses[slot] = memory.Copy(std::vector<std::int32_t>(values.begin(), values.end()));
    dtypes[slot] = 2;
  };
  floats(0, data[1]); floats(1, data[2]);
  addresses[2] = memory.Copy(std::vector<BF16>(data[3].begin(), data[3].end()));
  floats(3, data[4]); floats(4, data[5]); floats(5, data[6]);
  floats(6, data[7]); floats(7, data[8]); integers(8, data[9]);
  unsigned m = h[1], n = h[2], out_n = n / (Mode == 7 || Mode == 8 ? 2 : 1);
  layouts[3] = Dense({m, h[9], 2});
  std::vector<float> initial(data[10].size(), -123.f);
  if constexpr (S::kStore == Round::kBF16)
    addresses[9] = memory.Copy(std::vector<BF16>(initial.begin(), initial.end()));
  else floats(9, initial);
  if constexpr (Mode == 0 || Mode == 12) {
    auto& layout = layouts[9];
    layout = Dense({h[3], h[4], h[5], h[6]});
    layout.kind = DmLayout::kNHWC;
    layout.physical[3] = Mode == 0 ? 64 : 8;
    if constexpr (Mode == 0) {
      layout.halo_top = layout.halo_bottom = 1;
      layout.halo_left = layout.halo_right = 2;
      layout.physical[1] += 2; layout.physical[2] += 4;
    }
    std::uint64_t stride = 1;
    for (unsigned axis = 4; axis-- > 0;) {
      layout.strides[axis] = stride; stride *= layout.physical[axis];
    }
  }
  if constexpr (Mode == 13) layouts[9] = Dense({h[3], h[6], h[4], h[5]});
  DmEpilogueArguments p;
  p.chain = MakeChain(typename S::Chain{}); p.chain.store_rounding = S::kStore;
  if constexpr (Mode == 0) {
    p.chain.side_count = 5;
    p.chain.side[0] = {Side::kRowStats, 10, kDmNoIndex, 0};
    p.chain.side[1] = {Side::kChannelPartialSums, 11, kDmNoIndex, 0};
    p.chain.side[2] = {Side::kTopKPartial, 12, 13, 8};
    p.chain.side[3] = {Side::kArgmaxPartial, 14, 15, 0};
    p.chain.side[4] = {Side::kPartial, 16, kDmNoIndex, 0};
    for (unsigned slot = 10; slot < 17; ++slot) {
      std::vector<float> values(data[slot + 1].size(), slot == 16 ? 0.f : -123.f);
      if (slot == 13 || slot == 15) integers(slot, values); else floats(slot, values);
    }
    layouts[10] = Dense({m, 1, 2}); layouts[11] = Dense({3, 3, out_n});
    layouts[12] = layouts[13] = Dense({m, 1, 8});
    layouts[14] = layouts[15] = Dense({m, 1, 1}); layouts[16] = Dense({m, 1, 64});
  }
  std::vector<void*> pointer_values(addresses.begin(), addresses.end());
  p.buffers.data = memory.Copy(pointer_values);
  p.buffers.layouts = memory.Copy(std::vector<DmBufferLayout>(layouts.begin(), layouts.end()));
  p.buffers.dtypes = memory.Copy(std::vector<std::uint32_t>(dtypes.begin(), dtypes.end()));
  p.buffers.count = addresses.size(); p.output = addresses[9];
  p.m = m; p.n = n; p.output_stride = out_n; p.norm_width = h[8]; p.norm_eps = 1e-5f;
  p.image_rows = h[10]; p.write.kind = S::kWrite; p.write.factor = S::kFactor;
  if constexpr (Mode == 0 || Mode == 12 || Mode == 13) p.write.layout = 9;
  if constexpr (Mode == 14) p.write.rows = 8;
  auto input = memory.Copy(data[0]);
  Evaluate<S><<<dim3((m + 15) / 16, (n + 63) / 64), 128>>>(input, p);
  Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
  if constexpr (S::kStore == Round::kBF16)
    Compare(reinterpret_cast<BF16*>(addresses[9]), data[10], Mode, "output");
  else Compare(reinterpret_cast<float*>(addresses[9]), data[10], Mode, "output", true);
  if constexpr (Mode == 0) {
    for (unsigned slot = 10; slot < 17; ++slot)
      if (slot == 13 || slot == 15)
        Compare(reinterpret_cast<std::int32_t*>(addresses[slot]), data[slot + 1], Mode, "side-index", true);
      else Compare(reinterpret_cast<float*>(addresses[slot]), data[slot + 1], Mode, "side-value");
  }
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary); if (!input) return 2;
  if (Read<std::uint32_t>(input) != 0x444D4531) return 2;
  auto cases = Read<std::uint32_t>(input);
  std::uint32_t seen = 0;
  for (unsigned c = 0; c < cases; ++c) {
    auto header = Read<std::array<std::uint32_t, 11>>(input);
    if (header[0] >= 16 || (seen & (1u << header[0]))) return 2;
    seen |= 1u << header[0];
    std::vector<std::vector<float>> data;
    for (unsigned i = 0; i < (header[0] == 0 ? 18 : 11); ++i) data.push_back(Vector(input));
    switch (header[0]) {
#define DM_CASE(N) case N: Run<N>(header, data); break;
      DM_CASE(0) DM_CASE(1) DM_CASE(2) DM_CASE(3) DM_CASE(4) DM_CASE(5) DM_CASE(6) DM_CASE(7)
      DM_CASE(8) DM_CASE(9) DM_CASE(10) DM_CASE(11) DM_CASE(12) DM_CASE(13) DM_CASE(14) DM_CASE(15)
#undef DM_CASE
    }
  }
  if (seen != 0xffff || input.peek() != std::char_traits<char>::eof()) return 2;
  std::printf("DM epilogue tiles: %u PyTorch reference cases passed\n", cases);
}
