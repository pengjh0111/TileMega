// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <type_traits>

namespace tilemega::codegen {

inline constexpr std::uint32_t kDmNoIndex = 0xffffffffu;
enum class DmAAccess : std::uint32_t { kDense, kIm2Col, kRowGather };
enum class DmBAccess : std::uint32_t { kDense, kExpertIndirect };
enum class DmWriteKind : std::uint32_t { kDense, kNCHW, kPixelShuffle, kRowScatter };
enum class DmLayout : std::uint32_t { kRowMajor, kNHWC };
enum class DmFill : std::uint32_t { kZero, kNegativeInfinity };

struct DmWriteMap {
  DmWriteKind kind = DmWriteKind::kDense;
  std::uint32_t factor = 1;
  std::uint32_t layout = kDmNoIndex;
  std::uint32_t rows = kDmNoIndex;
};

struct ConvDesc {
  std::uint32_t n = 1, h = 0, w = 0, c = 0, k = 0, r = 1, s = 1;
  std::uint32_t stride_h = 1, stride_w = 1;
  std::uint32_t pad_h = 0, pad_w = 0;
  std::uint32_t dilation_h = 1, dilation_w = 1;
  std::uint32_t p = 0, q = 0;
  std::uint32_t input_layout = kDmNoIndex, output_layout = kDmNoIndex;
};

// NHWC shapes are N,H,W,C; row-major shapes use the first rank axes.
// Physical strides count elements, including halo
// and channel padding. Halo addresses have no producer and remain read-only.
struct DmBufferLayout {
  DmLayout kind = DmLayout::kRowMajor;
  std::uint32_t rank = 0;
  std::uint32_t logical[4]{};
  std::uint32_t physical[4]{};
  std::uint64_t strides[4]{};
  std::uint32_t halo_top = 0, halo_bottom = 0, halo_left = 0, halo_right = 0;
  DmFill fill = DmFill::kZero;
};

struct DmGemmAccess {
  DmAAccess a = DmAAccess::kDense;
  DmBAccess b = DmBAccess::kDense;
  std::uint32_t rows_per_batch = 0;  // Zero retains tokens/batch_rows semantics.
  // Dense row mapping: source_row = offset + row * (stride ? stride : 1).
  // This multiplier is distinct from ServingGemmOperands' element pitch.
  std::uint32_t a_row_stride = 0, a_row_offset = 0;
  std::uint32_t conv = kDmNoIndex, rows = kDmNoIndex, binding = kDmNoIndex;
  std::uint32_t a_scale = kDmNoIndex;
  std::uint64_t expert_stride = 0;
  DmWriteMap write{};
};

enum class DmEpilogueKind : std::uint32_t {
  kBias, kScale, kActivation, kResidual, kGatePair, kDeferredRMSNorm,
  kDeferredLayerNorm, kResidualLN
};
enum class DmActivation : std::uint32_t {
  kRelu, kRelu6, kGeluErf, kGeluTanh, kTanh, kSilu
};
enum class DmGatePair : std::uint32_t { kSwiGLU, kSimpleGate };
enum class DmRounding : std::uint32_t { kFP32, kBF16 };

// Rounding is explicit on both sides of every operation. Fused arithmetic
// may stay FP32 only where the plan's numerical contract permits it.
struct DmEpilogueOp {
  DmEpilogueKind kind = DmEpilogueKind::kBias;
  std::uint32_t parameter[4]{kDmNoIndex, kDmNoIndex, kDmNoIndex, kDmNoIndex};
  DmWriteMap residual_map{};
  DmActivation activation = DmActivation::kRelu;
  DmGatePair gate = DmGatePair::kSwiGLU;
  std::uint32_t unit = 16;
  DmRounding input_rounding = DmRounding::kFP32;
  DmRounding output_rounding = DmRounding::kFP32;
};

enum class DmSideOutputKind : std::uint32_t {
  kRowStats, kChannelPartialSums, kTopKPartial, kArgmaxPartial, kPartial
};
struct DmSideOutput {
  DmSideOutputKind kind = DmSideOutputKind::kRowStats;
  std::uint32_t buffer = kDmNoIndex, auxiliary = kDmNoIndex, count = 0;
};
struct DmEpilogueChain {
  std::uint32_t count = 0;
  DmEpilogueOp operations[8]{};
  std::uint32_t side_count = 0;
  DmSideOutput side[5]{};
  DmRounding store_rounding = DmRounding::kBF16;
};

struct DmBufferView {
  void* const* data = nullptr;
  DmBufferLayout const* layouts = nullptr;
  std::uint32_t const* dtypes = nullptr;
  std::uint32_t count = 0;
};

// Kind packs give codegen a finite compile-time dispatch surface. Numerical
// bodies specialize these packs rather than branch on kind per element.
template <DmEpilogueKind... Kinds>
struct DmEpilogueKinds {
  static_assert(sizeof...(Kinds) <= 8, "epilogue chain exceeds eight operations");
  static constexpr std::uint32_t kCount = sizeof...(Kinds);
};

static_assert(std::is_trivially_copyable_v<ConvDesc>);
static_assert(std::is_standard_layout_v<DmBufferLayout>);
static_assert(std::is_trivially_copyable_v<DmGemmAccess>);
static_assert(std::is_trivially_copyable_v<DmEpilogueChain>);

}  // namespace tilemega::codegen
