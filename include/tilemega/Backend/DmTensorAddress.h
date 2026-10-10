// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/DmDescriptors.h>
#include <cstdint>

#if defined(__CUDACC__)
#define TILEMEGA_DM_ADDRESS_HD __host__ __device__
#else
#define TILEMEGA_DM_ADDRESS_HD
#endif

namespace tilemega::backend {

// row numbers the flattened N,H,W axes for image tensors. NCHW and shuffle
// maps change physical ownership without changing that logical row space.
template <codegen::DmWriteKind Kind, unsigned Factor = 1>
struct DmTensorAddress {
  static_assert(Factor > 0);
  TILEMEGA_DM_ADDRESS_HD static std::uint64_t Offset(
      codegen::DmBufferLayout const& layout, std::uint64_t row,
      std::uint32_t column, std::uint64_t fallback_pitch,
      std::uint32_t const* scatter = nullptr) {
    using K = codegen::DmWriteKind;
    if constexpr (Kind == K::kRowScatter) row = scatter[row];
    if constexpr (Kind == K::kNCHW) {
      auto spatial = std::uint64_t(layout.logical[2]) * layout.logical[3];
      auto n = row / spatial, pixel = row % spatial;
      return n * layout.strides[0] + column * layout.strides[1] +
             (pixel / layout.logical[3]) * layout.strides[2] +
             (pixel % layout.logical[3]) * layout.strides[3];
    } else if constexpr (Kind == K::kPixelShuffle) {
      auto p = layout.logical[1] / Factor, q = layout.logical[2] / Factor;
      auto n = row / (std::uint64_t(p) * q);
      auto pixel = row % (std::uint64_t(p) * q);
      auto channel = column / (Factor * Factor);
      auto subpixel = column % (Factor * Factor);
      auto h = (pixel / q) * Factor + subpixel / Factor;
      auto w = (pixel % q) * Factor + subpixel % Factor;
      return n * layout.strides[0] + (h + layout.halo_top) * layout.strides[1] +
             (w + layout.halo_left) * layout.strides[2] + channel * layout.strides[3];
    } else {
      if (!layout.rank) return row * fallback_pitch + column;
      std::uint64_t offset = column * layout.strides[layout.rank - 1];
      for (unsigned axis = layout.rank - 1; axis-- > 0;) {
        auto coordinate = row % layout.logical[axis];
        row /= layout.logical[axis];
        if (layout.kind == codegen::DmLayout::kNHWC) {
          if (axis == 1) coordinate += layout.halo_top;
          if (axis == 2) coordinate += layout.halo_left;
        }
        offset += coordinate * layout.strides[axis];
      }
      return offset;
    }
  }
};

}  // namespace tilemega::backend
#undef TILEMEGA_DM_ADDRESS_HD
