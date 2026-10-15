// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tilemega::codegen::executor {
// xcore1000 has no TMA. The descriptor table keeps its 64-byte slots so the
// runtime's Params layout is unchanged, and encoding is a hard error.
struct TensorMapShape {
  std::uint64_t columns=0,rows=0,stride_bytes=0;
  std::uint32_t box_columns=64,box_rows=0;
};
struct alignas(64) TensorMap {
  unsigned char opaque[128];
};
inline int EncodeTensorMap(TensorMap&,void*,TensorMapShape const&) {
  throw std::runtime_error("tensor maps (TMA) do not exist on xcore1000");
}
} // namespace tilemega::codegen::executor
