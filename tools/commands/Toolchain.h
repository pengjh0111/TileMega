// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/BuildConfig.h>
#include <cstdlib>
#include <string>

namespace tilemega::commands {
inline std::string NvccPath() {
  auto* override = std::getenv("CUDACXX");
  return override ? override : TILEMEGA_CONFIG_NVCC;
}
/// cucc options shared by every generated-plan build: per-kernel register,
/// stack and smem report (parsed by ParsePtxasRegisters) in place of
/// -Xptxas=-v. cucc drops nvcc-only flags such as -arch on its own.
/// -fno-strict-aliasing: the bodies pun BF16 arrays through uint4; MACA's
/// clang at -O3 drops such stores under strict aliasing (verified: RMSNorm
/// produced zeros/NaN until this flag), which nvcc tolerated.
inline std::string DeviceBuildFlags() { return " -Xcompiler -resource-usage -fno-strict-aliasing"; }
inline std::string CudaLibraryDirectory() {
  auto* override = std::getenv("TILEMEGA_CUDA_LIBRARY_DIR");
  return override ? override : TILEMEGA_CONFIG_CUDA_LIBRARY_DIR;
}
}  // namespace tilemega::commands
