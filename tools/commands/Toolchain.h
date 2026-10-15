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
inline std::string DeviceBuildFlags() { return " -Xcompiler -resource-usage"; }
inline std::string CudaLibraryDirectory() {
  auto* override = std::getenv("TILEMEGA_CUDA_LIBRARY_DIR");
  return override ? override : TILEMEGA_CONFIG_CUDA_LIBRARY_DIR;
}
}  // namespace tilemega::commands
