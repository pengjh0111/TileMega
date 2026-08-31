// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Runtime/MegakernelRuntime.h>
#include <fstream>
namespace tilemega::runtime {
bool MegakernelRuntime::Load(std::string const& cubin_path) {
  // TODO(P2.4): load and retain the CUDA driver module/function.
  return static_cast<bool>(std::ifstream(cubin_path, std::ios::binary));
}
}  // namespace tilemega::runtime
