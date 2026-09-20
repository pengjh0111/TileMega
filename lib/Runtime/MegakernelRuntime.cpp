// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Runtime/MegakernelRuntime.h>
namespace tilemega::runtime {
MegakernelRuntime::~MegakernelRuntime() {
  if (module_) cuModuleUnload(module_);
}

bool MegakernelRuntime::Load(std::string const& cubin_path,
                            char const* kernel_name) {
  if (!kernel_name || cuInit(0) != CUDA_SUCCESS) return false;
  CUmodule module = nullptr;
  if (cuModuleLoad(&module, cubin_path.c_str()) != CUDA_SUCCESS) return false;
  CUfunction function = nullptr;
  if (cuModuleGetFunction(&function, module, kernel_name) != CUDA_SUCCESS) {
    cuModuleUnload(module);
    return false;
  }
  if (module_ && cuModuleUnload(module_) != CUDA_SUCCESS) {
    cuModuleUnload(module);
    return false;
  }
  module_ = module;
  function_ = function;
  return true;
}
}  // namespace tilemega::runtime
