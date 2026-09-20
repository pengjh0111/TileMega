// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.4 runtime integration.
#pragma once
#include <cuda.h>
#include <string>
#include <utility>
#include <tilemega/Runtime/Launcher.h>
namespace tilemega::runtime {
class MegakernelRuntime {
 public:
  explicit MegakernelRuntime(TargetSpec target) : target_(std::move(target)) {}
  ~MegakernelRuntime();
  MegakernelRuntime(MegakernelRuntime const&) = delete;
  MegakernelRuntime& operator=(MegakernelRuntime const&) = delete;
  TargetSpec const& target() const { return target_; }
  CUmodule module() const { return module_; }
  CUfunction function() const { return function_; }
  // Two preconditions the driver API does not check for us: a context must be
  // current (cuInit alone does not make one), and `kernel_name` is a symbol in
  // the cubin -- the generated kernels have C++ linkage, so that is the mangled
  // name of `tilemega_l1_kernel`, not the spelling above. No default: there is
  // no name that is right for every module.
  bool Load(std::string const& cubin_path, char const* kernel_name);
 private:
  TargetSpec target_;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
};
}  // namespace tilemega::runtime
