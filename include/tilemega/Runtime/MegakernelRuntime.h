// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.4 runtime integration (Phase 2 stub).
#pragma once
#include <string>
#include <utility>
#include <tilemega/Runtime/Launcher.h>
namespace tilemega::runtime {
class MegakernelRuntime {
 public:
  explicit MegakernelRuntime(TargetSpec target) : target_(std::move(target)) {}
  TargetSpec const& target() const { return target_; }
  bool Load(std::string const& cubin_path);
 private:
  TargetSpec target_;
};
}  // namespace tilemega::runtime
