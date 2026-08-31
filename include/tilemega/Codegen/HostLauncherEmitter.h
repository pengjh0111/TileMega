// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 host launch wrapper emission (Phase 2 stub).
#pragma once
#include <string>
namespace tilemega::codegen {
class HostLauncherEmitter { public: std::string Emit(std::string const& kernel_name) const; };
}  // namespace tilemega::codegen
