// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.3 and §8.1–§8.5 synchronization lowering.
#pragma once
#include <string>
namespace tilemega::codegen {
class SyncEmitter { public: std::string EmitWait(std::string const& event) const; std::string EmitSignal(std::string const& event) const; };
}  // namespace tilemega::codegen
