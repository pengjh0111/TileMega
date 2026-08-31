// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/HostLauncherEmitter.h>
#include <tilemega/Codegen/ScheduleTableEmitter.h>
#include <tilemega/Codegen/SyncEmitter.h>
#include <tilemega/Codegen/TaskBodyEmitter.h>
namespace tilemega::codegen {
std::string CouplingGraphToCUDA::Lower(std::string const& graph) const {
  return "// TODO(P2.1): lower Coupling Graph\n// " + graph;
}
std::string TaskBodyEmitter::Emit(std::string const& kind) const { return "// TaskBody: " + kind; }
std::string SyncEmitter::EmitWait(std::string const& event) const { return "wait(" + event + ")"; }
std::string SyncEmitter::EmitSignal(std::string const& event) const { return "signal(" + event + ")"; }
std::vector<ScheduleEntry> ScheduleTableEmitter::Emit(std::vector<int> const& order) const {
  std::vector<ScheduleEntry> result;
  for (int tile : order) result.push_back({0u, static_cast<std::uint32_t>(tile)});
  return result;
}
std::string HostLauncherEmitter::Emit(std::string const& kernel) const { return "launch(" + kernel + ")"; }
}  // namespace tilemega::codegen
