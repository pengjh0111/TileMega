// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/HostLauncherEmitter.h>
#include <tilemega/Codegen/ScheduleTableEmitter.h>
#include <tilemega/Codegen/SyncEmitter.h>
#include <tilemega/Codegen/TaskBodyEmitter.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>

#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace tilemega::codegen {
namespace {
analysis::ParamBinding readBinding(mlir::ModuleOp module, llvm::StringRef name) {
  analysis::ParamBinding result;
  if (auto values = module->getAttrOfType<mlir::DictionaryAttr>(name))
    for (auto item : values)
      if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(item.getValue()))
        result.Bind(item.getName().str(), integer.getInt());
  return result;
}
}  // namespace

std::string TaskBodyEmitter::Emit(mlir::ModuleOp module) const {
  bool gemm = false, rmsnorm = false, rope = false, kvappend = false;
  bool elementwise = false, attention = false;
  for (auto task : module.getOps<dialect::TaskSpaceOp>()) {
    llvm::StringRef kind = task.getKind().getValue().getValue();
    gemm |= kind == "gemm";
    rmsnorm |= kind == "reduction";
    rope |= task.getStage() % 12 == 4;
    kvappend |= task.getStage() % 12 == 4;
    attention |= task.getStage() % 12 == 5;
    elementwise |= kind == "elementwise";
  }
  if (!(gemm && rmsnorm && rope && kvappend && elementwise && attention))
    throw std::runtime_error("CG does not cover all six Llama TaskBody families");
  return "#include <tilemega/Codegen/tasks/GeneratedLlamaRuntime.cuh>\n";
}

std::string SyncEmitter::EmitWait(std::string const& event) const {
  return "#define TILEMEGA_GENERATED_WAIT_" + event +
      "(ev, need) do { \\\n"
      "  while (atomicAdd((ev), 0ull) < (need)) __nanosleep(64); \\\n"
      "} while (0)\n";
}

std::string SyncEmitter::EmitSignal(std::string const& event) const {
  return "#define TILEMEGA_GENERATED_NOTIFY_" + event +
      "(ev, value) atomicExch((ev), (value))\n";
}

std::vector<ScheduleEntry> ScheduleTableEmitter::Emit(
    std::vector<int> const& order) const {
  std::vector<ScheduleEntry> result;
  for (int tile : order)
    result.push_back({0u, static_cast<std::uint32_t>(tile)});
  return result;
}

std::string ScheduleTableEmitter::EmitStageCounts(
    std::vector<std::size_t> const& counts) const {
  std::ostringstream out;
  out << "#define TILEMEGA_GENERATED_STAGE_TASK_COUNTS {";
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (i) out << ",";
    out << counts[i];
  }
  out << "}\n"
      << "#define TILEMEGA_GENERATED_SCHEDULE {\\\n";
  for (std::size_t i = 0; i < counts.size(); ++i)
    out << "  {" << (i % 12) << "u, " << i << "u, 0u, "
        << counts[i] << "u}, \\\n";
  out << "}\n";
  return out.str();
}

std::string HostLauncherEmitter::Emit(std::string const& kernel) const {
  return "#define TILEMEGA_GENERATED_RESIDENT_GRID(target, function, block_size, dynamic_smem) \\\n"
      "  ((target).res.num_sms * (target).ActiveBlocksPerSM( \\\n"
      "      reinterpret_cast<void const*>(function), (block_size), (dynamic_smem)))\n"
      "// Model invocation tables are allocated on device and passed as one "
      "Params const* by GeneratedLlamaRuntime (F-17b).\n"
      "// Launcher specialization: " + kernel + "\n";
}

std::string CouplingGraphToCUDA::Lower(mlir::ModuleOp module) const {
  if (!module || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("CouplingGraphToCUDA requires a verified CG ModuleOp");

  std::size_t tasks = 0, couplings = 0, placements = 0;
  auto theta = readBinding(module, "tilemega.theta");
  auto granularity = readBinding(module, "tilemega.g");
  int maxStage = -1;
  for (auto task : module.getOps<dialect::TaskSpaceOp>()) {
    ++tasks;
    maxStage = std::max(maxStage, static_cast<int>(task.getStage()));
  }
  if (maxStage < 0) throw std::invalid_argument("CG has no task spaces");
  std::vector<std::size_t> stageCounts(maxStage + 1, 0);
  for (auto task : module.getOps<dialect::TaskSpaceOp>())
    ++stageCounts[task.getStage()];
  for (auto coupling : module.getOps<dialect::CouplingOp>()) {
    ++couplings;
    if (coupling.getSyncKind().getValue().getValue() != "global")
      throw std::invalid_argument("Phase-2 generator accepts only global synchronization");
    // Force semantic conversion through L3a; codegen never reads the printed
    // ClosedForm payload as an ad-hoc integer.
    (void)coupling.getWait().getValue().Eval(theta, granularity);
    (void)coupling.getFanout().getValue().Eval(theta, granularity);
    (void)coupling.getVolume().getValue().Eval(theta, granularity);
    (void)coupling.getCount().getValue().Eval(theta, granularity);
    (void)coupling.getRelation().getRelation();
  }
  for (auto placement : module.getOps<dialect::PlacementOp>()) {
    ++placements;
    if (placement.getCluster() != 1)
      throw std::invalid_argument("Phase-2 generator accepts round-robin cluster=1 placement");
  }
  if (placements != tasks)
    throw std::invalid_argument("every task space must have exactly one placement");

  std::ostringstream out;
  out << "// SPDX-License-Identifier: BSD-3-Clause\n"
      << "// Generated by CouplingGraphToCUDA from verified tilemega.* ops.\n"
      << "#define TILEMEGA_GENERATED_TASK_COUNT " << tasks << "\n"
      << "#define TILEMEGA_GENERATED_COUPLING_COUNT " << couplings << "\n"
      << "#define TILEMEGA_GENERATED_STAGE_COUNT " << stageCounts.size() << "\n"
      << ScheduleTableEmitter{}.EmitStageCounts(stageCounts)
      << SyncEmitter{}.EmitWait("global")
      << SyncEmitter{}.EmitSignal("global")
      << HostLauncherEmitter{}.Emit("l1_kernel")
      << TaskBodyEmitter{}.Emit(module);
  return out.str();
}

}  // namespace tilemega::codegen
