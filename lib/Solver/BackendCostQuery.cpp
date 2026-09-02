// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/BackendCostQuery.h>

#include <algorithm>
#include <cctype>

namespace tilemega::solver {
namespace {

int SmVersion(TargetSpec const& target) {
  return 10 * target.sm_major + target.sm_minor;
}

}  // namespace

bool BackendCandidate::isLegal(TargetSpec const& target) const {
  if (!traits_.shape_legal || traits_.smem_bytes <= 0 || traits_.threads <= 0)
    return false;
  if (traits_.smem_bytes > target.res.max_dynamic_smem_per_cta) return false;
  if (traits_.threads > target.res.max_threads_per_sm) return false;
  // sm_89 does not run an sm_90 collective, and sm_90 does not run an sm_100
  // one; the comparison is on the target's own version, never on a literal.
  if (traits_.arch_sm > SmVersion(target)) return false;
  if (traits_.cluster.size() > 1 &&
      (!target.caps.cluster || traits_.cluster.size() > target.res.max_cluster_size))
    return false;
  // A register count only exists after tier 3; when it does, it is a legality
  // input like any other budget.
  if (registers_ && *registers_ * traits_.threads > target.res.regs_per_sm)
    return false;
  return true;
}

int BackendCandidate::CoResidentPerSM(TargetSpec const& target) const {
  if (!isLegal(target)) return 0;
  int by_smem = traits_.smem_bytes > 0
                    ? target.res.max_smem_per_sm / traits_.smem_bytes
                    : 0;
  int by_threads = target.res.max_threads_per_sm / traits_.threads;
  int limit = std::min(by_smem, by_threads);
  if (registers_ && *registers_ > 0)
    limit = std::min(limit,
                     target.res.regs_per_sm / (*registers_ * traits_.threads));
  return std::max(0, limit);
}

bool BackendCandidate::RecordPtxas(std::string_view ptxas_log,
                                   std::string_view entry_filter) {
  int best = -1;
  std::string entry;
  for (auto const& [name, registers] : ParsePtxasRegisters(ptxas_log)) {
    if (!entry_filter.empty() && name.find(entry_filter) == std::string::npos)
      continue;
    if (registers > best) {
      best = registers;
      entry = name;
    }
  }
  if (best < 0) return false;
  registers_ = best;
  return true;
}

std::string BackendCandidate::Describe() const {
  return std::to_string(traits_.tile_m) + "x" + std::to_string(traits_.tile_n) +
         "x" + std::to_string(traits_.tile_k) + "s" +
         std::to_string(traits_.stages);
}

BackendTraits SimtF32Traits(int m, int n, int k, int stages) {
  BackendTraits traits;
  traits.tile_m = m;
  traits.tile_n = n;
  traits.tile_k = k;
  traits.stages = stages;
  traits.threads = kSimtF32Threads;
  traits.shape_legal = SimtF32ShapeLegal(m, n, k, stages);
  traits.smem_bytes = traits.shape_legal ? SimtF32SmemBytes(m, n, k, stages) : 0;
  traits.arch_sm = kSimtF32ArchSm;
  // cp.async moves one f32 per thread here, so neither operand needs more than
  // element alignment; the field exists because §4.2 makes it part of the
  // query, not because this family constrains it.
  traits.alignment = {1, 1};
  traits.cluster = {1, 1, 1};
  return traits;
}

std::vector<std::pair<std::string, int>> ParsePtxasRegisters(
    std::string_view log) {
  // ptxas -v prints one "Compiling entry function '<name>' for '<arch>'"
  // followed, after the function-properties block, by "Used N registers".
  std::vector<std::pair<std::string, int>> result;
  std::string current;
  std::size_t position = 0;
  auto next_line = [&](std::string_view* line) {
    if (position >= log.size()) return false;
    std::size_t end = log.find('\n', position);
    if (end == std::string_view::npos) end = log.size();
    *line = log.substr(position, end - position);
    position = end + 1;
    return true;
  };
  std::string_view line;
  while (next_line(&line)) {
    std::size_t entry = line.find("Compiling entry function '");
    if (entry != std::string_view::npos) {
      std::size_t start = entry + std::string_view("Compiling entry function '").size();
      std::size_t stop = line.find('\'', start);
      if (stop != std::string_view::npos)
        current = std::string(line.substr(start, stop - start));
      continue;
    }
    std::size_t used = line.find("Used ");
    if (used == std::string_view::npos || current.empty()) continue;
    std::size_t digit = used + 5;
    std::size_t end = digit;
    while (end < line.size() && std::isdigit(static_cast<unsigned char>(line[end])))
      ++end;
    if (end == digit || line.find(" registers", end) != end) continue;
    result.emplace_back(current, std::stoi(std::string(line.substr(digit, end - digit))));
    current.clear();
  }
  return result;
}

}  // namespace tilemega::solver
