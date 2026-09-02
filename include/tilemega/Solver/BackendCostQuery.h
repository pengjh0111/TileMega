// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §1.2 principle 2 (solver <-> backend cost channel),
//                §4.2 candidate legality, P4.1 cost query interface.
#pragma once

#include <tilemega/Target/TargetSpec.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tilemega::solver {

struct ClusterShape {
  int m = 1, n = 1, k = 1;
  constexpr int size() const { return m * n * k; }
};

struct AlignmentRequirement {
  int a = 1, b = 1;
};

/// Everything a candidate answers from compile-time traits alone: no device
/// code is generated and ptxas never runs, so a whole family costs one
/// translation unit (§1.2, tier 1).
struct BackendTraits {
  int tile_m = 0, tile_n = 0, tile_k = 0;
  int stages = 0;
  int threads = 0;
  int smem_bytes = 0;
  ClusterShape cluster;
  AlignmentRequirement alignment;
  int arch_sm = 0;  ///< minimum SM version, as 10 * major + minor
  bool shape_legal = false;
};

/// One point of the implementation search space, as the solver sees it.
///
/// Register pressure is deliberately not part of the trait record: CUTLASS's
/// constexpr traits do not carry it and no closed form predicts it, so
/// `estimatedRegisters()` stays empty until a tier-3 compile hands over real
/// ptxas output.
class BackendCandidate {
 public:
  BackendCandidate() = default;
  explicit BackendCandidate(BackendTraits traits) : traits_(traits) {}

  bool isLegal(TargetSpec const& target) const;
  int threads() const { return traits_.threads; }
  int smemBytes() const { return traits_.smem_bytes; }
  ClusterShape clusterShape() const { return traits_.cluster; }
  AlignmentRequirement alignmentRequirement() const { return traits_.alignment; }
  int architectureRequirement() const { return traits_.arch_sm; }
  std::optional<int> estimatedRegisters() const { return registers_; }

  /// Tier 3: take the maximum register count over the entry points named in a
  /// real `nvcc -Xptxas=-v` log. Returns false and leaves the candidate
  /// unchanged when the log names none, so a failed compile cannot be mistaken
  /// for a zero-register kernel.
  bool RecordPtxas(std::string_view ptxas_log, std::string_view entry_filter = {});

  BackendTraits const& traits() const { return traits_; }
  /// How many of these fit on one SM given the target's budgets; 0 when the
  /// candidate does not fit at all.
  int CoResidentPerSM(TargetSpec const& target) const;
  std::string Describe() const;

 private:
  BackendTraits traits_;
  std::optional<int> registers_;
};

/// Closed forms for the SIMT f32 TN family the megakernel dispatches to.
/// `Backend/CutlassGemmCandidate.h` static_asserts each of them against the
/// instantiated CUTLASS collective, which is what lets the host-side
/// enumerator prune without compiling anything.
constexpr int kSimtF32Threads = 256;
constexpr int kSimtF32ArchSm = 80;

constexpr int SimtF32SmemBytes(int m, int n, int k, int stages) {
  return 4 * stages * k * ((m + 1) + (n + 1));
}

constexpr bool SimtF32ShapeLegal(int m, int n, int k, int stages) {
  int const copy_k = k < 16 ? k : 16;
  int const copy_m = copy_k > 0 ? kSimtF32Threads / copy_k : 0;
  return m > 0 && n > 0 && k > 0 && stages > 1 && m % 16 == 0 && n % 16 == 0 &&
         copy_k > 0 && kSimtF32Threads % copy_k == 0 && k % copy_k == 0 &&
         copy_m > 0 && m % copy_m == 0 && n % copy_m == 0;
}

BackendTraits SimtF32Traits(int m, int n, int k, int stages);

/// `entry -> registers` for every entry point in a `-Xptxas=-v` log.
std::vector<std::pair<std::string, int>> ParsePtxasRegisters(std::string_view log);

}  // namespace tilemega::solver
