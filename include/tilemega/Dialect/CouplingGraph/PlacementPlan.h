// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.1 (Plan), §5.7.4 (who writes and who consumes it),
//                §8.11 (only the solver decides a schedule).
//
// The names and parameter arities of the placement plan carried by
// `tilemega.placement`.  The dialect verifier, codegen and the host
// materializer must agree on exactly this table, so it lives here rather than
// as three copies of a string comparison.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tilemega::dialect {

/// How the solver produces (pi, sigma) for one task space.
enum class PlacementMode {
  /// pi(task) = task % grid, queues stage-major.  What `map = [0]` alone has
  /// always meant, and the byte-identity reference for the plan contract.
  kLegacyGridStride = 0,
  /// pi(stage, task) = (task + base[stage]) % grid, `base` the prefix sum of
  /// the active task counts along the stage order, mod grid.
  kRotate,
  /// Affinity-first greedy under a queue cap.
  kBalanced,
  /// Earliest-finish-time list scheduling.  Materialized: the explicit
  /// (worker, slot) table travels in `kPlacementTableAttr` below.
  kEft,
  /// A closed form selected by `params[0]`; see `PlacementTemplate`.
  kTemplate,
};

/// `params[0]` of `kTemplate`.  Both forms come from the offline probe that
/// measured them (F-93, F-97).
enum class PlacementTemplate {
  /// Contiguous tiling of the outermost parallel band coordinate.
  kBand = 0,
  /// The second schedule dimension modulo the worker count.
  kWavefront = 1,
};

/// The module attribute an `eft` Plan's materialized (pi, sigma) travels in.
/// It is not on `tilemega.placement`: at a bound theta the table is one object
/// for the whole model, indexed by flat runtime node id, and every task space
/// would otherwise have to carry a copy of all of it.
inline constexpr char kPlacementTableAttr[] = "tilemega.placement_table";

/// The materialized form of a Plan (§5.7.1): pi as `worker` and sigma as
/// `slot`, per runtime node in flat node-id order, plus the theta and grid the
/// schedule was computed for.  The mode that produces it reads task durations,
/// which no layer below L2 has, so it can only travel already solved -- and
/// therefore only ever fits the one theta named here.
struct PlacementTable {
  std::vector<int> worker;
  std::vector<int> slot;
  long long seq = 0;
  long long past = 0;
  long long grid = 0;
};

/// False, with `*error` set, unless the table is a dense (pi, sigma) for the
/// grid it names: equal non-empty lengths, a positive grid and theta, every pi
/// inside the grid, and sigma a dense [0, n) per worker.  Callers treat that as
/// fatal (H5) -- a table that fails here was computed for a different bound
/// theta, and there is no cost model below L2 to recompute it with.
bool ValidatePlacementTable(PlacementTable const& table, std::string* error);

/// The attribute spelling, or nullptr for an out-of-range value.
char const* PlacementModeName(PlacementMode mode);
/// True when `name` is a known mode, in which case `*mode` is set.
bool ParsePlacementMode(char const* name, std::size_t length, PlacementMode* mode);

/// How many entries `params` must carry for `mode`.
std::size_t PlacementModeParamCount(PlacementMode mode);

/// The only dispatch policy the executor implements; `jit` is reserved for
/// duration-dependent stages (§5.7.1) and is rejected until EX-E5.
inline constexpr char kPlacementPolicyAot[] = "aot";

/// The only execution window the executor implements (§5.7.2 W = 1, strict
/// FIFO).  The dialect admits any positive W so that EX-E2 changes the
/// executor rather than the contract.
inline constexpr long kPlacementWindowImplemented = 1;

}  // namespace tilemega::dialect
