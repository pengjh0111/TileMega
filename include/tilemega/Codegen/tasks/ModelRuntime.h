// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.2 (only the TaskBody is handwritten), §5.3 (TaskBody ABI),
// §8 (synchronization and launch rules).
//
// This header is model independent by construction: it contains no dimension,
// no stage sequence and no operator name.  Everything a model contributes
// reaches it as generated tables (`ModelDims`, `BufferDesc`, `GemmDesc`,
// `StageDesc`, `WeightDesc`), which the code generator emits from the CG.
#pragma once

#include <cuda_runtime.h>

#include <tilemega/Codegen/tasks/TaskBase.h>

#include <cstdint>

namespace tilemega::codegen {

/// The symbolic dimensions, and only those. Everything static about a model
/// (widths, head counts and the stage sequence) is generated data carried by
/// the tables below. These values are bound from the workload at launch, so
/// one compiled binary serves every token count (invariant I1).
struct ModelDims {
  int seq = 0;
  int past = 0;
  int total = 0;
};

/// A GEMM instance as the generator describes it: M is the token count, which
/// stays symbolic, so only N and K are model data.  `a`/`b`/`c`/`d` are buffer ids.
struct GemmDesc {
  int n, k;
  std::uint32_t a, b, c, d;
  float beta;
};

/// One buffer the model needs.  `elements` is filled in by the generator;
/// `source` says where its contents come from.
enum class BufferSource : std::uint32_t {
  kZero = 0,      ///< scratch, zero initialized
  kFixture = 1,   ///< loaded from `file`
  kWeight = 2,    ///< loaded from `file` (a parameter)
};

/// A buffer's size is a closed form in the symbolic dimensions, not a scalar:
/// elements = constant + per_seq * seq + per_past * past + per_total * total.
/// The three symbolic terms are kept separate rather than folded through
/// `past = total - seq`, so every coefficient stays non-negative and the size
/// reads back as the expression the generator derived.
struct BufferDesc {
  char const* name;
  std::uint32_t constant;
  std::uint32_t per_seq;
  std::uint32_t per_past;
  std::uint32_t per_total;
  BufferSource source;
  char const* file;  ///< nullptr for scratch

  std::size_t Elements(ModelDims const& dims) const {
    return constant + static_cast<std::size_t>(per_seq) * dims.seq +
           static_cast<std::size_t>(per_past) * dims.past +
           static_cast<std::size_t>(per_total) * dims.total;
  }
};

/// The task families the generator can dispatch to.  A model that needs none
/// of the attention families simply never emits those stage kinds.
enum class TaskKind : std::uint32_t {
  kGemm = 0,
  kRMSNorm = 1,
  kRoPE = 2,
  kKVAppend = 3,
  kElementwise = 4,
  kAttention = 5,
  /// The combiner half of a split reduction (§2.4). It exists only in an
  /// instantiated stage list: the generator emits one kGemm stage, and the
  /// split transform rewrites it into a partial stage plus this one.
  kGemmCombine = 6,
};

inline constexpr std::uint32_t kNoOperand = 0xffffffffu;

/// One generated stage. Geometry belongs to the stage rather than a model
/// type: `extent` is the number of rows/heads/elements per token, `width` is
/// the inner width and `group` is the grouped-axis fan-in. `operand` holds
/// buffer ids whose meaning is fixed by `kind` and documented on each body.
struct StageDesc {
  TaskKind kind;
  std::uint32_t gemm;     ///< index into the generated GEMM table
  std::uint32_t extent;
  std::uint32_t width;
  std::uint32_t group;
  std::uint32_t operand[8];
};

/// A synchronization requirement synthesized from CG couplings: consumer
/// stage `consumer` waits for stage `producer` to complete.
///
/// The generated table is sorted by `consumer`, and ModelSpec carries a
/// per-stage offset array into it, so a consumer reads only its own
/// incoming edges. Scanning the whole table per stage instead is what made
/// the first L2 measurably slower than the L1 grid barrier: thread 0 of
/// every CTA re-read all of it once per stage (55 x 30 x 128 global reads
/// on the 2-layer model) purely to find its own handful of edges.
struct StageDependency {
  std::uint32_t producer;
  std::uint32_t consumer;
  enum class Map : std::uint32_t { kIdentity = 0, kAll = 1 } map;
};

/// A tensor the harness downloads and compares against the L0 reference.
struct OutputDesc {
  std::uint32_t buffer;
  char const* file;  ///< reference fixture
};

/// §8.4: one 128 B line per event so two counters never share a line.
struct alignas(128) EventCounter {
  unsigned long long arrivals;
  unsigned long long epoch;
  unsigned char padding[112];
};
static_assert(sizeof(EventCounter) == 128, "event cache-line padding");

/// The device-side view of a model.  Every large table is reached through a
/// device pointer (F-17b); nothing is passed by value into the kernel.
/// The device-side view of a model.  Every table is reached through a device
/// pointer (F-17b); nothing large is passed by value into the kernel.
struct Params {
  ModelDims dims;
  float** buffers;            ///< buffer id -> device pointer
  void const* gemms;          ///< GemmInvocation const*, opaque to this header
  StageDesc const* stages;
  std::uint32_t stage_count;
  StageDependency const* dependencies;
  std::uint32_t dependency_count;
  /// `dependency_offsets[stage] .. dependency_offsets[stage + 1]` is the
  /// slice of `dependencies` whose consumer is `stage`. Length is
  /// `stage_count + 1`.
  std::uint32_t const* dependency_offsets;
};

/// Everything the generator emits about one model.  The harness reads only
/// this; it never names a tensor, a stage or a dimension itself.
struct ModelSpec {
  ModelDims dims;
  BufferDesc const* buffers;
  std::uint32_t buffer_count;
  GemmDesc const* gemms;
  std::uint32_t gemm_count;
  StageDesc const* stages;
  std::uint32_t stage_count;
  OutputDesc const* outputs;
  std::uint32_t output_count;
  StageDependency const* dependencies;
  std::uint32_t dependency_count;
  std::uint32_t const* dependency_offsets;  ///< stage_count + 1 entries
};

}  // namespace tilemega::codegen
