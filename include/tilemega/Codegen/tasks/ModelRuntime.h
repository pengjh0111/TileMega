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
#include <tilemega/Codegen/RuntimeOwnership.h>
#include <tilemega/Codegen/AttentionPlan.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <cutlass/bfloat16.h>

#include <cstdint>

namespace tilemega::codegen {

enum class ScalarType : std::uint32_t { kF32 = 0, kBF16 = 1 };

#ifndef TILEMEGA_FP32_PARTIALS
#define TILEMEGA_FP32_PARTIALS 1
#endif

#if defined(TILEMEGA_MODEL_BF16) && TILEMEGA_MODEL_BF16
using ModelElement = cutlass::bfloat16_t;
inline constexpr ScalarType kCompiledScalarType = ScalarType::kBF16;
#else
using ModelElement = float;
inline constexpr ScalarType kCompiledScalarType = ScalarType::kF32;
#endif

#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
using ModelPartialElement = float;
#else
using ModelPartialElement = ModelElement;
#endif

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

/// One GEMM implementation selected by a runtime model variant.  The tile
/// shape is repeated here deliberately: C++ templates still require the
/// generator to instantiate a finite set of collectives, but ModelSpec is the
/// sole runtime source of truth for both the selected instantiation and the
/// granularity against which dependency windows were derived.
struct GemmRuntimeDesc {
  std::uint16_t compiled_variant;
  std::uint16_t split_k;
  std::uint16_t tile_m;
  std::uint16_t tile_n;
  std::uint16_t tile_k;
  std::uint16_t stages;
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
///
/// `map` says which producer tasks the consumer CTA actually waits for.
/// `kAll` is the I2 relaxation (the whole launch axis); the other two carry
/// the window fitted in the frontend (Analysis/DependencyForm.h): consumer
/// task `c` reads only producer tasks
/// `[(c / div) * scale + offset, ... + count)`, intersected with the
/// producer's live range. `kIdentity` is the `div=scale=count=1, offset=0`
/// case, kept as its own value so the emitted table stays readable.
/// A window is only ever emitted for an edge whose two TaskBodies both
/// declare `kTilePerBlock`, which is what makes `blockIdx.x` name the same
/// task on both ends.
struct StageDependency {
  std::uint32_t producer;
  std::uint32_t consumer;
  enum class Map : std::uint32_t {
    kIdentity = 0,
    kAll = 1,
    kWindow = 2
  } map;
  std::uint32_t div;
  std::int32_t scale;
  std::int32_t offset;
  std::uint32_t count;
};

/// One stage in the solver-produced order for a runtime variant.  Task counts
/// remain symbolic until launch, so codegen emits this compact schedule
/// program and the host materializes one concrete TaskRef queue per worker
/// after binding seq/past and the resident grid.
struct ScheduleStageDesc {
  std::uint32_t stage;
  std::uint32_t dependency_begin;
  std::uint32_t dependency_count;
};

/// One unique event a concrete task still has to observe.  The host removes
/// duplicates both inside a task and against earlier waits in the same worker
/// queue: epochs are monotone, so an event satisfied once stays satisfied.
struct TaskWait {
  std::uint32_t producer;
  std::uint32_t group;
};

/// Sentinel used by TaskWait for the producer-stage completion event.  A
/// positive kappa also owns fine logical-task groups; kAll edges wait on this
/// aggregate instead of expanding one poll per producer task.
inline constexpr std::uint32_t kWholeStageEventGroup =
    ~static_cast<std::uint32_t>(0);

enum EventRowFlag : std::uint32_t {
  kNeedsAggregateEvent = 1u << 0,
  kNeedsFineEvents = 1u << 1,
};

enum TaskRefFlag : std::uint32_t {
  kLastTaskOfStage = 1u << 0,
};

/// A concrete queue item consumed by the L2 persistent kernel.
struct TaskRef {
  std::uint32_t stage;
  std::uint32_t logical_task;
  /// The original CG incoming-edge interval, retained for diagnostics and to
  /// make the schedule/dependency relationship explicit.
  std::uint32_t dependency_begin;
  std::uint32_t dependency_count;
  /// The deduplicated event interval actually polled by this task.
  std::uint32_t wait_begin;
  std::uint32_t wait_count;
  std::uint32_t flags;
};

/// Optional instrumentation.  The sequence numbers come from one global
/// atomic counter, so unlike per-SM clock64 values they define a comparable
/// order across the whole device.
struct TaskTrace {
  unsigned long long start;
  unsigned long long end;
};

/// A tensor the harness downloads and compares against the L0 reference.
struct OutputDesc {
  std::uint32_t buffer;
  char const* file;  ///< reference fixture
};

/// All granularity-dependent data for one runtime interval.  In particular,
/// a dependency table never floats free of the GEMM plan that produced it.
struct RuntimeVariantDesc {
  GemmRuntimeDesc const* gemms;  ///< gemm_count entries
  StageDependency const* dependencies;
  std::uint32_t dependency_count;
  std::uint32_t const* dependency_offsets;  ///< stage_count + 1 entries
  ScheduleStageDesc const* schedule;
  std::uint32_t schedule_count;
  /// Maximum producer-to-consumer distance in the emitted stage order.  A
  /// non-positive dependency is rejected by codegen.  The runtime separately
  /// checks concrete worker-order spans after dimensions and placement bind;
  /// stage distance alone is not an over-residency proof.
  std::uint32_t max_dependency_span;
  std::uint32_t seq_begin;  ///< inclusive, for diagnostics
  std::uint32_t seq_end;    ///< inclusive, for diagnostics
  std::uint32_t ownership_flags;
  AttentionRuntimeRecord const* attention = nullptr;  ///< stage_count entries if present
  bool resident_only = true;
  bool balanced_placement = false;
  RuntimeExactDependencyDesc const* exact_dependencies = nullptr;
};

#ifndef TILEMEGA_EVENT_SPLIT_LINES
#define TILEMEGA_EVENT_SPLIT_LINES 0
#endif
/// T1.2: arrivals and epoch can each occupy their own 128 B line. Every user
/// addresses the named fields and allocates by sizeof(EventCounter).
struct alignas(128) EventCounter {
  unsigned long long arrivals;
#if TILEMEGA_EVENT_SPLIT_LINES
  alignas(128) unsigned long long epoch;
#else
  unsigned long long epoch;
  unsigned char padding[112];
#endif
};
static_assert(sizeof(EventCounter) == (TILEMEGA_EVENT_SPLIT_LINES ? 256 : 128),
              "event cache-line padding");
static_assert(alignof(EventCounter) == 128, "event cache-line alignment");

#ifndef TILEMEGA_EVENT_SHARDED
#define TILEMEGA_EVENT_SHARDED 0
#endif
#ifndef TILEMEGA_EVENT_SHARDS
#define TILEMEGA_EVENT_SHARDS 0
#endif
#ifndef TILEMEGA_EVENT_CLUSTER_FANIN
#define TILEMEGA_EVENT_CLUSTER_FANIN 0
#endif
#ifndef TILEMEGA_EVENT_CLUSTER_RESERVE
#define TILEMEGA_EVENT_CLUSTER_RESERVE TILEMEGA_EVENT_CLUSTER_FANIN
#endif
#ifndef TILEMEGA_EVENT_RELEASE_STORE
#define TILEMEGA_EVENT_RELEASE_STORE 0
#endif
static_assert(!TILEMEGA_EVENT_RELEASE_STORE,
              "T1.4 disabled pending a complete multi-level release proof");
static_assert(!TILEMEGA_EVENT_CLUSTER_FANIN || TILEMEGA_EVENT_SHARDED,
              "cluster fan-in requires the two-level event protocol");
struct alignas(128) ArrivalCounter {
  unsigned long long arrivals;
};
static_assert(sizeof(ArrivalCounter) == 128, "one first-level counter per line");
struct EventFanIn {
  std::uint32_t begin, modulus, nonempty, first_cluster;
};

/// The device-side view of a model.  Every large table is reached through a
/// device pointer (F-17b); nothing is passed by value into the kernel.
/// The device-side view of a model.  Every table is reached through a device
/// pointer (F-17b); nothing large is passed by value into the kernel.
struct Params {
  ModelDims dims;
  ModelElement** buffers;     ///< buffer id -> device pointer
  void const* gemms;          ///< GemmInvocation const*, opaque to this header
  StageDesc const* stages;
  std::uint32_t stage_count;
  StageDependency const* dependencies;
  std::uint32_t dependency_count;
  /// `dependency_offsets[stage] .. dependency_offsets[stage + 1]` is the
  /// slice of `dependencies` whose consumer is `stage`. Length is
  /// `stage_count + 1`.
  std::uint32_t const* dependency_offsets;
  TaskRef const* schedule;
  std::uint32_t schedule_count;
  /// `schedule_offsets[worker] .. schedule_offsets[worker+1]` is that
  /// physical worker's ordered queue. Length is grid + 1.
  std::uint32_t const* schedule_offsets;
  TaskWait const* task_waits;
  std::uint32_t task_wait_count;
  /// Prefix sum of L2 event groups per stage. L1 owns the first
  /// `stage_count` EventCounter rows; L2 rows begin after that region.
  std::uint32_t const* event_offsets;
  /// Per-stage EventRowFlag mask. Producers publish only rows referenced by
  /// at least one consumer, avoiding an unconditional second atomic stream.
  std::uint32_t const* event_flags;
  TaskTrace* task_trace;                 ///< nullptr unless profiling
  unsigned long long* trace_sequence;    ///< nullptr unless profiling
  std::uint32_t ownership_flags;
  EventFanIn const* event_fanin;
  ArrivalCounter* shard_arrivals;
  std::uint32_t const* shard_targets;
  std::uint32_t event_shard_count;
  std::uint32_t const* shard_local_offsets;
  std::uint32_t const* cluster_shard_offsets;
  std::uint32_t const* cluster_shard_indices;
};

/// Everything the generator emits about one model.  The harness reads only
/// this; it never names a tensor, a stage or a dimension itself.
struct ModelSpec {
  ModelDims dims;
  ScalarType dtype;
  BufferDesc const* buffers;
  std::uint32_t buffer_count;
  GemmDesc const* gemms;
  std::uint32_t gemm_count;
  StageDesc const* stages;
  std::uint32_t stage_count;
  OutputDesc const* outputs;
  std::uint32_t output_count;
  RuntimeVariantDesc const* runtime_variants;
  std::uint32_t runtime_variant_count;
  /// Direct seq -> runtime-variant table. Entry zero is unused because seq is
  /// positive; seq >= seq_variant_count is rejected instead of silently
  /// selecting a conservative plan. Selection is therefore strict O(1).
  std::uint16_t const* seq_variant;
  std::uint32_t seq_variant_count;
};

}  // namespace tilemega::codegen
