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

// The model's own `rms_norm_eps`, generated from the imported config. The
// default exists only so that a `.cu` generated before the epsilon became
// model data keeps its previous behaviour; it is never a dtype- or
// model-specific choice made here. `ModelSpec::norm_epsilon` carries the same
// number and the harness refuses a model whose table disagrees with it.
#ifndef TILEMEGA_NORM_EPSILON
#define TILEMEGA_NORM_EPSILON 1.0e-6f
#endif

// Whether the rotary phase is built in FP32: the position, the angle and the
// frequency table itself. Generated from the imported model, because the
// exported graph decides it -- a frequency table exported in model storage
// rounds the angle to that storage, one exported in FP32 does not.
#ifndef TILEMEGA_ROPE_FP32_PHASE
#define TILEMEGA_ROPE_FP32_PHASE 0
#endif

// Whether a GEMM element whose FP32 accumulator sits near a BF16 rounding
// boundary is recomputed in FP64 before it is rounded. Off by default: at zero
// refinements the generated code is what it was before the pass existed, which
// is what the default-build SASS identity checks.
#ifndef TILEMEGA_MIDPOINT_REFINE
#define TILEMEGA_MIDPOINT_REFINE 0
#endif

// How close to the boundary counts as near, in units of the BF16 ulp at that
// magnitude. The FP32 K-loop error grows with the running partial sums rather
// than with the result, so the guard is relative to the ulp and not absolute.
#ifndef TILEMEGA_MIDPOINT_GUARD
#define TILEMEGA_MIDPOINT_GUARD 0.015625f
#endif

// Whether this model normalizes queries and keys per head. Generated for the
// same reason as the gather below: a model without them must compile to what
// it compiled to before the family existed.
#ifndef TILEMEGA_QK_NORM_RUNTIME
#define TILEMEGA_QK_NORM_RUNTIME 0
#endif

// Whether this model gathers a token embedding. Generated: a model that starts
// at hidden states must compile to what it compiled to before the gather
// existed, which is what the default-build SASS identity checks.
#ifndef TILEMEGA_EMBEDDING_RUNTIME
#define TILEMEGA_EMBEDDING_RUNTIME 0
#endif

// The width of one token identifier in the generated buffer table. Generated
// from the imported model, because the exported index tensor decides it; the
// default matters only for a model that never gathers an embedding.
#ifndef TILEMEGA_TOKEN_ID_BITS
#define TILEMEGA_TOKEN_ID_BITS 32
#endif

// Ablation only, never generated: keeps `cosf`/`sinf` in FP32 instead of
// rounding them once to model storage. The reference implementations cast the
// cosine and sine to the model dtype before multiplying, so the rounded form
// is the faithful one; this switch exists to measure the difference.
#ifndef TILEMEGA_ROPE_FP32_TRIG
#define TILEMEGA_ROPE_FP32_TRIG 0
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
#if TILEMEGA_PREFETCH_RUNTIME
  /// No stage writes this buffer, so a task may fetch it before its
  /// dependencies resolve. Derived by the frontend from the write relation
  /// the producer edges come from; the generator never annotates it.
  bool no_producer;
#endif

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

/// The §5.7.1 Plan, as codegen hands it to the host.  Only pi and sigma are
/// decisions this round: `window` is W and the executor implements W = 1 only
/// (§5.7.2), `policy` is `aot`, and `sync`/`kappa` still ride on the coupling
/// attributes and `TILEMEGA_EVENT_KAPPA`.
///
/// The defaults are the legacy grid-stride plan, so a variant that carries no
/// plan reads exactly as the pre-plan generator emitted it.
///
/// `mode` is `tilemega::dialect::PlacementMode` and `policy` is fixed at 0
/// (`aot`); the names live in the dialect header because the CG is the layer
/// that decides them, and that header pulls in no MLIR.
struct RuntimePlanDesc {
  std::uint32_t mode = 0;
  std::int64_t const* params = nullptr;  ///< param_count entries, mode specific
  std::uint32_t param_count = 0;
  std::uint32_t window = 1;
  std::uint32_t policy = 0;
  /// `eft` only: pi and sigma per runtime node, flat node ids, `eft_nodes`
  /// entries each.  That mode prices task durations, so it cannot be evaluated
  /// here and travels already solved -- which pins it to the one theta and grid
  /// below.  The host compares all four and refuses a mismatch; it has no cost
  /// model with which to recompute the schedule (§5.7.1, §5.7.4).
  std::int32_t const* eft_worker = nullptr;
  std::int32_t const* eft_slot = nullptr;
  std::uint32_t eft_nodes = 0;
  std::uint32_t eft_seq = 0;
  std::uint32_t eft_past = 0;
  std::uint32_t eft_grid = 0;
  RuntimePlanDesc const* interval = nullptr;
  std::uint32_t interval_count = 0;
  std::uint32_t interval_begin = 0;
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
  /// Reserved, and zero in every schedule the harness builds: the one flag that
  /// ever lived here was written and never read.  The word stays so the queue
  /// item keeps the size and alignment the device kernel was compiled against.
  std::uint32_t flags;
};

/// Optional instrumentation.  The sequence numbers come from one global
/// atomic counter, so unlike per-SM clock64 values they define a comparable
/// order across the whole device.
struct TaskTrace {
  unsigned long long start;
  unsigned long long end;
};

#ifndef TILEMEGA_TRACE_PHASE
#define TILEMEGA_TRACE_PHASE 0
#endif

#ifndef TILEMEGA_TRACE_KLOOP
#define TILEMEGA_TRACE_KLOOP 0
#endif
#if TILEMEGA_TRACE_KLOOP && !TILEMEGA_TRACE_PHASE
#error "TILEMEGA_TRACE_KLOOP requires TILEMEGA_TRACE_PHASE"
#endif

// SIMT-side exposed wait (B0).  `TILEMEGA_TRACE_KLOOP` accounts for the waits
// inside an instrumented GEMM K-loop; this accounts for the waits inside the
// SIMT bodies, so that a wait share can be quoted against the whole pipeline
// instead of against the GEMM mainloops alone.  Only barriers the body already
// executed are timed: no barrier, atomic or polling-loop store is added.
#ifndef TILEMEGA_TRACE_SIMT
#define TILEMEGA_TRACE_SIMT 0
#endif
#if TILEMEGA_TRACE_SIMT && !TILEMEGA_TRACE_PHASE
#error "TILEMEGA_TRACE_SIMT requires TILEMEGA_TRACE_PHASE"
#endif

// Opt-in EX-S3 launch bound. Zero preserves occupancy-selected residency.
#ifndef TILEMEGA_RESIDENCY_CAP
#define TILEMEGA_RESIDENCY_CAP 0
#endif

#ifndef TILEMEGA_TRACE_V2
#define TILEMEGA_TRACE_V2 0
#endif

/// EX-D3 slot-private task phase boundaries, in both timer domains.
struct TaskPhase {
  unsigned long long ns[6]; // run, setup, first operand, mainloop, body return, run end
  unsigned long long cycles[6];
#if TILEMEGA_TRACE_KLOOP
  unsigned long long loop_begin_cycles, loop_end_cycles;
  unsigned long long operand_wait_cycles, iterations;
#endif
#if TILEMEGA_TRACE_SIMT
  /// Cycles thread zero spent inside the body's own `__syncthreads()` calls,
  /// and how many of them it passed through.  A thread that arrives last waits
  /// for nobody, so this is a lower bound on the CTA's idle time.
  unsigned long long simt_wait_cycles, simt_barriers;
#endif
};

/// Trace v2 (EX-D1).  Where TaskTrace orders task boundaries, this records
/// when they happened, so a per-hop latency can be reconstructed offline as
/// `ready` minus the producing event's publish stamp.  The field order is the
/// offline scripts' contract; the two clock64 columns are appended because the
/// measured %globaltimer tick is 1024 ns (TRACE_V2/resolution.md), far too
/// coarse for a task duration.
struct TaskTraceV2 {
  unsigned long long wait_begin;   ///< before WaitTaskDependencies
  unsigned long long ready;        ///< after WaitTaskDependencies returns
  unsigned long long run_begin;    ///< after the pre-run barrier
  unsigned long long run_end;      ///< after RunTask and its barrier
  unsigned long long publish_end;  ///< after NotifyTask returns
  unsigned int smid;               ///< %smid at run_begin
  unsigned int worker;             ///< blockIdx.x
  unsigned int stage;
  unsigned int logical_task;
  unsigned long long run_begin_clk;  ///< clock64 beside run_begin, same SM
  unsigned long long run_end_clk;    ///< clock64 beside run_end, same SM
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
  /// Last, and defaulted, so a legacy variant's initializer is unchanged.
  RuntimePlanDesc plan = {};
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
#ifndef TILEMEGA_UNSAFE_NO_NOTIFY_FENCE
#define TILEMEGA_UNSAFE_NO_NOTIFY_FENCE 0
#endif
#ifndef TILEMEGA_RELEASE_AFTER_BARRIER
#define TILEMEGA_RELEASE_AFTER_BARRIER 0
#endif
#ifndef TILEMEGA_ASYNC_PUBLISH
#define TILEMEGA_ASYNC_PUBLISH 0
#endif
#ifndef TILEMEGA_LOCAL_DEP_SMEM
#define TILEMEGA_LOCAL_DEP_SMEM 0
#endif
#ifndef TILEMEGA_CLUSTER_ARRIVE
#define TILEMEGA_CLUSTER_ARRIVE 0
#endif
static_assert(!TILEMEGA_EVENT_RELEASE_STORE,
              "T1.4 disabled pending a complete multi-level release proof");
static_assert(!TILEMEGA_EVENT_CLUSTER_FANIN || TILEMEGA_EVENT_SHARDED,
              "cluster fan-in requires the two-level event protocol");

// EX-E3 step 1: the executor's five per-task `__syncthreads()` (§5.5.1) cut to
// the two that carry an ordering nothing else carries -- the acquire after the
// wait, which is also the §8.6 union's WAR barrier, and the §8.5 release before
// thread 0 publishes.  Off by default (H2).  Trace v2 keeps the pair around
// RunTask: `run_begin`/`run_end` are thread 0 stamps and without the barriers
// they would time thread 0 rather than the CTA, so a traced v2 build has four
// and its task durations are the wider measurement -- recorded, not hidden.
#ifndef TILEMEGA_BARRIER_V2
#define TILEMEGA_BARRIER_V2 0
#endif
static_assert(!TILEMEGA_ASYNC_PUBLISH ||
                  (TILEMEGA_RELEASE_AFTER_BARRIER && TILEMEGA_BARRIER_V2),
              "async publication requires the single release and next-wait convergence");

// EX-E3 step 2: an event whose producer is a single CTA needs no arrival
// count -- that CTA is by construction the last arriver, so the atomic only
// ever returns `iteration` and the completion test is already true.  It
// publishes the epoch directly instead.  Off by default (H2).
#ifndef TILEMEGA_EVENT_SOLO
#define TILEMEGA_EVENT_SOLO 0
#endif

// EX-E3 step 3: the arrival add's return value is used for one thing only --
// finding the last arriver.  Dropping it makes the add a return-value-free
// release reduction and moves the completion test to the consumer, which
// polls `arrivals` against `triggers x (iteration + 1)`.  That counter is
// monotone across iterations exactly as `epoch` is (§8.2).  Off by default
// (H2).
#ifndef TILEMEGA_EVENT_RED_PUBLISH
#define TILEMEGA_EVENT_RED_PUBLISH 0
#endif

// EX-E2 (§5.7.2): the execution window.  W = 1 is strict FIFO, the only shape
// the executor implemented before this step; a larger W lets a worker start any
// of the next W slots whose waits are already satisfied, which is the only way
// the head-of-line stall time F-134 measured becomes reclaimable.  Off by
// default (H2).
//
// The bound is compile time because the executor's local-dependency mask is
// W - 1 bits wide; the runtime value comes from the Plan and nothing else
// (§8.11).  `kSlotWindowMax` is deliberately macro independent: RuntimePlanDesc
// is also compiled into libtilemega, through Solver/ModelDescription.h and
// lib/Target/Calibration.cu, which never see a model's macros -- a
// macro-dependent constant there would give the library and the model two
// different meanings for one struct.
#ifndef TILEMEGA_SLOT_WINDOW
#define TILEMEGA_SLOT_WINDOW 1
#endif
inline constexpr std::uint32_t kSlotWindowMax = 4;
static_assert(TILEMEGA_SLOT_WINDOW >= 1 &&
                  TILEMEGA_SLOT_WINDOW <= kSlotWindowMax,
              "the executor implements 1 <= W <= kSlotWindowMax");
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
#if TILEMEGA_TRACE_V2 || TILEMEGA_TRACE_PHASE
  /// Guarded so a default build keeps the layout, and therefore the constant
  /// bank offsets and the SASS, it had before trace v2 existed (H2).
#if TILEMEGA_TRACE_PHASE
  TaskPhase* task_phase = nullptr;
#endif
  TaskTraceV2* task_trace_v2;            ///< nullptr unless TILEMEGA_TRACE_V2=1
  /// Stamped by the last arriver of each event; the sole source of hop times.
  unsigned long long* event_publish;     ///< length event_count
#endif
  std::uint32_t ownership_flags;
  EventFanIn const* event_fanin;
  ArrivalCounter* shard_arrivals;
  std::uint32_t const* shard_targets;
  std::uint32_t event_shard_count;
  std::uint32_t const* shard_local_offsets;
  std::uint32_t const* cluster_shard_offsets;
  std::uint32_t const* cluster_shard_indices;
#if TILEMEGA_SLOT_WINDOW > 1
  /// Per-slot mask of this worker's own predecessors inside the window: bit k
  /// means slot - (k + 1) must be complete before this slot may start.  These
  /// are the same-worker edges W > 1 stops FIFO from discharging, so the host
  /// emits them instead of eliding them (§5.7.3 L-d, R3 H4).  Last, and
  /// guarded, so a default build keeps the layout and therefore the SASS (H2).
  std::uint32_t const* slot_local_deps;
  /// W as the Plan states it.  The executor reads it from here and never from
  /// TILEMEGA_SLOT_WINDOW, which only bounds what this build can implement.
  std::uint32_t window;
#endif
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
  /// Defaulted so that the fixed pre-generated sources used for the SASS
  /// identity check keep compiling unchanged.
  float norm_epsilon = TILEMEGA_NORM_EPSILON;
};

}  // namespace tilemega::codegen
