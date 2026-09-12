// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.2 (only TaskBodies are handwritten), §8 (sync and launch).
//
// The model-independent half of the runtime.  It contains no dimension, no
// stage sequence, no operator name and no layer count: a model reaches it only
// as the generated `ModelSpec` tables.  Adding a model means emitting new
// tables, never editing this file.
#pragma once

#include <cuda_runtime.h>
#include <tilemega/Codegen/tasks/EventSync.cuh>
#include <tilemega/Codegen/tasks/Benchmark.cuh>
#include <tilemega/Codegen/ResidentSchedule.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/BalancedPlacement.h>
#include <tilemega/Solver/ListScheduler.h>
#include <tilemega/Solver/PlanMaterialize.h>

#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/ElementwiseTaskBody.h>
#include <tilemega/Codegen/tasks/AddTaskBody.h>
#include <tilemega/Codegen/tasks/GemmCombineTaskBody.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/FusedGemmTaskBody.h>
#include <tilemega/Codegen/tasks/FusedRoPEKVTaskBody.h>
#include <tilemega/Codegen/tasks/KVAppendTaskBody.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/ClusterSync.cuh>
#include <tilemega/Codegen/tasks/RoPETaskBody.h>
#include <tilemega/Target/TargetSpec.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#if TILEMEGA_TRACE_V2
#include <filesystem>
#include <system_error>
#endif

#define TILEMEGA_CUDA_CHECK(expr) do { cudaError_t e = (expr); \
  if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

namespace tilemega::codegen {

#ifndef TILEMEGA_GENERATED_WAIT_global
#define TILEMEGA_GENERATED_WAIT_global(ev, need) do { \
  while (::tilemega::codegen::EventPoll((ev)) < (need)) __nanosleep(64); \
} while (0)
#endif
#ifndef TILEMEGA_GENERATED_NOTIFY_global
#define TILEMEGA_GENERATED_NOTIFY_global(ev, value) atomicExch((ev), (value))
#endif
/// P4.7: the cluster dimension the generator emitted for this model, 1 when
/// no coupling asked for cluster-scoped synchronization.  It is a compile-time
/// macro rather than a launch argument because the *barrier* changes with it,
/// and the barrier is inside the kernel.
#ifndef TILEMEGA_GENERATED_CLUSTER_DIM
#define TILEMEGA_GENERATED_CLUSTER_DIM 1
#endif
#ifndef TILEMEGA_NEGATIVE_OLD_CLAMP
#define TILEMEGA_NEGATIVE_OLD_CLAMP 0
#endif
#ifndef TILEMEGA_NEGATIVE_TASK_WAIT_CLAMP
#define TILEMEGA_NEGATIVE_TASK_WAIT_CLAMP 0
#endif
static_assert(!arch::kDevicePass || TILEMEGA_GENERATED_CLUSTER_DIM == 1 ||
                  arch::Caps<arch::CurrentArch>::kCluster,
              "a cluster stage barrier needs a cluster-capable target; a "
              "cluster-shaped kernel must never fall back to the flat grid "
              "barrier and keep reporting itself as a cluster result");
static_assert(!arch::kDevicePass || !TILEMEGA_EVENT_CLUSTER_FANIN ||
                  arch::Caps<arch::CurrentArch>::kCluster,
              "cluster event aggregation requires caps.cluster");

#ifndef TILEMEGA_GENERATED_RESIDENT_GRID
#define TILEMEGA_GENERATED_RESIDENT_GRID(target, function, block_size, dynamic_smem) \
  ((target).res.num_sms * (target).ActiveBlocksPerSM( \
      reinterpret_cast<void const*>(function), (block_size), (dynamic_smem)))
#endif

inline constexpr int kHarnessThreads = kGemmThreads;

#ifndef TILEMEGA_FUSION_RUNTIME
#define TILEMEGA_FUSION_RUNTIME 0
#endif
#ifndef TILEMEGA_FUSION_GEMM_RUNTIME
#define TILEMEGA_FUSION_GEMM_RUNTIME TILEMEGA_FUSION_RUNTIME
#endif
#ifndef TILEMEGA_FUSION_ROPE_RUNTIME
#define TILEMEGA_FUSION_ROPE_RUNTIME 0
#endif
#ifndef TILEMEGA_FUSION_ROPE_MAX_WIDTH
#define TILEMEGA_FUSION_ROPE_MAX_WIDTH 0
#endif

/// §8.6: one explicit union covering every family the dispatch can reach.
union TaskSmem {
  SimtTaskResources<TaskKind::kRMSNorm,kHarnessThreads>::SharedStorage rms;
  SimtTaskResources<TaskKind::kAttention,kHarnessThreads>::SharedStorage attention;
  SimtTaskResources<TaskKind::kElementwise,kHarnessThreads>::SharedStorage pointwise;
  GemmVariantSmem gemm;
#if TILEMEGA_FUSION_GEMM_RUNTIME
  alignas(16) unsigned char fused_gemm[
      FusedGemmStorageBytes<arch::CurrentArch,kHarnessThreads>()];
#endif
#if TILEMEGA_FUSION_ROPE_RUNTIME
  ModelElement fused_rope[std::max(2*kHarnessThreads,TILEMEGA_FUSION_ROPE_MAX_WIDTH)];
#endif
};
inline constexpr std::size_t kNonGemmTaskSmem =
    std::max({sizeof(TaskSmem::rms), sizeof(TaskSmem::attention), sizeof(TaskSmem::pointwise)});
inline constexpr std::size_t kExpectedTaskSmem =
    std::max({sizeof(GemmVariantSmem),kNonGemmTaskSmem
#if TILEMEGA_FUSION_GEMM_RUNTIME
        ,sizeof(TaskSmem::fused_gemm)
#endif
#if TILEMEGA_FUSION_ROPE_RUNTIME
        ,sizeof(TaskSmem::fused_rope)
#endif
    });
static_assert(sizeof(TaskSmem) == kExpectedTaskSmem,
              "one explicit union must equal max_i(TaskBody::SharedStorage)");

using HarnessArch = cutlass::arch::Sm80;
using T_Gemm = GemmStageTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Norm = RMSNormTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_RoPE = RoPETaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_KV = KVAppendTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Elementwise = ElementwiseTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Add = AddTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_Attention = AttentionTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
using T_GemmCombine = GemmCombineTaskBody<HarnessArch, TaskSmem, kHarnessThreads>;
#if TILEMEGA_FUSION_GEMM_RUNTIME
using T_FusedGemm = FusedGemmStageTaskBody<HarnessArch,TaskSmem,kHarnessThreads>;
static_assert(T_FusedGemm::kLegal && DeclaresOwnership<T_FusedGemm>::value);
#endif
#if TILEMEGA_FUSION_ROPE_RUNTIME
using T_FusedRoPE = FusedRoPEKVTaskBody<HarnessArch,TaskSmem,kHarnessThreads>;
static_assert(T_FusedRoPE::kLegal && DeclaresOwnership<T_FusedRoPE>::value);
#endif
static_assert(T_Gemm::kLegal && T_Norm::kLegal && T_RoPE::kLegal &&
              T_KV::kLegal && T_Elementwise::kLegal && T_Attention::kLegal &&
              T_GemmCombine::kLegal,
              "every dispatched TaskBody must be legal at this granularity");
static_assert(DeclaresOwnership<T_Gemm>::value &&
              DeclaresOwnership<T_Norm>::value &&
              DeclaresOwnership<T_RoPE>::value &&
              DeclaresOwnership<T_KV>::value &&
              DeclaresOwnership<T_Elementwise>::value &&
              DeclaresOwnership<T_Attention>::value &&
              DeclaresOwnership<T_GemmCombine>::value,
              "every dispatched TaskBody must declare its CTA->task "
              "ownership (§5.3); L2 skips a stage's waits for CTAs at or "
              "above the declared count");

/// The dispatch is over the TaskBody families, which are a property of the
/// library, not of any model.  A model that needs no attention simply never
/// emits those stage kinds.
__device__ inline void RunStage(Params const& p, std::uint32_t index,
                                TaskSmem& smem) {
  StageDesc const& stage = p.stages[index];
  switch (stage.kind) {
    case TaskKind::kGemm: T_Gemm{}(p, stage, smem); break;
    case TaskKind::kRMSNorm: T_Norm{}(p, stage, smem); break;
    case TaskKind::kRoPE: T_RoPE{}(p, stage, smem); break;
    case TaskKind::kKVAppend: T_KV{}(p, stage, smem); break;
    case TaskKind::kElementwise: T_Elementwise{}(p, stage, smem); break;
    case TaskKind::kAdd: T_Add{}(p, stage, smem); break;
    case TaskKind::kAttention: T_Attention{}(p, stage, smem); break;
    case TaskKind::kGemmCombine: T_GemmCombine{}(p, stage, smem); break;
    case TaskKind::kGemmAdd:
    case TaskKind::kGemmRMSNorm:
#if TILEMEGA_FUSION_GEMM_RUNTIME
      T_FusedGemm{}(p,stage,smem); break;
#else
      asm volatile("trap;"); break;
#endif
    case TaskKind::kRoPEKVAppend:
#if TILEMEGA_FUSION_ROPE_RUNTIME
      T_FusedRoPE{}(p,stage,smem); break;
#else
      asm volatile("trap;"); break;
#endif
  }
}

/// Dispatch exactly one logical task.  L0.5/L1 keep calling RunStage, whose
/// grid-stride loops reuse these same single-task entries; L2 consumes them
/// directly from the materialized worker queue.
__device__ inline void RunTask(Params const& p, std::uint32_t index,
                               std::uint32_t logical_task, TaskSmem& smem) {
  StageDesc const& stage = p.stages[index];
  int const task = static_cast<int>(logical_task);
  switch (stage.kind) {
    case TaskKind::kGemm:
      T_Gemm::RunLogicalTask(p, stage, smem, task);
      break;
    case TaskKind::kRMSNorm: T_Norm::RunTask(p, stage, smem, task); break;
    case TaskKind::kAdd: T_Add::RunTask(p, stage, smem, task); break;
    case TaskKind::kRoPE: T_RoPE::RunTask(p, stage, smem, task); break;
    case TaskKind::kKVAppend: T_KV::RunTask(p, stage, smem, task); break;
    case TaskKind::kElementwise:
      T_Elementwise::RunTask(p, stage, smem, task);
      break;
    case TaskKind::kAttention:
      T_Attention::RunTask(p, stage, smem, task);
      break;
    case TaskKind::kGemmCombine:
      T_GemmCombine::RunTask(p, stage, smem, task);
      break;
    case TaskKind::kGemmAdd:
    case TaskKind::kGemmRMSNorm:
#if TILEMEGA_FUSION_GEMM_RUNTIME
      T_FusedGemm::RunTask(p,stage,smem,task); break;
#else
      asm volatile("trap;"); break;
#endif
    case TaskKind::kRoPEKVAppend:
#if TILEMEGA_FUSION_ROPE_RUNTIME
      T_FusedRoPE::RunTask(p,stage,smem,task); break;
#else
      asm volatile("trap;"); break;
#endif
  }
}

__host__ __device__ inline int CeilDiv(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

/// §2.3's event granularity kappa, as a compile-time knob on the L2 path.
///
/// 0 is the aggregate-only control: one event per producer stage. A positive
/// value groups kappa consecutive logical tasks
/// of a stage into one event, so a stage publishes ceil(task_count/kappa) of
/// them even when task_count exceeds the resident grid. L1 has no such knob:
/// its grid barrier is one
/// event per stage by construction.
///
/// A consumer waits on only the groups its `StageDependency` window spans, so
/// with a narrowed edge kappa now has a benefit side as well as a cost side:
/// smaller kappa means more events published but a tighter wait set. A kAll
/// edge uses the producer's aggregate completion event rather than expanding
/// every fine group, keeping one task's descriptor count O(CG in-edges).
#ifndef TILEMEGA_EVENT_KAPPA
#define TILEMEGA_EVENT_KAPPA 1
#endif

#if TILEMEGA_TRACE_V2
/// %globaltimer is one counter broadcast to every SM, so unlike clock64 it is
/// directly comparable across workers.  It ticks in 1024 ns steps on sm_89
/// (TRACE_V2/resolution.md), which is why clock64 is recorded beside it for
/// the run interval, where the two reads are on one SM and need no calibration.
__device__ inline unsigned long long TraceNow() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}

__device__ inline unsigned int TraceSmid() {
  unsigned int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
#endif

/// L1 uses the first stage_count counters. L2 follows with a runtime-sized
/// prefix table. A stage owns an aggregate completion row and/or (for positive
/// kappa) one row per logical-task group, according to what its consumers
/// reference. This is what makes κ group *tasks*, not worker ids, without
/// making kAll fan-in linear in tasks or publishing unused rows.
__device__ inline std::uint32_t EventIndex(Params const& p,
                                           std::uint32_t stage,
                                           std::uint32_t group) {
  std::uint32_t const base = p.stage_count + p.event_offsets[stage];
#if TILEMEGA_EVENT_KAPPA > 0
  return group == kWholeStageEventGroup
             ? base
             : base + ((p.event_flags[stage] & kNeedsAggregateEvent) ? 1u : 0u) +
                   group;
#else
  (void)group;
  return base;
#endif
}

/// Active CTAs of a stage, clamped to the grid (legacy stage-wait helper).
__device__ inline int ActiveBlocksClamped(Params const& p, std::uint32_t stage);

/// Cardinality of image(C_kappa) along the launch axis.  CTAs beyond this
/// bound participate only in control flow and own no producer event.
///
/// L2 relies on a stronger reading than "owns no event": a CTA with
/// `blockIdx.x >= ActiveBlocks(stage)` neither reads nor writes anything in
/// that stage, so it may skip the stage's waits and fences entirely. That
/// holds because every dispatched TaskBody guards on exactly this bound --
/// `tile_n >= invocation.tiles_n` (GemmStage), `token < p.dims.seq`
/// (RMSNorm), `query < seq*heads` (AttentionChunk), and a grid-stride loop
/// whose trip count is the same numerator for RoPE/KVAppend/Elementwise.
/// That is no longer a duplicated guard the harness has to keep in step:
/// each TaskBody declares it through the ABI's `Ownership` entry (§5.3,
/// TaskBase.h) and this switch only dispatches. A TaskBody that does not
/// declare it fails the static_assert below rather than silently breaking
/// the skip.
__device__ inline int ActiveBlocks(Params const& p, StageDesc const& stage) {
  switch (stage.kind) {
    case TaskKind::kGemm: return T_Gemm::Ownership(p, stage).count;
    case TaskKind::kRMSNorm: return T_Norm::Ownership(p, stage).count;
    case TaskKind::kAdd: return T_Add::Ownership(p, stage).count;
    case TaskKind::kRoPE: return T_RoPE::Ownership(p, stage).count;
    case TaskKind::kKVAppend: return T_KV::Ownership(p, stage).count;
    case TaskKind::kElementwise: return T_Elementwise::Ownership(p, stage).count;
    case TaskKind::kAttention: return T_Attention::Ownership(p, stage).count;
    case TaskKind::kGemmCombine: return T_GemmCombine::Ownership(p, stage).count;
    case TaskKind::kGemmAdd:
    case TaskKind::kGemmRMSNorm:
#if TILEMEGA_FUSION_GEMM_RUNTIME
      return T_FusedGemm::Ownership(p,stage).count;
#else
      asm volatile("trap;"); return 0;
#endif
    case TaskKind::kRoPEKVAppend:
#if TILEMEGA_FUSION_ROPE_RUNTIME
      return T_FusedRoPE::Ownership(p,stage).count;
#else
      asm volatile("trap;"); return 0;
#endif
  }
  return 0;
}

__device__ inline int ActiveBlocksClamped(Params const& p,
                                          std::uint32_t stage) {
  int active = ActiveBlocks(p, p.stages[stage]);
  if (active > static_cast<int>(gridDim.x)) active = gridDim.x;
  return active < 1 ? 1 : active;
}

/// How many CTAs must arrive before stage `producer` counts as complete at
/// iteration `iteration` -- §8.2's `needed = num_triggers x iteration_num`.
/// `num_triggers` is the stage's own active CTA count, not the whole grid:
/// a stage whose task space is smaller than the grid is finished when its
/// own CTAs are, and the idle CTAs never trigger.
__device__ inline unsigned long long StageArrivalTarget(
    Params const& p, std::uint32_t producer, unsigned long long iteration) {
  int triggers = ActiveBlocks(p, p.stages[producer]);
  if (triggers > static_cast<int>(gridDim.x)) triggers = gridDim.x;
  if (triggers < 1) triggers = 1;
  return static_cast<unsigned long long>(triggers) * (iteration + 1ull);
}

/// §8 wait sequence over the events synthesized from CG couplings.
///
/// One event per producer *stage*, waited on once per incoming edge. Two
/// earlier shapes were measured and rejected, and both are worth recording
/// because each looked correct:
///
///  1. One event per producer *tile*, with the consumer spinning over every
///     one of them. That computes the same predicate ("all of this
///     producer's CTAs have published") at O(grid) spin-waits per edge, all
///     serialized in thread 0.
///  2. A single monotonic arrival counter per stage, polled directly by the
///     consumers. O(1) waits, but consumers then poll the very line the
///     producers are incrementing -- read-write sharing of one cache line
///     across the whole grid. Measured: no better than (1).
///
/// What works is the split this shares with GridBarrier: producers
/// accumulate into `arrivals`, and the CTA whose arrival completes the stage
/// publishes `epoch` once. Consumers poll `epoch`, a line that is written
/// exactly once per stage and read by everyone -- read-mostly sharing, which
/// is the regime the hardware is good at. §8.2's monotonicity is what lets
/// both be compared with `>=` and never reset between iterations.
__device__ inline void WaitDependencies(Params const& p, EventCounter* events,
                                        std::uint32_t consumer, bool active,
                                        unsigned long long iteration) {
#if TILEMEGA_UNSAFE_NO_EVENT_WAIT
  // A cost probe, exactly like TILEMEGA_UNSAFE_NO_GRID_SYNC above and never a
  // build anyone ships: the notify side still runs, so this arm isolates what
  // the *polling* costs.  Its output is wrong by construction.
  // Nothing at all, not even a fence: the arm has to isolate the polling, and
  // a fence left in would be charged to it.  `active` is still computed, which
  // is L2's own structure rather than the event scheme's.
  (void)p; (void)events; (void)consumer; (void)iteration; (void)active;
  return;
#endif
  // `active` depends only on blockIdx, so it is block-uniform and the early
  // return cannot split a __syncthreads. A CTA that owns no tile in this
  // stage reads nothing the producers wrote, so it needs neither the wait
  // nor the acquire fence -- and skipping them is what lets it run ahead to
  // the stage where it does own work.
  if (!active) return;
  // The polls are spread over the CTA's threads.  Measured on this machine the
  // wait was the *entire* L2-vs-L1 gap at seq=128 (102.9% of it on the GQA
  // model, 102.7% on the MHA one), and it was a serial walk on thread 0 while
  // 255 threads idled.  Every epoch is monotone (§8.2), so a poll depends on
  // nothing another poll does: the loop nest below is walked by every thread
  // for its cheap arithmetic, and the expensive spin is taken only on the
  // iterations that belong to this thread.  The `__syncthreads` that already
  // ended this function is what makes the union complete.
  unsigned poll_index = 0;
  auto poll = [&](std::uint32_t producer, int group) {
#if TILEMEGA_SERIAL_POLL
    // The pre-optimization shape, kept compilable so the two can be measured
    // against each other in one session rather than across two.
    if (threadIdx.x != 0) return;
    (void)poll_index;
#else
    if (poll_index++ % blockDim.x != threadIdx.x) return;
#endif
    TILEMEGA_GENERATED_WAIT_global(
        &events[EventIndex(p, producer,
                           static_cast<std::uint32_t>(group))].epoch,
        iteration + 1ull);
  };
  {
    std::uint32_t first = p.dependency_offsets[consumer];
    std::uint32_t last = p.dependency_offsets[consumer + 1];
    for (std::uint32_t edge = first; edge < last; ++edge) {
      StageDependency const& dep = p.dependencies[edge];
      std::uint32_t const producer = dep.producer;
#if TILEMEGA_EVENT_KAPPA > 0
      // The producer tasks this CTA actually reads.  `kAll` is the whole
      // launch axis; a window is `[(c / div) * scale + offset, ... + count)`
      // for c = this CTA's task, which is `PlacedBlock()` because a window is
      // only ever emitted when both TaskBodies declare `kTilePerBlock`.  The
      // clamp to the producer's live range is part of the fit, not a
      // widening: a window fitted at prefill points past the last live task
      // at decode, and those tasks produce nothing to wait for.
      int const grid = static_cast<int>(gridDim.x);
      int const produced = ActiveBlocks(p, p.stages[producer]);
      int const live = ActiveBlocksClamped(p, producer);
      if (dep.map == StageDependency::Map::kAll) {
        for (int group = 0; group <= (live - 1) / TILEMEGA_EVENT_KAPPA; ++group)
          poll(producer, group);
      } else {
#if TILEMEGA_NEGATIVE_OLD_CLAMP
        // Deliberately reproduce the obsolete implementation for SEQSCAN's
        // negative control: it treats blockIdx as the only consumer task and
        // truncates producer tasks to the resident grid.  This is incorrect
        // whenever either stage is grid-strided.
        int const task = PlacedBlock();
        int const at =
            (task / static_cast<int>(dep.div)) * dep.scale + dep.offset;
        int const begin = at < 0 ? 0 : at;
        int const truncated = produced < grid ? produced : grid;
        int const past = at + static_cast<int>(dep.count);
        int const end = past < truncated ? past : truncated;
        for (int group = begin / TILEMEGA_EVENT_KAPPA;
             begin < end && group <= (end - 1) / TILEMEGA_EVENT_KAPPA;
             ++group)
          poll(producer, group);
#else
        // A stage whose task space is wider than the grid is run grid-strided
        // (GemmStageTaskBody), so one CTA owns tasks b, b+grid, ... and task t
        // is published by CTA t % grid. Both have to be undone here: the wait
        // is the union over the tasks this CTA owns, and each window is mapped
        // back onto CTA indices, wrapping when it is wider than one stride.
        // Only kTilePerBlock stages get a window at all, so `task` is the
        // fitted consumer index.
        int const owned = ActiveBlocks(p, p.stages[consumer]);
        for (int task = PlacedBlock(); task < owned; task += grid) {
          int const at =
              (task / static_cast<int>(dep.div)) * dep.scale + dep.offset;
          int const begin = at < 0 ? 0 : at;
          int const past = at + static_cast<int>(dep.count);
          int const end = past < produced ? past : produced;
          if (begin >= end) continue;
          int first = begin % grid, last = (end - 1) % grid;
          if (end - begin >= grid) { first = 0; last = live - 1; }
          int const wrapped = first > last;
          for (int group = first / TILEMEGA_EVENT_KAPPA;
               group <= (wrapped ? live - 1 : last) / TILEMEGA_EVENT_KAPPA;
               ++group)
            poll(producer, group);
          for (int group = 0; wrapped && group <= last / TILEMEGA_EVENT_KAPPA;
               ++group)
            poll(producer, group);
        }
#endif
      }
#else
      poll(producer, 0);
#endif
    }
  }
  __syncthreads();
  __threadfence();
}

/// Queue form of the wait: every row was already narrowed, converted to event
/// groups, deduplicated, and lifted past earlier waits by the host materializer.
/// Device work is therefore linear only in the unique, not-yet-known-satisfied
/// events of this task.
__device__ inline void WaitTaskDependencies(Params const& p,
                                            EventCounter* events,
                                            TaskRef const& task,
                                            unsigned long long iteration) {
#if TILEMEGA_UNSAFE_NO_EVENT_WAIT
  (void)p;
  (void)events;
  (void)task;
  (void)iteration;
  return;
#endif
  for (std::uint32_t i = threadIdx.x; i < task.wait_count;
       i += blockDim.x) {
    TaskWait const& wait = p.task_waits[task.wait_begin + i];
    TILEMEGA_GENERATED_WAIT_global(
        &events[EventIndex(p, wait.producer, wait.group)].epoch,
        iteration + 1ull);
  }
  if (task.wait_count != 0) {
    __syncthreads();
    __threadfence();
  }
}

// T1.3-A: called after every writer's release fence and CTA convergence.
// Last arrivals at each level carry the previous writers into the next
// release. No counter is reset while iterations are using this allocation.
__device__ inline void ArriveEvent(Params const& p, EventCounter* events,
                                    std::uint32_t index, int members,
                                    unsigned long long iteration) {
  unsigned long long triggers = static_cast<unsigned long long>(members);
#if TILEMEGA_EVENT_SHARDED
  // The one-member and S=1 cases are the exact one-level degeneracy. Avoid
  // adding a second atomic when there is no fan-in to combine.
  if (members > 1 && p.event_shard_count > 1) {
    EventFanIn const plan = p.event_fanin[index];
    std::uint32_t shard;
    unsigned long long* counter;
    using CS = ClusterSync<arch::CurrentArch>;
    if constexpr (TILEMEGA_EVENT_CLUSTER_FANIN && CS::kEnabled) {
      shard = plan.begin + blockIdx.x / CS::Size() - plan.first_cluster;
      extern __shared__ unsigned char event_bytes[];
      auto* local = reinterpret_cast<unsigned long long*>(event_bytes + sizeof(TaskSmem));
      counter = CS::CounterPeer(local + p.shard_local_offsets[shard], 0);
    } else {
      shard = plan.begin + blockIdx.x % plan.modulus;
      counter = &p.shard_arrivals[shard].arrivals;
    }
    unsigned long long ticket = atomicAdd(counter, 1ull);
    if (ticket + 1ull != static_cast<unsigned long long>(p.shard_targets[shard]) *
                              (iteration + 1ull)) return;
    __threadfence();
    triggers = plan.nonempty;
  }
#endif
  unsigned long long ticket = atomicAdd(&events[index].arrivals, 1ull);
  if (ticket + 1ull == triggers * (iteration + 1ull)) {
    __threadfence();
    TILEMEGA_GENERATED_NOTIFY_global(&events[index].epoch, iteration + 1ull);
#if TILEMEGA_TRACE_V2
    // Only the last arriver reaches here, so this plain store has exactly one
    // writer per event and needs no atomic.  It is the sole source of the
    // instant a consumer could first have been released.
    if (p.event_publish != nullptr) p.event_publish[index] = TraceNow();
#endif
  }
}

/// §8.5 CTA-cooperative release, then one monotonic arrival (§8.2), with the
/// completing CTA publishing the stage's epoch.  The fence is per writer and
/// precedes the CTA barrier, so every thread's writes are visible before
/// thread 0 publishes (F-1); the second fence orders the arrival before the
/// epoch that releases the consumers.
/// Cost probe: publish nothing.  Only meaningful together with
/// TILEMEGA_UNSAFE_NO_EVENT_WAIT -- on its own it deadlocks every consumer.
__device__ inline void NotifyTask(Params const& p, EventCounter* events,
                                  std::uint32_t producer,
                                  std::uint32_t logical_task,
                                  unsigned long long iteration) {
#if TILEMEGA_UNSAFE_NO_EVENT_NOTIFY
  (void)p; (void)events; (void)producer; (void)logical_task; (void)iteration;
  return;
#endif
  std::uint32_t const event_flags = p.event_flags[producer];
  if (event_flags == 0) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    int const produced = ActiveBlocks(p, p.stages[producer]);
#if TILEMEGA_EVENT_KAPPA > 0
    if (event_flags & kNeedsFineEvents) {
      int const group =
          static_cast<int>(logical_task) / TILEMEGA_EVENT_KAPPA;
      int const members = produced - group * TILEMEGA_EVENT_KAPPA
                              < TILEMEGA_EVENT_KAPPA
                          ? produced - group * TILEMEGA_EVENT_KAPPA
                          : TILEMEGA_EVENT_KAPPA;
      ArriveEvent(p, events, EventIndex(p, producer, static_cast<std::uint32_t>(group)),
                  members, iteration);
    }
#endif
    // A task contributes to the aggregate completion row only when some kAll
    // consumer references it. Such consumers pay one poll per incoming CG
    // edge, while narrowed consumers observe the fine group above.
    if (event_flags & kNeedsAggregateEvent) {
      ArriveEvent(p, events, EventIndex(p, producer, kWholeStageEventGroup),
                  produced, iteration);
    }
  }
  __syncthreads();
}

/// §8.1/§8.2/§8.3: single-thread polling with backoff on a monotonic counter,
/// release fence before CTA convergence (F-1), acquire fence after.
///
/// §8.2: `needed = num_triggers x iteration_num`. Here every CTA in the grid
/// triggers, so num_triggers is gridDim.x, and `iteration` is the caller's
/// iteration index. Because the target scales with the iteration rather than
/// the counter being cleared, the counters are never reset between
/// iterations and a late CTA from iteration i can never be mistaken for an
/// early one from iteration i+1 (the ABA the rule exists to prevent).
/// Speed-of-light probe: drop the *grid* half of the barrier and keep the CTA
/// half.  The result is numerically wrong by construction -- stages read
/// buffers the previous stage has not finished writing -- and exists only to
/// bound what any synchronization change (kappa, clusters, placement) could
/// ever be worth.  Off by default; the harness refuses to report PASS with it
/// on, so it can never be mistaken for a measurement of the real kernel.
#ifndef TILEMEGA_UNSAFE_NO_GRID_SYNC
#define TILEMEGA_UNSAFE_NO_GRID_SYNC 0
#endif

__device__ inline void GridBarrier(EventCounter* events, std::uint32_t stage,
                                   unsigned long long iteration) {
#if TILEMEGA_UNSAFE_NO_GRID_SYNC
  (void)events; (void)stage; (void)iteration;
  __threadfence();
  __syncthreads();
  return;
#elif TILEMEGA_GENERATED_CLUSTER_DIM > 1
  // The cluster closes over itself in hardware and only its rank 0 pays the
  // global round trip, so the arrival count is clusters, not CTAs.  The grid
  // is an exact multiple of the cluster dimension by construction (RunModel
  // trims it), which is what makes that division the true cluster count.
  ClusterSync<arch::CurrentArch>::StageBarrier(
      &events[stage].arrivals, &events[stage].epoch, iteration,
      gridDim.x / TILEMEGA_GENERATED_CLUSTER_DIM);
#else
  unsigned long long needed =
      static_cast<unsigned long long>(gridDim.x) * (iteration + 1ull);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long ticket = atomicAdd(&events[stage].arrivals, 1ull);
    if (ticket + 1 == needed) {
      __threadfence();
      TILEMEGA_GENERATED_NOTIFY_global(&events[stage].epoch, iteration + 1ull);
    } else {
      TILEMEGA_GENERATED_WAIT_global(&events[stage].epoch, iteration + 1ull);
    }
  }
  __syncthreads();
  __threadfence();
#endif
}

__global__ __launch_bounds__(kHarnessThreads, TILEMEGA_MIN_BLOCKS_PER_SM)
void tilemega_stage_kernel(Params const* params, std::uint32_t stage) {
  extern __shared__ unsigned char bytes[];
  RunStage(*params, stage, *reinterpret_cast<TaskSmem*>(bytes));
}

/// The L1 megakernel.  The stage loop is a run-time loop over the generated
/// table: its trip count is data, so one compiled kernel serves every model.
__global__ __launch_bounds__(kHarnessThreads, TILEMEGA_MIN_BLOCKS_PER_SM)
void tilemega_l1_kernel(Params const* params, EventCounter* events,
                        unsigned long long iteration) {
  extern __shared__ unsigned char bytes[];
  auto& smem = *reinterpret_cast<TaskSmem*>(bytes);
  for (std::uint32_t stage = 0; stage < params->stage_count; ++stage) {
    RunStage(*params, stage, smem);
    GridBarrier(events, stage, iteration);
  }
}

/// L2 is worker-queue driven. Adjacent rows may name different stages; only
/// the concrete task's precomputed event slice constrains progress.
__global__ __launch_bounds__(kHarnessThreads, TILEMEGA_MIN_BLOCKS_PER_SM)
void tilemega_l2_kernel(Params const* params, EventCounter* events,
                        unsigned long long iteration) {
  extern __shared__ unsigned char bytes[];
  auto& smem = *reinterpret_cast<TaskSmem*>(bytes);
  using CS = ClusterSync<arch::CurrentArch>;
  if constexpr (TILEMEGA_EVENT_CLUSTER_FANIN && CS::kEnabled) {
    unsigned const cluster = blockIdx.x / CS::Size();
    unsigned const begin = params->cluster_shard_offsets[cluster];
    unsigned const end = params->cluster_shard_offsets[cluster + 1];
    auto* local = reinterpret_cast<unsigned long long*>(bytes + sizeof(TaskSmem));
    if (CS::Rank() == 0)
      for (unsigned i = threadIdx.x; begin + i < end; i += blockDim.x)
        local[i] = params->shard_arrivals[params->cluster_shard_indices[begin + i]].arrivals;
    CS::Sync();
  }
  std::uint32_t const worker = static_cast<std::uint32_t>(blockIdx.x);
  std::uint32_t const first = params->schedule_offsets[worker];
  std::uint32_t const last = params->schedule_offsets[worker + 1];
  for (std::uint32_t slot = first; slot < last; ++slot) {
    TaskRef const task = params->schedule[slot];
#if TILEMEGA_TRACE_V2
    // Every stamp is thread 0's own plain store into this slot's row: no
    // atomic, no added barrier, and nothing written inside a polling loop.
    if (params->task_trace_v2 != nullptr && threadIdx.x == 0)
      params->task_trace_v2[slot].wait_begin = TraceNow();
#endif
    WaitTaskDependencies(*params, events, task, iteration);
#if TILEMEGA_TRACE_V2
    // `ready` follows the wait's own __syncthreads()/__threadfence(), so it
    // includes them; the offline report says so rather than subtracting them.
    if (params->task_trace_v2 != nullptr && threadIdx.x == 0)
      params->task_trace_v2[slot].ready = TraceNow();
#endif
    if (params->task_trace != nullptr && threadIdx.x == 0)
      params->task_trace[slot].start =
          atomicAdd(params->trace_sequence, 1ull);
    __syncthreads();
#if TILEMEGA_TRACE_V2
    if (params->task_trace_v2 != nullptr && threadIdx.x == 0) {
      TaskTraceV2& row = params->task_trace_v2[slot];
      row.run_begin = TraceNow();
      row.run_begin_clk = static_cast<unsigned long long>(clock64());
      row.smid = TraceSmid();
      row.worker = worker;
      row.stage = task.stage;
      row.logical_task = task.logical_task;
    }
#endif
    RunTask(*params, task.stage, task.logical_task, smem);
    __syncthreads();
#if TILEMEGA_TRACE_V2
    if (params->task_trace_v2 != nullptr && threadIdx.x == 0) {
      TaskTraceV2& row = params->task_trace_v2[slot];
      row.run_end = TraceNow();
      row.run_end_clk = static_cast<unsigned long long>(clock64());
    }
#endif
    if (params->task_trace != nullptr && threadIdx.x == 0)
      params->task_trace[slot].end =
          atomicAdd(params->trace_sequence, 1ull);
    NotifyTask(*params, events, task.stage, task.logical_task, iteration);
#if TILEMEGA_TRACE_V2
    if (params->task_trace_v2 != nullptr && threadIdx.x == 0)
      params->task_trace_v2[slot].publish_end = TraceNow();
#endif
  }
  if constexpr (TILEMEGA_EVENT_CLUSTER_FANIN && CS::kEnabled) {
    // No CTA may leave while another still accesses its DSMEM allocation.
    CS::Sync();
    unsigned const cluster = blockIdx.x / CS::Size();
    unsigned const begin = params->cluster_shard_offsets[cluster];
    unsigned const end = params->cluster_shard_offsets[cluster + 1];
    auto* local = reinterpret_cast<unsigned long long*>(bytes + sizeof(TaskSmem));
    if (CS::Rank() == 0)
      for (unsigned i = threadIdx.x; begin + i < end; i += blockDim.x)
        params->shard_arrivals[params->cluster_shard_indices[begin + i]].arrivals = local[i];
  }
}

/// How wide each edge's wait set actually is, in the same units the device
/// pays for: one line per dependency, `polls` summed over the consumer's own
/// CTAs.  `relaxed` is what the same edge would cost as kAll, so the pair is
/// the narrowing factor Part 3.1 attributes the L2/L1 ratio with.  Printing
/// happens on the device because the active-CTA count is a TaskBody property
/// (`Ownership`) and gridDim has to be the real launch grid.
__global__ __launch_bounds__(kHarnessThreads, TILEMEGA_MIN_BLOCKS_PER_SM)
void tilemega_wait_profile_kernel(Params const* params, int seq) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  // `seq` is a counting what-if, not an execution: every active-CTA count is
  // a function of dims.seq through the TaskBodies' own Ownership, so
  // substituting it answers "how wide would this wait set be at that sequence
  // length" without a fixture at that length.  Nothing is dispatched.
  Params scaled = *params;
  if (seq > 0) scaled.dims.seq = seq;
  Params const& p = scaled;
  int const kappa = TILEMEGA_EVENT_KAPPA;
  // Only reached when kappa > 0; kept off zero so the kappa = 0 build does not
  // constant-fold a division by it.
  int const per_group = kappa > 0 ? kappa : 1;
  for (std::uint32_t consumer = 0; consumer < p.stage_count; ++consumer) {
    int const consumers = ActiveBlocksClamped(p, consumer);
    for (std::uint32_t edge = p.dependency_offsets[consumer];
         edge < p.dependency_offsets[consumer + 1]; ++edge) {
      StageDependency const& dep = p.dependencies[edge];
      int const grid = static_cast<int>(gridDim.x);
      int const produced = ActiveBlocks(p, p.stages[dep.producer]);
      int const producers = ActiveBlocksClamped(p, dep.producer);
      int const groups = kappa > 0 ? CeilDiv(producers, per_group) : 1;
      // Counted exactly as WaitDependencies polls it, duplicates included:
      // the grid-stride union can name one group twice and the device pays
      // for both.
      long long polls = 0;
      int const owned = ActiveBlocks(p, p.stages[consumer]);
      for (int c = 0; c < consumers; ++c) {
        if (kappa == 0) { polls += 1; continue; }
        if (dep.map == StageDependency::Map::kAll) {
          polls += groups;
          continue;
        }
        for (int task = c; task < owned; task += grid) {
          int const at = (task / static_cast<int>(dep.div)) * dep.scale + dep.offset;
          int const begin = at < 0 ? 0 : at;
          int const past = at + static_cast<int>(dep.count);
          int const end = past < produced ? past : produced;
          if (begin >= end) continue;
          int first = begin % grid, last = (end - 1) % grid;
          if (end - begin >= grid) { first = 0; last = producers - 1; }
          if (first > last) {
            polls += (producers - 1) / per_group - first / per_group + 1;
            polls += last / per_group + 1;
          } else {
            polls += last / per_group - first / per_group + 1;
          }
        }
      }
      std::printf("E2E_WAITSET producer=%u consumer=%u map=%u div=%u scale=%d "
                  "offset=%d count=%u p_active=%d c_active=%d groups=%d "
                  "polls=%lld relaxed=%lld p_tasks=%d c_tasks=%d grid=%d "
                  "seq=%d\n",
                  dep.producer, consumer, static_cast<unsigned>(dep.map),
                  dep.div, dep.scale, dep.offset, dep.count, producers,
                  consumers, groups, polls,
                  static_cast<long long>(consumers) * groups, produced, owned,
                  grid, p.dims.seq);
    }
  }
}

namespace harness {

inline std::vector<ModelElement> Load(std::string const& path,
                                      std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<ModelElement> value(count);
  input.read(reinterpret_cast<char*>(value.data()), count * sizeof(ModelElement));
  if (input.gcount() !=
      static_cast<std::streamsize>(count * sizeof(ModelElement))) {
    std::fprintf(stderr, "wrong fixture size: %s\n", path.c_str());
    std::exit(2);
  }
  return value;
}

struct DeviceModel {
  ModelSpec const* spec = nullptr;
  RuntimeVariantDesc const* runtime_variant = nullptr;
  std::uint32_t runtime_variant_index = 0;
  /// The instantiated stage list. It equals `spec->stages` unless §2.4's
  /// Split was applied, which rewrites one GEMM stage into a partial stage
  /// plus its combiner.
  std::vector<StageDesc> stages;
  std::vector<ModelElement*> buffers;
  std::vector<std::vector<ModelElement>> host_sources;
  ModelElement** device_buffers = nullptr;
  GemmInvocation* device_gemms = nullptr;
  StageDesc* device_stages = nullptr;
  StageDependency* device_dependencies = nullptr;
  std::uint32_t* device_dependency_offsets = nullptr;
  std::vector<TaskRef> schedule;
  std::vector<std::uint32_t> schedule_offsets;
  std::vector<TaskWait> task_waits;
  TaskRef* device_schedule = nullptr;
  std::uint32_t* device_schedule_offsets = nullptr;
  TaskWait* device_task_waits = nullptr;
  std::vector<std::uint32_t> event_offsets;
  std::uint32_t* device_event_offsets = nullptr;
  std::vector<std::uint32_t> event_flags;
  std::uint32_t* device_event_flags = nullptr;
  std::vector<EventFanIn> event_fanin;
  std::vector<std::uint32_t> shard_targets;
  EventFanIn* device_event_fanin = nullptr;
  ArrivalCounter* device_shard_arrivals = nullptr;
  std::uint32_t* device_shard_targets = nullptr;
  std::vector<std::uint32_t> shard_local_offsets, cluster_shard_offsets, cluster_shard_indices;
  std::uint32_t* device_shard_local_offsets = nullptr;
  std::uint32_t* device_cluster_shard_offsets = nullptr;
  std::uint32_t* device_cluster_shard_indices = nullptr;
  std::size_t l2_smem_bytes = sizeof(TaskSmem);
  TaskTrace* device_task_trace = nullptr;
  unsigned long long* device_trace_sequence = nullptr;
#if TILEMEGA_TRACE_V2
  TaskTraceV2* device_task_trace_v2 = nullptr;
  unsigned long long* device_event_publish = nullptr;
  bool trace_v2_enabled = false;
  /// active_tasks(stage) kept for the dump: events.tsv reports each row's
  /// fan-in, which is what the last-arriver condition is counted against.
  std::vector<std::uint32_t> trace_v2_active_tasks;
#endif
  std::vector<std::uint32_t> stage_order;
  std::uint32_t schedule_max_span = 0;
  std::uint32_t schedule_max_worker_span = 0;
  bool schedule_has_global_fanin = false;
  std::size_t schedule_raw_polls = 0;
  std::size_t schedule_waiting_tasks = 0;
  std::size_t normalization_dummy_lower_bound = 0;
  Params params{};
  Params* device_params = nullptr;
  EventCounter* events = nullptr;
  std::size_t event_count = 0;
};

#if TILEMEGA_TRACE_V2
/// Trace v2 allocates and dumps only when asked at run time, so one build
/// serves both the traced and the untraced arm of the perturbation pairing.
inline bool TraceV2Enabled() {
  char const* setting = std::getenv("TILEMEGA_TRACE_V2");
  return setting != nullptr && std::atoi(setting) != 0;
}

inline void ZeroTraceV2(DeviceModel& model) {
  if (model.device_task_trace_v2 != nullptr)
    TILEMEGA_CUDA_CHECK(cudaMemset(model.device_task_trace_v2, 0,
                                   model.schedule.size() * sizeof(TaskTraceV2)));
  if (model.device_event_publish != nullptr)
    TILEMEGA_CUDA_CHECK(cudaMemset(model.device_event_publish, 0,
                                   model.event_count * sizeof(unsigned long long)));
}
#endif

/// Bind the symbolic dimensions.  The generated tables never carry a token
/// count; it arrives with the workload, here from the fixture manifest.
inline ModelDims BindDims(ModelDims dims, std::string const& dir) {
  std::ifstream input(dir + "/manifest.json");
  if (!input) {
    std::fprintf(stderr, "cannot open %s/manifest.json\n", dir.c_str());
    std::exit(2);
  }
  std::string text((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  auto field = [&](char const* name) {
    std::string key = std::string("\"") + name + "\"";
    std::size_t at = text.find(key);
    if (at == std::string::npos) {
      std::fprintf(stderr, "manifest has no %s\n", name);
      std::exit(2);
    }
    at = text.find(':', at) + 1;
    return std::atoi(text.c_str() + at);
  };
  dims.seq = field("seq");
  dims.past = field("past");
  dims.total = dims.seq + dims.past;
  return dims;
}

inline DeviceModel Create(ModelSpec const& spec,
                          RuntimeVariantDesc const& runtime_variant,
                          std::uint32_t runtime_variant_index,
                          ModelDims const& dims, std::string const& dir,
                          int grid, int blocks_per_sm, TargetSpec const& target,
                          std::size_t l2_smem_bytes) {
  DeviceModel model;
  model.spec = &spec;
  model.runtime_variant = &runtime_variant;
  model.runtime_variant_index = runtime_variant_index;
  model.params.ownership_flags = runtime_variant.ownership_flags;
#if TILEMEGA_EVENT_CLUSTER_RESERVE
  if (!target.caps.cluster) {
    std::fprintf(stderr, "cluster counter storage requires caps.cluster\n");
    std::exit(2);
  }
  // Reserve the advertised per-CTA shared-memory budget before selecting
  // residency. Both the cluster treatment and its storage-matched control
  // use this reservation; occupancy is never computed using a smaller size.
  model.l2_smem_bytes = l2_smem_bytes;
#endif
  model.host_sources.resize(spec.buffer_count);
  for (std::uint32_t i = 0; i < spec.buffer_count; ++i) {
    BufferDesc const& desc = spec.buffers[i];
    std::size_t elements = desc.Elements(dims);
    ModelElement* pointer = nullptr;
    TILEMEGA_CUDA_CHECK(cudaMalloc(&pointer, elements * sizeof(ModelElement)));
    if (desc.file != nullptr) {
      model.host_sources[i] = Load(dir + "/" + desc.file, elements);
      TILEMEGA_CUDA_CHECK(cudaMemcpy(pointer, model.host_sources[i].data(),
                                     elements * sizeof(ModelElement),
                                     cudaMemcpyHostToDevice));
    } else {
      TILEMEGA_CUDA_CHECK(cudaMemset(pointer, 0,
                                     elements * sizeof(ModelElement)));
    }
    model.buffers.push_back(pointer);
  }

  // §2.4 Split applied to the instantiated task graph: each GEMM's `k` is cut
  // into `chunks` contributions writing their own partial, and one combiner
  // stage reduces them. Splitting on the host keeps every chunk an ordinary
  // CUTLASS invocation, so no TaskBody knows it is part of a split.
  std::vector<GemmInvocation> gemms;
  std::vector<std::uint32_t> gemm_base(spec.gemm_count);
  std::vector<std::uint32_t> gemm_partial(spec.gemm_count, kNoOperand);
  std::vector<int> gemm_chunks(spec.gemm_count, 1);
  std::size_t partial_bytes = 0;
  for (std::uint32_t i = 0; i < spec.gemm_count; ++i) {
    GemmDesc const& desc = spec.gemms[i];
    int m = dims.seq;
    GemmRuntimeDesc const& runtime = runtime_variant.gemms[i];
    int variant = runtime.compiled_variant;
    if (variant < 0 || variant >= kGemmVariantCount) {
      std::fprintf(stderr,
                   "runtime variant %u names invalid compiled GEMM variant %d\n",
                   runtime_variant_index, variant);
      std::exit(2);
    }
    GemmVariantInfo const& tiling = kGemmVariantInfo[variant];
    if (tiling.tile_m != runtime.tile_m || tiling.tile_n != runtime.tile_n ||
        tiling.tile_k != runtime.tile_k || tiling.stages != runtime.stages) {
      std::fprintf(stderr,
                   "ModelSpec/template mismatch in runtime variant %u GEMM %u\n",
                   runtime_variant_index, i);
      std::exit(2);
    }
    int split = runtime.split_k;
    int k_tiles = CeilDiv(desc.k, tiling.tile_k);
    int chunks = split < k_tiles ? split : k_tiles;
    if (chunks < 1) chunks = 1;
    gemm_chunks[i] = chunks;
    gemm_base[i] = static_cast<std::uint32_t>(gemms.size());
    if (chunks > 1) {
      partial_bytes += static_cast<std::size_t>(chunks) * m * desc.n * sizeof(ModelPartialElement);
      ModelPartialElement* partial = nullptr;
      TILEMEGA_CUDA_CHECK(cudaMalloc(
          &partial, static_cast<std::size_t>(chunks) * m * desc.n *
                        sizeof(ModelPartialElement)));
      gemm_partial[i] = static_cast<std::uint32_t>(model.buffers.size());
      // This table carries addresses; only the combiner interprets a partial
      // entry, explicitly as ModelPartialElement, never as model storage.
      model.buffers.push_back(reinterpret_cast<ModelElement*>(partial));
      model.host_sources.emplace_back();
    }
    for (int chunk = 0; chunk < chunks; ++chunk) {
      // Tiles are distributed as evenly as the count allows, so no chunk is
      // empty whenever chunks <= k_tiles -- an empty CUTLASS problem is not a
      // legal invocation.
      int k_begin = chunk * k_tiles / chunks * tiling.tile_k;
      int k_end = (chunk + 1) * k_tiles / chunks * tiling.tile_k;
      if (k_end > desc.k) k_end = desc.k;
      GemmProblem problem{m, desc.n, k_end - k_begin, 1};
      // The chunk's A/B are the same matrices seen from a K offset: the row
      // stride is still the full k, so only the base pointer moves.
      auto stride_a = cutlass::make_cute_packed_stride(
          typename GemmMainloop::StrideA{}, cute::make_shape(m, desc.k, 1));
      auto stride_b = cutlass::make_cute_packed_stride(
          typename GemmMainloop::StrideB{}, cute::make_shape(desc.n, desc.k, 1));
      auto stride_c = cutlass::make_cute_packed_stride(
          typename GemmEpilogue::StrideC{}, cute::make_shape(m, desc.n, 1));
      auto stride_d = cutlass::make_cute_packed_stride(
          typename GemmEpilogue::StrideD{}, cute::make_shape(m, desc.n, 1));
      GemmMainloopOperands main_args{
          model.buffers[desc.a] + k_begin, stride_a,
          model.buffers[desc.b] + k_begin, stride_b};
      // Only the first chunk applies beta*C; the combiner adds no residual, so
      // the split result differs from the unsplit one only by association.
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
      ModelElement* destination = model.buffers[desc.d];
#else
      ModelElement* destination = chunks > 1
          ? model.buffers[gemm_partial[i]] +
                static_cast<std::size_t>(chunk) * m * desc.n
          : model.buffers[desc.d];
#endif
      typename GemmEpilogue::Arguments epilogue_args{
          {1.0f, chunk == 0 ? desc.beta : 0.0f}, model.buffers[desc.c], stride_c,
          destination, stride_d};
      GemmInvocation invocation;
      invocation.problem = problem;
      invocation.mainloop = main_args;
      invocation.epilogue =
          GemmEpilogue::to_underlying_arguments(problem, epilogue_args, nullptr);
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
      if (chunks > 1) {
        auto* partial = reinterpret_cast<float*>(model.buffers[gemm_partial[i]]) +
                        static_cast<std::size_t>(chunk) * m * desc.n;
        PartialEpilogue::Arguments args{{1.0f, 0.0f}, nullptr, stride_c, partial, stride_d};
        invocation.partial_epilogue = PartialEpilogue::to_underlying_arguments(problem, args, nullptr);
        invocation.residual = model.buffers[desc.c];
        invocation.residual_beta = desc.beta;
      }
#endif
      invocation.tiles_m = CeilDiv(m, tiling.tile_m);
      invocation.tiles_n = CeilDiv(desc.n, tiling.tile_n);
      invocation.tile_m = tiling.tile_m;
      invocation.tile_n = tiling.tile_n;
      invocation.chunks = chunks;
      invocation.variant = variant;
      gemms.push_back(invocation);
    }
  }

  // Rewrite the stage list and the dependency graph around the combiners.
  std::printf("E2E_PARTIALS fp32=%d element_bytes=%zu total_bytes=%zu\n",
              TILEMEGA_FP32_PARTIALS && kCompiledScalarType == ScalarType::kBF16,
              sizeof(ModelPartialElement), partial_bytes);
  std::vector<std::uint32_t> entry(spec.stage_count), done(spec.stage_count);
  std::vector<std::uint32_t> attention_chunks(spec.stage_count,1);
  for (std::uint32_t i = 0; i < spec.stage_count; ++i) {
    StageDesc stage = spec.stages[i];
    bool const fused_gemm = stage.kind == TaskKind::kGemmAdd ||
                            stage.kind == TaskKind::kGemmRMSNorm;
    if (fused_gemm) {
      if (!TILEMEGA_FUSION_RUNTIME || !TILEMEGA_FUSION_GEMM_RUNTIME ||
          !runtime_variant.exact_dependencies || stage.gemm >= spec.gemm_count ||
          stage.operand[0] >= model.buffers.size() || stage.operand[1] >= model.buffers.size()) {
        std::fprintf(stderr,"invalid or disabled fused GEMM runtime stage\n");
        std::exit(2);
      }
      auto const& invocation = gemms[gemm_base[stage.gemm]];
      if (invocation.chunks != 1 ||
          (stage.kind == TaskKind::kGemmRMSNorm && invocation.tiles_n != 1)) {
        std::fprintf(stderr,"fused GEMM requires unsplit accumulation and a full-row norm tile\n");
        std::exit(2);
      }
    }
    if (stage.kind==TaskKind::kRoPEKVAppend) {
      bool valid=TILEMEGA_FUSION_RUNTIME && TILEMEGA_FUSION_ROPE_RUNTIME &&
          runtime_variant.exact_dependencies && stage.width>0 && stage.width%2==0 && stage.extent>0;
      if ((model.params.ownership_flags & kRoPETileOwnership) &&
          stage.width>std::max(2*kHarnessThreads,TILEMEGA_FUSION_ROPE_MAX_WIDTH)) valid=false;
      for (int operand=0;operand<4;++operand)
        if (stage.operand[operand]>=model.buffers.size()) valid=false;
      if (!valid) {
        std::fprintf(stderr,"invalid or disabled fused RoPE/KV runtime stage\n");
        std::exit(2);
      }
    }
    entry[i] = static_cast<std::uint32_t>(model.stages.size());
    if (stage.kind == TaskKind::kAttention && runtime_variant.attention)
      attention_chunks[i] = runtime_variant.attention[i].chunks;
    if (attention_chunks[i] > 1) {
      auto allocate_float = [&](std::size_t elements) {
        float* buffer = nullptr;
        TILEMEGA_CUDA_CHECK(cudaMalloc(&buffer,elements*sizeof(float)));
        auto id = static_cast<std::uint32_t>(model.buffers.size());
        model.buffers.push_back(reinterpret_cast<ModelElement*>(buffer));
        model.host_sources.emplace_back();
        return id;
      };
      std::size_t queries = static_cast<std::size_t>(dims.seq)*stage.extent;
      stage.operand[4] = allocate_float(queries*dims.total);
      stage.operand[5] = allocate_float(queries*attention_chunks[i]*stage.width);
      stage.operand[6] = attention_chunks[i];
      for (auto phase : kAttentionExpandedPhases) {
        stage.operand[7] = static_cast<std::uint32_t>(phase);
        model.stages.push_back(stage);
      }
      done[i] = static_cast<std::uint32_t>(model.stages.size())-1;
      continue;
    }
    int chunks = stage.kind == TaskKind::kGemm ? gemm_chunks[stage.gemm] : 1;
    if (stage.kind == TaskKind::kGemm || stage.kind == TaskKind::kAdd || fused_gemm)
      stage.gemm = gemm_base[stage.gemm];
    model.stages.push_back(stage);
    done[i] = entry[i];
    if (chunks <= 1) continue;
    StageDesc combine = spec.stages[i];
    combine.kind = TaskKind::kGemmCombine;
    combine.gemm = gemm_base[spec.stages[i].gemm];
    combine.group = static_cast<std::uint32_t>(chunks);
    combine.width = spec.gemms[spec.stages[i].gemm].n;
    combine.operand[0] = gemm_partial[spec.stages[i].gemm];
    combine.operand[1] = spec.gemms[spec.stages[i].gemm].d;
    done[i] = static_cast<std::uint32_t>(model.stages.size());
    model.stages.push_back(combine);
  }
  std::vector<StageDependency> dependencies;
  for (std::uint32_t i = 0; i < spec.stage_count; ++i) {
    if (attention_chunks[i] > 1) {
      int chunks = static_cast<int>(attention_chunks[i]);
      for (auto const& dep : AttentionInternalDependencies(chunks))
        dependencies.push_back({entry[i]+dep.producer,entry[i]+dep.consumer,
            StageDependency::Map::kWindow,static_cast<std::uint32_t>(dep.div),
            dep.scale,0,static_cast<std::uint32_t>(dep.count)});
    } else if (done[i] != entry[i]) {
      if (model.params.ownership_flags & kCombinerTileOwnership) {
#if TILEMEGA_CG_SPLIT_TASK_ORDER
        // The chunk axis is contiguous in the CG task order, not storage order.
        int const chunks = gemm_chunks[spec.stages[i].gemm];
        dependencies.push_back(
            {entry[i], done[i], StageDependency::Map::kWindow, 1u, chunks,
             0, static_cast<std::uint32_t>(chunks)});
#else
        GemmInvocation const& invocation = gemms[model.stages[entry[i]].gemm];
        int const tiles = invocation.tiles_m * invocation.tiles_n;
        for (int chunk = 0; chunk < gemm_chunks[spec.stages[i].gemm]; ++chunk)
          dependencies.push_back(
              {entry[i], done[i], StageDependency::Map::kWindow, 1u, 1,
               chunk * tiles, 1u});
#endif
      } else {
        dependencies.push_back(
            {entry[i], done[i], StageDependency::Map::kAll, 1u, 0, 0, 1u});
      }
    }
    for (std::uint32_t e = runtime_variant.dependency_offsets[i];
         e < runtime_variant.dependency_offsets[i + 1]; ++e) {
      StageDependency edge = runtime_variant.dependencies[e];
      std::uint32_t const producer = edge.producer;
      edge.producer = done[producer];
      edge.consumer = entry[i];
      if (attention_chunks[i] > 1 && edge.map != StageDependency::Map::kAll) {
        if (edge.div > std::numeric_limits<std::uint32_t>::max()/attention_chunks[i]) {
          std::fprintf(stderr,"attention dependency divisor overflow\n"); std::exit(2);
        }
        edge.div *= attention_chunks[i];
      }
      // Split-K moves the producer event onto the combiner, which owns its
      // tasks by element chunk -- blockIdx no longer names the tile the
      // window was fitted against, so the edge falls back to kAll.
      if (spec.stages[producer].kind == TaskKind::kGemm && done[producer] != entry[producer] &&
          !(model.params.ownership_flags & kCombinerTileOwnership)) {
        edge.map = StageDependency::Map::kAll;
        edge.div = 1u;
        edge.scale = 0;
        edge.offset = 0;
        edge.count = 1u;
      }
      dependencies.push_back(edge);
    }
  }
  std::sort(dependencies.begin(), dependencies.end(),
            [](StageDependency const& a, StageDependency const& b) {
              return a.consumer < b.consumer;
            });
  std::vector<std::uint32_t> offsets(model.stages.size() + 1, 0);
  for (auto const& edge : dependencies) ++offsets[edge.consumer + 1];
  for (std::size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];

  // Expand the solver's variant schedule around host-inserted split-K
  // combiners, then prove the concrete stage order is acyclic before a kernel
  // can be launched.  Runtime dimensions change queue lengths, not this order.
  if (runtime_variant.schedule_count != spec.stage_count) {
    std::fprintf(stderr, "runtime variant schedule has %u/%u stages\n",
                 runtime_variant.schedule_count, spec.stage_count);
    std::exit(2);
  }
  std::vector<bool> scheduled_once(spec.stage_count, false);
  model.stage_order.reserve(model.stages.size());
  for (std::uint32_t i = 0; i < runtime_variant.schedule_count; ++i) {
    std::uint32_t const original = runtime_variant.schedule[i].stage;
    if (original >= spec.stage_count || scheduled_once[original]) {
      std::fprintf(stderr, "runtime variant schedule is not a permutation\n");
      std::exit(2);
    }
    scheduled_once[original] = true;
    model.stage_order.push_back(entry[original]);
    for (auto expanded = entry[original]+1; expanded <= done[original]; ++expanded)
      model.stage_order.push_back(expanded);
  }
  // Experimental control for Part 3.3.  Stage ids are emitted in the
  // frontend's original topological order, including an immediately-following
  // split-K combiner, so this is the deterministic round-robin baseline to
  // compare with the solver's critical-path priority.  The switch is consumed
  // only while materializing host queues; both arms execute identical device
  // code and are validated below before any launch.
  if (char const* policy = std::getenv("TILEMEGA_SCHEDULE_POLICY")) {
    if (std::strcmp(policy, "critical_path") == 0) {
      // The generated order already is the critical-path schedule.
    } else if (std::strcmp(policy, "round_robin") == 0) {
      std::iota(model.stage_order.begin(), model.stage_order.end(), 0u);
    } else {
      std::fprintf(stderr, "unknown TILEMEGA_SCHEDULE_POLICY=%s\n", policy);
      std::exit(2);
    }
  }
  std::vector<std::uint32_t> stage_position(model.stages.size());
  for (std::uint32_t i = 0; i < model.stage_order.size(); ++i)
    stage_position[model.stage_order[i]] = i;
  for (auto const& edge : dependencies) {
    if (stage_position[edge.producer] >= stage_position[edge.consumer]) {
      std::fprintf(stderr,
                   "task queue would wait backwards: stage %u -> %u\n",
                   edge.producer, edge.consumer);
      std::exit(2);
    }
    model.schedule_max_span = std::max(
        model.schedule_max_span,
        stage_position[edge.consumer] - stage_position[edge.producer]);
  }

  auto active_tasks = [&](std::uint32_t index) {
    StageDesc const& stage = model.stages[index];
    switch (stage.kind) {
      case TaskKind::kGemm:
      case TaskKind::kGemmAdd: {
        GemmInvocation const& invocation = gemms[stage.gemm];
        return invocation.tiles_m * invocation.tiles_n * invocation.chunks;
      }
      case TaskKind::kAdd: {
        GemmInvocation const& invocation = gemms[stage.gemm];
        return invocation.tiles_m * invocation.tiles_n;
      }
      case TaskKind::kRMSNorm:
      case TaskKind::kGemmRMSNorm: return dims.seq;
      case TaskKind::kRoPE:
        if (model.params.ownership_flags & kRoPETileOwnership)
          return dims.seq * static_cast<int>(stage.extent);
        return CeilDiv(dims.seq * static_cast<int>(stage.extent) *
                           (static_cast<int>(stage.width) / 2),
                       kHarnessThreads);
      case TaskKind::kKVAppend:
      case TaskKind::kRoPEKVAppend:
        if (model.params.ownership_flags & kKVTileOwnership)
          return dims.seq * static_cast<int>(stage.extent);
        return CeilDiv(std::max(dims.seq, dims.past) *
                           static_cast<int>(stage.extent) *
                           static_cast<int>(stage.width),
                       kHarnessThreads);
      case TaskKind::kElementwise:
        if (model.params.ownership_flags & kActivationTileOwnership)
          return dims.seq;
        return CeilDiv(dims.seq * static_cast<int>(stage.extent),
                       kHarnessThreads);
      case TaskKind::kAttention:
        return stage.operand[7] == kNoOperand || stage.operand[7] == 0
            ? dims.seq * static_cast<int>(stage.extent)
            : AttentionPhaseTasks(static_cast<AttentionPhase>(stage.operand[7]),
                dims.seq * static_cast<int>(stage.extent),stage.operand[6]);
      case TaskKind::kGemmCombine:
        if (model.params.ownership_flags & kCombinerTileOwnership) {
          GemmInvocation const& invocation = gemms[stage.gemm];
          return invocation.tiles_m * invocation.tiles_n;
        }
        return CeilDiv(dims.seq * static_cast<int>(stage.width),
                       kHarnessThreads);
    }
    return 0;
  };

  // Materialize one queue per physical CTA.  Event requirements are first
  // deduplicated for the task and then lifted out of later tasks in the same
  // worker queue.  The latter is valid only because epoch never decreases.
  model.schedule_offsets.resize(static_cast<std::size_t>(grid) + 1, 0);
#if TILEMEGA_FUSION_RUNTIME
  std::vector<int> exact_stage_offsets;
  std::vector<std::vector<std::pair<int,int>>> exact_predecessors;
  if (runtime_variant.exact_dependencies) {
    if (runtime_variant.balanced_placement) {
      std::fprintf(stderr,"fused exact dependencies require their selected stage-major placement\n");
      std::exit(2);
    }
    std::vector<int> counts;
    for (std::uint32_t stage=0;stage<model.stages.size();++stage) counts.push_back(active_tasks(stage));
    auto graph=MaterializeExactRuntimeTaskGraph(counts,*runtime_variant.exact_dependencies,
                                               dims.seq,dims.past,grid);
    exact_stage_offsets=graph.stage_offsets;
    exact_predecessors.resize(graph.successors.size());
    std::set<std::pair<int,int>> declared;
    for (auto const& edge:dependencies) declared.emplace(edge.producer,edge.consumer);
    for (std::size_t producer=0;producer<graph.successors.size();++producer) {
      int ps=std::upper_bound(graph.stage_offsets.begin(),graph.stage_offsets.end(),producer)-
             graph.stage_offsets.begin()-1;
      for (int consumer:graph.successors[producer]) {
        int cs=std::upper_bound(graph.stage_offsets.begin(),graph.stage_offsets.end(),consumer)-
               graph.stage_offsets.begin()-1;
        if (!declared.count({ps,cs})) {
          std::fprintf(stderr,"exact task dependency is missing its generated stage policy\n");
          std::exit(2);
        }
        exact_predecessors[consumer].emplace_back(ps,producer-graph.stage_offsets[ps]);
      }
    }
    std::printf("E2E_EXACT_FUSION task_refs=%zu stage_count=%zu\n",
                graph.successors.size(),counts.size());
  }
#else
  if (runtime_variant.exact_dependencies) {
    std::fprintf(stderr,"exact fusion schedule requires TILEMEGA_FUSION_RUNTIME=1\n");
    std::exit(2);
  }
#endif
  std::vector<std::set<std::pair<std::uint32_t, std::uint32_t>>> seen(grid);
#if TILEMEGA_EVENT_KAPPA > 0
  bool const force_all_dependencies =
      std::getenv("TILEMEGA_FORCE_ALL_DEPENDENCIES") != nullptr;
#endif
  std::vector<int> physical_worker(grid);
  for (int worker = 0; worker < grid; ++worker)
    physical_worker[HostPlacedBlock(worker, grid, blocks_per_sm)] = worker;

  // §5.7.4: the Plan is the only schedule source here.  TILEMEGA_PLACEMENT
  // still selects the CTA-to-SM map (Placement.cuh); 4 and 5 additionally name
  // an ownership decision, so a Plan that disagrees with them is a build error
  // rather than a silent override.
  auto plan_mode = dialect::PlacementMode::kLegacyGridStride;
#if TILEMEGA_PLACEMENT == 4
  plan_mode = dialect::PlacementMode::kBalanced;
#elif TILEMEGA_PLACEMENT == 5
  plan_mode = dialect::PlacementMode::kRotate;
#endif
  std::vector<std::int64_t> plan_params;
  {
    auto const carried =
        static_cast<dialect::PlacementMode>(runtime_variant.plan.mode);
    if (carried != dialect::PlacementMode::kLegacyGridStride) {
      if (plan_mode != dialect::PlacementMode::kLegacyGridStride &&
          plan_mode != carried) {
        std::fprintf(stderr,"compiled placement %d contradicts plan mode %s\n",
                     TILEMEGA_PLACEMENT,dialect::PlacementModeName(carried));
        std::exit(2);
      }
      plan_mode = carried;
      plan_params.assign(runtime_variant.plan.params,
                         runtime_variant.plan.params + runtime_variant.plan.param_count);
    }
    if (runtime_variant.plan.window != dialect::kPlacementWindowImplemented) {
      std::fprintf(stderr,"the executor implements window=1 only (§5.7.2)\n");
      std::exit(2);
    }
    if (runtime_variant.plan.policy != 0) {
      std::fprintf(stderr,"the executor implements the aot dispatch policy only\n");
      std::exit(2);
    }
  }

  // One runtime task DAG for the placement, the statistics and the legality
  // checks: three copies of the same projection used to be built per launch.
  std::vector<int> plan_counts;
  for (std::uint32_t stage=0;stage<model.stages.size();++stage)
    plan_counts.push_back(active_tasks(stage));
  std::vector<RuntimeDependencyWindow> plan_windows;
  for (auto const& edge:dependencies)
    plan_windows.push_back({static_cast<int>(edge.producer),static_cast<int>(edge.consumer),
        edge.map==StageDependency::Map::kAll,edge.div,edge.scale,edge.offset,edge.count});
  auto const runtime_graph=MaterializeRuntimeTaskGraph(plan_counts,plan_windows,grid);

#if TILEMEGA_PLACEMENT == 4
  if (!runtime_variant.balanced_placement) {
    std::fprintf(stderr,"placement=4 requires balanced L-sched writeback\n");
    std::exit(2);
  }
#else
  if (runtime_variant.balanced_placement &&
      plan_mode != dialect::PlacementMode::kBalanced) {
    std::fprintf(stderr,"balanced L-sched requires TILEMEGA_PLACEMENT=4\n");
    std::exit(2);
  }
#endif

  solver::PlanRequest plan_request;
  plan_request.mode=plan_mode;
  plan_request.params=plan_params;
  plan_request.grid=grid;
  plan_request.counts=plan_counts;
  plan_request.stage_order.assign(model.stage_order.begin(),model.stage_order.end());
  plan_request.physical_worker=physical_worker;
  plan_request.graph=&runtime_graph;
  solver::MaterializedPlan plan;
  std::string plan_error;
  if (!solver::MaterializePlanPlacement(plan_request,&plan,&plan_error)) {
    std::fprintf(stderr,"placement plan rejected: %s\n",plan_error.c_str());
    std::exit(2);
  }
  auto const& task_owner=plan.owner;
  if (plan.has_balanced_stats)
    std::printf("E2E_BALANCED max_queue=%d baseline_max_queue=%d same_worker_edges=%ld "
                "fence_free_producers=%ld resident_only=1\n",plan.balanced.max_queue,
                plan.balanced.baseline_max_queue,plan.balanced.same_worker_edges,
                plan.balanced.fence_free_producers);
  if (!plan.rotate_base.empty() &&
      std::getenv("TILEMEGA_PLACEMENT_BASE_DUMP") != nullptr)
    for (std::uint32_t i = 0; i < model.stage_order.size(); ++i)
      std::printf("E2E_PLACE_BASE pos=%u stage=%u active=%d base=%d\n", i,
                  model.stage_order[i], active_tasks(model.stage_order[i]),
                  plan.rotate_base[model.stage_order[i]]);
  // L-a and L-c on the materialized plan.  L-b was checked before the grid was
  // fixed (`ResidentScheduleLegal`), L-e below with the event tables, and L-d
  // collapses to the current hoisting rules only because W = 1 (§5.7.3).
  if (!solver::CheckPlanLegality(runtime_graph,plan,&plan_error)) {
    std::fprintf(stderr,"placement plan is illegal: %s\n",plan_error.c_str());
    std::exit(2);
  }
  {
    auto owner_of=[&](int node){
      int const stage=static_cast<int>(std::upper_bound(runtime_graph.stage_offsets.begin(),
          runtime_graph.stage_offsets.end(),node)-runtime_graph.stage_offsets.begin()-1);
      return task_owner[stage][node-runtime_graph.stage_offsets[stage]];
    };
    long same_worker_edges=0,cross_worker_edges=0;
    for (std::size_t producer=0;producer<runtime_graph.successors.size();++producer)
      for (int consumer:runtime_graph.successors[producer])
        (owner_of(static_cast<int>(producer))==owner_of(consumer) ? same_worker_edges
                                                                 : cross_worker_edges)++;
    std::vector<int> queue_length(grid,0);
    for (auto const& stage:task_owner)
      for (int owner:stage) ++queue_length[owner];
    int max_queue=0;
    for (int length:queue_length) max_queue=std::max(max_queue,length);
    std::printf("E2E_PLACE_STATS placement=%d max_queue=%d same_worker_edges=%ld "
                "cross_worker_edges=%ld base_rotation=%d plan=%s\n",
                TILEMEGA_PLACEMENT,max_queue,same_worker_edges,cross_worker_edges,
                plan_mode==dialect::PlacementMode::kRotate?1:0,
                dialect::PlacementModeName(plan_mode));
  }
  std::vector<int> stage_max_producer_worker(model.stages.size(), -1);
  for (std::uint32_t stage = 0; stage < model.stages.size(); ++stage) {
    for (int owner : task_owner[stage])
      stage_max_producer_worker[stage] = std::max(
          stage_max_producer_worker[stage], owner);
  }
#if TILEMEGA_EVENT_KAPPA > 0
  int const per_group = TILEMEGA_EVENT_KAPPA;
#endif
  model.event_offsets.resize(model.stages.size() + 1, 0);
  model.event_flags.resize(model.stages.size(), 0);
  for (StageDependency const& dep : dependencies) {
#if TILEMEGA_EVENT_KAPPA > 0
    if (force_all_dependencies || dep.map == StageDependency::Map::kAll)
      model.event_flags[dep.producer] |= kNeedsAggregateEvent;
    else
      model.event_flags[dep.producer] |= kNeedsFineEvents;
#else
    model.event_flags[dep.producer] |= kNeedsAggregateEvent;
#endif
  }
  for (std::uint32_t stage = 0; stage < model.stages.size(); ++stage) {
    int const count = active_tasks(stage);
#if TILEMEGA_TRACE_V2
    model.trace_v2_active_tasks.push_back(static_cast<std::uint32_t>(count));
#endif
#if TILEMEGA_EVENT_KAPPA > 0
    std::uint32_t groups = 0;
    if (model.event_flags[stage] & kNeedsAggregateEvent) ++groups;
    if (model.event_flags[stage] & kNeedsFineEvents)
      groups += static_cast<std::uint32_t>(
          (count + TILEMEGA_EVENT_KAPPA - 1) / TILEMEGA_EVENT_KAPPA);
#else
    std::uint32_t const groups =
        (model.event_flags[stage] & kNeedsAggregateEvent) ? 1u : 0u;
#endif
    model.event_offsets[stage + 1] = model.event_offsets[stage] + groups;
  }
#if TILEMEGA_EVENT_SHARDED
  static_assert(TILEMEGA_EVENT_SHARDS >= 0, "negative shard count");
  // Automatic choice: the largest power of two no greater than num_sms.
  // Explicit values are experimental controls and may not exceed hardware.
  std::uint32_t shards = 1;
  while (shards <= static_cast<std::uint32_t>(target.res.num_sms) / 2) shards *= 2;
  if (TILEMEGA_EVENT_SHARDS > 0) shards = TILEMEGA_EVENT_SHARDS;
  if (shards > static_cast<std::uint32_t>(target.res.num_sms)) {
    std::fprintf(stderr, "requested shard count exceeds TargetSpec::Res::num_sms\n");
    std::exit(2);
  }
#if TILEMEGA_EVENT_CLUSTER_FANIN
  shards = grid / TILEMEGA_GENERATED_CLUSTER_DIM;
  std::vector<std::vector<std::uint32_t>> cluster_shards(shards);
#endif
  model.params.event_shard_count = shards;
  model.event_fanin.resize(model.stages.size() + model.event_offsets.back());
  for (std::uint32_t stage = 0; stage < model.stages.size(); ++stage) {
    int const produced = active_tasks(stage);
    auto add = [&](std::uint32_t row, int begin, int end) {
      EventFanIn plan{};
      plan.begin = static_cast<std::uint32_t>(model.shard_targets.size());
      plan.modulus = std::min(shards, static_cast<std::uint32_t>(end - begin));
      if (plan.modulus == 0) { model.event_fanin[row] = plan; return; }
#if TILEMEGA_EVENT_CLUSTER_FANIN
      // Singleton events publish directly and need no DSMEM slot. For other
      // events store only the interval of clusters containing producers.
      if (end - begin == 1) { model.event_fanin[row] = {}; return; }
      std::uint32_t first = shards, last = 0;
      for (int logical = begin; logical < end; ++logical) {
        unsigned const cluster = task_owner[stage][logical] / TILEMEGA_GENERATED_CLUSTER_DIM;
        first = std::min(first, cluster); last = std::max(last, cluster);
      }
      plan.first_cluster = first;
      plan.modulus = last - first + 1;
#endif
      model.shard_targets.resize(plan.begin + plan.modulus, 0);
      for (int logical = begin; logical < end; ++logical) {
        int const worker = task_owner[stage][logical];
#if TILEMEGA_EVENT_CLUSTER_FANIN
        ++model.shard_targets[plan.begin + worker / TILEMEGA_GENERATED_CLUSTER_DIM - plan.first_cluster];
#else
        ++model.shard_targets[plan.begin + worker % plan.modulus];
#endif
      }
      for (std::uint32_t i = 0; i < plan.modulus; ++i)
        if (model.shard_targets[plan.begin + i]) {
          ++plan.nonempty;
#if TILEMEGA_EVENT_CLUSTER_FANIN
          cluster_shards[plan.first_cluster + i].push_back(plan.begin + i);
#endif
        }
      model.event_fanin[row] = plan;
    };
    std::uint32_t row = static_cast<std::uint32_t>(model.stages.size()) + model.event_offsets[stage];
    if (model.event_flags[stage] & kNeedsAggregateEvent) add(row++, 0, produced);
#if TILEMEGA_EVENT_KAPPA > 0
    if (model.event_flags[stage] & kNeedsFineEvents)
      for (int begin = 0; begin < produced; begin += TILEMEGA_EVENT_KAPPA)
        add(row++, begin, std::min(produced, begin + TILEMEGA_EVENT_KAPPA));
#endif
  }
#if TILEMEGA_EVENT_CLUSTER_FANIN
  model.shard_local_offsets.resize(model.shard_targets.size(), 0);
  model.cluster_shard_offsets.push_back(0);
  std::size_t max_local = 0;
  for (auto const& local : cluster_shards) {
    max_local = std::max(max_local, local.size());
    for (std::size_t i = 0; i < local.size(); ++i) {
      model.shard_local_offsets[local[i]] = static_cast<std::uint32_t>(i);
      model.cluster_shard_indices.push_back(local[i]);
    }
    model.cluster_shard_offsets.push_back(model.cluster_shard_indices.size());
  }
  std::size_t const needed = sizeof(TaskSmem) + max_local * sizeof(unsigned long long);
  if (needed > model.l2_smem_bytes) {
    std::fprintf(stderr, "cluster counters need %zu shared bytes, TargetSpec budget is %zu\n",
                 needed, model.l2_smem_bytes);
    std::exit(2);
  }
  std::printf("E2E_CLUSTER_FANIN dim=%d local_counters=%zu needed_smem=%zu reserved_smem=%zu\n",
              TILEMEGA_GENERATED_CLUSTER_DIM, max_local, needed, model.l2_smem_bytes);
#endif
  std::printf("E2E_FANIN shards=%u first_level_counters=%zu bytes=%zu\n", shards,
              model.shard_targets.size(), model.shard_targets.size() * sizeof(ArrivalCounter));
#else
  (void)target;
  model.params.event_shard_count = 1;
#endif
  for (int worker = 0; worker < grid; ++worker) {
    // sigma, not stage-major: the queue is whatever order the Plan asked for,
    // and `plan.queue[worker]` is already sorted by it (§5.7.2).
    for (auto const& item : plan.queue[worker]) {
      std::uint32_t const stage = item.stage;
      int const logical = item.logical;
      {
        TaskRef task{};
        task.stage = stage;
        task.logical_task = static_cast<std::uint32_t>(logical);
        task.dependency_begin = offsets[stage];
        task.dependency_count = offsets[stage + 1] - offsets[stage];
        task.wait_begin = static_cast<std::uint32_t>(model.task_waits.size());
        std::set<std::pair<std::uint32_t, std::uint32_t>> desired;
        for (std::uint32_t e = offsets[stage]; e < offsets[stage + 1]; ++e) {
          // Queue-era negative control at the point L2 actually consumes:
          // truncate every TaskWait interval to zero. Never enabled in a
          // shipped build; SEQSCAN requires this build to fail.
          if (TILEMEGA_NEGATIVE_TASK_WAIT_CLAMP) break;
          StageDependency const& dep = dependencies[e];
          int const produced = active_tasks(dep.producer);
#if TILEMEGA_EVENT_KAPPA > 0
          auto observe_owner = [&](int owner) {
            int const producer_worker = owner;
            if (producer_worker > worker)
              model.schedule_max_worker_span = std::max(
                  model.schedule_max_worker_span,
                  static_cast<std::uint32_t>(producer_worker - worker));
          };
          if (force_all_dependencies || dep.map == StageDependency::Map::kAll) {
            model.schedule_has_global_fanin = true;
            desired.emplace(dep.producer, kWholeStageEventGroup);
            int const latest = stage_max_producer_worker[dep.producer];
            if (latest > worker)
              model.schedule_max_worker_span = std::max(
                  model.schedule_max_worker_span,
                  static_cast<std::uint32_t>(latest - worker));
          } else {
            int const at =
                (logical / static_cast<int>(dep.div)) * dep.scale + dep.offset;
            int const begin = std::max(at, 0);
            int const end = std::min(
                at + static_cast<int>(dep.count), produced);
            auto require_task = [&](int producer_task) {
              int const owner = task_owner[dep.producer][producer_task];
              int const group = producer_task / per_group;
              int const group_end = std::min((group + 1) * per_group, produced);
              for (int member = group * per_group; member < group_end; ++member)
                observe_owner(task_owner[dep.producer][member]);
              // With one worker per event, an earlier entry in *this* queue is
              // already a proof; no global-memory poll is needed.  The test is
              // on sigma, not on the stage order, because a Plan may interleave
              // stages on one worker.  L-c has already rejected a same-worker
              // producer that does not come first, so this cannot silently drop
              // a poll that was actually needed.
              if (per_group == 1 && owner == worker &&
                  plan.slot[dep.producer][producer_task] <
                      plan.slot[stage][logical])
                return;
              desired.emplace(dep.producer,
                              static_cast<std::uint32_t>(group));
            };
#if TILEMEGA_FUSION_RUNTIME
            if (runtime_variant.exact_dependencies) {
              for (auto const& incoming:exact_predecessors[exact_stage_offsets[stage]+logical])
                if (incoming.first==int(dep.producer)) require_task(incoming.second);
            } else
#endif
              for (int producer_task=begin;producer_task<end;++producer_task)
                require_task(producer_task);
          }
#else
          (void)produced;
          model.schedule_has_global_fanin = true;
          int const latest = stage_max_producer_worker[dep.producer];
          if (latest > worker)
            model.schedule_max_worker_span = std::max(
                model.schedule_max_worker_span,
                static_cast<std::uint32_t>(latest - worker));
          desired.emplace(dep.producer, kWholeStageEventGroup);
#endif
        }
        model.schedule_raw_polls += desired.size();
        for (auto const& wait : desired) {
          // L-e: a consumer may only wait on a row its producer publishes.
          std::uint32_t const needed = wait.second == kWholeStageEventGroup
                                           ? kNeedsAggregateEvent
                                           : kNeedsFineEvents;
          if ((model.event_flags[wait.first] & needed) == 0) {
            std::fprintf(stderr,
                         "L-e: stage %u waits on an event stage %u never publishes\n",
                         stage, wait.first);
            std::exit(2);
          }
          if (seen[worker].insert(wait).second)
            model.task_waits.push_back({wait.first, wait.second});
        }
        task.wait_count = static_cast<std::uint32_t>(model.task_waits.size()) -
                          task.wait_begin;
        if (task.wait_count != 0) ++model.schedule_waiting_tasks;
        if (task.wait_count > 1)
          model.normalization_dummy_lower_bound += task.wait_count - 1;
        model.schedule.push_back(task);
      }
    }
    model.schedule_offsets[worker + 1] =
        static_cast<std::uint32_t>(model.schedule.size());
  }

  // A deterministic view of what the Plan materialized, written before any
  // device work so the H2/H3 byte identities compare tables, not timings.
  if (char const* dump_dir = std::getenv("TILEMEGA_PLAN_DUMP")) {
    std::string const base = std::string(dump_dir) + "/";
    auto open_dump = [&](char const* name) {
      std::FILE* f = std::fopen((base + name).c_str(), "w");
      if (f == nullptr) {
        std::fprintf(stderr,"cannot write %s%s\n",base.c_str(),name);
        std::exit(2);
      }
      return f;
    };
    std::FILE* f = open_dump("schedule.tsv");
    std::fprintf(f,"worker\tslot\tstage\tlogical_task\tdependency_begin\t"
                   "dependency_count\twait_begin\twait_count\n");
    for (int worker = 0; worker < grid; ++worker)
      for (std::uint32_t i = model.schedule_offsets[worker];
           i < model.schedule_offsets[worker + 1]; ++i) {
        TaskRef const& t = model.schedule[i];
        std::fprintf(f,"%d\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",worker,
                     i - model.schedule_offsets[worker],t.stage,t.logical_task,
                     t.dependency_begin,t.dependency_count,t.wait_begin,t.wait_count);
      }
    std::fclose(f);
    f = open_dump("waits.tsv");
    std::fprintf(f,"wait_index\tproducer\tgroup\n");
    for (std::size_t i = 0; i < model.task_waits.size(); ++i)
      std::fprintf(f,"%zu\t%u\t%u\n",i,model.task_waits[i].producer,
                   model.task_waits[i].group);
    std::fclose(f);
    f = open_dump("events.tsv");
    std::fprintf(f,"stage\tevent_offset\tevent_flags\n");
    for (std::size_t stage = 0; stage < model.stages.size(); ++stage)
      std::fprintf(f,"%zu\t%u\t%u\n",stage,model.event_offsets[stage],
                   model.event_flags[stage]);
    std::fclose(f);
  }

  auto upload = [](void const* host, std::size_t bytes) {
    void* device = nullptr;
    TILEMEGA_CUDA_CHECK(cudaMalloc(&device, bytes));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    return device;
  };
  model.device_buffers = static_cast<ModelElement**>(
      upload(model.buffers.data(), model.buffers.size() * sizeof(ModelElement*)));
  model.device_gemms = static_cast<GemmInvocation*>(
      upload(gemms.data(), gemms.size() * sizeof(GemmInvocation)));
  model.device_stages = static_cast<StageDesc*>(upload(
      model.stages.data(), model.stages.size() * sizeof(StageDesc)));
  if (!dependencies.empty())
    model.device_dependencies = static_cast<StageDependency*>(upload(
        dependencies.data(), dependencies.size() * sizeof(StageDependency)));
  model.device_dependency_offsets = static_cast<std::uint32_t*>(
      upload(offsets.data(), offsets.size() * sizeof(std::uint32_t)));
  if (!model.schedule.empty())
    model.device_schedule = static_cast<TaskRef*>(upload(
        model.schedule.data(), model.schedule.size() * sizeof(TaskRef)));
  model.device_schedule_offsets = static_cast<std::uint32_t*>(upload(
      model.schedule_offsets.data(),
      model.schedule_offsets.size() * sizeof(std::uint32_t)));
  if (!model.task_waits.empty())
    model.device_task_waits = static_cast<TaskWait*>(upload(
        model.task_waits.data(), model.task_waits.size() * sizeof(TaskWait)));
  model.device_event_offsets = static_cast<std::uint32_t*>(upload(
      model.event_offsets.data(),
      model.event_offsets.size() * sizeof(std::uint32_t)));
  model.device_event_flags = static_cast<std::uint32_t*>(upload(
      model.event_flags.data(),
      model.event_flags.size() * sizeof(std::uint32_t)));
#if TILEMEGA_EVENT_SHARDED
  model.device_event_fanin = static_cast<EventFanIn*>(upload(
      model.event_fanin.data(), model.event_fanin.size() * sizeof(EventFanIn)));
  model.device_shard_targets = static_cast<std::uint32_t*>(upload(
      model.shard_targets.data(), model.shard_targets.size() * sizeof(std::uint32_t)));
  TILEMEGA_CUDA_CHECK(cudaMalloc(&model.device_shard_arrivals,
      model.shard_targets.size() * sizeof(ArrivalCounter)));
#if TILEMEGA_EVENT_CLUSTER_FANIN
  model.device_shard_local_offsets = static_cast<std::uint32_t*>(upload(
      model.shard_local_offsets.data(), model.shard_local_offsets.size() * sizeof(std::uint32_t)));
  model.device_cluster_shard_offsets = static_cast<std::uint32_t*>(upload(
      model.cluster_shard_offsets.data(), model.cluster_shard_offsets.size() * sizeof(std::uint32_t)));
  model.device_cluster_shard_indices = static_cast<std::uint32_t*>(upload(
      model.cluster_shard_indices.data(), model.cluster_shard_indices.size() * sizeof(std::uint32_t)));
#endif
#endif
#if TILEMEGA_TRACE_V2
  // Allocation waits for PrepareEvents, which is where event_count is known.
  model.trace_v2_enabled = TraceV2Enabled();
#endif
  if (std::getenv("TILEMEGA_TASK_TRACE") != nullptr) {
    TILEMEGA_CUDA_CHECK(cudaMalloc(
        &model.device_task_trace, model.schedule.size() * sizeof(TaskTrace)));
    TILEMEGA_CUDA_CHECK(cudaMemset(
        model.device_task_trace, 0, model.schedule.size() * sizeof(TaskTrace)));
    TILEMEGA_CUDA_CHECK(
        cudaMalloc(&model.device_trace_sequence, sizeof(unsigned long long)));
    TILEMEGA_CUDA_CHECK(
        cudaMemset(model.device_trace_sequence, 0, sizeof(unsigned long long)));
  }

  model.params.dims = dims;
  model.params.buffers = model.device_buffers;
  model.params.gemms = model.device_gemms;
  model.params.stages = model.device_stages;
  model.params.stage_count = static_cast<std::uint32_t>(model.stages.size());
  model.params.dependencies = model.device_dependencies;
  model.params.dependency_count =
      static_cast<std::uint32_t>(dependencies.size());
  model.params.dependency_offsets = model.device_dependency_offsets;
  model.params.schedule = model.device_schedule;
  model.params.schedule_count = static_cast<std::uint32_t>(model.schedule.size());
  model.params.schedule_offsets = model.device_schedule_offsets;
  model.params.task_waits = model.device_task_waits;
  model.params.task_wait_count =
      static_cast<std::uint32_t>(model.task_waits.size());
  model.params.event_offsets = model.device_event_offsets;
  model.params.event_flags = model.device_event_flags;
  model.params.event_fanin = model.device_event_fanin;
  model.params.shard_arrivals = model.device_shard_arrivals;
  model.params.shard_targets = model.device_shard_targets;
  model.params.shard_local_offsets = model.device_shard_local_offsets;
  model.params.cluster_shard_offsets = model.device_cluster_shard_offsets;
  model.params.cluster_shard_indices = model.device_cluster_shard_indices;
  model.params.task_trace = model.device_task_trace;
  model.params.trace_sequence = model.device_trace_sequence;
  model.params.ownership_flags = runtime_variant.ownership_flags;
  TILEMEGA_CUDA_CHECK(cudaMalloc(&model.device_params, sizeof(Params)));
  TILEMEGA_CUDA_CHECK(cudaMemcpy(model.device_params, &model.params,
                                 sizeof(Params), cudaMemcpyHostToDevice));
  return model;
}

inline void PrepareEvents(DeviceModel& model, int grid) {
  (void)grid;
  if (model.events) TILEMEGA_CUDA_CHECK(cudaFree(model.events));
  model.event_count = static_cast<std::size_t>(model.params.stage_count) +
                      model.event_offsets.back();
  TILEMEGA_CUDA_CHECK(
      cudaMalloc(&model.events, sizeof(EventCounter) * model.event_count));
#if TILEMEGA_TRACE_V2
  if (model.trace_v2_enabled && model.device_task_trace_v2 == nullptr) {
    TILEMEGA_CUDA_CHECK(cudaMalloc(&model.device_task_trace_v2,
                                   model.schedule.size() * sizeof(TaskTraceV2)));
    TILEMEGA_CUDA_CHECK(cudaMalloc(&model.device_event_publish,
                                   model.event_count * sizeof(unsigned long long)));
    model.params.task_trace_v2 = model.device_task_trace_v2;
    model.params.event_publish = model.device_event_publish;
    // device_params was uploaded by Create, before event_count existed.
    TILEMEGA_CUDA_CHECK(cudaMemcpy(model.device_params, &model.params,
                                   sizeof(Params), cudaMemcpyHostToDevice));
  }
  ZeroTraceV2(model);
#endif
}

/// Restore every buffer to its pre-run contents so two launches see the same
/// input.  File-backed buffers are re-uploaded, scratch is zeroed.
/// Restore every buffer to its launch state, leaving the event counters
/// alone. §8.2's monotonic counters are meant to survive across iterations,
/// so the repeat-iteration check must not clear them.
inline void ResetBuffersOnly(DeviceModel& model) {
  ModelSpec const& spec = *model.spec;
  for (std::uint32_t i = 0; i < spec.buffer_count; ++i) {
    std::size_t bytes = spec.buffers[i].Elements(model.params.dims) *
                        sizeof(ModelElement);
    if (spec.buffers[i].file != nullptr)
      TILEMEGA_CUDA_CHECK(cudaMemcpy(model.buffers[i],
                                     model.host_sources[i].data(), bytes,
                                     cudaMemcpyHostToDevice));
    else
      TILEMEGA_CUDA_CHECK(cudaMemset(model.buffers[i], 0, bytes));
  }
  if (model.params.ownership_flags & kKVTileOwnership) {
    for (StageDesc const& stage : model.stages) {
      if ((stage.kind != TaskKind::kKVAppend && stage.kind != TaskKind::kRoPEKVAppend) ||
          model.params.dims.past == 0)
        continue;
      int const heads = static_cast<int>(stage.extent);
      int const dim = static_cast<int>(stage.width);
      std::size_t const head_bytes =
          static_cast<std::size_t>(model.params.dims.past) * dim *
          sizeof(ModelElement);
      for (int head = 0; head < heads; ++head)
        TILEMEGA_CUDA_CHECK(cudaMemcpy(
            model.buffers[stage.operand[stage.kind==TaskKind::kRoPEKVAppend ? 3 : 2]] +
                static_cast<std::size_t>(head) * model.params.dims.total * dim,
            model.buffers[stage.operand[stage.kind==TaskKind::kRoPEKVAppend ? 2 : 1]] +
                static_cast<std::size_t>(head) * model.params.dims.past * dim,
            head_bytes, cudaMemcpyDeviceToDevice));
    }
  }
}

inline void Reset(DeviceModel& model) {
  ResetBuffersOnly(model);
  if (model.events)
    TILEMEGA_CUDA_CHECK(cudaMemset(model.events, 0,
                                   sizeof(EventCounter) * model.event_count));
  if (model.device_shard_arrivals)
    TILEMEGA_CUDA_CHECK(cudaMemset(model.device_shard_arrivals, 0,
        sizeof(ArrivalCounter) * model.shard_targets.size()));
}

inline std::vector<std::vector<ModelElement>> Download(DeviceModel const& model) {
  ModelSpec const& spec = *model.spec;
  std::vector<std::vector<ModelElement>> output(spec.output_count);
  for (std::uint32_t i = 0; i < spec.output_count; ++i) {
    output[i].resize(
        spec.buffers[spec.outputs[i].buffer].Elements(model.params.dims));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(output[i].data(),
                                   model.buffers[spec.outputs[i].buffer],
                                   output[i].size() * sizeof(ModelElement),
                                   cudaMemcpyDeviceToHost));
  }
  return output;
}

#if TILEMEGA_TRACE_V2
/// Write the four tables §3.4 fixes.  Nothing here is summarised: the offline
/// analysis is a separate script so the raw stamps stay auditable.
inline void DumpTraceV2(DeviceModel const& model, char const* fixture_dir,
                        int grid, float l1_ms, float l2_ms,
                        char const* traced_launch) {
  if (!model.trace_v2_enabled || model.device_task_trace_v2 == nullptr) return;
  char const* dir = std::getenv("TILEMEGA_TRACE_V2_OUT");
  if (dir == nullptr) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  std::vector<TaskTraceV2> slots(model.schedule.size());
  TILEMEGA_CUDA_CHECK(cudaMemcpy(slots.data(), model.device_task_trace_v2,
                                 slots.size() * sizeof(TaskTraceV2),
                                 cudaMemcpyDeviceToHost));
  std::vector<unsigned long long> publish(model.event_count);
  TILEMEGA_CUDA_CHECK(cudaMemcpy(publish.data(), model.device_event_publish,
                                 publish.size() * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost));

  std::string const base = std::string(dir) + "/";
  auto open = [&](char const* name) {
    std::FILE* f = std::fopen((base + name).c_str(), "w");
    if (f == nullptr) {
      std::fprintf(stderr, "cannot write %s%s\n", base.c_str(), name);
      std::exit(2);
    }
    return f;
  };

  std::FILE* f = open("slots.tsv");
  std::fprintf(f, "slot\tworker\tstage\tlogical_task\tsmid\twait_begin\tready\t"
                  "run_begin\trun_end\tpublish_end\twait_begin_idx\twait_count\t"
                  "dependency_begin\tdependency_count\trun_begin_clk\trun_end_clk\n");
  for (std::size_t i = 0; i < slots.size(); ++i) {
    TaskTraceV2 const& r = slots[i];
    TaskRef const& t = model.schedule[i];
    std::fprintf(f, "%zu\t%u\t%u\t%u\t%u\t%llu\t%llu\t%llu\t%llu\t%llu\t%u\t%u\t%u\t%u\t%llu\t%llu\n",
                 i, r.worker, t.stage, t.logical_task, r.smid, r.wait_begin,
                 r.ready, r.run_begin, r.run_end, r.publish_end, t.wait_begin,
                 t.wait_count, t.dependency_begin, t.dependency_count,
                 r.run_begin_clk, r.run_end_clk);
  }
  std::fclose(f);

  // TaskWait carries only (producer, group); the row a device poll lands on is
  // recomputed here with EventIndex's formula so the join is explicit offline.
  auto event_index = [&](std::uint32_t stage, std::uint32_t group) {
    std::uint32_t const row =
        static_cast<std::uint32_t>(model.stages.size()) + model.event_offsets[stage];
#if TILEMEGA_EVENT_KAPPA > 0
    return group == kWholeStageEventGroup
               ? row
               : row + ((model.event_flags[stage] & kNeedsAggregateEvent) ? 1u : 0u) +
                     group;
#else
    (void)group;
    return row;
#endif
  };

  f = open("waits.tsv");
  std::fprintf(f, "wait_index\tproducer\tgroup\tevent_index\n");
  for (std::size_t i = 0; i < model.task_waits.size(); ++i) {
    TaskWait const& w = model.task_waits[i];
    std::fprintf(f, "%zu\t%u\t%u\t%u\n", i, w.producer, w.group,
                 event_index(w.producer, w.group));
  }
  std::fclose(f);

  f = open("events.tsv");
  std::fprintf(f, "event_index\tstage\tgroup\tfanin\tpublish_ns\n");
  // L1 owns the first stage_count rows and L2 never publishes them; they are
  // listed with fan-in 0 so event_index stays the array index.
  for (std::uint32_t stage = 0; stage < model.stages.size(); ++stage)
    std::fprintf(f, "%u\t%u\t%u\t0\t%llu\n", stage, stage,
                 kWholeStageEventGroup, publish[stage]);
  for (std::uint32_t stage = 0; stage < model.stages.size(); ++stage) {
    std::uint32_t const produced = model.trace_v2_active_tasks[stage];
    std::uint32_t row =
        static_cast<std::uint32_t>(model.stages.size()) + model.event_offsets[stage];
    if (model.event_flags[stage] & kNeedsAggregateEvent) {
      std::fprintf(f, "%u\t%u\t%u\t%u\t%llu\n", row, stage,
                   kWholeStageEventGroup, produced, publish[row]);
      ++row;
    }
#if TILEMEGA_EVENT_KAPPA > 0
    if (model.event_flags[stage] & kNeedsFineEvents)
      for (std::uint32_t begin = 0; begin < produced;
           begin += TILEMEGA_EVENT_KAPPA, ++row)
        std::fprintf(f, "%u\t%u\t%u\t%u\t%llu\n", row, stage,
                     begin / TILEMEGA_EVENT_KAPPA,
                     std::min<std::uint32_t>(TILEMEGA_EVENT_KAPPA, produced - begin),
                     publish[row]);
#endif
  }
  std::fclose(f);

  char const* policy = std::getenv("TILEMEGA_SCHEDULE_POLICY");
  char const* name = std::getenv("TILEMEGA_MODEL_NAME");
  char const* commit = std::getenv("TILEMEGA_COMMIT");
  char const* tick = std::getenv("TILEMEGA_GLOBALTIMER_NS");
  benchmark::Settings const timing;
  f = open("meta.tsv");
  std::fprintf(f, "key\tvalue\n");
  std::fprintf(f, "model\t%s\n", name != nullptr ? name : "unset");
  std::fprintf(f, "fixture\t%s\n", fixture_dir);
  std::fprintf(f, "seq\t%d\n", model.params.dims.seq);
  std::fprintf(f, "past\t%d\n", model.params.dims.past);
  std::fprintf(f, "grid\t%d\n", grid);
  std::fprintf(f, "kappa\t%d\n", TILEMEGA_EVENT_KAPPA);
  std::fprintf(f, "placement\t%d\n", TILEMEGA_PLACEMENT);
  std::fprintf(f, "schedule_policy\t%s\n",
               policy != nullptr ? policy : "critical_path");
  std::fprintf(f, "stage_count\t%zu\n", model.stages.size());
  std::fprintf(f, "slot_count\t%zu\n", model.schedule.size());
  std::fprintf(f, "event_count\t%zu\n", model.event_count);
  std::fprintf(f, "l2_ms\t%.6f\n", l2_ms);
  std::fprintf(f, "l1_ms\t%.6f\n", l1_ms);
  std::fprintf(f, "traced_launch\t%s\n", traced_launch);
  std::fprintf(f, "benchmark_warmup\t%d\n", timing.warmup);
  std::fprintf(f, "benchmark_repeat\t%d\n", timing.repeat);
  std::fprintf(f, "globaltimer_resolution_ns\t%s\n",
               tick != nullptr ? tick : "unset");
  std::fprintf(f, "commit\t%s\n", commit != nullptr ? commit : "unset");
  std::fclose(f);

  std::printf("E2E_TRACE_V2 slots=%zu events=%zu traced_launch=%s out=%s\n",
              slots.size(), model.event_count, traced_launch, dir);
}
#endif

inline void ReportTaskTrace(DeviceModel const& model) {
  if (model.device_task_trace == nullptr) return;
  std::vector<TaskTrace> trace(model.schedule.size());
  TILEMEGA_CUDA_CHECK(cudaMemcpy(trace.data(), model.device_task_trace,
                                 trace.size() * sizeof(TaskTrace),
                                 cudaMemcpyDeviceToHost));
  std::vector<unsigned long long> stage_end(model.stages.size(), 0);
  std::vector<std::uint32_t> position(model.stages.size(), 0);
  for (std::uint32_t i = 0; i < model.stage_order.size(); ++i)
    position[model.stage_order[i]] = i;
  for (std::size_t i = 0; i < trace.size(); ++i)
    stage_end[model.schedule[i].stage] =
        std::max(stage_end[model.schedule[i].stage], trace[i].end);
  std::size_t early = 0;
  std::size_t overlap_pairs = 0;
  for (std::size_t i = 0; i < trace.size(); ++i) {
    bool any = false;
    std::uint32_t const rank = position[model.schedule[i].stage];
    for (std::uint32_t prior = 0; prior < rank; ++prior) {
      std::uint32_t const stage = model.stage_order[prior];
      if (stage_end[stage] > trace[i].start) {
        any = true;
        ++overlap_pairs;
      }
    }
    if (any) ++early;
  }
  std::size_t transitions = 0;
  for (std::size_t worker = 0; worker + 1 < model.schedule_offsets.size();
       ++worker)
    for (std::uint32_t i = model.schedule_offsets[worker] + 1;
         i < model.schedule_offsets[worker + 1]; ++i)
      if (model.schedule[i - 1].stage != model.schedule[i].stage)
        ++transitions;
  double const pct = trace.empty() ? 0.0 : 100.0 * early / trace.size();
  std::printf("E2E_OVERLAP task_starts=%zu cross_stage_early=%zu "
              "cross_stage_early_pct=%.4f overlap_pairs=%zu "
              "queue_stage_transitions=%zu stage_control_early=0\n",
              trace.size(), early, pct, overlap_pairs, transitions);
}

inline void DumpBuffers(DeviceModel const& model, char const* directory) {
  if (!directory || !*directory) return;
  ModelSpec const& spec = *model.spec;
  for (std::uint32_t i = 0; i < spec.buffer_count; ++i) {
    std::vector<ModelElement> host(
        spec.buffers[i].Elements(model.params.dims));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(host.data(), model.buffers[i],
                                   host.size() * sizeof(ModelElement),
                                   cudaMemcpyDeviceToHost));
    std::string path = std::string(directory) + "/buffer_" +
                       std::to_string(i) + ".bin";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<char const*>(host.data()),
                 static_cast<std::streamsize>(host.size() *
                                              sizeof(ModelElement)));
  }
}

struct Difference {
  std::size_t mismatch = 0;
  float max_abs = 0;
  float max_rel = 0;
};

/// `max_abs` and `max_rel` are independent maxima over the whole comparison, so
/// neither identifies the element that actually exceeded the bound.  Set
/// TILEMEGA_DIFF_DUMP=<n> to print the first `n` offending elements with both
/// values; without it this is one integer compare per element.
inline int DiffDumpLimit() {
  char const* setting = std::getenv("TILEMEGA_DIFF_DUMP");
  return setting == nullptr ? 0 : std::atoi(setting);
}

inline Difference Compare(std::vector<std::vector<ModelElement>> const& actual,
                          std::vector<std::vector<ModelElement>> const& expected,
                          char const* what = nullptr) {
  Difference result;
  int const dump = what == nullptr ? 0 : DiffDumpLimit();
  int dumped = 0;
  for (std::size_t tensor = 0; tensor < actual.size(); ++tensor)
    for (std::size_t i = 0; i < actual[tensor].size(); ++i) {
      float a = static_cast<float>(actual[tensor][i]);
      float e = static_cast<float>(expected[tensor][i]);
      float delta = std::fabs(a - e);
      float relative = delta / std::max(std::fabs(e), 1.0e-6f);
      float const tolerance = kCompiledScalarType == ScalarType::kBF16
          ? 1.6e-2f + 1.6e-2f * std::fabs(e)
          : 3.0e-5f + 3.0e-5f * std::fabs(e);
      if (delta > tolerance) {
        ++result.mismatch;
        if (dumped < dump) {
          ++dumped;
          std::printf("E2E_DIFF_ELEM pair=%s tensor=%zu index=%zu actual=%.9g "
                      "expected=%.9g delta=%.9g tolerance=%.9g\n",
                      what, tensor, i, a, e, delta, tolerance);
        }
      }
      result.max_abs = std::max(result.max_abs, delta);
      result.max_rel = std::max(result.max_rel, relative);
    }
  return result;
}

inline unsigned long long BitHash(
    std::vector<std::vector<ModelElement>> const& values) {
  unsigned long long hash = 1469598103934665603ull;
  for (auto const& tensor : values) {
    auto const* bytes = reinterpret_cast<unsigned char const*>(tensor.data());
    for (std::size_t i = 0; i < tensor.size() * sizeof(ModelElement); ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

inline float LaunchL05(DeviceModel& model, int grid, bool timed = true) {
  return benchmark::Time([&] {
  for (std::uint32_t stage = 0; stage < model.params.stage_count; ++stage)
    tilemega_stage_kernel<<<grid, kHarnessThreads, sizeof(TaskSmem)>>>(
        model.device_params, stage);
  }, timed);
}

/// Per-stage L0.5 timing, used by the partition oracle to attribute an
/// end-to-end delta to individual operators. Off unless TILEMEGA_STAGE_PROFILE
/// is set, and always run after the timed launches so it cannot perturb them.
inline void ProfileStages(DeviceModel& model, ModelSpec const& spec, int grid) {
  if (!std::getenv("TILEMEGA_STAGE_PROFILE")) return;
  std::uint32_t count = model.params.stage_count;
  std::vector<float> best(count, 3.4e38f);
  cudaEvent_t start, stop;
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&start));
  TILEMEGA_CUDA_CHECK(cudaEventCreate(&stop));
  for (int repeat = 0; repeat < 5; ++repeat) {
    ResetBuffersOnly(model);
    for (std::uint32_t stage = 0; stage < count; ++stage) {
      TILEMEGA_CUDA_CHECK(cudaEventRecord(start));
      tilemega_stage_kernel<<<grid, kHarnessThreads, sizeof(TaskSmem)>>>(
          model.device_params, stage);
      TILEMEGA_CUDA_CHECK(cudaEventRecord(stop));
      TILEMEGA_CUDA_CHECK(cudaEventSynchronize(stop));
      TILEMEGA_CUDA_CHECK(cudaGetLastError());
      float ms = 0;
      TILEMEGA_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      best[stage] = std::min(best[stage], ms);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  for (std::uint32_t stage = 0; stage < count; ++stage)
    std::printf("E2E_STAGE idx=%u kind=%u gemm=%u extent=%u width=%u "
                "min_ms=%.6f\n", stage,
                static_cast<unsigned>(model.stages[stage].kind),
                model.stages[stage].gemm, model.stages[stage].extent,
                model.stages[stage].width, best[stage]);
}

/// One launch site for both persistent kernels.  A cluster launch is the same
/// kernel plus one attribute, but it cannot be a runtime branch on the ordinary
/// `<<<>>>` form: `cudaLaunchKernelEx` takes its arguments by value through a
/// different entry point, and the driver rejects a grid that does not divide
/// into whole clusters rather than truncating it.
template <typename Kernel, typename... Args>
inline void LaunchPersistent(Kernel kernel, int grid, std::size_t smem_bytes, Args... args) {
#if TILEMEGA_GENERATED_CLUSTER_DIM > 1
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kHarnessThreads);
  config.dynamicSmemBytes = smem_bytes;
  cudaLaunchAttribute attribute[1] = {};
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = TILEMEGA_GENERATED_CLUSTER_DIM;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  TILEMEGA_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
#else
  kernel<<<grid, kHarnessThreads, smem_bytes>>>(args...);
#endif
}

inline float LaunchL1(DeviceModel& model, int grid,
                      unsigned long long iteration = 0, bool timed = true) {
  return benchmark::Time([&] {
  LaunchPersistent(tilemega_l1_kernel, grid, sizeof(TaskSmem), model.device_params,
                   model.events, iteration);
  }, timed);
}

inline float LaunchL2(DeviceModel& model, int grid,
                      unsigned long long iteration = 0, bool timed = true) {
  return benchmark::Time([&] {
  LaunchPersistent(tilemega_l2_kernel, grid, model.l2_smem_bytes, model.device_params,
                   model.events, iteration);
  }, timed);
}

}  // namespace harness

/// The whole generated `main` is this call: everything model specific is in
/// `spec`, which the code generator emitted from the CG.
inline int RunModel(ModelSpec const& spec, char const* fixture_dir) {
  using namespace harness;
  if (spec.dtype != kCompiledScalarType) {
    std::fprintf(stderr, "ModelSpec dtype does not match compiled TaskBodies\n");
    return 2;
  }
  ModelDims dims = BindDims(spec.dims, fixture_dir);
  if (dims.seq <= 0 || static_cast<std::uint32_t>(dims.seq) >=
                           spec.seq_variant_count) {
    std::fprintf(stderr,
                 "seq=%d is outside the generated runtime-variant table [1,%u]\n",
                 dims.seq, spec.seq_variant_count ? spec.seq_variant_count - 1 : 0);
    return 2;
  }
  std::uint32_t const runtime_variant_index = spec.seq_variant[dims.seq];
  if (runtime_variant_index >= spec.runtime_variant_count) {
    std::fprintf(stderr, "seq=%d selects invalid runtime variant %u\n",
                 dims.seq, runtime_variant_index);
    return 2;
  }
  RuntimeVariantDesc const& runtime_variant =
      spec.runtime_variants[runtime_variant_index];
  for (std::uint32_t s=0; s<spec.stage_count; ++s) {
    if (spec.stages[s].kind != TaskKind::kAttention) continue;
    AttentionRuntimeRecord choice = runtime_variant.attention
        ? runtime_variant.attention[s] : AttentionRuntimeRecord{};
    if (!choice.chunks || (choice.chunks > 1 && !TILEMEGA_CHUNKED_ATTENTION) ||
        (choice.chunks > 1 && !choice.chunk_extent)) {
      std::fprintf(stderr,"attention plan is invalid or chunk execution is disabled\n"); return 2;
    }
    std::size_t extent = (static_cast<std::size_t>(dims.total)+choice.chunks-1)/choice.chunks;
    std::size_t capacity = sizeof(TaskSmem::attention)/sizeof(float);
    if (extent > capacity || (choice.chunk_extent && extent > choice.chunk_extent) ||
        static_cast<std::size_t>(dims.seq)*spec.stages[s].extent*choice.chunks >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      std::fprintf(stderr,"attention plan exceeds compiled scratch or task-count capacity\n"); return 2;
    }
  }
  if (static_cast<std::uint32_t>(dims.seq) < runtime_variant.seq_begin ||
      static_cast<std::uint32_t>(dims.seq) > runtime_variant.seq_end) {
    std::fprintf(stderr, "runtime variant table/interval mismatch for seq=%d\n",
                 dims.seq);
    return 2;
  }
  std::vector<std::vector<ModelElement>> reference(spec.output_count);
  for (std::uint32_t i = 0; i < spec.output_count; ++i)
    reference[i] = Load(std::string(fixture_dir) + "/" + spec.outputs[i].file,
                        spec.buffers[spec.outputs[i].buffer].Elements(dims));

  auto target = tilemega::TargetSpec::Probe();
  cudaFuncAttributes l2_attributes{};
  TILEMEGA_CUDA_CHECK(cudaFuncGetAttributes(&l2_attributes, tilemega_l2_kernel));
  cudaFuncAttributes l1_attributes{}, l05_attributes{};
  TILEMEGA_CUDA_CHECK(cudaFuncGetAttributes(&l1_attributes, tilemega_l1_kernel));
  TILEMEGA_CUDA_CHECK(cudaFuncGetAttributes(&l05_attributes, tilemega_stage_kernel));
  std::size_t const l2_smem_bytes = TILEMEGA_EVENT_CLUSTER_RESERVE
      ? target.res.max_dynamic_smem_per_cta - l2_attributes.sharedSizeBytes : sizeof(TaskSmem);
  TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(
      tilemega_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(TaskSmem)));
  TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(
      tilemega_l1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(TaskSmem)));
  TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(
      tilemega_l2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      l2_smem_bytes));
  // One grid serves every launch, and both persistent kernels spin, so the
  // resident bound is the *minimum* over them. L2 carries the event epochs in
  // registers and can cost a whole CTA per SM more than L1 (128 vs 144
  // registers at 128x16x32, stages=2): sizing the grid from L1 alone launches
  // CTAs that are not resident, and a resident CTA then waits forever for an
  // arrival only a non-resident one can make.
  int grid = TILEMEGA_GENERATED_RESIDENT_GRID(target, tilemega_l1_kernel,
                                              kHarnessThreads, sizeof(TaskSmem));
  int l2_grid = TILEMEGA_GENERATED_RESIDENT_GRID(target, tilemega_l2_kernel,
                                                 kHarnessThreads, l2_smem_bytes);
  if (l2_grid < grid) grid = l2_grid;
  int const l1_ctas = target.ActiveBlocksPerSM(
      reinterpret_cast<void const*>(tilemega_l1_kernel), kHarnessThreads, sizeof(TaskSmem));
  int const l2_ctas = target.ActiveBlocksPerSM(
      reinterpret_cast<void const*>(tilemega_l2_kernel), kHarnessThreads, l2_smem_bytes);
#if TILEMEGA_GENERATED_CLUSTER_DIM > 1
  // The capability table is a compile-time policy; this is the device in front
  // of us.  Refusing here is the point: a cluster kernel that quietly ran with
  // a smaller cluster would still print timings.
  if (target.res.max_cluster_size < TILEMEGA_GENERATED_CLUSTER_DIM) {
    std::fprintf(stderr,
                 "cluster dim %d exceeds the device's max_cluster_size %d\n",
                 TILEMEGA_GENERATED_CLUSTER_DIM, target.res.max_cluster_size);
    return 2;
  }
  grid -= grid % TILEMEGA_GENERATED_CLUSTER_DIM;
  cudaLaunchConfig_t cluster_config{};
  cluster_config.gridDim = dim3(grid);
  cluster_config.blockDim = dim3(kHarnessThreads);
  cudaLaunchAttribute cluster_attribute{};
  cluster_attribute.id = cudaLaunchAttributeClusterDimension;
  cluster_attribute.val.clusterDim = {TILEMEGA_GENERATED_CLUSTER_DIM, 1, 1};
  cluster_config.attrs = &cluster_attribute;
  cluster_config.numAttrs = 1;
  for (bool l2 : {false, true}) {
    cluster_config.dynamicSmemBytes = l2 ? l2_smem_bytes : sizeof(TaskSmem);
    int clusters = 0;
    TILEMEGA_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
        &clusters, l2 ? tilemega_l2_kernel : tilemega_l1_kernel, &cluster_config));
    grid = std::min(grid, clusters * TILEMEGA_GENERATED_CLUSTER_DIM);
  }
  if (grid == 0) {
    std::fprintf(stderr, "resident grid is smaller than one cluster\n");
    return 2;
  }
#endif
  int const resident_limit = std::min(l1_ctas,l2_ctas)*target.res.num_sms;
  if (!codegen::ResidentScheduleLegal(runtime_variant.resident_only,grid,resident_limit)) {
    std::fprintf(stderr,"L-sched resident-only constraint rejected grid=%d limit=%d\n",
                 grid,resident_limit);
    return 2;
  }
  int blocks_per_sm = std::max(1, grid / target.res.num_sms);
#if TILEMEGA_PLACEMENT == 1
  // The `pair` placement needs the residency the grid was sized from.
  TILEMEGA_CUDA_CHECK(cudaMemcpyToSymbol(tilemega_blocks_per_sm, &blocks_per_sm,
                                         sizeof(int)));
#endif
  DeviceModel model = Create(spec, runtime_variant, runtime_variant_index,
                             dims, fixture_dir, grid, blocks_per_sm, target, l2_smem_bytes);
  PrepareEvents(model, grid);

  benchmark::Settings const timing;
  std::printf("E2E_TIMING cold=%d warmup=%d repeat=%d statistic=median reset=outside_timing\n",
              TILEMEGA_COLD_START_TIMING, timing.warmup, timing.repeat);
  auto reset_forward = [&] { Reset(model); };
  float l05_ms = benchmark::Forward(timing, reset_forward,
      [&](bool timed) { return LaunchL05(model, grid, timed); });
  auto l05 = Download(model);
  DumpBuffers(model, std::getenv("TILEMEGA_DUMP_BUFFERS"));
  float l1_ms = benchmark::Forward(timing, reset_forward,
      [&](bool timed) { return LaunchL1(model, grid, 0, timed); });
  auto l1 = Download(model);
#if TILEMEGA_TRACE_V2
  // Zeroed outside the timed region; every sample of the Forward below
  // overwrites the rows, so what is dumped is its last timed launch, and
  // Reset() clears the event counters before each, keeping the slot stamps and
  // the publish stamps from one and the same launch.
  ZeroTraceV2(model);
#endif
  float l2_ms = benchmark::Forward(timing, reset_forward,
      [&](bool timed) { return LaunchL2(model, grid, 0, timed); });
  auto l2 = Download(model);
  ReportTaskTrace(model);
#if TILEMEGA_TRACE_V2
  DumpTraceV2(model, fixture_dir, grid, l1_ms, l2_ms,
              "l2_forward_last_timed_sample");
#endif

  // §8.2: the counters are monotonic, so a second iteration must be correct
  // *without* clearing them -- `needed` scales with the iteration instead.
  // This is the property the rule exists for: if the target were fixed and
  // the counters reset, a CTA still finishing iteration i could be counted
  // as an early arrival for iteration i+1. Only the buffers are reset here;
  // the event memory is deliberately carried over.
  // One repeat is enough to catch a counter that was reset; it is not enough
  // to catch one whose *target* does not advance, because at iteration 1 the
  // epoch is still only one ahead.  `TILEMEGA_ITERATIONS` runs the persistent
  // kernel N times over the same event memory, so an arrival from iteration i
  // can only be mistaken for one from i+1 if the monotone target is wrong --
  // which is exactly what the negative build below removes.
  int iterations = 1;
  if (char const* setting = std::getenv("TILEMEGA_ITERATIONS")) {
    iterations = std::atoi(setting);
    if (iterations < 1) iterations = 1;
  }
  ResetBuffersOnly(model);
  float l2_again_ms = LaunchL2(model, grid, /*iteration=*/1);
  auto l2_again = Download(model);
  Difference l2_iter = Compare(l2_again, l2);
  std::size_t iteration_mismatch = l2_iter.mismatch;
  for (int step = 2; step <= iterations; ++step) {
#if TILEMEGA_NEGATIVE_RESET_EVENTS
    // §8.2's negative control, and the shape a naive implementation takes:
    // clear the counters between iterations and always announce iteration 0.
    // A CTA still finishing iteration i then satisfies iteration i+1's wait
    // from the cleared counter -- the ABA the monotone target exists to
    // prevent.  Never a build anyone ships; its output is wrong on purpose.
    Reset(model);
    l2_again_ms = LaunchL2(model, grid, 0ull);
#else
    ResetBuffersOnly(model);
    l2_again_ms = LaunchL2(model, grid, static_cast<unsigned long long>(step));
#endif
    auto repeated = Download(model);
    Difference const step_diff = Compare(repeated, l2);
    iteration_mismatch += step_diff.mismatch;
    if (step_diff.mismatch != 0)
      std::printf("E2E_ITER_STEP step=%d mismatch=%zu max_abs=%.8g\n", step,
                  step_diff.mismatch, step_diff.max_abs);
  }
  if (iterations > 1)
    std::printf("E2E_ITER_SWEEP iterations=%d total_mismatch=%zu\n", iterations,
                iteration_mismatch);

  Difference l05_l0 = Compare(l05, reference, "l05_vs_l0");
  Difference l1_l05 = Compare(l1, l05, "l1_vs_l05");
  Difference l2_l1 = Compare(l2, l1, "l2_vs_l1");
  std::printf("E2E_RESOURCE block=%d reg=%d smem=%zu task_smem=%zu occupancy_smem=%zu "
              "static_smem=%zu regs_per_sm=%d smem_per_sm=%d threads_per_sm=%d gemm_union=%zu "
              "variant_count=%d ctas_per_sm=%d num_sms=%d grid=%d "
              "l1_reg=%d l05_reg=%d l1_ctas=%d l2_ctas=%d min_blocks=%d warp_size=%d "
              "resident_formula=ctas_per_sm*num_sms\n",
              kHarnessThreads, l2_attributes.numRegs, l2_smem_bytes, sizeof(TaskSmem),
              l2_smem_bytes, l2_attributes.sharedSizeBytes, target.res.regs_per_sm,
              target.res.max_smem_per_sm, target.res.max_threads_per_sm, sizeof(GemmVariantSmem),
              TILEMEGA_GEMM_VARIANT_COUNT, blocks_per_sm,
              target.res.num_sms, grid, l1_attributes.numRegs, l05_attributes.numRegs,
              l1_ctas, l2_ctas, TILEMEGA_MIN_BLOCKS_PER_SM, target.res.warp_size);
  std::uint32_t max_worker_task_refs = 0;
  for (std::size_t worker = 0; worker + 1 < model.schedule_offsets.size(); ++worker)
    max_worker_task_refs = std::max(max_worker_task_refs,
        model.schedule_offsets[worker + 1] - model.schedule_offsets[worker]);
  std::printf("E2E_SCHEDULE workers=%d variant_stages=%u task_refs=%zu "
              "task_ref_bytes=%zu waits=%zu wait_bytes=%zu raw_polls=%zu "
              "lifted_polls=%zu waiting_tasks=%zu normalization_dummies_lb=%zu "
              "max_span=%u generated_max_span=%u "
              "max_worker_span=%u resident_limit=%d global_fanin=%d "
              "i3_current=pass i3_overresident=reject max_worker_task_refs=%u\n",
              grid, runtime_variant.schedule_count, model.schedule.size(),
              model.schedule.size() * sizeof(TaskRef), model.task_waits.size(),
              model.task_waits.size() * sizeof(TaskWait),
              model.schedule_raw_polls,
              model.schedule_raw_polls - model.task_waits.size(),
              model.schedule_waiting_tasks,
              model.normalization_dummy_lower_bound,
              model.schedule_max_span, runtime_variant.max_dependency_span,
              model.schedule_max_worker_span, grid,
              model.schedule_has_global_fanin ? 1 : 0, max_worker_task_refs);
  std::printf("E2E_TIME l05_ms=%.6f l1_ms=%.6f ratio=%.6f l2_ms=%.6f "
              "l2_over_l1=%.6f\n", l05_ms, l1_ms, l1_ms / l05_ms, l2_ms,
              l2_ms / l1_ms);
  std::printf("E2E_DIFF l05_vs_l0_mismatch=%zu max_abs=%.8g max_rel=%.8g "
              "l1_vs_l05_mismatch=%zu max_abs=%.8g max_rel=%.8g "
              "l2_vs_l1_mismatch=%zu max_abs=%.8g max_rel=%.8g\n",
              l05_l0.mismatch, l05_l0.max_abs, l05_l0.max_rel,
              l1_l05.mismatch, l1_l05.max_abs, l1_l05.max_rel,
              l2_l1.mismatch, l2_l1.max_abs, l2_l1.max_rel);
  for (std::uint32_t i = 0; i < spec.output_count; ++i) {
    Difference per = Compare({l05[i]}, {reference[i]});
    std::printf("E2E_OUTPUT_DIFF index=%u buffer=%u mismatch=%zu "
                "max_abs=%.8g max_rel=%.8g\n",
                i, spec.outputs[i].buffer, per.mismatch, per.max_abs,
                per.max_rel);
  }
  std::printf("E2E_HASH l05=%016llx l1=%016llx l2=%016llx\n", BitHash(l05),
              BitHash(l1), BitHash(l2));
  std::printf("E2E_ITER l2_iter1_ms=%.6f l2_iter1_vs_iter0_mismatch=%zu "
              "max_abs=%.8g\n", l2_again_ms, l2_iter.mismatch,
              l2_iter.max_abs);
  ProfileStages(model, spec, grid);
  if (char const* profile = std::getenv("TILEMEGA_WAIT_PROFILE")) {
    // A comma-separated list of sequence lengths to count at; "1" or any
    // non-numeric value means the fixture's own.
    for (char const* at = profile; at && *at;) {
      int seq = std::atoi(at);
      // No dynamic shared memory: TaskSmem is over the 48 KB opt-in threshold
      // and this kernel touches none of it.
      tilemega_wait_profile_kernel<<<grid, kHarnessThreads>>>(
          model.device_params, seq);
      TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
      TILEMEGA_CUDA_CHECK(cudaGetLastError());
      char const* comma = std::strchr(at, ',');
      at = comma ? comma + 1 : nullptr;
    }
  }
  bool pass = l05_l0.mismatch == 0 && l1_l05.mismatch == 0 &&
              l2_l1.mismatch == 0 && iteration_mismatch == 0;
#if TILEMEGA_UNSAFE_NO_GRID_SYNC
  // A build without the grid half of the barrier is a timing probe, not a
  // kernel; it must never be able to print PASS, whatever the comparison says.
  std::printf("E2E_UNSAFE no_grid_sync=1\n");
  pass = false;
#endif
#if TILEMEGA_EVENT_KAPPA > 0
  std::printf("E2E_KAPPA event_kappa=%d dependency_table=variant_exact\n",
              TILEMEGA_EVENT_KAPPA);
#endif
  std::printf("E2E_VARIANT index=%u seq_begin=%u seq_end=%u\n",
              runtime_variant_index, runtime_variant.seq_begin,
              runtime_variant.seq_end);
#if TILEMEGA_PLACEMENT != 0
  std::printf("E2E_PLACEMENT placement=%d\n", TILEMEGA_PLACEMENT);
#endif
  std::printf("RESULT status=%s\n", pass ? "PASS" : "MISMATCH");
  return pass ? 0 : 1;
}

}  // namespace tilemega::codegen
