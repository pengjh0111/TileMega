// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.3 TaskBody ABI and §8 code-generation rules.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

/// Host tools read the same ownership table the device does, and TaskBase.h
/// is the one task header that compiles without CUDA.
#if defined(__CUDACC__)
#define TILEMEGA_TASK_HD __host__ __device__
#else
#define TILEMEGA_TASK_HD
#endif

namespace tilemega::codegen {

/// Architecture-independent arguments passed to every generated TaskBody.
struct TaskContext {
  void const* input0 = nullptr;
  void const* input1 = nullptr;
  void* output = nullptr;
  std::int64_t logical_tile = 0;
  std::int32_t iteration = 0;
  std::int32_t extent = 0;
};

/// Generated schedule descriptor.  Large model parameter tables are never
/// embedded here; generated kernels receive a device pointer to Params (F-17b).
struct TaskDesc {
  std::uint32_t kind = 0;
  std::uint32_t stage = 0;
  std::uint32_t logical_tile = 0;
  std::uint32_t extent = 0;
};

/// §5.3 TaskBody ABI: how `blockIdx.x` maps to the tasks a TaskBody owns.
///
/// Two facts the megakernel needs and could not previously ask for:
///
///   * `count` -- how many CTAs own work in this stage. Everything at or
///     above it neither reads nor writes, so L2 lets those CTAs skip the
///     stage's waits, fences and arrival entirely. Before this entry existed
///     the harness restated each TaskBody's own guard in a switch, and a new
///     TaskKind whose guard disagreed would have broken the skip silently.
///   * `kind` -- *what* CTA `b` owns, which is what decides whether
///     `kIdentity` is admissible. `kTilePerBlock` means CTA `b` owns task
///     `b` of the stage's task space, so two stages that both declare it and
///     agree on `count` have the same CTA->task map and an identity coupling
///     between them may be lowered to a per-CTA wait. `kElementChunk` means
///     CTA `b` owns a grid-stride slice of a flat element range, which is a
///     function of `gridDim` rather than of the task space -- so the same
///     `b` names different tasks in two stages even when the coupling is the
///     identity, and `kIdentity` is inadmissible.
///
/// This is the ownership entry skeleton §1.5.1 records as missing; it is a
/// declaration by the TaskBody, not an inference by the harness.
enum class TaskOwnershipKind : std::uint32_t {
  kTilePerBlock = 0,
  kElementChunk = 1,
};

struct TaskOwnership {
  TaskOwnershipKind kind;
  int count;  ///< |image(C_kappa)| along the launch axis
};

/// Which TaskBody runs a stage. Declared here rather than beside the runtime
/// tables so a host-only analysis tool can ask the same questions the device
/// asks without pulling in CUDA.
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

/// The ownership each TaskKind's TaskBody declares. Every TaskBody's
/// `Ownership` returns this rather than repeating a literal, so a host tool
/// reading it is reading the TaskBody's own declaration and the two cannot
/// drift.
TILEMEGA_TASK_HD constexpr TaskOwnershipKind OwnershipOf(TaskKind kind) {
  switch (kind) {
    case TaskKind::kGemm:
    case TaskKind::kRMSNorm:
    case TaskKind::kAttention:
      return TaskOwnershipKind::kTilePerBlock;
    case TaskKind::kRoPE:
    case TaskKind::kKVAppend:
    case TaskKind::kElementwise:
    case TaskKind::kGemmCombine:
      return TaskOwnershipKind::kElementChunk;
  }
  return TaskOwnershipKind::kElementChunk;
}

/// Compile-time resource information consumed by candidate pruning (§5.3).
template <int Threads, std::size_t SharedBytes>
struct TaskTraits {
  static constexpr int kThreads = Threads;
  static constexpr std::size_t kSharedStorageBytes = SharedBytes;
  static_assert(Threads > 0, "a TaskBody needs at least one thread");
};

template <class Body>
inline constexpr bool kTaskLegal = Body::kLegal;

/// Aligned byte storage whose lifetime belongs to the enclosing megakernel.
template <std::size_t Bytes>
struct alignas(16) SharedStorage {
  unsigned char data[Bytes == 0 ? 1 : Bytes];
};

template <class Task, class = void>
struct IsTaskBody : std::false_type {};

template <class Task>
struct IsTaskBody<Task, std::void_t<typename Task::Traits,
                                    typename Task::SharedStorage>>
    : std::true_type {};

/// A TaskBody satisfies the ownership half of the ABI when it declares
/// `static __device__ TaskOwnership Ownership(Params const&, StageDesc const&)`.
template <class Task, class = void>
struct DeclaresOwnership : std::false_type {};

template <class Task>
struct DeclaresOwnership<Task, std::void_t<decltype(&Task::Ownership)>>
    : std::true_type {};

}  // namespace tilemega::codegen
