// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.3 TaskBody ABI and §8 code-generation rules.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

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
