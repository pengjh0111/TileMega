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

}  // namespace tilemega::codegen
