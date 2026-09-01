// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 generic elementwise body.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch, class TileShape = void>
struct ElementwiseTaskBody {
  using Traits = TaskTraits<256, 0>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Traits::kThreads;
  static constexpr bool kLegal = true;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    auto i = c.logical_tile * blockDim.x + threadIdx.x;
    if (c.output && i < c.extent) static_cast<int*>(c.output)[i] = c.iteration;
  }
  template <class Params>
  __device__ void operator()(TaskDesc const& task, char* smem, Params const& p) const {
    Run(p.context(task), *reinterpret_cast<SharedStorage*>(smem));
  }
};
}  // namespace tilemega::codegen
