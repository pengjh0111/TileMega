// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 rotary-position elementwise body.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch>
struct RoPETaskBody {
  using Traits = TaskTraits<256, 0>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    if (threadIdx.x == 0 && c.output) static_cast<int*>(c.output)[c.logical_tile] = c.iteration;
  }
};
}  // namespace tilemega::codegen
