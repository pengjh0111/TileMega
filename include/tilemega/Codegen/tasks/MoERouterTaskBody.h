// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 MoE top-k routing body.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch>
struct MoERouterTaskBody {
  using Traits = TaskTraits<256, 6 * 1024>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    if (threadIdx.x == 0 && c.output) static_cast<int*>(c.output)[c.logical_tile] = c.extent;
  }
};
}  // namespace tilemega::codegen
