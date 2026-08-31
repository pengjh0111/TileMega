// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 generic elementwise body.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch>
struct ElementwiseTaskBody {
  using Traits = TaskTraits<256, 0>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    auto i = c.logical_tile * blockDim.x + threadIdx.x;
    if (c.output && i < c.extent) static_cast<int*>(c.output)[i] = c.iteration;
  }
};
}  // namespace tilemega::codegen
