// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.3 and §8.1 scheduler/event body.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch>
struct SchedulerTaskBody {
  using Traits = TaskTraits<256, 1024>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    if (threadIdx.x == 0 && c.output) atomicAdd(static_cast<unsigned int*>(c.output), 1u);
  }
};
}  // namespace tilemega::codegen
