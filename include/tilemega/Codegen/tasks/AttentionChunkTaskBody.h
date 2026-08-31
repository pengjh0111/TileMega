// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 online-softmax attention chunk.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch, int Stages>
struct AttentionChunkTaskBody {
  using Traits = TaskTraits<256, static_cast<std::size_t>(Stages) * 12 * 1024>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    if (threadIdx.x == 0 && c.output) static_cast<int*>(c.output)[c.logical_tile] = c.extent;
  }
};
}  // namespace tilemega::codegen
