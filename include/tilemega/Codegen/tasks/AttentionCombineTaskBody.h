// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 attention state combination.  Kept as the standalone
// family used by the architecture cross-compile matrix; the generated model
// runtime fuses chunk/combine in AttentionTaskBody.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template <class Arch, class TileShape = void>
struct AttentionCombineTaskBody {
  using Traits = TaskTraits<256, 8 * 1024>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Traits::kThreads;
  static constexpr bool kLegal = true;
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    if (threadIdx.x == 0 && c.output)
      static_cast<int*>(c.output)[c.logical_tile] = c.iteration;
  }
  template <class Params>
  __device__ void operator()(TaskDesc const& task, char* smem,
                             Params const& p) const {
    Run(p.context(task), *reinterpret_cast<SharedStorage*>(smem));
  }
};
}  // namespace tilemega::codegen
