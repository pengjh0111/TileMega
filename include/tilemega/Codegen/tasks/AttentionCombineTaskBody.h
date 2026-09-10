// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3. Partial PV sums use FP32 storage; normalization has
// already rounded probabilities to model dtype before the partial PV phase.
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
struct AttentionCombineShape {
  int chunks, width;
};
template <class Arch, class TileShape = void, int Threads = 256>
struct AttentionCombineTaskBody {
  using Traits = TaskTraits<Threads, 0>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Traits::kThreads;
  static constexpr bool kLegal = true;
  template <class Element>
  __device__ static void Reduce(float const* partials, Element* output,
      int query, int chunks, int width) {
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
      float value = 0.0f;
      for (int chunk = 0; chunk < chunks; ++chunk)
        value += partials[(static_cast<std::size_t>(query) * chunks + chunk) * width + d];
      output[static_cast<std::size_t>(query) * width + d] = Element(value);
    }
  }
  __device__ static void Run(TaskContext const& c, SharedStorage&) {
    auto const& shape = *static_cast<AttentionCombineShape const*>(c.input1);
    Reduce(static_cast<float const*>(c.input0), static_cast<float*>(c.output),
           static_cast<int>(c.logical_tile), shape.chunks, shape.width);
  }
  template <class Params>
  __device__ void operator()(TaskDesc const& task, char* smem,
                             Params const& p) const {
    Run(p.context(task), *reinterpret_cast<SharedStorage*>(smem));
  }
};
}  // namespace tilemega::codegen
