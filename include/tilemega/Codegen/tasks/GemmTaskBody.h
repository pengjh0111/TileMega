// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 CUTLASS/CuTe GEMM TaskBody.
#pragma once

#include <tilemega/Codegen/tasks/TaskBase.h>
#include <tilemega/Target/ArchDispatch.h>

namespace tilemega::codegen {

/// GEMM body contract. `Stages` is selected with TargetSpec::ComputeStages;
/// the body itself only consumes the enclosing kernel's shared storage.
template <class Arch, int Stages, int BytesPerStage = 16 * 1024,
          class TileShape = void>
struct GemmTaskBody {
  static_assert(Stages > 0, "GEMM requires a non-empty pipeline");
  using Traits = TaskTraits<256, static_cast<std::size_t>(Stages) * BytesPerStage>;
  using SharedStorage = codegen::SharedStorage<Traits::kSharedStorageBytes>;
  static constexpr char const* kCollective = arch::Caps<Arch>::kCollective;
  static constexpr int kSmemBytes = static_cast<int>(sizeof(SharedStorage));
  static constexpr int kNumThreads = Traits::kThreads;
  static constexpr bool kLegal = Stages > 0;
  // CUTLASS coordinate contract (F-17a): A is logical (M,K), B is logical
  // (N,K). A contiguous PyTorch [N,K] weight is exposed as B strides (K,1),
  // represented by CUTLASS ColumnMajor in the direct cp.async adapter.
  static constexpr char const* kLogicalA = "(M,K), strides (K,1)";
  static constexpr char const* kLogicalB = "(N,K), strides (K,1)";

  /// Minimal executable ABI used by the megakernel skeleton. V-B supplies the
  /// CUTLASS collective specialization without changing this interface.
  __device__ static void Run(TaskContext const& context, SharedStorage&) {
    if (threadIdx.x == 0 && context.output != nullptr) {
      static_cast<int*>(context.output)[context.logical_tile] = context.iteration;
    }
  }

  template <class Params>
  __device__ void operator()(TaskDesc const& task, char* smem,
                             Params const& params) const {
    auto& storage = *reinterpret_cast<SharedStorage*>(smem);
    TaskContext context = params.context(task);
    Run(context, storage);
  }
};

}  // namespace tilemega::codegen
