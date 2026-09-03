// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.4 (cost model input), §5.3 (generated model tables).
//
// The host-side mirror of the tables `CouplingGraphToCUDA` emits: the same
// GEMM shapes and the same stage sequence the megakernel executes, in a form
// the solver can evaluate without a compiler or a device.
//
// It is read back out of a generated .cu rather than re-derived from the CG:
// the generated file is the artifact the measured configurations were built
// from, so a cost model validated against those measurements must describe
// exactly it.
#pragma once

#include <string>
#include <vector>

namespace tilemega::solver {

/// The symbolic dimensions, bound at launch (ModelRuntime.h `ModelDims`).
struct ModelDims {
  int seq = 0;
  int past = 0;
  int total = 0;
};

/// M stays symbolic, so a GEMM contributes only N and K.
struct GemmOp {
  int n = 0;
  int k = 0;
};

/// Mirrors `TaskKind`; kGemmCombine is absent because it exists only after the
/// host-side split rewrite, which the cost model performs itself.
enum class StageKind {
  kGemm,
  kRMSNorm,
  kRoPE,
  kKVAppend,
  kElementwise,
  kAttention,
};

struct ModelStage {
  StageKind kind = StageKind::kGemm;
  int gemm = -1;  ///< index into ModelDescription::gemms, -1 when not a GEMM
  int extent = 0;
  int width = 0;
  int group = 0;
};

struct ModelDescription {
  std::string name;
  ModelDims dims;
  std::vector<GemmOp> gemms;
  std::vector<ModelStage> stages;

  /// Parse the `kGemms` and `kStages` tables out of a generated .cu.  Throws
  /// std::runtime_error when either table is missing or malformed -- a silent
  /// empty model would validate a cost model against nothing.
  static ModelDescription FromGeneratedCuda(std::string const& path,
                                            ModelDims dims,
                                            std::string name);

  /// Bytes of parameter and activation storage the model keeps live, which is
  /// what the L2 must hold for the weight stream to stay resident (§2.2(e)).
  double LiveFootprintBytes() const;
};

}  // namespace tilemega::solver
