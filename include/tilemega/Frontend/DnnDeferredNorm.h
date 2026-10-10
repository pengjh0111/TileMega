// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <set>

namespace tilemega::frontend {
// Rewrite only channel-complete, pixel-preserving normalization consumers.
// Unsupported or externally observed normalized values retain explicit LN.
// A null selection enables every proved edge. An explicit set names GEMM
// instances, allowing a solver to retain explicit LN for other consumers.
unsigned ApplyDnnDeferredLayerNorm(ModelPlan& plan,
    std::set<unsigned> const* selected_gemms = nullptr,
    std::set<DeferredLayerNormEdge> const* selected_edges = nullptr);
}
