// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>

namespace tilemega::frontend {
// Rewrite only channel-complete, pixel-preserving normalization consumers.
// Unsupported or externally observed normalized values retain explicit LN.
unsigned ApplyDnnDeferredLayerNorm(ModelPlan& plan);
}
