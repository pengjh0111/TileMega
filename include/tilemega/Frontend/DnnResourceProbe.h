// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <string>

namespace tilemega::frontend {
// The translation unit instantiates exactly the non-GEMM templates in a plan.
// ProbeArch is supplied by the architecture-specific compilation driver.
std::string DnnNonGemmProbeSource(ModelPlan const&);
std::uint64_t DnnNonGemmSharedBytes(ModelPlan const&);
}
