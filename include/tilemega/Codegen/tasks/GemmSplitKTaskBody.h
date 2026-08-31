// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 split-K GEMM producer/reduction body.
#pragma once
#include <tilemega/Codegen/tasks/GemmTaskBody.h>
namespace tilemega::codegen {
template <class Arch, int Stages>
struct GemmSplitKTaskBody : GemmTaskBody<Arch, Stages> {};
}  // namespace tilemega::codegen
