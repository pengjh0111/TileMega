// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <set>
namespace tilemega::frontend {
// Select pointwise GEMM instances. A null selection enables every legal pair.
unsigned ApplyDnnDwPwFusion(ModelPlan&,std::set<unsigned> const* selected=nullptr);
}
