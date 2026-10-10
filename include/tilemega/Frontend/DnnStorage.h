// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/SemanticLifting.h>

namespace tilemega::frontend {
void MaterializeDnnStorage(ModelPlan& plan,std::vector<GemmGranularity> const& gemms,int batch);
} // namespace tilemega::frontend
