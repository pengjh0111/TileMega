// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/TaskInstantiation.h>

namespace tilemega::analysis {
void ValidateTileStorage(SemanticOp const& op);
SemanticGraph MaterializeTaskStorage(SemanticGraph const& graph, Granularity const& g);
} // namespace tilemega::analysis
