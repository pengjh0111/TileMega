// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/Semantics.h>
#include <tilemega/Analysis/TaskReductionIndex.h>

namespace tilemega::analysis {
// Versioned, lossless CG payload, decoded before candidate-specific tiling.
std::string EncodeSemanticOp(SemanticOp const& op);
SemanticOp DecodeSemanticOp(std::string const& payload);
std::string EncodeTaskReductionIndex(TaskReductionIndex const& index);
TaskReductionIndex DecodeTaskReductionIndex(std::string const& payload);
}  // namespace tilemega::analysis
