// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/TaskWork.h>
namespace tilemega::analysis {
TaskWork DeriveExactTaskWork(OperatorNode const& task, ParamBinding const& known,
                            TaskWorkOptions const& options);
}  // namespace tilemega::analysis
