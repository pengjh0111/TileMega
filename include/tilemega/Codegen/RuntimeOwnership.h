// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <tilemega/Codegen/tasks/TaskBase.h>

#ifndef TILEMEGA_CG_SPLIT_TASK_ORDER
#define TILEMEGA_CG_SPLIT_TASK_ORDER 1
#endif

namespace tilemega::codegen {
struct SplitTaskCoordinate { int tile, chunk; };
TILEMEGA_TASK_HD constexpr SplitTaskCoordinate DecodeSplitTask(
    int task, int tiles, int chunks, bool cg_order = TILEMEGA_CG_SPLIT_TASK_ORDER) {
  return cg_order ? SplitTaskCoordinate{task/chunks,task%chunks}
                  : SplitTaskCoordinate{task%tiles,task/tiles};
}

enum RuntimeOwnershipFlag : std::uint32_t {
  kRoPETileOwnership = 1u << 0,
  kKVTileOwnership = 1u << 1,
  kActivationTileOwnership = 1u << 2,
  kCombinerTileOwnership = 1u << 3,
};
}  // namespace tilemega::codegen
