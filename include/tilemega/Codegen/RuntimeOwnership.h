// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <tilemega/Codegen/tasks/TaskBase.h>

#ifndef TILEMEGA_CG_SPLIT_TASK_ORDER
#define TILEMEGA_CG_SPLIT_TASK_ORDER 1
#endif

namespace tilemega::codegen {
inline constexpr std::uint32_t kHandoffAutoCombine = 0xfffffffeu;
struct SplitTaskCoordinate { int tile, chunk; };
TILEMEGA_TASK_HD constexpr SplitTaskCoordinate DecodeSplitTask(
    int task, int tiles, int chunks, bool cg_order = TILEMEGA_CG_SPLIT_TASK_ORDER) {
  return cg_order ? SplitTaskCoordinate{task/chunks,task%chunks}
                  : SplitTaskCoordinate{task%tiles,task/tiles};
}

// Only the worker traversal changes; dependencies still name row-major tasks.
TILEMEGA_TASK_HD constexpr int NMajorGemmTask(int ordinal,int tiles_m,int tiles_n,
    int chunks,bool cg_order=TILEMEGA_CG_SPLIT_TASK_ORDER) {
  if(tiles_m<1 || tiles_n<1 || chunks<1 || ordinal<0)return -1;
  auto tiles=std::int64_t(tiles_m)*tiles_n;
  if(tiles>INT32_MAX/chunks)return -1;
  int count=int(tiles)*chunks;
  if(ordinal>=count)return ordinal;
  auto point=DecodeSplitTask(ordinal,tiles_m*tiles_n,chunks,cg_order);
  int tile=(point.tile%tiles_m)*tiles_n+point.tile/tiles_m;
  return cg_order?tile*chunks+point.chunk:point.chunk*(tiles_m*tiles_n)+tile;
}

enum RuntimeOwnershipFlag : std::uint32_t {
  kRoPETileOwnership = 1u << 0,
  kKVTileOwnership = 1u << 1,
  kActivationTileOwnership = 1u << 2,
  kCombinerTileOwnership = 1u << 3,
};
}  // namespace tilemega::codegen
