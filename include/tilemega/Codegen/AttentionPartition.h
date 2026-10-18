// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>

namespace tilemega::codegen {
struct AttentionPartition {
  int blocks=0;
  int extent=0;
  int live_blocks=0;
};
// Keep the partial allocation fixed while redistributing the live history.
TILEMEGA_TASK_HD constexpr AttentionPartition DecodeAttentionPartition(
    int capacity,int past,int static_extent,bool balanced,int minimum=32) {
  if(capacity<1 || past<0 || past>=capacity || static_extent<1 || minimum<1)return {};
  int blocks=(capacity+static_extent-1)/static_extent,total=past+1;
  int active=balanced?(total+minimum-1)/minimum:blocks;
  if(active>blocks)active=blocks;
  int extent=balanced?(total+active-1)/active:static_extent;
  return {blocks,extent,(total+extent-1)/extent};
}
} // namespace tilemega::codegen
