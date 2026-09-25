// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingTaskIndex.h>

#include <cassert>

int main() {
  // Prefill: B=16, QPerKV*S/Rq=4, eight KV groups, one cache block.
  for (int b = 0; b < 16; ++b)
    for (int qb = 0; qb < 4; ++qb)
      for (int g = 0; g < 8; ++g) {
        int task = (b * 4 + qb) * 8 + g;
        auto c = tilemega::codegen::DecodeServingAttentionTask(task, 4, 8, 1);
        assert(c.batch == b && c.query_block == qb && c.group == g &&
               c.cache_block == 0);
        // Three consecutive 128-column QKV tiles cover each packed group.
        int first_producer = (b * 4 + qb) * 24 + 3 * g;
        assert(first_producer == 3 * task);
      }
  // Decode has one query block; its task order is unchanged.
  for (int b = 0; b < 16; ++b)
    for (int g = 0; g < 8; ++g)
      for (int c = 0; c < 5; ++c) {
        int task = (b * 8 + g) * 5 + c;
        auto x = tilemega::codegen::DecodeServingAttentionTask(task, 1, 8, 5);
        assert(x.batch == b && x.query_block == 0 && x.group == g &&
               x.cache_block == c);
      }
}
