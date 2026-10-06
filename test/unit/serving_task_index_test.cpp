// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingTaskIndex.h>

#include <cassert>
#include <vector>
#include <algorithm>
#include <initializer_list>

namespace tilemega::tests::serving_task_index_test {

int TestServingTaskIndex(int argc, char** argv) {
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
  // Decode is group-major. The attention->merge fiber is C consecutive
  // tasks, and each group's B requests form one contiguous interval.
  for(int batch : {1,16})
    for(int g=0;g<8;++g)
      for(int b=0;b<batch;++b)
        for(int c=0;c<5;++c) {
          int task=(g*batch+b)*5+c;
          auto x=tilemega::codegen::DecodeServingAttentionTaskGMajor(
              task,batch,5);
          assert(x.batch==b && x.group==g && x.cache_block==c);
          assert(task/5==g*batch+b);
          assert(task/(batch*5)==g);
        }

  for(int batch:{1,16})for(int chunk:{32,64})for(int grid:{128,170}) {
    int blocks=(1088+chunk-1)/chunk,total=batch*8*blocks;
    std::vector<int> seen(total);
    for(int ordinal=0;ordinal<total;++ordinal) {
      int task=tilemega::codegen::ServingAttentionL1Task(ordinal,batch,8,blocks,chunk);
      assert(task>=0 && task<total);++seen[task];
      assert(tilemega::codegen::ServingAttentionL1Task(ordinal,batch,8,blocks,chunk,64)==ordinal);
    }
    assert(std::all_of(seen.begin(),seen.end(),[](int n){return n==1;}));
    for(int past:{64,575,1000}) {
      std::vector<int> live(grid);
      for(int ordinal=0;ordinal<total;++ordinal) {
        int task=tilemega::codegen::ServingAttentionL1Task(ordinal,batch,8,blocks,chunk);
        if((task%blocks)*chunk<=past)++live[ordinal%grid];
      }
      assert(*std::max_element(live.begin(),live.end())-*std::min_element(live.begin(),live.end())<=1);
    }
  }
  return 0;
}

}  // namespace tilemega::tests::serving_task_index_test
