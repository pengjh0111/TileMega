// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tilemega::codegen {
enum class ScalarPhase { kLoad, kStore, kArithmetic, kBlockReduction, kPublish, kLocalLoad, kLocalStore };
struct ScalarFlowNode {
  ScalarPhase phase;
  std::vector<int> inputs;
  // Empty means every input operand; otherwise indices follow the TaskBody.
  std::vector<int> read_operands;
};
struct ScalarDataflow {
  std::vector<ScalarFlowNode> nodes;
  int Add(ScalarPhase phase,std::vector<int> inputs={}) {
    nodes.push_back({phase,std::move(inputs),{}});
    return int(nodes.size())-1;
  }
  // The latency depth counts dependent global-memory phases, not FLOPs.
  // A block reduction publishes partials then halves the active set, with
  // the same barrier at every level as RMSNormTaskBody::RunTask.
  std::pair<int,int> MemoryDepthAndBarriers(int threads) const {
    if (nodes.empty() || threads<=0 || (threads&(threads-1)))
      throw std::invalid_argument("invalid scalar dataflow or reduction width");
    std::vector<int> depth;
    int barriers=0;
    for (int i=0;i<int(nodes.size());++i) {
      int d=0;
      for (int p:nodes[i].inputs) {
        if (p<0 || p>=i) throw std::invalid_argument("scalar dataflow is not a topological DAG");
        d=std::max(d,depth[p]);
      }
      auto phase=nodes[i].phase;
      if (phase==ScalarPhase::kLoad || phase==ScalarPhase::kStore) ++d;
      if (phase==ScalarPhase::kPublish) ++barriers;
      if (phase==ScalarPhase::kBlockReduction)
        for (int live=threads;live;live/=2) ++barriers;
      depth.push_back(d);
    }
    return {*std::max_element(depth.begin(),depth.end()),barriers};
  }
};

// TaskBody control structure, not fitted per-kind latency constants. The
// scalar expressions themselves remain in the single arithmetic schema.
inline ScalarDataflow ScalarTaskDataflow(TaskKind kind) {
  ScalarDataflow flow;
  int input=flow.Add(ScalarPhase::kLoad);
  switch (kind) {
    case TaskKind::kRMSNorm: {
      flow.nodes[input].read_operands={0};
      int sum=flow.Add(ScalarPhase::kBlockReduction,{input});
      int scale=flow.Add(ScalarPhase::kArithmetic,{sum});
      int weighted=flow.Add(ScalarPhase::kLoad,{scale});
      flow.nodes[weighted].read_operands={0,1};
      flow.Add(ScalarPhase::kStore,{weighted});
      return flow;
    }
    case TaskKind::kAttention: {
      int qk=flow.Add(ScalarPhase::kArithmetic,{input});
      int scores=flow.Add(ScalarPhase::kPublish,{qk});
      int softmax=flow.Add(ScalarPhase::kArithmetic,{scores});
      int probabilities=flow.Add(ScalarPhase::kPublish,{softmax});
      int values=flow.Add(ScalarPhase::kLoad,{probabilities});
      int pv=flow.Add(ScalarPhase::kArithmetic,{values});
      flow.Add(ScalarPhase::kStore,{pv});
      return flow;
    }
    case TaskKind::kRoPE:
    case TaskKind::kKVAppend:
    case TaskKind::kElementwise:
    case TaskKind::kAdd:
      flow.Add(ScalarPhase::kStore,{flow.Add(ScalarPhase::kArithmetic,{input})});
      return flow;
    default: throw std::invalid_argument("TaskBody has no scalar dataflow");
  }
}
}  // namespace tilemega::codegen
