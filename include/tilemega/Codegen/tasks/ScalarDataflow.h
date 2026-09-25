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
  int extra_flops_per_output=0;
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

/// The operand a kind's TaskBody names in its §5.3.1 `Prefetch()`, or -1 for
/// a kind that declares none.  `RMSNormTaskBody::Prefetch`,
/// `QKNormTaskBody::Prefetch` and `ModelHarness.cuh`'s `PrefetchFor` are the
/// device-side spelling of this same table: the solver credits only what the
/// executor will issue, and the executor issues only what the frontier allows.
inline int ScalarPrefetchOperand(TaskKind kind) {
  switch (kind) {
    case TaskKind::kRMSNorm:
    case TaskKind::kQKNorm: return 1;
    default: return -1;
  }
}

// TaskBody control structure, not fitted per-kind latency constants. The
// scalar expressions themselves remain in the single arithmetic schema.
inline ScalarDataflow ScalarTaskDataflow(TaskKind kind) {
  ScalarDataflow flow;
  int input=flow.Add(ScalarPhase::kLoad);
  switch (kind) {
    case TaskKind::kQKNorm:
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
    case TaskKind::kFusedAttention: {
      int qk=flow.Add(ScalarPhase::kArithmetic,{input});
      int softmax=flow.Add(ScalarPhase::kArithmetic,{qk});
      int values=flow.Add(ScalarPhase::kLoad,{softmax});
      flow.Add(ScalarPhase::kStore,{flow.Add(ScalarPhase::kArithmetic,{values})});
      return flow;
    }
    case TaskKind::kAttentionMerge:
    case TaskKind::kArgmaxReduce:
      flow.Add(ScalarPhase::kStore,{flow.Add(ScalarPhase::kBlockReduction,{input})});
      return flow;
    case TaskKind::kGemmCombine:
      flow.extra_flops_per_output=1; // Zero-seeded chunk summation.
      flow.Add(ScalarPhase::kStore,{flow.Add(ScalarPhase::kArithmetic,{input})});
      return flow;
    case TaskKind::kEmbedding: {
      // Two dependent global reads: the row address is the first load's value.
      flow.nodes[input].read_operands={0};
      int row=flow.Add(ScalarPhase::kLoad,{input});
      flow.nodes[row].read_operands={1};
      flow.Add(ScalarPhase::kStore,{row});
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
