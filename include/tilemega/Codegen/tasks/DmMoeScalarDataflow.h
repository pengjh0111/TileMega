// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ScalarDataflow.h>
#include <tilemega/Codegen/DmDescriptors.h>
#include <cstdint>

namespace tilemega::codegen {
// Control phases follow RunMoe and MoEDispatchTaskBody. For scatter's last
// chunk the static barrier count is an upper bound; no measured cost is implied.
inline ScalarDataflow DmMoeScalarTaskDataflow(DmMoeStage const& cfg,unsigned tokens) {
  if(!tokens || !cfg.experts || cfg.experts>128 || !cfg.top_k || cfg.top_k>32 ||
      cfg.top_k>cfg.experts || !cfg.chunk_tokens)
    throw std::invalid_argument("invalid MoE scalar control geometry");
  ScalarDataflow flow;int tail=-1;
  auto add=[&](ScalarPhase phase) {
    tail=flow.Add(phase,tail<0?std::vector<int>{}:std::vector<int>{tail});
  };
  auto publish=[&](unsigned n) {while(n--)add(ScalarPhase::kPublish);};
  auto select=[&] {add(ScalarPhase::kLoad);add(ScalarPhase::kArithmetic);add(ScalarPhase::kStore);};
  auto histogram=[&] {
    publish(1);add(ScalarPhase::kLoad);add(ScalarPhase::kArithmetic);
    publish(1);add(ScalarPhase::kStore);publish(1);
  };
  auto prefix=[&] {
    add(ScalarPhase::kLoad);add(ScalarPhase::kArithmetic);add(ScalarPhase::kStore);publish(1);
    for(unsigned stride=1;stride<cfg.experts;stride*=2) {
      add(ScalarPhase::kArithmetic);publish(2);
    }
    add(ScalarPhase::kStore);publish(1);add(ScalarPhase::kStore);publish(1);
  };
  auto scatter=[&](unsigned rows) {
    if(cfg.grouped)publish(1);
    auto waves=(std::uint64_t(rows)*cfg.top_k+127)/128;
    for(std::uint64_t wave=0;wave<waves;++wave) {
      if(cfg.grouped)publish(1);
      add(ScalarPhase::kLoad);add(ScalarPhase::kArithmetic);
      if(cfg.grouped) {
        publish(1);add(ScalarPhase::kLoad);add(ScalarPhase::kArithmetic);
      }
      add(ScalarPhase::kStore);
      if(cfg.grouped)publish(2);
    }
    publish(1);
  };
  switch(cfg.step) {
    case DmMoeStep::kSelect:select();break;
    case DmMoeStep::kSelectAndDispatch:
      if(std::uint64_t(tokens)*cfg.top_k>4096)
        throw std::invalid_argument("small dispatch exceeds its runtime capacity");
      select();publish(1);
      if(cfg.grouped) {
        for(unsigned begin=0;begin<tokens;begin+=cfg.chunk_tokens)histogram();
        prefix();
      }
      for(unsigned begin=0;begin<tokens;begin+=cfg.chunk_tokens)
        scatter(std::min(cfg.chunk_tokens,tokens-begin));
      break;
    case DmMoeStep::kHistogram:if(cfg.grouped)histogram();break;
    case DmMoeStep::kPrefix:if(cfg.grouped)prefix();break;
    case DmMoeStep::kScatter:scatter(std::min(cfg.chunk_tokens,tokens));break;
    case DmMoeStep::kCombine:
      add(ScalarPhase::kLoad);add(ScalarPhase::kArithmetic);add(ScalarPhase::kStore);
      publish(1);add(ScalarPhase::kArithmetic);add(ScalarPhase::kStore);publish(1);break;
    default:throw std::invalid_argument("invalid MoE scalar control step");
  }
  if(flow.nodes.empty())throw std::invalid_argument("empty MoE scalar control step");
  return flow;
}
}
