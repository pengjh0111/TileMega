// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <stdexcept>
#include <vector>

namespace tilemega::codegen {

// These edges name the expanded runtime table. Generated stage indices cannot
// be used after split-K inserts combine stages between the original stages.
inline std::vector<LagDependency> ServingLagDependencies(
    std::vector<StageDesc> const& stages) {
  std::vector<LagDependency> result;
  std::uint32_t embedding = kNoOperand, argmax = kNoOperand;
  for (std::uint32_t i = 0; i < stages.size(); ++i) {
    if (stages[i].kind == TaskKind::kEmbedding) {
      if (embedding != kNoOperand) throw std::invalid_argument("multiple serving embeddings");
      embedding = i;
    }
    if (stages[i].kind == TaskKind::kArgmaxReduce) {
      if (argmax != kNoOperand) throw std::invalid_argument("multiple serving argmax stages");
      argmax = i;
    }
  }
  if (embedding != kNoOperand) {
    if (argmax == kNoOperand) throw std::invalid_argument("serving embedding has no argmax");
    result.push_back({argmax, embedding, LagDependency::Kind::kToken});
  }
  for (std::uint32_t i = 0; i < stages.size(); ++i) {
    if (stages[i].kind != TaskKind::kFusedAttention) continue;
    std::uint32_t producer = i;
    if (stages[i].handoff_reduce_stage != kNoOperand) {
      auto const reducer = stages[i].handoff_reduce_stage;
      if (reducer >= stages.size() ||
          stages[reducer].kind != TaskKind::kAttentionMerge)
        throw std::invalid_argument("attention lag reducer is not a merge");
      if (stages[reducer].handoff_elided) producer = reducer;
    }
    result.push_back({producer, i, LagDependency::Kind::kHistoricalKv});
  }
  return result;
}

inline void ValidateServingLagDependencies(
    std::vector<StageDesc> const& stages,
    std::vector<std::uint32_t> const& event_flags,
    std::vector<std::uint32_t> const& event_offsets,
    std::vector<LagDependency> const& lags) {
  auto expected = ServingLagDependencies(stages);
  if (event_flags.size() != stages.size() || event_offsets.size() != stages.size() + 1 ||
      lags.size() != expected.size())
    throw std::invalid_argument("serving lag table does not match runtime stages");
  for (std::size_t i = 0; i < lags.size(); ++i)
    if (lags[i].producer != expected[i].producer ||
        lags[i].consumer != expected[i].consumer || lags[i].kind != expected[i].kind)
      throw std::invalid_argument("serving lag table does not match runtime stages");
  for (auto const& lag : lags) {
    if (lag.producer >= stages.size() || lag.consumer >= stages.size() ||
        !(event_flags[lag.producer] & kNeedsAggregateEvent) ||
        event_offsets[lag.producer] >= event_offsets[lag.producer + 1])
      throw std::invalid_argument("serving lag producer has no aggregate event row");
  }
}
} // namespace tilemega::codegen
