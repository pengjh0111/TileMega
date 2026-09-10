// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/FusionAccess.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <stdexcept>

using namespace tilemega::analysis;
int main() try {
  IslContext context;
  auto relation = [](char const* text) { return CouplingRelation::FromIslText(text); };
  TaskAccesses producer, consumer;
  producer.writes["mid"] = relation("[S] -> { [p] -> [i] : S > 0 and 0 <= p < S and 2p <= i < 2p+2 }");
  producer.reads["input"] = producer.writes.at("mid");
  consumer.reads["mid"] = relation("[S] -> { [c] -> [i] : S > 0 and 0 <= c < 2S and i=c }");
  consumer.writes["out"] = consumer.reads.at("mid");
  auto fused = ComposeFusionAccesses(producer, consumer, {"mid"}, {});
  auto expected = relation("[S] -> { [c] -> [i] : S > 0 and 0 <= c < 2S and 2*floor(c/2) <= i < 2*floor(c/2)+2 }");
  if (!fused.task.reads.at("input").IsSubset(expected) ||
      !expected.IsSubset(fused.task.reads.at("input")) || fused.task.reads.count("mid") ||
      fused.task.writes.count("mid")) throw std::runtime_error("incorrect fusion access composition");
  for (int seq : {1, 4, 128}) {
    ParamBinding binding; binding.Bind("S", seq);
    if (fused.recompute_tasks.Eval(binding) != seq || fused.task_count.Eval(binding) != 2*seq)
      throw std::runtime_error("fanout replication not charged");
  }
  int errors = 0;
  auto reject = [&](auto action) {
    bool rejected = false;
    try { action(); } catch (std::invalid_argument const&) { rejected = true; }
    if (!rejected) throw std::runtime_error("illegal fusion accepted");
    ++errors;
  };
  reject([&] { ComposeFusionAccesses(producer,consumer,{},{}); });
  reject([&] { ComposeFusionAccesses(producer,consumer,{"absent"},{}); });
  reject([&] { ComposeFusionAccesses(producer,consumer,{"mid"},{"mid"}); });
  auto wide = consumer;
  wide.reads["mid"] = relation("[S] -> { [c] -> [i] : S > 0 and 0 <= c < S and 0 <= i < 2S }");
  wide.writes["out"] = relation("[S] -> { [c] -> [i] : S > 0 and 0 <= c < S and i=c }");
  reject([&] { ComposeFusionAccesses(producer,wide,{"mid"},{}); });
  auto missing = consumer;
  missing.reads["mid"] = relation("[S] -> { [c] -> [i] : S > 0 and 0 <= c < 2S and i=c+2S }");
  reject([&] { ComposeFusionAccesses(producer,missing,{"mid"},{}); });
  consumer.reads["mid"] = producer.writes.at("mid");
  consumer.writes["out"] = producer.writes.at("mid");
  auto retained = ComposeFusionAccesses(producer,consumer,{"mid"},{"mid"});
  if (!retained.task.writes.count("mid") || !retained.recompute_tasks.IsZero())
    throw std::runtime_error("external output was dropped or unique producer recomputed");
  std::cout << "FUSION_ACCESS symbolic_composition=1 external_retention=1 errors=" << errors << '\n';
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
