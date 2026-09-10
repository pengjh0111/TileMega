// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/FusionAccess.h>
#include <tilemega/Analysis/ISLContext.h>
#include <stdexcept>

namespace tilemega::analysis {
FusionAccesses ComposeFusionAccesses(TaskAccesses const& producer,
    TaskAccesses const& consumer, std::set<std::string> const& internal_tensors,
    std::set<std::string> const& externally_read_tensors) {
  IslReferenceAudit audit(__func__);
#if !TILEMEGA_FUSION_ACCESS
  throw std::runtime_error("fusion access composition disabled");
#endif
  if (internal_tensors.empty() || producer.writes.empty() || consumer.writes.empty())
    throw std::invalid_argument("fusion requires a producer, consumer and internal tensor");
  FusionAccesses out;
  for (auto const& name : internal_tensors) {
    auto write = producer.writes.find(name), read = consumer.reads.find(name);
    if (write == producer.writes.end() || read == consumer.reads.end())
      throw std::invalid_argument("fusion internal tensor is not a producer-consumer edge");
    auto edge = read->second.ApplyRange(write->second.Reverse());
    if (!read->second.IsSubset(edge.ApplyRange(write->second)))
      throw std::invalid_argument("fusion internal read is not fully supplied by producer");
    out.consumer_to_producer = out.consumer_to_producer.empty()
        ? edge : out.consumer_to_producer.Union(edge);
  }
  auto const& coupling = out.consumer_to_producer;
  if (!coupling.IsSingleValued())
    throw std::invalid_argument("fusion tile constraint: consumer spans multiple producer tasks");
  auto consumer_domain = consumer.writes.begin()->second.Reverse().Image();
  for (auto const& [name, write] : consumer.writes)
    consumer_domain = consumer_domain.Union(write.Reverse().Image());
  if (!consumer_domain.IsSubset(coupling.Reverse().Image()) ||
      !coupling.Reverse().Image().IsSubset(consumer_domain))
    throw std::invalid_argument("fusion edge does not cover the complete consumer task space");

  // I1: composition/union preserve the parameterized relation. No sampled
  // task counts or newly derived operator coupling replace the original C.
  out.task.writes = consumer.writes;
  auto insert = [](auto& table, std::string const& name, CouplingRelation access) {
    auto found = table.find(name);
    if (found == table.end()) table.emplace(name, std::move(access));
    else found->second = found->second.Union(access);
  };
  for (auto const& [name, read] : producer.reads)
    insert(out.task.reads, name, coupling.ApplyRange(read));
  for (auto const& [name, read] : consumer.reads)
    if (!internal_tensors.count(name)) insert(out.task.reads, name, read);
  for (auto const& [name, write] : producer.writes) {
    if (!internal_tensors.count(name) || externally_read_tensors.count(name)) {
      if (!write.Reverse().Image().IsSubset(coupling.Image()) ||
          !coupling.Reverse().IsSingleValued())
        throw std::invalid_argument("fusion retained write requires complete, unique producer ownership");
      insert(out.task.writes, name, coupling.ApplyRange(write));
      out.retained_intermediates.insert(name);
    }
  }
  out.fanout = coupling.FanoutCard();
  out.recompute_tasks = out.fanout.SumDomain().Add(coupling.ImageCard().Scale(-1));
  out.task_count = consumer_domain.ImageCard();
  if (!coupling.Card().SumDomain().SemanticallyEqual(out.fanout.SumDomain(), {}))
    throw std::runtime_error("fusion coupling violates wait/fanout conservation");
  return out;
}
}  // namespace tilemega::analysis
