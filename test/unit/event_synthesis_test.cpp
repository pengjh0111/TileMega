// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/EventSynthesis.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cassert>

int main() {
  using namespace tilemega::analysis;
  DecoderShape shape;
  auto edges = CouplingDerivation{}.Derive(LlamaDecoderLayer(shape));
  auto events = EventSynthesis{}.Synthesize(edges);
  assert(events.size() == edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    assert(events[i].producer == edges[i].src);
    assert(events[i].consumer == edges[i].dst);
    assert(events[i].shape.size() == edges[i].event_shape.size());
    assert(events[i].wait.ToString() == edges[i].metrics.wait.ToString());
    assert(events[i].tier == edges[i].tier);
  }

  auto gather = CouplingDerivation{}.Derive(GatherModel(shape, true));
  auto relaxed = EventSynthesis{}.Synthesize(gather);
  assert(relaxed.size() == 1);
  assert(relaxed[0].tier == Tier::kDataDependent);
  assert(!relaxed[0].exact);
  assert(!relaxed[0].shape.empty());
  return 0;
}
