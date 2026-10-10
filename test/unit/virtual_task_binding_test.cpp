// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/VirtualTaskBinding.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/StorageHazards.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace tilemega::tests::virtual_task_binding_test {
namespace {
using namespace analysis;
ClosedForm F(long value) { return ClosedForm::Constant(value); }
SemanticOp Virtual(char const* name, char const* tensor, bool group) {
  auto t = ClosedForm::Symbol("T");
  auto capacity = group ? (t * F(8)).CeilDiv(F(16)) + F(128).Min(t * F(8)) : t * F(8);
  SemanticOp op; op.name = name; op.exact_task_access = true;
  IterationDim v; v.name = "v"; v.extent = ClosedForm::Symbol("live_slots");
  v.runtime = true; v.capacity = capacity; v.binding_source = "bindings";
  v.binding_requirement = group ? "prefix_sum" : "tensor_values";
  op.domain = {v, {"c", F(64)}};
  op.result = {tensor, {{"v", capacity}, {"c", F(64)}}};
  op.result_map.results = {IndexResult::Dim("v"), IndexResult::Dim("c")};
  op.task_space = {tensor, {{"v", v.extent, F(0), true}, {"c", F(64)}}};
  op.task_map = op.result_map; return op;
}
void CapacityAndProvenance(bool group) {
  auto producer = Virtual("gate_up", "intermediate", group);
  auto consumer = Virtual("down", "partial", group);
  consumer.operands.push_back({producer.name, producer.result, producer.result_map, {}});
  auto encoded = EncodeSemanticOp(consumer);
  auto decoded = DecodeSemanticOp(encoded);
  assert(EncodeSemanticOp(decoded) == encoded && decoded.Serialize() == consumer.Serialize());
  auto bindings = VirtualBindings(decoded);
  assert(bindings.size() == 1 && bindings[0].source == "bindings");
  assert(bindings[0].requirement == (group ? "prefix_sum" : "tensor_values"));
  Granularity g;
  for (auto const* name : {"gate_up", "down"}) g.Tile(name, "v", F(1)).Tile(name, "c", F(32));
  auto graph = Instantiate({{producer, consumer}}, g);
  assert(!graph.nodes[0].HasRuntimeTaskSpace() && !graph.nodes[1].HasRuntimeTaskSpace());
  for (long tokens : {1, 17}) {
    ParamBinding known; known.Bind("T", tokens);
    auto capacity = group ? (tokens * 8 + 15) / 16 + std::min(128L, tokens * 8) : tokens * 8;
    assert(graph.nodes[0].Count().Eval(known, {}) == capacity * 2);
    auto edges = CouplingDerivation{}.Derive(graph, known);
    assert(edges.size() == 1);
    auto const& edge = edges[0];
    assert(edge.exact && edge.tier == Tier::kAffine);
    assert(edge.attributes.extent_kind == ExtentKind::kSymbolicStatic);
    assert(edge.attributes.runtime_requirement == (group ? RuntimeRequirement::kPrefixSum
                                                        : RuntimeRequirement::kTensorValues));
    assert(edge.metrics.wait.Eval({}) == 1);
    for (auto const& [to, from] : edge.C.Points()) assert(to == from);
    assert(edge.C.ToString().find("live_slots") == std::string::npos);
    auto work = DeriveTaskWork(consumer, graph.nodes[1], known);
    assert(work.task_count.Eval({}) == capacity * 2);
    assert(work.write_elements.SumDomain().Eval({}) == capacity * 64);
    assert(work.read_elements.SumDomain().Eval({}) == capacity * 64);
    auto const& p = graph.nodes[0]; auto const& c = graph.nodes[1];
    auto writes = ProjectTaskElements(producer, p, p.element_access->partition,
                                     producer.result, producer.result_map, {}, known);
    auto reads = ProjectTaskRead(consumer, c, c.element_access->partition,
                                producer.result, producer.result_map, {}, known);
    auto hazards = DeriveStorageReuseHazards({&p, writes, {}}, {{&c, reads, {}}}, {&p, writes, {}}, known);
    assert(hazards.size() == 1 && hazards[0].kind == StorageHazardKind::kWAR);
    assert(hazards[0].coupling.tier == Tier::kAffine);
    assert(hazards[0].coupling.attributes.runtime_requirement == edge.attributes.runtime_requirement);
  }
  consumer.operands[0].map.results[0] = IndexResult::DataDependent("rows");
  auto gathered = DecodeSemanticOp(EncodeSemanticOp(consumer));
  assert(gathered.operands[0].map.results[0].binding_source == "rows");
  auto indirect = Instantiate({{producer, gathered}}, g);
  assert(indirect.nodes[1].operands[0].axes[0].binding_source == "rows");
  ParamBinding known; known.Bind("T", 1);
  auto relaxed = CouplingDerivation{}.Derive(indirect, known);
  assert(relaxed.size() == 1 && !relaxed[0].exact);
  assert(relaxed[0].tier == Tier::kDataDependent);
  assert(relaxed[0].attributes.runtime_requirement == RuntimeRequirement::kTensorValues);
  auto exact = CouplingDerivation{}.Derive(graph, known);
  assert(Contains(relaxed[0].C, exact[0].C) && !Contains(exact[0].C, relaxed[0].C));
  for (auto const& [to, from] : relaxed[0].C.Points()) assert(to[1] == from[1]);
}
void InvalidBindings() {
  auto reject = [](SemanticOp op) {
    bool rejected = false;
    try { (void)BindCapacityTaskSpace(op); } catch (std::invalid_argument const&) { rejected = true; }
    assert(rejected);
  };
  auto valid = Virtual("task", "output", false);
  auto invalid = valid; invalid.domain[0].capacity = F(0); reject(invalid);
  invalid = valid; invalid.domain[0].binding_source.clear(); reject(invalid);
  invalid = valid; invalid.domain[0].runtime = false; reject(invalid);
  invalid = valid; invalid.domain[0].binding_requirement = "none"; reject(invalid);
  invalid = valid; invalid.domain[0].capacity.reset(); reject(invalid);
  invalid = valid; invalid.domain[0].capacity = ClosedForm::Symbol("v"); reject(invalid);
  invalid = valid; invalid.task_map.results[0] = IndexResult::Dim("v", F(2)); reject(invalid);
  ParamBinding zero; zero.Bind("T", 0);
  auto graph = Instantiate({{valid}}, {});
  bool rejected = false;
  try { (void)ProjectTaskElements(valid, graph.nodes[0], graph.nodes[0].element_access->partition,
                                  valid.result, valid.result_map, {}, zero); }
  catch (std::invalid_argument const&) { rejected = true; }
  assert(rejected);
  SemanticOp legacy; legacy.name = "legacy"; legacy.domain = {{"m", F(4)}};
  auto encoded = EncodeSemanticOp(legacy);
  assert(encoded.find("capacity") == std::string::npos && encoded.find("binding_source") == std::string::npos);
  assert(DecodeSemanticOp(encoded).Serialize() == legacy.Serialize());
}
}
int TestVirtualTaskBinding(int, char**) {
  IslContext context;
  CapacityAndProvenance(false); CapacityAndProvenance(true); InvalidBindings(); return 0;
}
}
