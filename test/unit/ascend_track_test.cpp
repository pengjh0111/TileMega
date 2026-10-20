// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/Ascend/Track.h>
#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace tilemega::ascend;
template<class F> static void Reject(F f) {
  bool failed = false;
  try { f(); } catch (std::invalid_argument const&) { failed = true; }
  assert(failed);
}
static TargetCapability Fixture() {
  TargetCapability t;
  t.arch_tag = "ascend_3510"; t.product_sku = "950PR"; t.cann_version = "fixture";
  t.tracks = {{"aic", TrackRole::kAic, 4, {1024, 128, 128, 512, 0, 0, 0},
               {ImplementationRole::kAicOnly, ImplementationRole::kMix1_2}, "mix_group"},
              {"aiv", TrackRole::kAiv, 8, {0, 0, 0, 0, 768, 0, 0},
               {ImplementationRole::kAivOnly, ImplementationRole::kMix1_2}, "mix_group"}};
  t.max_verified_mix_groups = 4;
  t.implementations = {{"cpu_gemm_fixture", tilemega::codegen::TaskKind::kGemm},
                       {"cpu_add_fixture", tilemega::codegen::TaskKind::kAdd},
                       {"cpu_mix_fixture", tilemega::codegen::TaskKind::kGemm}};
  t.aic_only_kernel = t.aiv_only_kernel = t.mix_1_2 = true;
  return t;
}
static TaskImplementation Impl(ImplementationRole role) {
  TaskImplementation i;
  i.impl_id = role == ImplementationRole::kAicOnly ? "cpu_gemm_fixture" :
              role == ImplementationRole::kAivOnly ? "cpu_add_fixture" : "cpu_mix_fixture";
  if (role == ImplementationRole::kAivOnly) i.kind = tilemega::codegen::TaskKind::kAdd;
  i.input_layout = i.output_layout = "ND";
  i.required_role = role;
  return i;
}
int main() {
  auto target = Fixture();
  tilemega::TargetSpec outer;
  outer.ascend_capability = target;
  assert(outer.ascend_capability->TrackCount(TrackRole::kAic) == 4);

  // T-1: the placement domain is the resource class, independent of global IDs.
  auto aic = WorkerMap::Build(target, LaunchMode::kAicOnly, 2);
  auto aiv = WorkerMap::Build(target, LaunchMode::kAivOnly, 3);
  assert(Home(aic.Domain(ImplementationRole::kAicOnly), 0, 3) == 1);
  assert(Home(aiv.Domain(ImplementationRole::kAivOnly), 1, 2) == 0);
  Reject([&] { aic.Domain(ImplementationRole::kAivOnly); });
  Reject([&] { Spread({2, 3, 4}, 0, 0, 2); });

  // T-2: both small and the fixture's maximum MIX map are bijective.
  for (int g : {1, 2, 4}) {
    auto m = WorkerMap::Build(target, LaunchMode::kMix1_2, g);
    std::vector<uint32_t> offsets(m.workers.size()+1);
    m.Validate(target, offsets, 0);
    assert(m.launch.num_blocks == g && m.workers.size() == unsigned(3*g));
    for (int w = g; w < 3*g; ++w) assert(m.At(w).group == (w-g)/2);
  }
  auto map = WorkerMap::Build(target, LaunchMode::kMix1_2, 2);
  assert(map.Domain(ImplementationRole::kAicOnly) == std::vector<int>({0, 1}));
  assert(map.Domain(ImplementationRole::kAivOnly) == std::vector<int>({2, 3, 4, 5}));
  assert(Spread(map.Domain(ImplementationRole::kAivOnly), 0, 0, 2) == std::vector<int>({2, 4}));
  auto malformed = map; malformed.workers[3].sub_block_id = 0;
  Reject([&] { malformed.Validate(target, std::vector<uint32_t>(7), 0); });
  Reject([&] { map.Validate(target, {0, 0, 0, 2, 1, 2, 2}, 2); });

  // T-3 and owner-once events: a composite is one task with three participants.
  std::vector<TaskAssignment> tasks = {
      {0, Impl(ImplementationRole::kAicOnly), {0}, {}, {0}},
      {2, Impl(ImplementationRole::kAivOnly), {2}, {0}, {1}},
      {1, Impl(ImplementationRole::kMix1_2), {1, 4, 5}, {1}, {2}}};
  std::vector<std::vector<unsigned>> queues = {{0}, {2}, {1}, {}, {2}, {2}};
  auto audit = ValidatePlan(target, map, tasks, queues, {1, 1, 1});
  assert(audit.topological_order == std::vector<unsigned>({0, 1, 2}));
  assert(audit.event_triggers == std::vector<unsigned>({1, 1, 1}));
  auto wrong = tasks; wrong[0].owner = 2;
  Reject([&] { ValidatePlan(target, map, wrong, queues, {1, 1, 1}); });
  wrong = tasks; wrong[2].participants = {0, 4, 5};
  Reject([&] { ValidatePlan(target, map, wrong, queues, {1, 1, 1}); });
  Reject([&] { ValidatePlan(target, map, tasks, queues, {1, 1, 3}); });

  // T-4: affinity does not change identity semantics or the direct-route scope.
  Reject([&] { ValidateColocation(map, 0, 2, true); });
  ValidateColocation(map, 0, 2, false);
  auto handoff = HandoffContract{HandoffRoute::kLocalL0cToUb, SyncKind::kGroupFlag,
                                0, 2, 0, "ND", "fixture", 2};
  Reject([&] { handoff.Validate(target, map); });
  auto direct = target; direct.same_group_l0c_to_ub = direct.group_flag = true;
  handoff.Validate(direct, map);
  handoff.consumer = 4;
  Reject([&] { handoff.Validate(direct, map); });

  // T-5: each buffer has its own owner and bound, not an occupancy quotient.
  auto impl = Impl(ImplementationRole::kAicOnly);
  impl.aic_scratch = {1024, 128, 128, 512, 0, 0, 0};
  impl.Validate(target);
  ++impl.aic_scratch.l0c;
  Reject([&] { impl.Validate(target); });
  impl = Impl(ImplementationRole::kAivOnly); impl.aiv_scratch.ub = 768;
  impl.Validate(target);
  impl.aiv_scratch.l1 = 1;
  Reject([&] { impl.Validate(target); });

  // T-6: one device-global rate services the sum of both roles' GM traffic.
  assert(!SharedGmTimeNs({120, 80}, std::nullopt));
  assert(*SharedGmTimeNs({120, 80}, 10.0) == 20.0);
  Reject([&] { SharedGmTimeNs({1}, 0.0); });
  assert(!Impl(ImplementationRole::kAicOnly).calibrated);
  tilemega::TargetSpec actual_target;
  actual_target.arch_tag = "ascend_3510"; actual_target.ascend_capability = target;
  tilemega::solver::BackendTraits traits;
  traits.arch_tag = "ascend_3510"; traits.shape_legal = true;
  traits.ascend_implementation = Impl(ImplementationRole::kAicOnly);
  tilemega::solver::BackendCandidate candidate(traits);
  assert(candidate.isLegal(actual_target));
  assert(candidate.ResidentWorkersPerTrack(actual_target) == 1);
  Reject([&] { candidate.CoResidentPerSM(actual_target); });
  Reject([&] { candidate.RecordPtxas("Used 32 registers"); });
  traits.smem_bytes = 1;
  assert(!tilemega::solver::BackendCandidate(traits).isLegal(actual_target));
  auto unsupported = Impl(ImplementationRole::kAicOnly);
  unsupported.kind = tilemega::codegen::TaskKind::kEncoderAttention;
  Reject([&] { unsupported.Validate(target); });
  traits.smem_bytes = 0; traits.ascend_implementation->aic_scratch.ub = 1;
  assert(!tilemega::solver::BackendCandidate(traits).isLegal(actual_target));

  // T-16: the task graph is a DAG; adding its legal worker FIFO can still cycle.
  std::vector<TaskAssignment> deadlock = {
      {0, Impl(ImplementationRole::kAicOnly), {0}, {}, {}},
      {1, Impl(ImplementationRole::kAicOnly), {1}, {0}, {}},
      {0, Impl(ImplementationRole::kAicOnly), {0}, {1}, {}}};
  Reject([&] { ValidatePlan(target, map, deadlock, {{2, 0}, {1}, {}, {}, {}, {}}); });
  std::cout << "{\"test\":\"Ascend host contracts\",\"passed\":[\"T-1\",\"T-2\",\"T-3\",\"T-4\",\"T-5\",\"T-6\",\"T-16\"],\"device_execution\":false,\"solver_integration\":false}\n";
}
