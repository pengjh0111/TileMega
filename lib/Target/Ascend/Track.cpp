// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/Ascend/Track.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace tilemega::ascend {
namespace {
void Require(bool ok, char const* message) {
  if (!ok) throw std::invalid_argument(message);
}
void Fits(LocalResources const& use, LocalResources const& budget) {
  Require(use.l1 <= budget.l1 && use.l0a <= budget.l0a && use.l0b <= budget.l0b &&
              use.l0c <= budget.l0c && use.ub <= budget.ub &&
              use.registers <= budget.registers && use.ssbuffer <= budget.ssbuffer,
          "Implementation exceeds a per-role local buffer budget");
}
} // namespace

int TargetCapability::TrackCount(TrackRole role) const {
  int count = 0;
  for (auto const& track : tracks) if (track.role == role) count += track.count;
  return count;
}
LocalResources const& TargetCapability::Budget(TrackRole role) const {
  for (auto const& track : tracks) if (track.role == role) return track.local_resources;
  throw std::invalid_argument("Missing TrackClass budget");
}
void TargetCapability::Validate() const {
  Require(arch_tag == "ascend_3510" && product_sku == "950PR", "Unsupported architecture or SKU");
  Require(!cann_version.empty(), "Missing CANN version");
  std::set<TrackRole> roles;
  for (auto const& track : tracks) {
    Require(track.count > 0 && !track.name.empty(), "Invalid TrackClass");
    Require(roles.insert(track.role).second, "Duplicate TrackClass role");
    auto const& b = track.local_resources;
    if (track.role == TrackRole::kAic) Require(!b.ub && !b.registers, "AIC cannot own AIV resources");
    else Require(!b.l1 && !b.l0a && !b.l0b && !b.l0c, "AIV cannot own AIC resources");
  }
  Require(max_verified_mix_groups >= 0, "Negative verified group count");
  Require(max_verified_mix_groups <= TrackCount(TrackRole::kAic) &&
              max_verified_mix_groups * 2 <= TrackCount(TrackRole::kAiv), "MIX exceeds role counts");
  Require(!mix_1_2 || max_verified_mix_groups > 0, "MIX is unverified");
}
WorkerMap WorkerMap::Build(TargetCapability const& target, LaunchMode mode, int blocks) {
  target.Validate();
  Require(mode == LaunchMode::kAicOnly || mode == LaunchMode::kAivOnly || mode == LaunchMode::kMix1_2,
          "Unsupported launch mode");
  Require(blocks > 0, "Launch requires positive block count");
  WorkerMap map;
  map.launch = {mode, mode == LaunchMode::kMix1_2 ? blocks : 0, blocks, 1};
  if (mode == LaunchMode::kMix1_2) {
    Require(target.mix_1_2 && blocks <= target.max_verified_mix_groups, "Unverified MIX launch shape");
    for (int g = 0; g < blocks; ++g) map.workers.push_back({g, g, TrackRole::kAic, g, 0});
    for (int g = 0; g < blocks; ++g)
      for (int lane = 0; lane < 2; ++lane)
        map.workers.push_back({blocks+2*g+lane, g, TrackRole::kAiv, 2*g+lane, lane});
  } else {
    const bool aic = mode == LaunchMode::kAicOnly;
    Require(aic ? target.aic_only_kernel : target.aiv_only_kernel, "Unverified single-role launch mode");
    auto role = aic ? TrackRole::kAic : TrackRole::kAiv;
    Require(blocks <= target.TrackCount(role), "Launch exceeds TrackClass count");
    for (int i = 0; i < blocks; ++i) map.workers.push_back({i, -1, role, i, 0});
  }
  return map;
}
WorkerPlacement const& WorkerMap::At(int worker) const {
  Require(worker >= 0 && size_t(worker) < workers.size(), "Worker outside map");
  Require(workers[worker].global_worker_id == worker, "Noncanonical worker ID");
  return workers[worker];
}
std::vector<int> WorkerMap::Domain(ImplementationRole role) const {
  std::vector<int> result;
  for (auto const& worker : workers) {
    if ((role == ImplementationRole::kAivOnly) == (worker.role == TrackRole::kAiv))
      result.push_back(worker.global_worker_id);
  }
  Require(role != ImplementationRole::kMix1_2 || launch.mode == LaunchMode::kMix1_2,
          "Composite placement requires MIX groups");
  Require(!result.empty(), "Implementation has no legal owner domain");
  return result;
}
void WorkerMap::Validate(TargetCapability const& target,
                         std::vector<std::uint32_t> const& offsets, std::uint32_t refs) const {
  auto expected = Build(target, launch.mode, launch.num_blocks);
  Require(launch.residency == 1 && launch.num_groups == expected.launch.num_groups,
          "Launch topology conflates groups, residency or blocks");
  Require(workers.size() == expected.workers.size(), "Wrong worker count");
  Require(offsets.size() == workers.size()+1 && offsets.front() == 0 && offsets.back() == refs,
          "Queue offsets do not cover TaskRefs exactly");
  for (size_t i = 0; i < workers.size(); ++i) {
    auto const& w = At(i);
    auto const& e = expected.workers[i];
    Require(w.group == e.group && w.role == e.role && w.local_block_id == e.local_block_id &&
                w.sub_block_id == e.sub_block_id, "Worker hardware/group/role mapping mismatch");
    Require(offsets[i] <= offsets[i+1] && offsets[i+1] <= refs, "Invalid worker queue bounds");
  }
}
void TaskImplementation::Validate(TargetCapability const& target) const {
  target.Validate();
  Require(required_role == ImplementationRole::kAicOnly || required_role == ImplementationRole::kAivOnly ||
              required_role == ImplementationRole::kMix1_2, "Unsupported implementation role");
  Require(!impl_id.empty() && arch_tag == target.arch_tag, "Missing or incompatible implementation");
  auto found = target.implementations.find(impl_id);
  Require(found != target.implementations.end() && found->second == kind,
          "Unbound implementation or unsupported TaskKind");
  Require(!input_layout.empty() && !output_layout.empty(), "Missing implementation layout contract");
  if (required_role != ImplementationRole::kAivOnly) Fits(aic_scratch, target.Budget(TrackRole::kAic));
  else Fits(aic_scratch, {});
  if (required_role != ImplementationRole::kAicOnly) Fits(aiv_scratch, target.Budget(TrackRole::kAiv));
  else Fits(aiv_scratch, {});
  if (required_role == ImplementationRole::kMix1_2) Require(target.mix_1_2, "MIX implementation unavailable");
  for (auto const& track : target.tracks) {
    if ((required_role == ImplementationRole::kAicOnly && track.role == TrackRole::kAiv) ||
        (required_role == ImplementationRole::kAivOnly && track.role == TrackRole::kAic)) continue;
    Require(std::find(track.available_impl_types.begin(), track.available_impl_types.end(), required_role) !=
                track.available_impl_types.end(), "TrackClass does not offer implementation role");
  }
  std::set<unsigned> flags;
  for (auto flag : internal_flags) Require(flags.insert(flag).second, "Internal flag conflict");
}
void HandoffContract::Validate(TargetCapability const& target, WorkerMap const& map) const {
  auto const& p = map.At(producer);
  auto const& c = map.At(consumer);
  const bool group = p.group >= 0 && p.group == c.group;
  Require(buffer_epochs > 0 && !layout.empty() && !visibility_evidence.empty(), "Incomplete handoff lifetime/visibility contract");
  if (route != HandoffRoute::kGmEvent) {
    Require(group && map.launch.mode == LaunchMode::kMix1_2, "Local route requires same MIX group");
    Require(sync == SyncKind::kGroupFlag && target.group_flag, "Local route needs verified group synchronization");
  }
  if (route == HandoffRoute::kLocalL0cToUb)
    Require(target.same_group_l0c_to_ub && p.role == TrackRole::kAic && c.role == TrackRole::kAiv,
            "L0C-to-UB unavailable or wrong direction");
  else if (route == HandoffRoute::kLocalUbToL1)
    Require(target.same_group_ub_to_l1 && p.role == TrackRole::kAiv && c.role == TrackRole::kAic,
            "UB-to-L1 unavailable or wrong direction");
  else if (route == HandoffRoute::kLocalControl) Require(target.ssbuffer_control, "SSBuffer control unavailable");
  else {
    Require(workspace_bytes > 0, "GM route has no storage");
    Require((sync == SyncKind::kGmEvent && target.gm_event) ||
                (sync == SyncKind::kStageBarrier && target.cross_group_stage_barrier) ||
                (sync == SyncKind::kGroupFlag && target.group_flag && group), "Unverified GM visibility/synchronization route");
  }
}
int Home(std::vector<int> const& domain, int base, int tile) {
  Require(!domain.empty() && base >= 0 && tile >= 0, "Invalid placement domain or coordinate");
  Require(std::set<int>(domain.begin(), domain.end()).size() == domain.size(), "Duplicate placement workers");
  Require(*std::min_element(domain.begin(), domain.end()) >= 0, "Negative worker ID");
  return domain[(uint64_t(base)+uint64_t(tile)) % domain.size()];
}
std::vector<int> Spread(std::vector<int> const& domain, int base, int tile, int width) {
  Home(domain, base, tile);
  Require(width > 0 && size_t(width) <= domain.size() && domain.size() % width == 0, "Invalid spread width");
  std::vector<int> result;
  for (int i = 0; i < width; ++i)
    result.push_back(Home(domain, base, tile+i*(domain.size()/width)));
  return result;
}
void ValidateColocation(WorkerMap const& map, int producer, int consumer, bool identity) {
  auto const& p = map.At(producer);
  auto const& c = map.At(consumer);
  if (identity) Require(producer == consumer, "Identity colocation means the same worker");
  else Require(p.group >= 0 && p.group == c.group, "Affinity requires the same MIX group");
}
std::optional<double> SharedGmTimeNs(std::vector<std::uint64_t> const& bytes,
                                   std::optional<double> rate) {
  if (!rate) return std::nullopt;
  Require(std::isfinite(*rate) && *rate > 0, "Invalid calibrated shared GM rate");
  long double total = 0;
  for (auto size : bytes) total += size;
  return static_cast<double>(total / *rate);
}
PlanAudit ValidatePlan(TargetCapability const& target, WorkerMap const& map,
                       std::vector<TaskAssignment> const& tasks,
                       std::vector<std::vector<unsigned>> const& queues,
                       std::vector<unsigned> const& expected_event_triggers) {
  Require(queues.size() == map.workers.size(), "Wrong number of worker queues");
  std::vector<uint32_t> offsets{0};
  for (auto const& q : queues) {
    Require(uint64_t(offsets.back())+q.size() <= std::numeric_limits<uint32_t>::max(), "TaskRef count overflow");
    offsets.push_back(offsets.back()+q.size());
  }
  map.Validate(target, offsets, offsets.back());
  std::vector<std::set<int>> seen(tasks.size());
  std::vector<std::set<unsigned>> graph(tasks.size());
  std::vector<unsigned> indegree(tasks.size(), 0);
  PlanAudit audit;
  audit.event_triggers.resize(expected_event_triggers.size());
  auto edge = [&](unsigned p, unsigned c, bool fifo) {
    Require(p < tasks.size() && c < tasks.size() && p != c, "Invalid dependency or duplicate FIFO task");
    if (graph[p].insert(c).second) {
      ++indegree[c];
      if (fifo) ++audit.fifo_edges; else ++audit.dependency_edges;
    }
  };
  for (size_t worker = 0; worker < queues.size(); ++worker) {
    auto const& queue = queues[worker];
    for (size_t i = 0; i < queue.size(); ++i) {
      Require(queue[i] < tasks.size(), "TaskRef outside task domain");
      Require(seen[queue[i]].insert(worker).second, "Duplicate TaskRef on worker");
      if (i) edge(queue[i-1], queue[i], true);
    }
  }
  for (size_t task = 0; task < tasks.size(); ++task) {
    auto const& t = tasks[task];
    t.implementation.Validate(target);
    auto const& owner = map.At(t.owner);
    auto domain = map.Domain(t.implementation.required_role);
    Require(std::find(domain.begin(), domain.end(), t.owner) != domain.end(), "Task owner has wrong role");
    std::set<int> participants(t.participants.begin(), t.participants.end());
    Require(participants.size() == t.participants.size() && participants == seen[task] &&
                participants.count(t.owner), "Task ownership or queue participation mismatch");
    if (t.implementation.required_role == ImplementationRole::kMix1_2) {
      Require(participants.size() == 3 && owner.group >= 0, "Composite task requires three same-group lanes");
      int aic = 0, aiv = 0;
      for (int worker : participants) {
        auto const& w = map.At(worker);
        Require(w.group == owner.group, "Composite task mixes unrelated groups");
        w.role == TrackRole::kAic ? ++aic : ++aiv;
      }
      Require(aic == 1 && aiv == 2, "Invalid composite role tuple");
    } else Require(participants.size() == 1, "Single-role task has extra participants");
    for (auto p : t.dependencies) edge(p, task, false);
    std::set<unsigned> published;
    for (auto event : t.publishes) {
      Require(event < audit.event_triggers.size(), "Event index outside plan event domain");
      Require(published.insert(event).second, "Task publishes external event twice");
      ++audit.event_triggers[event];
    }
  }
  std::vector<unsigned> ready;
  for (size_t i = 0; i < tasks.size(); ++i) if (!indegree[i]) ready.push_back(i);
  for (size_t i = 0; i < ready.size(); ++i) {
    auto task = ready[i];
    audit.topological_order.push_back(task);
    for (auto consumer : graph[task]) if (!--indegree[consumer]) ready.push_back(consumer);
  }
  Require(audit.topological_order.size() == tasks.size(), "Task dependencies plus worker FIFO form a deadlock cycle");
  Require(audit.event_triggers == expected_event_triggers, "Event trigger count differs from producer ownership");
  return audit;
}
} // namespace tilemega::ascend
