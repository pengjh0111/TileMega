// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace tilemega::solver {
struct Sm90DenseFlowEvents;
enum class Sm90DenseSchedulePhase {kStart,kDone,kRmw,kContinue,kProgress0,kProgress1,kProgress2,kProgress3,kProgress4,kProgress5,kProgress6,kProgress7};
struct Sm90DenseScheduleAnchor {std::size_t operation;Sm90DenseSchedulePhase phase=Sm90DenseSchedulePhase::kDone;};
struct Sm90DenseScheduleOperation {
  std::string name,service,resource;
  std::size_t source_event=0;
  std::vector<Sm90DenseScheduleAnchor> after;
  bool structural_join=false;
  std::optional<int> tie_priority;
  std::optional<double> not_before;
};
struct Sm90DenseScheduleService {
  double completion_delay=0,admission_occupancy=0;
  std::optional<double> rmw_offset;
  std::array<std::optional<double>,8> progress_offsets;
};
struct Sm90DenseScheduleResource {int capacity=0;std::string arbitration;};
struct Sm90DenseScheduleRmwMember {std::size_t rmw,publication_return,merge;};
struct Sm90DenseScheduleMapping {
  std::vector<Sm90DenseScheduleOperation> operations;
  std::vector<std::vector<Sm90DenseScheduleRmwMember>> groups;
  std::vector<std::size_t> source_scope;
  std::size_t finish=0;
  double origin=0;
  std::vector<std::string> unpriced;
  std::map<std::string,std::vector<double>> initial_resource_available;
};
struct Sm90DenseSchedulePolicies {
  std::map<std::string,std::optional<Sm90DenseScheduleService>> services;
  std::map<std::string,std::optional<Sm90DenseScheduleResource>> resources;
};
struct Sm90DenseScheduleResult {
  std::vector<std::optional<double>> starts,completions,rmw,continuations;
  std::vector<std::optional<std::size_t>> owners;
  std::vector<std::array<std::optional<double>,8>> progress;
  std::optional<double> scope_time;
  std::vector<std::string> pending;
  std::map<std::string,std::vector<double>> resource_available;
};
// Explicitly mapped source phases only. Unknown service/resource policies or
// unresolved arbitration ties produce no time. Inferred FIFO policies are
// explicit inputs; no physical defaults or full-program price are supplied.
Sm90DenseScheduleResult ScheduleSm90DenseFlowScope(Sm90DenseFlowEvents const&,
    Sm90DenseScheduleMapping const&,Sm90DenseSchedulePolicies const&);
}  // namespace tilemega::solver
