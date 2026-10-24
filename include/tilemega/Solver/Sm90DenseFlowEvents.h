// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseWorkGraph.h>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace tilemega::solver {
using Sm90DenseEventId=std::size_t;
inline constexpr Sm90DenseEventId kSm90DenseNoEvent=std::numeric_limits<Sm90DenseEventId>::max();
enum class Sm90DenseFlowEventKind {
  kKernelBegin,kRingSlotInitialize,kLaunchInitialize,kStageBegin,kStageInitialize,
  kTaskBegin,kEmptyWait,kExpectTransaction,kBulkArrival,kBulkIssue,kBulkComplete,
  kActivationIssue,kActivationComplete,kFullReady,kFullWait,kQueryCurrentRow,kPatchAndPad,
  kOperandReadersDone,kEmptyArrivals,kFinalize,kEpilogue,kPartial,
  kAtomicPublish,kGroupAllPublished,kMerge,kTaskContinue,kTokenWork,
  kProducerGenerations,kConsumerGenerations,kProduceDone,kTailBegin,kRetireSlot,
  kPrefetchL2Issue,kTailIssueDone,kCtaJoin,kOperandRecordPublish,kCtaArrival,kAllCtasArrived,kV3Release,
  kStageEnvelopeEnd,kRingInvalidate,kKernelEnd,kProducerSetup,kProducerRendezvous
};
enum class Sm90DenseFlowResource {
  kSourceOrder,kCtaProducerIssue,kSmConsumerIssue,kDeviceMemoryHierarchy,
  kResidentGrid,kSharedArena,kAtomicPublication,kGridSynchronization
};
enum class Sm90DenseFlowService {
  kRingInitialization,kInactiveRows,kProducerIssue,kBulkCompletion,
  kActivationIssue,kActivationCompletion,kConsumerState,kConsumer,kQueryCurrentRow,kPatchAndPad,
  kFinalize,kEpilogue,kPartial,kAtomicPublication,kMerge,kTokenWork,
  kGenerationPublication,kBarrierWait,kBarrierPublication,kPrefetchIssue,kGridArrival,kGridRelease,kCtaPlacement,
  kSharedSmArbitration,kMemoryHierarchyArbitration,kProducerSetup,kProducerReturn
};
struct Sm90DenseFlowEvent {
  Sm90DenseFlowEventKind kind;
  int step=-1,stage=-1,target_stage=-1,cta=-1,task=-1,sequence=-1,bank=-1,slot=-1;
  std::uint64_t generation=0;
  std::int64_t bytes=0;
  int arrivals=0,expected_transactions=0,history_rows=0,valid_rows=0,compute_rows=0,current_rows=0;
  int request_index=-1,request_count=0;
  int local_tasks=-1;
  int condition=-1,owner=-1;
};
struct Sm90DenseFlowEdge {
  Sm90DenseEventId before,after;
  // Conditional edges apply only to the actual last acq_rel publisher.
  int condition=-1,owner=-1;
};
struct Sm90DenseMergeAlternative {
  int cta=-1,task=-1;
  Sm90DenseEventId publication=kSm90DenseNoEvent,merge=kSm90DenseNoEvent,
      continuation=kSm90DenseNoEvent;
};
struct Sm90DenseMergeCondition {
  int step=-1,stage=-1,group=-1;
  Sm90DenseEventId all_published=kSm90DenseNoEvent;
  std::vector<Sm90DenseMergeAlternative> alternatives;
};
struct Sm90DenseFlowRequirement {
  Sm90DenseFlowService service;
  Sm90DenseFlowResource resource;
  std::string key;
  std::vector<Sm90DenseEventId> events;
};
// Caller-provided references. The caller must authenticate the referenced
// measurements/artifacts; this module performs no file or device queries.
struct Sm90DenseFlowServiceEvidence {
  std::string requirement_key,measurement_id,record_sha256,measured_binary_sha256,measured_source_digest,cache_state;
  int processes=0;
  bool verified_measurement=false;
};
struct Sm90DenseFlowPending {
  std::string requirement,reason;
};
struct Sm90DenseFlowCoverage {
  std::vector<Sm90DenseFlowPending> pending;
  json::Value ToJson() const;
};
// Nodes are completion milestones and edges are necessary source ordering.
// They are not service-start constraints or a duration-bearing schedule.
struct Sm90DenseFlowEvents {
  int grid=0,num_sms=0,requested_residency=0,shared_bytes=0,steps=0;
  std::vector<Sm90DenseFlowEvent> events;
  std::vector<Sm90DenseFlowEdge> edges;
  std::vector<Sm90DenseMergeCondition> merge_conditions;
  std::vector<Sm90DenseFlowRequirement> requirements;
  std::vector<std::string> pending_stage_bindings;
  std::vector<std::array<std::array<std::uint64_t,8>,2>> final_generations;
  std::vector<Sm90DenseEventId> stage_envelope_ends;
  Sm90DenseEventId kernel_end=kSm90DenseNoEvent;

  // Only unconditional ordering is checked; choosing every merge alternative
  // simultaneously is not a legal graph interpretation.
  std::vector<Sm90DenseEventId> UnconditionalOrder() const;
  Sm90DenseFlowCoverage Coverage(Sm90DenseResidentEvidence const&,
      std::string const& cache_state,std::vector<Sm90DenseFlowServiceEvidence> const&) const;
  json::Value ToJson(bool include_events=false) const;
};
Sm90DenseFlowEvents BuildSm90DenseFlowEvents(Sm90DenseWorkGraph const&);
// Steps share the same launch: ring initialization occurs once, generation
// ordinals persist, and the final stage never preissues across the step edge.
Sm90DenseFlowEvents BuildSm90DenseFlowEvents(std::vector<Sm90DenseWorkGraph> const&);
}  // namespace tilemega::solver
