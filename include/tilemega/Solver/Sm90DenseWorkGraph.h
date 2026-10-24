// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Target/TargetSpec.h>
#include <map>
#include <vector>

namespace tilemega::solver {
struct Sm90DenseWorkload;

enum class Sm90DenseTailOrder { kNone, kBeforeProduce, kAfterProduce, kRetireOperands };

// Logical source operations, not independent bandwidth allowances or prices.
// Slot reuse waits for the last operand reader, before finalize/epilogue.
struct Sm90DenseBlockWork {
  int sequence=0,slot=0;
  std::uint64_t generation=0;
  int previous_reader=-1;
  int history_rows=0,valid_rows=0,compute_rows=0,current_rows=0;
  std::int64_t weight_bytes=0,activation_bytes=0,kv_bytes=0;
  std::int64_t prefix_bytes=0;
  int prefix_requests=0,remainder_requests=0;
  int expected_transactions=0,explicit_bulk_arrivals=0;
  int activation_arrivals=32,empty_arrivals=128;
};
struct Sm90DenseTaskWork {
  int task=0,group=-1;
  bool contains_current_row=false;
  std::vector<Sm90DenseBlockWork> blocks;
};
struct Sm90DenseCtaWork {
  int cta=0;
  std::vector<Sm90DenseTaskWork> tasks;
  std::int64_t l2_hint_bytes=0;
  int l2_hint_requests=0;
};
struct Sm90DenseStageWork {
  std::string kind;
  int layer=-1,tasks=0,ring_bank=0,ring_slots=0;
  int prefix_predecessor=-1;
  int merge_chunks=1;
  Sm90DenseTailOrder incoming_order=Sm90DenseTailOrder::kNone;
  backend::Sm90DenseLayout layout;
  backend::Sm90DensePayload prefix_payload;
  int workspace_offset=0;
  std::int64_t inactive_zero_bytes_per_cta=0;
  std::vector<Sm90DenseCtaWork> ctas;
  std::map<int,int> LocalTaskHistogram() const;
};
struct Sm90DenseResidentEvidence {
  int grid=0,dynamic_shared_bytes=0,resident_ctas_per_sm=0;
  std::string binary_sha256,source_digest;
};
struct Sm90DenseWorkGraph {
  int grid=0,num_sms=0,requested_residency=0;
  int shared_bytes=0,shared_budget=0,past=0;
  std::int64_t launch_inactive_zero_bytes_per_cta=0;
  bool activation_reuse=false;
  std::vector<Sm90DenseStageWork> stages;

  // Shared-only legality never proves that a persistent barrier grid fits.
  // A missing final-binary occupancy witness remains explicitly pending.
  void RequireResident(Sm90DenseResidentEvidence const&) const;
  json::Value ToJson(bool include_blocks=false) const;
};

Sm90DenseWorkGraph BuildSm90DenseWorkGraph(Sm90DenseWorkload const&,
    Sm90DenseCoordinates const&,TargetSpec const&,int past=575,int capacity=1088);
}  // namespace tilemega::solver
