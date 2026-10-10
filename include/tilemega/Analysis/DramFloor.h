// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/TaskWork.h>
#include <map>
#include <set>
#include <optional>

namespace tilemega::analysis {
struct DramTensorFootprint {
  CouplingRelation reads, writes, no_producer, external_writes;
  QuasiPolynomial read_bytes, write_bytes;
  int element_bytes=0;
  bool state=false, output=false;
  // Statistical unique-weight cardinalities carry no claimed physical image.
  std::optional<QuasiPolynomial> expected_read_elements;
  std::string expectation_source;
};
struct DramFloorOptions {
  double dram_gbps=0, tc_gflops=0;
  std::map<std::string,int> element_bytes;
  std::set<std::string> outputs;
  // Runtime gathers cannot supply unique physical addresses from theta alone.
  // Callers may bind their actual read image; otherwise the audit rejects them.
  std::map<std::string,CouplingRelation> indirect_read_images;
  // Scatter images carry actual element coordinates, not I2 envelopes.
  std::map<std::string,CouplingRelation> indirect_write_images;
  struct ExpectedRead {
    QuasiPolynomial elements;
    std::string source;
  };
  // Bound-theta expectation for an entirely read-only, indirect-only tensor.
  // It is not a runtime-address witness and cannot justify synchronization.
  std::map<std::string,ExpectedRead> expected_indirect_reads;
};
struct DramFloor {
  std::map<std::string,DramTensorFootprint> tensors;
  QuasiPolynomial no_producer_bytes, output_state_bytes, matmul_flops;
  QuasiPolynomial dram_ns, compute_ns;
  double dram_rate=0, compute_rate=0;
  struct Value { double read_bytes,write_bytes,flops,dram_ns,compute_ns,floor_ns; };
  Value Evaluate(ParamBinding const& theta) const;
};
DramFloor DeriveDramFloor(SemanticGraph const& graph,DramFloorOptions const& options,
                          ParamBinding const& fixed={});
} // namespace tilemega::analysis
