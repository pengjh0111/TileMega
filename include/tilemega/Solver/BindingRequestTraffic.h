// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <set>
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/TaskElementRelation.h>

namespace tilemega::solver {
struct BindingRequestTraffic {
  analysis::QuasiPolynomial read_bytes, write_bytes;
  analysis::QuasiPolynomial no_producer_read_bytes, external_write_bytes;
  std::uint64_t produced_live_bytes=0;
  std::set<std::string> produced_tensors;
};
// Counts potential accesses over capacity. Routing occupancy is a subsequent
// expectation; I2 images establish provenance, never issued byte counts.
BindingRequestTraffic DeriveBindingRequestTraffic(analysis::OperatorNode const& task,
    analysis::DramFloor const& floor,analysis::ParamBinding const& known={});
} // namespace tilemega::solver
