// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DependencyTable.h>
#include <cstdint>

namespace tilemega::analysis {
struct CountedDependencyForm {
  std::string binding_source;
  // Counts logical contributions, not the number of predecessor tasks.
  // A producer spanning several rows may publish several arrivals.
  // Every target unit must have one writer. Separate column writers require
  // a column partition coordinate in the unit, or aligned producer tiles.
  std::vector<std::uint32_t> expected;
  CouplingRelation target_units;
};

CountedDependencyForm BindCountedTaskDependency(OperatorNode const& consumer,
    CouplingRelation const& reads, std::vector<unsigned> const& unit_axes,
    std::string binding_source, ParamBinding const& known = {});

// Requires a permutation of logical units from the named runtime binding.
// Static axes must match the consumer partition; dispatch establishes the
// permutation, while this proof excludes duplicate column contributions.
CountedDependencyForm BindAlignedCountedScatterDependency(
    OperatorNode const& producer, OperatorNode const& consumer,
    std::string const& tensor, std::vector<unsigned> const& unit_axes,
    std::string const& binding_source, ParamBinding const& known = {});
} // namespace tilemega::analysis
