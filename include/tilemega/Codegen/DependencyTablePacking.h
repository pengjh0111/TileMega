// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DependencyTable.h>
#include <stdexcept>
#include <vector>

namespace tilemega::codegen {
// Return zero unless a complete sequence of interval rows repeats exactly.
// Prefix matching compares interval contents, never hashes or sampled rows.
inline unsigned DependencyTableRowPeriod(analysis::DependencyTable const& table) {
  if (std::uint64_t(table.consumers)*table.stride != table.intervals.size())
    throw std::invalid_argument("dependency packing requires complete interval rows");
  if (!table.consumers || !table.stride) return 0;
  auto equal = [&](unsigned a, unsigned b) {
    for (unsigned i=0;i<table.stride;++i) {
      auto x=table.intervals[std::size_t(a)*table.stride+i];
      auto y=table.intervals[std::size_t(b)*table.stride+i];
      if (x.first!=y.first || x.count!=y.count) return false;
    }
    return true;
  };
  std::vector<unsigned> prefix(table.consumers);
  for (unsigned row=1;row<table.consumers;++row) {
    unsigned match=prefix[row-1];
    while (match && !equal(row,match)) match=prefix[match-1];
    if (equal(row,match)) ++match;
    prefix[row]=match;
  }
  unsigned period=table.consumers-prefix.back();
  return period<table.consumers && table.consumers%period==0 ? period : 0;
}
} // namespace tilemega::codegen
