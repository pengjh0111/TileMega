// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <cmath>
#include <string>
#include <vector>

namespace tilemega::solver {
struct RelationBounds { double work_ns=0, critical_path_ns=0; };

/// Work and the semantic critical path read straight off the projected
/// dependency relation `[consumer stage,task] -> [producer stage,task]`.
///
/// A dense piece of that relation is a complete bipartite block, and the
/// longest-path relaxation such a block induces is one maximum over its
/// producers applied to each of its consumers. Keeping the block as its two
/// intervals therefore costs the side lengths instead of their product, which
/// is the whole of §6 C1-b: the bound never needs the edges themselves, only
/// this aggregate. Pieces with coupled coordinates or modular holes still
/// arrive edge by edge, so the walk below is the same DAG `PreparePlanBounds`
/// walks on the materialized graph and returns the same two numbers.
///
/// Overlapping pieces may repeat an edge. A repeat raises the consumer's
/// indegree and lowers it again, so it changes neither termination nor the
/// maxima -- the materialized path deduplicates instead, and the two agree.
inline bool PrepareRelationBounds(std::vector<int> const& stage_offsets,
    std::vector<int> const& counts, std::vector<double> const& task_ns,
    std::string const& dependencies, RelationBounds* out, std::string* error) {
  auto fail=[&](char const* message){if(error)*error=message;return false;};
  if (stage_offsets.size()!=counts.size()+1 || stage_offsets.empty())
    return fail("relation bounds need one offset per stage boundary");
  int const nodes=stage_offsets.back();
  if (nodes<0 || task_ns.size()!=std::size_t(nodes))
    return fail("relation bounds graph/cost size mismatch");
  for (double price:task_ns)
    if (!std::isfinite(price) || price<0)
      return fail("relation bounds require finite nonnegative node costs");

  struct Block { int consumer_lo,consumer_hi,remaining; double producer_end; };
  std::vector<Block> blocks;
  std::vector<std::vector<int>> producer_blocks(nodes),point_successors(nodes);
  std::vector<int> indegree(nodes,0);
  std::string domain_error;
  auto node=[&](long stage,long task)->int {
    if (stage<0 || stage>=long(counts.size()) || task<0 || task>=counts[stage]) {
      domain_error="projected relation outside task domain";
      return -1;
    }
    return stage_offsets[stage]+int(task);
  };
  auto region=[&](long const* low,long const* high) {
    for (long cs=low[0];cs<=high[0];++cs)
      for (long ps=low[2];ps<=high[2];++ps) {
        int const consumer_lo=node(cs,low[1]),consumer_hi=node(cs,high[1]);
        int const producer_lo=node(ps,low[3]),producer_hi=node(ps,high[3]);
        if (consumer_lo<0 || consumer_hi<0 || producer_lo<0 || producer_hi<0) return;
        int const index=int(blocks.size());
        blocks.push_back({consumer_lo,consumer_hi,producer_hi-producer_lo+1,0});
        for (int p=producer_lo;p<=producer_hi;++p) producer_blocks[p].push_back(index);
        for (int c=consumer_lo;c<=consumer_hi;++c) ++indegree[c];
      }
  };
  auto point=[&](long const* edge) {
    int const consumer=node(edge[0],edge[1]),producer=node(edge[2],edge[3]);
    if (consumer<0 || producer<0) return;
    point_successors[producer].push_back(consumer);++indegree[consumer];
  };
  // A block is consumed at its side lengths here, so the width the point
  // expansion needs does not apply. Measured on this machine an ISL slice query
  // costs about 44 us against about 1.6 us per enumerated point, so a slice has
  // to be about 28 wide to repay itself; 16 is the lowest threshold that does
  // not regress the short cells and is the fastest on the long ones.
  analysis::VisitFiniteRegions(analysis::SharedIslContext(),dependencies,4,region,point,16);
  if (!domain_error.empty()) return fail(domain_error.c_str());

  std::vector<double> end(nodes,0);
  std::vector<int> ready;
  double work=0,critical_path=0;
  for (int n=0;n<nodes;++n) {work+=task_ns[n];if(!indegree[n])ready.push_back(n);}
  std::size_t visited=0;
  auto relax=[&](int consumer,double at) {
    end[consumer]=std::max(end[consumer],at);
    if (--indegree[consumer]==0) ready.push_back(consumer);
  };
  while (visited<ready.size()) {
    int const n=ready[visited++];
    end[n]+=task_ns[n];
    critical_path=std::max(critical_path,end[n]);
    for (int consumer:point_successors[n]) relax(consumer,end[n]);
    for (int index:producer_blocks[n]) {
      Block& block=blocks[index];
      block.producer_end=std::max(block.producer_end,end[n]);
      if (--block.remaining) continue;
      for (int c=block.consumer_lo;c<=block.consumer_hi;++c) relax(c,block.producer_end);
    }
  }
  if (visited!=std::size_t(nodes)) return fail("relation bounds task graph is cyclic");
  *out={work,critical_path};
  return true;
}
}  // namespace tilemega::solver
