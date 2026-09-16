// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/ParametricPlacement.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <map>

namespace tilemega::solver {
struct AffineCountPiece {
  int begin=0,end=0;
  std::vector<std::string> counts; // exact affine expressions in S, proved against task domains
  std::vector<int> band_width;
};
inline std::string SumExpressions(std::vector<std::string> const& xs) {
  std::string result="0";for (auto const& x:xs) result+="+("+x+")";return result;
}
/// Grid is a finite target-domain parameter: modulo a variable divisor is
/// not Presburger. Each calibrated target grid has its own constant-divisor
/// branch in the SAME map, rather than shipping a materialized worker table.
inline ParametricPlacement BuildParametricTemplate(
    analysis::CouplingRelation tasks,analysis::CouplingRelation dependencies,
    std::vector<std::uint32_t> const& stage_order,
    std::vector<AffineCountPiece> const& pieces,std::vector<int> const& grids,
    int resident_limit,std::string const& family,std::vector<int> const& levels={}) {
  if (pieces.empty() || grids.empty()) throw std::invalid_argument("empty symbolic template domain");
  int n=int(pieces.front().counts.size());
  if (stage_order.size()!=std::size_t(n)) throw std::invalid_argument("incomplete symbolic stage order");
  std::vector<int> positions(n,-1);
  for (int i=0;i<n;++i) {
    auto s=stage_order[i];if (s>=unsigned(n) || positions[s]>=0) throw std::invalid_argument("invalid symbolic stage order");positions[s]=i;
  }
  if (family=="wavefront" && levels.size()!=std::size_t(n)) throw std::invalid_argument("wavefront needs proved task levels");
  std::vector<std::string> mappings,ranks,limits,domains,theta_filters;
  for (int grid:grids) {
    if (grid<=0) throw std::invalid_argument("invalid symbolic grid");
    for (auto const& piece:pieces) {
      std::string condition=std::to_string(piece.begin)+"<=S<="+std::to_string(piece.end)+" and G="+std::to_string(grid);
      theta_filters.push_back("[s,t] -> [s,t] : "+condition);
      limits.push_back("[] -> ["+std::to_string(grid)+","+std::to_string(resident_limit)+"] : "+condition);
      for (int s=0;s<n;++s) {
        auto count=piece.counts.at(s);std::string owner,slot,rank;
        auto ceil_worker=[&](std::string const& x){return "floord(("+x+")+"+std::to_string(grid-1)+"-w,"+std::to_string(grid)+")";};
        std::vector<std::string> prior,slots;
        if (family=="wavefront") {
          std::map<int,std::vector<std::string>> groups;
          for (int p=0;p<n;++p) {
            if (levels[p]<levels[s]) groups[levels[p]].push_back(piece.counts[p]);
            else if (levels[p]==levels[s] && p<s) prior.push_back(piece.counts[p]);
          }
          for (auto const& [level,counts]:groups) slots.push_back(ceil_worker(SumExpressions(counts)));
          auto within="t+("+SumExpressions(prior)+")";
          owner="("+within+")%"+std::to_string(grid);
          slot=SumExpressions(slots)+"+floord("+within+","+std::to_string(grid)+")";
          rank=std::to_string(levels[s])+","+within;
        } else {
          for (int i=0;i<positions[s];++i) prior.push_back(piece.counts[stage_order[i]]);
          rank=std::to_string(positions[s])+",t";
          if (family=="rotate") {
            auto linear="t+("+SumExpressions(prior)+")";
            owner="("+linear+")%"+std::to_string(grid);slot="floord("+linear+","+std::to_string(grid)+")";
          } else if (family=="legacy_grid_stride") {
            owner="t%"+std::to_string(grid);
            for (auto const& c:prior) slots.push_back(ceil_worker(c));
            slot=SumExpressions(slots)+"+floord(t,"+std::to_string(grid)+")";
          } else if (family=="band") {
            if (piece.band_width.size()!=std::size_t(n)) throw std::invalid_argument("band widths must be constant on the proved piece");
            int width=piece.band_width[s];
            auto actual_width=analysis::CouplingRelation::FromIslText("[S,G] -> { [] -> [w] : "+
                condition+" and w=max(1,floord(("+count+")+"+std::to_string(grid-1)+","+std::to_string(grid)+")) }");
            if (!actual_width.IsSubset(actual_width.IntersectRange("{ [w] : w="+std::to_string(width)+" }")))
              throw std::invalid_argument("band width is not constant on count piece");
            owner="floord(t,"+std::to_string(width)+")";
            for (int i=0;i<positions[s];++i) {
              int p=stage_order[i],pw=piece.band_width[p];
              slots.push_back("min("+std::to_string(pw)+",max(0,("+piece.counts[p]+")-w*"+std::to_string(pw)+"))");
            }
            slot=SumExpressions(slots)+"+t%"+std::to_string(width);
          } else throw std::invalid_argument("placement has no closed-form template family: "+family);
        }
        auto domain=condition+" and 0<=t<("+count+")";
        domains.push_back("[] -> ["+std::to_string(s)+",t] : "+domain);
        mappings.push_back("["+std::to_string(s)+",t] -> [w,q] : "+domain+" and w="+owner+" and q="+slot);
        ranks.push_back("["+std::to_string(s)+",t] -> ["+rank+"] : "+domain);
      }
    }
  }
  auto relation=[](std::vector<std::string> const& parts) {
    std::string text="[S,G] -> { ";for (auto const& p:parts) text+=p+"; ";text+="}";
    return analysis::CouplingRelation::FromIslText(text);
  };
  auto domain=relation(domains);
  // Count-piece construction is only admitted after symbolic equality, so
  // deriving breakpoints is not an unproved fit to endpoint observations.
  auto restriction=domain.ImageIdentity();
  auto restricted_tasks=tasks.ApplyRange(relation(theta_filters));
  if (!domain.IsSubset(restricted_tasks) || !restricted_tasks.IsSubset(domain))
    throw std::invalid_argument("symbolic count pieces disagree with projected task domain");
  dependencies=restriction.ApplyRange(dependencies).ApplyRange(restriction);
  return {domain,dependencies,relation(mappings),relation(ranks),relation(limits),family};
}
} // namespace tilemega::solver
