// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/BoundDependencyForm.h>
#include <tilemega/Analysis/ISLContext.h>
#include <map>
#include <set>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
CouplingRelation EncodeWindow(WaitWindow window, long producers, long consumers) {
  if (!window.narrowed) { window.div = 1; window.scale = 0; window.offset = 0; window.count = producers; }
  if (window.div <= 0 || window.count < 0)
    throw std::invalid_argument("invalid exact dependency window");
  auto begin = std::to_string(window.scale) + " * floord(c, " + std::to_string(window.div) + ") + " + std::to_string(window.offset);
  return CouplingRelation::FromIslText("{ [c] -> [p] : 0 <= c < " + std::to_string(consumers) +
      " and 0 <= p < " + std::to_string(producers) + " and (" + begin + ") <= p < (" + begin +
      ") + " + std::to_string(window.count) + " }");
}
std::optional<WaitWindow> FitClampedWindow(CouplingRelation const& linear,
    long producers, long consumers,DependencyTable const* table=nullptr) {
  // Solve the unclamped endpoints. Fitting only observed minima mistakes a
  // clipped halo (0,0,1,2,...) for a non-affine access.
  if (consumers <= 0 || producers <= 0 || consumers > 65536) return std::nullopt;
  std::map<long, long> first, last;
  if(table) {
    for(long c=0;c<consumers;++c)for(unsigned i=0;i<table->stride;++i) {
      auto interval=table->intervals[c*table->stride+i];if(!interval.count)continue;
      // A row with a genuine hole cannot be encoded by any one window.
      if(first.count(c))return std::nullopt;
      first[c]=interval.first;last[c]=std::uint64_t(interval.first)+interval.count;
    }
  }else {
    for (auto const& [c, p] : linear.LexMin().Points()) first[c.at(0)] = p.at(0);
    for (auto const& [c, p] : linear.LexMax().Points()) last[c.at(0)] = p.at(0) + 1;
  }
  if (first.empty()) return std::nullopt;
  long width = 0, period = 0;
  auto changes = [&](std::map<long, long> const& endpoints, bool lower) {
    long previous_change = -1;
    for (long c = 1; c < consumers; ++c) {
      auto current = endpoints.find(c), previous = endpoints.find(c - 1);
      if (current == endpoints.end() || previous == endpoints.end() || current->second == previous->second ||
          (lower ? current->second == 0 : current->second == producers)) continue;
      if (previous_change >= 0) period = std::gcd(period, c - previous_change);
      previous_change = c;
    }
  };
  changes(first, true); changes(last, false);
  for (auto const& [c, begin] : first) width = std::max(width, last.at(c) - begin);
  std::set<long> divisors{1, consumers};
  for (long divisor = 1; divisor * divisor <= period; ++divisor)
    if (period % divisor == 0) { divisors.insert(divisor); divisors.insert(period / divisor); }
  for (auto divisor : divisors) {
    auto constraint = [&](long c) {
      auto begin = std::to_string(c / divisor) + " * scale + offset";
      auto end = begin + " + count";
      if (auto at = first.find(c); at != first.end()) {
        auto text = " and (" + begin + (at->second == 0 ? " <= 0" : " = " + std::to_string(at->second)) + ")";
        auto past = last.at(c);
        return text + " and (" + end + (past == producers ? " >= " : " = ") + std::to_string(past) + ")";
      }
      return " and ((" + end + ") <= 0 or (" + begin + ") >= " + std::to_string(producers) + ")";
    };
    // A few endpoint constraints determine ordinary affine windows. Refine
    // from violating consumers instead of parsing thousands of redundant
    // inequalities. Acceptance still proves equality with the full relation.
    std::set<long> selected{0,consumers-1};
    for(long c=1;c<consumers;c*=2)selected.insert(c);
    for(unsigned iteration=0;iteration<32;++iteration) {
      std::string constraints = "count >= " + std::to_string(width) +
          " and abs_scale >= scale and abs_scale >= -scale and abs_offset >= offset and abs_offset >= -offset";
      for(auto c:selected)constraints+=constraint(c);
      auto feasible = CouplingRelation::FromIslText("{ [] -> [count,abs_scale,abs_offset,scale,offset] : " + constraints + " }");
      auto point = feasible.LexMin().Points();
      if(point.empty())break;
      auto const& p=point.front().second;
      WaitWindow window{true,divisor,p[3],p[4],p[0]};
      bool refined=false;
      for(long c=0;c<consumers;++c) {
        auto begin=std::max(0L,(c/divisor)*window.scale+window.offset);
        auto end=std::min(producers,(c/divisor)*window.scale+window.offset+window.count);
        auto at=first.find(c);
        bool equal=at==first.end()?begin>=end:begin==at->second && end==last.at(c);
        if(!equal) {refined=selected.insert(c).second;break;}
      }
      if(refined)continue;
      // The table has already proved both inclusions for every bounded row.
      // Equal contiguous endpoints therefore prove this window as well,
      // without counting all relation pairs or parsing their cardinality.
      if(table)return window;
      auto encoded=EncodeWindow(window,producers,consumers);
      if(Contains(encoded,linear) && Contains(linear,encoded))return window;
      break;
    }
  }
  return std::nullopt;
}
}
BoundDependencyForm BindExactTaskDependency(CouplingEdge const& edge,
    OperatorNode const& producer, OperatorNode const& consumer, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  BoundDependencyForm result;
  auto linear = LinearizeTaskCoupling(edge.C, producer, consumer, known);
  auto candidate = FitWaitWindowSymbolic(edge, producer, consumer, known, known);
  auto try_window = [&](WaitWindow window) {
    auto producers = producer.Count().Eval(known, {}), consumers = consumer.Count().Eval(known, {});
    auto encoded = EncodeWindow(window, producers, consumers);
    if (!Contains(encoded, linear) || !Contains(linear, encoded)) return false;
    if (!window.narrowed) window = {true, 1, 0, 0, producers};
    result.encoding = BoundDependencyForm::Encoding::kWindow;
    result.window = window; result.encoded_relation = std::move(encoded); return true;
  };
  if (candidate && try_window(*candidate)) return result;
  if (try_window(FitWaitWindow(edge, producer, consumer, known))) return result;
  if (auto clipped = FitClampedWindow(linear, producer.Count().Eval(known, {}),
                                     consumer.Count().Eval(known, {})); clipped && try_window(*clipped))
    return result;
  result.table = BuildDependencyTable(edge.C, producer, consumer, known);
  result.encoded_relation = result.table->encoded_relation;
  return result;
}
BoundDependencyForm BindExactTaskDependencyLinear(CouplingRelation const& relation,
    std::uint32_t producers, std::uint32_t consumers) {
  if (!producers || !consumers || relation.DomainDimNames().size() != 1 ||
      relation.RangeDimNames().size() != 1)
    throw std::invalid_argument("invalid bound linear dependency dimensions");
  BoundDependencyForm result;
  auto table=BuildDependencyTableLinear(relation,producers,consumers);
  if(auto window=FitClampedWindow(relation,producers,consumers,&table)) {
    result.encoding=BoundDependencyForm::Encoding::kWindow;
    result.window=*window;result.encoded_relation=EncodeWindow(*window,producers,consumers);
    return result;
  }
  result.encoded_relation=table.encoded_relation;result.table=std::move(table);
  return result;
}
}  // namespace tilemega::analysis
