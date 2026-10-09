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
    long producers, long consumers) {
  // Solve the unclamped endpoints. Fitting only observed minima mistakes a
  // clipped halo (0,0,1,2,...) for a non-affine access.
  if (consumers <= 0 || producers <= 0 || consumers > 65536) return std::nullopt;
  std::map<long, long> first, last;
  for (auto const& [c, p] : linear.LexMin().Points()) first[c.at(0)] = p.at(0);
  for (auto const& [c, p] : linear.LexMax().Points()) last[c.at(0)] = p.at(0) + 1;
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
    std::string constraints = "count >= " + std::to_string(width) +
        " and abs_scale >= scale and abs_scale >= -scale and abs_offset >= offset and abs_offset >= -offset";
    for (long c = 0; c < consumers; ++c) {
      auto begin = std::to_string(c / divisor) + " * scale + offset";
      auto end = begin + " + count";
      if (auto at = first.find(c); at != first.end()) {
        constraints += " and (" + begin + (at->second == 0 ? " <= 0" : " = " + std::to_string(at->second)) + ")";
        auto past = last.at(c);
        constraints += " and (" + end + (past == producers ? " >= " : " = ") + std::to_string(past) + ")";
      } else constraints += " and ((" + end + ") <= 0 or (" + begin + ") >= " + std::to_string(producers) + ")";
    }
    auto feasible = CouplingRelation::FromIslText("{ [] -> [count,abs_scale,abs_offset,scale,offset] : " + constraints + " }");
    auto point = feasible.LexMin().Points();
    if (point.empty()) continue;
    auto const& p = point.front().second;
    WaitWindow window{true, divisor, p[3], p[4], p[0]};
    auto encoded = EncodeWindow(window, producers, consumers);
    if (Contains(encoded, linear) && Contains(linear, encoded)) return window;
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
  // A linear runtime ID stays one-dimensional when its extent is one.
  // OperatorNode elides whole axes, so that synthetic task representation
  // cannot be used to re-linearize singleton producer/consumer spaces.
  if(producers==1 || consumers==1) {
    BoundDependencyForm result;
    auto table=BuildDependencyTableLinear(relation,producers,consumers);
    if(auto window=FitClampedWindow(relation,producers,consumers)) {
      auto encoded=EncodeWindow(*window,producers,consumers);
      if(Contains(encoded,relation) && Contains(relation,encoded)) {
        result.encoding=BoundDependencyForm::Encoding::kWindow;
        result.window=*window;result.encoded_relation=std::move(encoded);
        return result;
      }
    }
    result.encoded_relation=table.encoded_relation;result.table=std::move(table);
    return result;
  }
  OperatorNode producer, consumer;
  producer.output = {"producer", {{relation.RangeDimNames()[0], ClosedForm::Constant(producers)}}};
  consumer.output = {"consumer", {{relation.DomainDimNames()[0], ClosedForm::Constant(consumers)}}};
  producer.tile = consumer.tile = {ClosedForm::Constant(1)};
  CouplingEdge edge; edge.C = relation;
  return BindExactTaskDependency(edge, producer, consumer, {});
}
}  // namespace tilemega::analysis
