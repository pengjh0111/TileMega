// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DependencyForm.h>

#include <tilemega/Analysis/CouplingDerivation.h>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

namespace tilemega::analysis {
namespace {

/// Row-major over the node's tiled axes: the last coordinate varies fastest,
/// which is the order `blockIdx.x` walks a `kTilePerBlock` task space in.
long LinearId(OperatorNode const& node, ParamBinding const& known,
              std::vector<long> const& point) {
  long id = 0;
  std::size_t at = 0;
  for (std::size_t axis = 0; axis < node.tile.size(); ++axis) {
    if (!node.IsTiled(axis)) continue;
    long const extent = node.CoordinateExtent(axis).Eval(known, {});
    id = id * extent + (at < point.size() ? point[at] : 0);
    ++at;
  }
  return id;
}

WaitWindow Relaxed() { return WaitWindow{}; }

std::string Join(std::vector<std::string> const& names) {
  std::ostringstream out;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i) out << ',';
    out << names[i];
  }
  return out.str();
}

/// Map a task-space coordinate tuple to the exact row-major id used by its
/// TaskBody.  A symbolic leading extent is harmless: only *trailing* extents
/// become coefficients.  If one of those is symbolic, multiplication of a
/// parameter by a coordinate would leave Presburger arithmetic and the caller
/// must use the concrete fallback for this edge.
std::optional<CouplingRelation> LinearIdMap(
    OperatorNode const& node, std::vector<std::string> const& dimensions,
    ParamBinding const& known, std::string const& output) {
  std::vector<ClosedForm> extents;
  for (std::size_t axis = 0; axis < node.output.axes.size(); ++axis)
    if (node.IsTiled(axis)) extents.push_back(node.CoordinateExtent(axis));
  if (extents.size() != dimensions.size()) return std::nullopt;

  std::vector<long> stride(extents.size(), 1);
  long suffix = 1;
  for (std::size_t i = extents.size(); i-- > 0;) {
    stride[i] = suffix;
    if (i == 0) break;
    ClosedForm extent = extents[i].Substitute(known);
    if (!extent.IsConstant()) return std::nullopt;
    long const value = extent.Eval({}, {});
    if (value <= 0 || suffix > std::numeric_limits<long>::max() / value)
      return std::nullopt;
    suffix *= value;
  }

  std::ostringstream expression;
  if (dimensions.empty()) {
    expression << '0';
  } else {
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
      if (i) expression << " + ";
      expression << stride[i] << '*' << dimensions[i];
    }
  }
  std::ostringstream text;
  text << "{ [" << Join(dimensions) << "] -> [" << output << "] : "
       << output << " = " << expression.str() << " }";
  try {
    return CouplingRelation::FromIslText(text.str());
  } catch (std::exception const&) {
    return std::nullopt;
  }
}

}  // namespace

std::string WaitWindow::ToString() const {
  if (!narrowed) return "all";
  if (IsIdentity()) return "identity";
  std::ostringstream text;
  text << "window(" << div << "," << scale << "," << offset << "," << count
       << ")";
  return text.str();
}

bool operator==(WaitWindow const& a, WaitWindow const& b) {
  if (a.narrowed != b.narrowed) return false;
  if (!a.narrowed) return true;
  return a.div == b.div && a.scale == b.scale && a.offset == b.offset &&
         a.count == b.count;
}

WaitWindow ParseWaitWindow(std::string const& text) {
  if (text == "identity") {
    WaitWindow window;
    window.narrowed = true;
    window.div = 1;
    window.scale = 1;
    window.offset = 0;
    window.count = 1;
    return window;
  }
  if (text.compare(0, 7, "window(") != 0) return Relaxed();
  WaitWindow window;
  char comma = 0;
  std::istringstream in(text.substr(7));
  if (!(in >> window.div >> comma >> window.scale >> comma >> window.offset >>
        comma >> window.count))
    return Relaxed();
  window.narrowed = true;
  return window;
}

WaitWindow FitWaitWindow(CouplingEdge const& edge, OperatorNode const& producer,
                         OperatorNode const& consumer,
                         ParamBinding const& known) {
  if (edge.C.empty()) return Relaxed();
  // C's range is deliberately left unbounded by the producer's task space
  // (CouplingDerivation.cpp documents the barvinok regression that folding it
  // in causes on the wait side), so at a small instantiation it still carries
  // whole-tile producer coordinates -- rmsnorm -> GEMM offers p0 up to Tm-1
  // with only S tasks in existence. A window built on those would block on
  // tasks no launch creates, so the clamp is applied here.
  CouplingRelation clamped = edge.C;
  try {
    clamped =
        edge.C.IntersectRange(ProducerTaskSpaceText(edge.C, producer, known));
    if (clamped.empty()) return Relaxed();
    // Enumerating C costs one point at a time, and the edges whose point count
    // is large here are exactly the ones an element-chunk endpoint already
    // disqualifies. Exceeding the budget is reported as the relaxation, never
    // as a guess.
    if (clamped.Card().Eval(known) > (1L << 20)) return Relaxed();
  } catch (std::exception const&) {
    return Relaxed();
  }

  long producers = 0, consumers = 0;
  try {
    producers = producer.Count().Eval(known, {});
    consumers = consumer.Count().Eval(known, {});
  } catch (std::exception const&) {
    return Relaxed();
  }
  if (producers <= 0 || consumers <= 0) return Relaxed();

  std::map<long, std::vector<long>> wait;
  try {
    for (auto const& [to, from] : clamped.Points())
      wait[LinearId(consumer, known, to)].push_back(
          LinearId(producer, known, from));
  } catch (std::exception const&) {
    return Relaxed();
  }
  if (wait.empty()) return Relaxed();

  long observed_count = 0;
  for (auto& [task, set] : wait) {
    // A linear id outside its own task space means the row-major walk here and
    // the node's coordinate order have diverged; no window read off it would
    // be sound.
    if (task < 0 || task >= consumers) return Relaxed();
    std::sort(set.begin(), set.end());
    set.erase(std::unique(set.begin(), set.end()), set.end());
    if (set.front() < 0 || set.back() >= producers) return Relaxed();
    // A window is an interval; a wait set with a hole is not one of these
    // shapes and is not going to be rounded up to one.
    for (std::size_t k = 1; k < set.size(); ++k)
      if (set[k] != set[k - 1] + 1) return Relaxed();
    observed_count = std::max(observed_count, static_cast<long>(set.size()));
  }

  auto first_of = [&](long task) -> long {
    auto it = wait.find(task);
    return it == wait.end() ? -1 : it->second.front();
  };
  auto verify = [&](WaitWindow const& window) {
    for (long task = 0; task < consumers; ++task) {
      long begin = (task / window.div) * window.scale + window.offset;
      long end = begin + window.count;
      begin = std::max(begin, 0L);
      end = std::min(end, producers);
      auto it = wait.find(task);
      if (it == wait.end()) {
        if (begin < end) return false;
        continue;
      }
      if (begin >= end) return false;
      if (it->second.front() != begin ||
          static_cast<long>(it->second.size()) != end - begin)
        return false;
    }
    return true;
  };

  for (long div = 1; div <= consumers; ++div) {
    if (consumers % div != 0) continue;
    // `first` must be constant within a block of `div` consecutive consumers
    // and affine in the block index; two blocks fix the line, the rest verify.
    long base_block = -1, base_first = 0, scale = 0;
    bool consistent = true;
    for (long task = 0; task < consumers && consistent; ++task) {
      long const first = first_of(task);
      if (first < 0) continue;
      long const block = task / div;
      if (base_block < 0) {
        base_block = block;
        base_first = first;
        continue;
      }
      if (block == base_block) {
        consistent = first == base_first;
        continue;
      }
      long const delta = first - base_first, span = block - base_block;
      if (delta % span != 0) { consistent = false; break; }
      long const candidate = delta / span;
      if (scale == 0) scale = candidate;
      else if (candidate != scale) consistent = false;
    }
    if (!consistent || base_block < 0) continue;
    WaitWindow window;
    window.narrowed = true;
    window.div = div;
    window.scale = scale;
    window.offset = base_first - base_block * scale;
    window.count = observed_count;
    if (!verify(window)) continue;
    // A window covering the producer's whole task space is the relaxation
    // wearing a fitted shape; reporting it as narrowed would claim a saving
    // that is not there, and its `count` would track S rather than the map.
    if (window.scale == 0 && window.offset <= 0 &&
        window.offset + window.count >= producers)
      return Relaxed();
    return window;
  }
  return Relaxed();
}

std::optional<WaitWindow> FitWaitWindowSymbolic(
    CouplingEdge const& edge, OperatorNode const& producer,
    OperatorNode const& consumer, ParamBinding const& known,
    ParamBinding const& witness) {
  try {
    CouplingRelation const clamped = edge.C.IntersectRange(
        ProducerTaskSpaceText(edge.C, producer, known));
    auto consumer_id =
        LinearIdMap(consumer, clamped.DomainDimNames(), known, "tc");
    auto producer_id =
        LinearIdMap(producer, clamped.RangeDimNames(), known, "tp");
    if (!consumer_id || !producer_id) return std::nullopt;
    CouplingRelation const linear =
        consumer_id->Reverse().ApplyRange(clamped).ApplyRange(*producer_id);

    // Discover only the two interval endpoints at the witness.  Enumerating
    // C itself is proportional to fan-in (hundreds of thousands of points on
    // the real-width model); lexmin/lexmax keep at most two rows per consumer.
    CouplingRelation const concrete = linear.BindParams(witness);
    std::map<long, long> first, last;
    for (auto const& [from, to] : concrete.LexMin().Points()) {
      if (from.size() != 1 || to.size() != 1) return std::nullopt;
      first[from[0]] = to[0];
    }
    for (auto const& [from, to] : concrete.LexMax().Points()) {
      if (from.size() != 1 || to.size() != 1) return std::nullopt;
      last[from[0]] = to[0];
    }
    long const consumer_count = consumer.Count().Eval(witness, {});
    if (consumer_count <= 0 || first.empty() || first.size() != last.size())
      return Relaxed();
    long observed_count = 0;
    for (auto const& [task, begin] : first) {
      auto found = last.find(task);
      if (found == last.end() || found->second < begin) return Relaxed();
      observed_count = std::max(observed_count, found->second - begin + 1);
    }
    auto first_of = [&](long task) -> long {
      auto found = first.find(task);
      return found == first.end() ? -1 : found->second;
    };
    WaitWindow candidate;
    for (long div = 1; div <= consumer_count && !candidate.narrowed; ++div) {
      if (consumer_count % div != 0) continue;
      long base_block = -1, base_first = 0, scale = 0;
      bool consistent = true;
      for (long task = 0; task < consumer_count && consistent; ++task) {
        long const at = first_of(task);
        if (at < 0) continue;
        long const block = task / div;
        if (base_block < 0) {
          base_block = block;
          base_first = at;
        } else if (block == base_block) {
          consistent = at == base_first;
        } else {
          long const delta = at - base_first, span = block - base_block;
          if (delta % span != 0) {
            consistent = false;
          } else {
            long const value = delta / span;
            if (scale == 0) scale = value;
            else if (value != scale) consistent = false;
          }
        }
      }
      if (!consistent || base_block < 0) continue;
      candidate.narrowed = true;
      candidate.div = div;
      candidate.scale = scale;
      candidate.offset = base_first - base_block * scale;
      candidate.count = observed_count;
    }
    if (!candidate.narrowed || candidate.div <= 0 || candidate.count <= 0)
      return Relaxed();

    ClosedForm const consumers = consumer.Count().Substitute(known);
    ClosedForm const producers = producer.Count().Substitute(known);
    std::set<std::string> parameter_set;
    for (auto const& name : consumers.FreeSymbols()) parameter_set.insert(name);
    for (auto const& name : producers.FreeSymbols()) parameter_set.insert(name);
    std::vector<std::string> parameters(parameter_set.begin(),
                                        parameter_set.end());
    std::ostringstream begin;
    begin << candidate.scale << "*floord(tc," << candidate.div << ")";
    if (candidate.offset > 0) begin << '+' << candidate.offset;
    if (candidate.offset < 0) begin << candidate.offset;
    std::ostringstream expected;
    if (!parameters.empty()) expected << '[' << Join(parameters) << "] -> ";
    expected << "{ [tc] -> [tp] : 0 <= tc and tc < "
             << consumers.ToIslText() << " and 0 <= tp and tp < "
             << producers.ToIslText() << " and " << begin.str()
             << " <= tp and tp < " << begin.str() << '+' << candidate.count
             << " }";
    CouplingRelation const window =
        CouplingRelation::FromIslText(expected.str());
    if (linear.IsSubset(window) && window.IsSubset(linear)) {
      long const producer_count = producer.Count().Eval(witness, {});
      if (candidate.scale == 0 && candidate.offset <= 0 &&
          candidate.offset + candidate.count >= producer_count)
        return Relaxed();
      return candidate;
    }

    CouplingRelation const concrete_window = window.BindParams(witness);
    if (!concrete.IsSubset(concrete_window) ||
        !concrete_window.IsSubset(concrete))
      return Relaxed();

    // A dynamic full-fan-in edge looks like a fixed-size interval at the
    // witness, but that count grows with the producer task space and thus
    // cannot equal the constant candidate above.  Prove the kAll form only
    // for this case; running the product query for every narrow edge costs
    // more than the window proof itself on production dimensions.
    long const producer_count = producer.Count().Eval(witness, {});
    if (candidate.scale == 0 && candidate.offset <= 0 &&
        candidate.offset + candidate.count >= producer_count &&
        clamped.CouplesEveryDomainPointTo(
            ProducerTaskSpaceText(edge.C, producer, known)))
      return Relaxed();
  } catch (std::exception const&) {
    // A rejected symbolic construction is not evidence that the window is
    // inexact.  The caller runs the concrete fallback for this edge only.
  }
  return std::nullopt;
}

}  // namespace tilemega::analysis
