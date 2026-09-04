// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DependencyForm.h>

#include <tilemega/Analysis/CouplingDerivation.h>

#include <algorithm>
#include <map>
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

}  // namespace tilemega::analysis
