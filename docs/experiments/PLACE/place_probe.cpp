// SPDX-License-Identifier: BSD-3-Clause
//
// P4.8 Place, analytic half: is there anything for a placement heuristic to
// optimize?
//
// A placement is a bijection from CTAs to the tasks of a stage.  Every
// locality objective on that bijection has the same shape -- sum over
// co-located task pairs of their temporal-locality weight
//
//     w(c1, c2) = |R(c1) intersect R(c2)|
//
// -- so the objective can only distinguish two placements if `w` distinguishes
// two pairs.  If `w(c1, c2)` is the same number for every c1 != c2, every
// bijection scores identically and the search is degenerate before it starts.
// That is a property of the derived access relations, not of the hardware, so
// it is decidable here with barvinok and no GPU.
//
// `w` is computed exactly: R(c) is `AccessRelation`'s element box, the pair
// overlap is an isl map from (c1, c2) to the elements both read, and the count
// is isl_map_card.  No sampling, no rectangle approximation on top of the one
// AccessRelation already documents.
#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Analysis/TensorSpace.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace tilemega::analysis;

namespace {

std::string Join(std::vector<std::string> const& parts) {
  std::string out;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) out += ", ";
    out += parts[i];
  }
  return out;
}

AffineExpr Rename(AffineExpr expr, std::string const& suffix) {
  for (auto& term : expr.terms) term.coordinate += suffix;
  return expr;
}

/// `{ [c1, c2] -> [e] : e in R(c1) and e in R(c2) }`, counted per (c1, c2).
/// Returns false when the operand is data dependent, which is the one case
/// where the element set is not a box and no exact count exists.
bool PairOverlap(OperatorNode const& op, AccessRelation const& read,
                 ParamBinding const& known, QuasiPolynomial* out) {
  if (read.data_dependent) return false;
  std::vector<std::string> const coords = op.Coordinates();
  if (coords.empty()) return false;

  std::set<std::string> symbols;
  auto collect = [&](std::vector<std::string> const& names) {
    for (auto const& name : names) symbols.insert(name);
  };
  std::vector<std::string> constraints;
  std::vector<std::string> elements;
  for (std::size_t axis = 0; axis < read.index.size(); ++axis) {
    std::string const e = "e" + std::to_string(axis);
    elements.push_back(e);
    collect(read.index[axis].base.FreeSymbols());
    collect(read.index[axis].span.FreeSymbols());
    std::string const span =
        read.index[axis].span.Substitute(known).ToIslText();
    for (char const* suffix : {"__a", "__b"}) {
      std::string const base =
          Rename(read.index[axis].base, suffix).ToIslText(known);
      constraints.push_back("(" + base + ") <= " + e);
      constraints.push_back(e + " < (" + base + ") + (" + span + ")");
    }
  }

  std::size_t index = 0;
  for (std::size_t axis = 0; axis < op.output.axes.size(); ++axis) {
    if (!op.IsTiled(axis)) continue;
    ClosedForm const extent = op.CoordinateExtent(axis);
    collect(extent.FreeSymbols());
    std::string const bound = extent.Substitute(known).ToIslText();
    for (char const* suffix : {"__a", "__b"}) {
      std::string const name = coords[index] + suffix;
      constraints.push_back("0 <= " + name);
      constraints.push_back(name + " < " + bound);
    }
    ++index;
  }
  // The pair coordinates are isl *parameters*, not domain dimensions: only a
  // parameter can be pinned to a literal before the count is evaluated, and
  // pinning is the whole point -- w is wanted at one (c1, c2), not maximized
  // over all of them.
  for (char const* suffix : {"__a", "__b"})
    for (auto const& name : coords) symbols.insert(name + suffix);

  std::ostringstream text;
  if (!symbols.empty()) {
    std::vector<std::string> names(symbols.begin(), symbols.end());
    text << "[" << Join(names) << "] -> ";
  }
  text << "{ [] -> [" << Join(elements) << "] : ";
  for (std::size_t i = 0; i < constraints.size(); ++i) {
    if (i) text << " and ";
    text << constraints[i];
  }
  text << " }";
  if (getenv("PLACE_DEBUG")) std::printf("    MAP %s\n", text.str().c_str());
  try {
    *out = QuasiPolynomial::Card(CouplingRelation::FromIslText(text.str()));
    if (getenv("PLACE_DEBUG")) std::printf("    CARD %s\n", out->ToString().c_str());
  } catch (std::exception const& error) {
    std::printf("    PARSE-FAIL %s\n", error.what());
    return false;
  }
  return true;
}

/// `w` at one (c1, c2): coordinate `axis` differs by `distance`, every other
/// coordinate is held equal, because two CTAs of one stage differ in the
/// linearized task index and that linearization walks one axis at a time.
long At(QuasiPolynomial const& card, std::vector<std::string> const& coords,
        ParamBinding known, std::size_t axis, long distance) {
  for (std::size_t i = 0; i < coords.size(); ++i) {
    known.Bind(coords[i] + "__a", 0);
    known.Bind(coords[i] + "__b", i == axis ? distance : 0);
  }
  try {
    return card.Eval(known);
  } catch (std::exception const&) {
    // An empty intersection has no domain left to take a max over, which is
    // the same isl failure as an unbound parameter.  The two are told apart
    // by asking whether the pinned quantity is identically zero.
    QuasiPolynomial const pinned = card.SubstituteParams(known);
    return pinned.IsZero() ? 0 : -1;
  }
}

struct AxisVerdict {
  long self = -1;
  long low = -1;   ///< smallest off-diagonal weight seen on this axis
  long high = -1;  ///< largest
  bool counted = false;
};

AxisVerdict Sweep(QuasiPolynomial const& card,
                  std::vector<std::string> const& coords, std::size_t axis,
                  long extent, ParamBinding const& known) {
  AxisVerdict out;
  out.self = At(card, coords, known, axis, 0);
  for (long d = 1; d < extent && d <= 64; ++d) {
    long const value = At(card, coords, known, axis, d);
    if (value < 0) return out;
    if (!out.counted) { out.low = out.high = value; out.counted = true; }
    out.low = std::min(out.low, value);
    out.high = std::max(out.high, value);
  }
  return out;
}

void Report(char const* label, OperatorGraph const& graph,
            ParamBinding const& known) {
  int constant = 0, varying = 0, uncounted = 0, single = 0;
  for (auto const& op : graph.nodes) {
    std::vector<std::string> const coords = op.Coordinates();
    if (coords.empty()) continue;
    std::vector<long> extents;
    for (std::size_t axis = 0; axis < op.output.axes.size(); ++axis) {
      if (!op.IsTiled(axis)) continue;
      try {
        extents.push_back(op.CoordinateExtent(axis).Eval(known, known));
      } catch (std::exception const&) {
        extents.push_back(1);
      }
    }
    for (std::size_t k = 0; k < op.operands.size(); ++k) {
      AccessRelation const read = BuildReadMap(op, k);
      QuasiPolynomial card;
      if (!PairOverlap(op, read, known, &card)) {
        ++uncounted;
        std::printf("OPERAND model=%s op=%s operand=%zu status=uncounted "
                    "reason=%s\n", label, op.name.c_str(), k,
                    read.data_dependent ? "data_dependent" : "not_a_box");
        continue;
      }
      bool all_counted = true, any_off = false;
      long self = -1, low = -1, high = -1;
      std::string detail;
      for (std::size_t axis = 0; axis < coords.size(); ++axis) {
        AxisVerdict const v = Sweep(card, coords, axis, extents[axis], known);
        self = v.self;
        detail += " " + coords[axis] + "=";
        if (extents[axis] <= 1) { detail += "single"; continue; }
        if (!v.counted) { all_counted = false; detail += "uncounted"; continue; }
        any_off = true;
        if (low < 0 || v.low < low) low = v.low;
        if (v.high > high) high = v.high;
        detail += std::to_string(v.low);
        if (v.low != v.high) detail += ".." + std::to_string(v.high);
      }
      char const* verdict = "degenerate";
      if (!all_counted) { verdict = "uncounted"; ++uncounted; }
      else if (!any_off) { verdict = "single_task"; ++single; }
      else if (low != high) { verdict = "structured"; ++varying; }
      else ++constant;
      std::printf("OPERAND model=%s op=%s operand=%zu tensor=%s self=%ld "
                  "off=[%ld,%ld] share=%.4f affinity=%s per_axis:%s\n",
                  label, op.name.c_str(), k, read.tensor.name.c_str(), self,
                  low, high, self > 0 && high >= 0 ? double(high) / double(self)
                                                   : 0.0,
                  verdict, detail.c_str());
    }
  }
  std::printf("SUMMARY model=%s degenerate=%d structured=%d single_task=%d "
              "uncounted=%d\n", label, constant, varying, single, uncounted);
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  known.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);
  Report("gqa2", LlamaStack(shape, /*layers=*/2), known);
  Report("mha4", MhaModel(shape, /*layers=*/4), known);
  return 0;
}
