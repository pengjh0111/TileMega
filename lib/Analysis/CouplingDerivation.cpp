// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingDerivation.h>

#include <cctype>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace tilemega::analysis {
namespace {

/// ceildiv that recognises the exact cases first.  `ceildiv(Tm, Tm)` must come
/// out as the literal 1, otherwise wait for §2.7 edge 1 prints as a division.
ClosedForm Divide(ClosedForm const& numerator, ClosedForm const& tile) {
  ClosedForm exact = ClosedForm::Constant(0);
  if (numerator.TryExactDivide(tile, &exact)) return exact;
  return numerator.CeilDiv(tile);
}

/// A conservative minimum.  Equal expressions and constants are decided; for
/// anything else the producer tile is returned, which is an upper bound on the
/// intersection because a producer task never writes more than its own tile.
ClosedForm MinExtent(ClosedForm const& tile, ClosedForm const& span) {
  if (tile.ToString() == span.ToString()) return tile;
  if (tile.IsConstant() && span.IsConstant())
    return ClosedForm::Constant(std::min(tile.Eval({}, {}), span.Eval({}, {})));
  return tile;
}

/// Clamp a read-derived producer-task count to the number of tasks the
/// producer axis actually has.  A read of Tkv cache rows cannot touch more
/// append tasks than the append created; without this, `wait` for the KV edge
/// reports the chunk width even when a single token was appended.
ClosedForm ClampCount(ClosedForm const& count, ClosedForm const& available) {
  if (count.ToString() == available.ToString()) return count;
  if (count.IsConstant() && available.IsConstant())
    return ClosedForm::Constant(
        std::min(count.Eval({}, {}), available.Eval({}, {})));
  // Extents are >= 1, so a literal 1 on either side decides the minimum.
  if (available.IsLiteral(1)) return available;
  if (count.IsLiteral(1)) return count;
  // Undecided: keep the read-derived count.  It is an upper bound, so C is
  // widened rather than narrowed, which I2 permits.
  return count;
}

/// Collects every free (not in `known`) symbol seen while assembling one
/// isl map's constraints, for the `[params] -> { ... }` prefix isl's parser
/// requires (it does not infer parameters from unbound identifiers).
class ParamCollector {
 public:
  explicit ParamCollector(ParamBinding const& known) : known_(known) {}
  void Add(std::vector<std::string> const& symbols) {
    for (auto const& name : symbols) {
      if (known_.Contains(name)) continue;
      if (std::find(names_.begin(), names_.end(), name) != names_.end())
        continue;
      names_.push_back(name);
    }
  }
  std::vector<std::string> const& names() const { return names_; }

 private:
  ParamBinding const& known_;
  std::vector<std::string> names_;
};

std::string JoinTuple(std::vector<std::string> const& names) {
  std::ostringstream out;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i) out << ",";
    out << names[i];
  }
  return out.str();
}

std::string FreshName(std::size_t index) { return "p" + std::to_string(index); }

/// `{ [coords] : 0 <= coord_i < extent_i for every tiled axis }`, the box
/// every consumer coordinate is bound to before DeriveCoupling returns C.
/// Without this, a coordinate this edge's own constraints never reference
/// (e.g. a GEMM's "n" when only "m" ties back to its producer) is isl-
/// unbounded -- ranging over every integer -- which makes fanout diverge.
std::string DomainBoxText(OperatorNode const& consumer, ParamBinding const& known) {
  std::vector<std::string> domain = consumer.Coordinates();
  ParamCollector params(known);
  std::vector<std::string> bounds;
  std::size_t domain_index = 0;
  for (std::size_t axis = 0; axis < consumer.output.axes.size(); ++axis) {
    if (!consumer.IsTiled(axis)) continue;
    std::string const& name = domain[domain_index++];
    ClosedForm extent = consumer.CoordinateExtent(axis);
    params.Add(extent.FreeSymbols());
    bounds.push_back("0 <= " + name);
    bounds.push_back(name + " < " + extent.Substitute(known).ToIslText());
  }
  std::ostringstream text;
  if (!params.names().empty()) text << "[" << JoinTuple(params.names()) << "] -> ";
  text << "{ [" << JoinTuple(domain) << "]";
  if (!bounds.empty()) {
    text << " : ";
    for (std::size_t i = 0; i < bounds.size(); ++i) {
      if (i) text << " and ";
      text << bounds[i];
    }
  }
  text << " }";
  return text.str();
}

/// True when any extent or tile this edge's two task spaces are built from
/// still names a symbol. Deliberately independent of the derivation-time
/// `known` binding: I1 is about what L-sem says, not about what happened to
/// be bound when the edge was derived.
bool AnySymbolicExtent(OperatorNode const& node) {
  for (auto const& axis : node.output.axes)
    if (!axis.extent.FreeSymbols().empty() || !axis.origin.FreeSymbols().empty())
      return true;
  for (auto const& tile : node.tile)
    if (!tile.FreeSymbols().empty()) return true;
  return false;
}

/// Classify a derived count by the shape of the isl form barvinok printed.
/// isl separates cells with ';' and prints a periodic coefficient with
/// `floor`, so either marks a quantity with no single value; a single cell
/// whose value expression is a literal is the constant case, and anything
/// else is a genuine (single-cell) function of the position.
Countability ClassifyCount(QuasiPolynomial const& count) {
  std::string const& text = count.ToString();
  if (text.find(';') != std::string::npos ||
      text.find("floor") != std::string::npos)
    return Countability::kPiecewiseQuasiPolynomial;
  std::size_t open = text.rfind('{');
  if (open == std::string::npos) return Countability::kQuasiPolynomial;
  std::string body = text.substr(open + 1);
  std::size_t stop = body.find_first_of(":}");
  if (stop != std::string::npos) body = body.substr(0, stop);
  // isl always prints wait/fanout as a function of the task coordinate, so a
  // `[m, n] ->` prefix is not itself evidence of position dependence; what
  // decides is whether the value on the other side of it is a literal.
  std::size_t arrow = body.rfind("->");
  if (arrow != std::string::npos) body = body.substr(arrow + 2);
  std::string value;
  for (char c : body)
    if (!std::isspace(static_cast<unsigned char>(c))) value.push_back(c);
  if (value.empty()) return Countability::kQuasiPolynomial;
  std::size_t digits = value[0] == '-' ? 1 : 0;
  for (std::size_t i = digits; i < value.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(value[i])))
      return Countability::kQuasiPolynomial;
  return Countability::kConstant;
}

}  // namespace

std::string ToString(Tier tier) {
  switch (tier) {
    case Tier::kAffine: return "0";
    case Tier::kSharedInjectiveLayout: return "1";
    case Tier::kStructuredRagged: return "2";
    case Tier::kDataDependent: return "3";
  }
  return "?";
}

std::string ToString(RelationKind kind) {
  switch (kind) {
    case RelationKind::kAffine: return "affine";
    case RelationKind::kLayoutMediated: return "layout_mediated";
    case RelationKind::kDataDependent: return "data_dependent";
  }
  return "?";
}

std::string ToString(ExtentKind kind) {
  switch (kind) {
    case ExtentKind::kStaticLiteral: return "static_literal";
    case ExtentKind::kSymbolicStatic: return "symbolic_static";
    case ExtentKind::kRuntimeDynamic: return "runtime_dynamic";
  }
  return "?";
}

std::string ToString(Exactness exactness) {
  return exactness == Exactness::kExact ? "exact" : "relaxed";
}

std::string ToString(RuntimeRequirement requirement) {
  switch (requirement) {
    case RuntimeRequirement::kNone: return "none";
    case RuntimeRequirement::kPrefixSum: return "prefix_sum";
    case RuntimeRequirement::kTensorValues: return "tensor_values";
  }
  return "?";
}

std::string ToString(Countability countability) {
  switch (countability) {
    case Countability::kConstant: return "constant";
    case Countability::kQuasiPolynomial: return "quasipoly";
    case Countability::kPiecewiseQuasiPolynomial: return "piecewise_quasipoly";
    case Countability::kUncountable: return "uncountable";
  }
  return "?";
}

std::string CouplingAttributes::ToString() const {
  return tilemega::analysis::ToString(relation_kind) + " + " +
         tilemega::analysis::ToString(extent_kind) + " + " +
         tilemega::analysis::ToString(exactness) + " + " +
         tilemega::analysis::ToString(runtime_requirement) + " + " +
         tilemega::analysis::ToString(countability);
}

bool ParseCouplingAttributes(std::string const& relation_kind,
                             std::string const& extent_kind,
                             std::string const& exactness,
                             std::string const& runtime_requirement,
                             std::string const& countability,
                             CouplingAttributes* out) {
  CouplingAttributes parsed;
  if (relation_kind == "affine")
    parsed.relation_kind = RelationKind::kAffine;
  else if (relation_kind == "layout_mediated")
    parsed.relation_kind = RelationKind::kLayoutMediated;
  else if (relation_kind == "data_dependent")
    parsed.relation_kind = RelationKind::kDataDependent;
  else
    return false;
  if (extent_kind == "static_literal")
    parsed.extent_kind = ExtentKind::kStaticLiteral;
  else if (extent_kind == "symbolic_static")
    parsed.extent_kind = ExtentKind::kSymbolicStatic;
  else if (extent_kind == "runtime_dynamic")
    parsed.extent_kind = ExtentKind::kRuntimeDynamic;
  else
    return false;
  if (exactness == "exact")
    parsed.exactness = Exactness::kExact;
  else if (exactness == "relaxed")
    parsed.exactness = Exactness::kRelaxed;
  else
    return false;
  if (runtime_requirement == "none")
    parsed.runtime_requirement = RuntimeRequirement::kNone;
  else if (runtime_requirement == "prefix_sum")
    parsed.runtime_requirement = RuntimeRequirement::kPrefixSum;
  else if (runtime_requirement == "tensor_values")
    parsed.runtime_requirement = RuntimeRequirement::kTensorValues;
  else
    return false;
  if (countability == "constant")
    parsed.countability = Countability::kConstant;
  else if (countability == "quasipoly")
    parsed.countability = Countability::kQuasiPolynomial;
  else if (countability == "piecewise_quasipoly")
    parsed.countability = Countability::kPiecewiseQuasiPolynomial;
  else if (countability == "uncountable")
    parsed.countability = Countability::kUncountable;
  else
    return false;
  *out = parsed;
  return true;
}

Tier DeriveTier(CouplingAttributes const& attributes) {
  // The order is §2.4's, read as "what is the strongest obstacle to solving
  // this edge": an index out of a tensor, then an extent or a projection that
  // is not available in closed form, then an indirection that cancels.
  if (attributes.relation_kind == RelationKind::kDataDependent ||
      attributes.countability == Countability::kUncountable)
    return Tier::kDataDependent;
  if (attributes.extent_kind == ExtentKind::kRuntimeDynamic ||
      attributes.exactness == Exactness::kRelaxed)
    return Tier::kStructuredRagged;
  if (attributes.relation_kind == RelationKind::kLayoutMediated)
    return Tier::kSharedInjectiveLayout;
  return Tier::kAffine;
}

std::string CouplingEdge::EventShapeString() const {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < event_shape.size(); ++i) {
    if (i) out << "x";
    out << event_shape[i].ToString();
  }
  out << "]";
  return out.str();
}

CouplingRelation DeriveCoupling(AccessRelation const& W, AccessRelation const& R,
                                OperatorNode const& producer,
                                OperatorNode const& consumer,
                                ParamBinding const& known,
                                CouplingDetail* detail) {
  CouplingDetail local;
  CouplingDetail& info = detail ? *detail : local;
  info = CouplingDetail{};

  if (W.index.size() != R.index.size())
    throw std::invalid_argument("W and R must address the same tensor rank");

  std::vector<std::string> domain = consumer.Coordinates();
  std::vector<std::string> outputs;
  std::vector<std::string> constraints;
  ParamCollector params(known);

  // The relaxation fallback: every producer task on this axis.  Used only when
  // no exact projection rule applies; it widens C, which I2 permits, and the
  // caller raises the tier so nothing downstream reads it as a closed form.
  auto relaxAxis = [&](std::size_t axis, std::string const& why) {
    std::string name = FreshName(outputs.size());
    outputs.push_back(name);
    ClosedForm extent = producer.CoordinateExtent(axis);
    params.Add(extent.FreeSymbols());
    constraints.push_back("0 <= " + name + " and " + name + " < " +
                          extent.Substitute(known).ToIslText());
    info.exact = false;
    // One reason per distinct cause: relaxing three axes for the same reason
    // is still one reason in the P3 report.
    if (info.relaxation.find(why) != std::string::npos) return;
    if (!info.relaxation.empty()) info.relaxation += "; ";
    info.relaxation += why;
  };

  // Every consumer coordinate is bound to its own task-space extent
  // (DomainBoxText), not just the ones the producer-axis constraints above
  // happen to mention -- otherwise a coordinate this edge does not use
  // (e.g. a GEMM's "n" when only "m" ties back to its producer) is isl-
  // unbounded, and fanout (which counts *all* domain points mapping to one
  // producer coordinate) diverges. Which coordinates the constraints *do*
  // reference is tracked separately in info.occurring as they are added
  // above, for ComputeEventShape -- isl has no reliable "does this
  // dimension actually affect the output" query once the domain is bounded
  // (isl_map_involves_dims is syntactic: it also flags a dimension that
  // merely bounds the domain, confirmed empirically), so that answer is
  // recorded here, where it is unambiguous, rather than re-derived later.
  auto assemble = [&]() -> CouplingRelation {
    std::ostringstream text;
    if (!params.names().empty()) text << "[" << JoinTuple(params.names()) << "] -> ";
    text << "{ [" << JoinTuple(domain) << "] -> [" << JoinTuple(outputs) << "]";
    if (!constraints.empty()) {
      text << " : ";
      for (std::size_t i = 0; i < constraints.size(); ++i) {
        if (i) text << " and ";
        text << constraints[i];
      }
    }
    text << " }";
    return CouplingRelation::FromIslText(text.str())
        .IntersectDomain(DomainBoxText(consumer, known));
  };

  if (R.data_dependent) {
    // Tier 3: no affine inverse exists.  Relax the whole producer task space.
    for (std::size_t a = 0; a < producer.output.axes.size(); ++a)
      if (producer.IsTiled(a)) relaxAxis(a, "data-dependent index");
    return assemble();
  }

  auto mergeOccurring = [&](std::vector<std::string> const& coordinates) {
    for (auto const& name : coordinates) {
      if (std::find(info.occurring.begin(), info.occurring.end(), name) ==
          info.occurring.end())
        info.occurring.push_back(name);
    }
  };

  // A producer that writes only a sub-window of a tensor axis (an append)
  // couples to a consumer only where the consumer's read meets that window.
  // The guard is recorded rather than folded into C: widening C to the whole
  // consumer domain would be sound by I2 but would inflate wait.
  auto recordGuard = [&](TensorAxis const& axis, ElementInterval const& read) {
    if (!info.guard.empty()) return;
    if (read.base.terms.size() != 1) return;
    if (read.base.terms.front().coefficient.ToString() !=
        read.span.ToString())
      return;
    info.guard = read.base.terms.front().coordinate + " == floordiv(" +
                 axis.origin.ToString() + ", " + read.span.ToString() + ")";
  };

  for (std::size_t a = 0; a < producer.output.axes.size(); ++a) {
    TensorAxis const& axis_all = producer.output.axes[a];
    if (!producer.IsTiled(a)) {
      // Whole axis: no task coordinate, but a sub-window still guards the edge.
      bool sub_window = !axis_all.origin.IsLiteral(0);
      if (sub_window) recordGuard(axis_all, R.index[a]);
      continue;
    }
    TensorAxis const& axis = producer.output.axes[a];
    ClosedForm const& tile = producer.tile[a];
    ElementInterval const& read = R.index[a];
    std::string name = FreshName(outputs.size());

    // A producer axis with a single task contributes the constant coordinate 0.
    // This is where an appended window lands: the whole write is one task, and
    // the consumer coordinates that miss the window are excluded by the guard.
    ClosedForm coordinate_extent = producer.CoordinateExtent(a);
    if (coordinate_extent.IsLiteral(1)) {
      outputs.push_back(name);
      constraints.push_back(name + " = 0");
      if (!axis.origin.IsLiteral(0)) recordGuard(axis, read);
      continue;
    }

    // Shift the read into the producer's own index space before inverting.
    AffineExpr shifted = read.base;
    if (!axis.origin.IsLiteral(0)) {
      recordGuard(axis, read);
      shifted = shifted +
                AffineExpr::Constant(ClosedForm::Constant(-1) * axis.origin);
    }

    AffineExpr base;
    if (shifted.TryExactDivide(tile, &base)) {
      ClosedForm count = ClampCount(Divide(read.span, tile), coordinate_extent);
      outputs.push_back(name);
      params.Add(base.FreeSymbols());
      mergeOccurring(base.Coordinates());
      if (count.IsLiteral(1)) {
        constraints.push_back(name + " = " + base.ToIslText(known));
      } else {
        params.Add(count.FreeSymbols());
        std::string base_text = base.ToIslText(known);
        constraints.push_back(base_text + " <= " + name);
        constraints.push_back(name + " < " + base_text + " + " +
                              count.Substitute(known).ToIslText());
      }
      continue;
    }

    if (read.span.IsLiteral(1)) {
      // A single element: floor(base / tile) is exact even though base does not
      // divide the tile.  §2.7 edge 5 is exactly this case.
      AffineExpr projected = shifted;
      projected.divisor = tile;
      outputs.push_back(name);
      params.Add(projected.FreeSymbols());
      mergeOccurring(projected.Coordinates());
      constraints.push_back(name + " = " + projected.ToIslText(known));
      continue;
    }

    // Not tile aligned, and more than one element: the read interval
    // [base, base + span) straddles producer tiles. This is exact anyway --
    // producer task p covers [p*tile, p*tile + tile) in the shifted index
    // space, so p is coupled exactly when the two intervals overlap:
    //
    //     p*tile < base + span   and   base < p*tile + tile
    //
    // Both are affine in the consumer coordinates once `tile` is a literal
    // (which `known` guarantees, or ToIslText throws), so isl can hold the
    // condition directly. The pre-migration path had to relax here instead:
    // ClosedForm could express neither the two-sided overlap condition nor
    // the count over it, and that count is genuinely a *piecewise
    // quasi-polynomial* whenever span and tile are misaligned -- wait then
    // varies periodically with the consumer coordinate (see
    // docs/experiments/P3_ISL/ and MisalignedTileModel in ReferenceModels.h).
    // Keeping the edge exact here is a precision gain the migration unlocks,
    // not a behaviour change for any previously covered access pattern:
    // the exact-quotient and single-element branches above still fire first,
    // and no reference model reached this branch before.
    {
      ClosedForm literal_tile = tile.Substitute(known);
      if (!literal_tile.IsConstant()) {
        relaxAxis(a, "read interval is not tile aligned");
        continue;
      }
      outputs.push_back(name);
      params.Add(shifted.FreeSymbols());
      params.Add(read.span.FreeSymbols());
      mergeOccurring(shifted.Coordinates());
      std::string base_text = shifted.ToIslText(known);
      std::string tile_text = literal_tile.ToIslText();
      std::string span_text = read.span.Substitute(known).ToIslText();
      constraints.push_back("(" + name + " * " + tile_text + ") < (" +
                            base_text + ") + (" + span_text + ")");
      constraints.push_back("(" + base_text + ") < (" + name + " * " +
                            tile_text + ") + " + tile_text);
    }
  }

  return assemble();
}

namespace {
/// `{ [p0,...] : 0 <= p_i < producer_extent_i }`, the producer-side twin of
/// DomainBoxText, named after C's actual range dimensions.
///
/// This is applied *only* when counting fanout (and by the relaxation
/// check), never folded into C itself.
/// Both halves of that split are load-bearing, and each was established by a
/// failure:
///   * Without any range bound, isl_map_card's own piecewise decomposition of
///     the reversed map keeps a "reachable only for some other parameter
///     value" tail whose formula evaluates to 0 (attn_combine->wo retains a
///     p0-in-[S, 126+S] piece), so fanout looks position-dependent (max 32,
///     min 0) when it is really a uniform 32.
///   * With the bound folded into C, the *wait* direction regresses instead:
///     counting a relation that carries both a genuine isl parameter (S) and
///     an inequality-range-derived producer coordinate bounded on both sides
///     drives barvinok into "unexpected missing (bounded) solution"
///     (basis_reduction_tab.c) and an incomplete result.
/// Restricting only the reversed map keeps each direction in the regime its
/// own counting problem is tractable in. See TileMega_skeleton.md §1.5.1.
}  // namespace

std::string ProducerTaskSpaceText(CouplingRelation const& C,
                                  OperatorNode const& producer,
                                  ParamBinding const& known) {
  std::vector<std::string> range = C.RangeDimNames();
  ParamCollector params(known);
  std::vector<std::string> bounds;
  std::size_t range_index = 0;
  for (std::size_t axis = 0;
       axis < producer.output.axes.size() && range_index < range.size(); ++axis) {
    if (!producer.IsTiled(axis)) continue;
    std::string const& name = range[range_index++];
    ClosedForm extent = producer.CoordinateExtent(axis);
    params.Add(extent.FreeSymbols());
    bounds.push_back("0 <= " + name);
    bounds.push_back(name + " < " + extent.Substitute(known).ToIslText());
  }
  std::ostringstream text;
  if (!params.names().empty()) text << "[" << JoinTuple(params.names()) << "] -> ";
  text << "{ [" << JoinTuple(range) << "]";
  if (!bounds.empty()) {
    text << " : ";
    for (std::size_t i = 0; i < bounds.size(); ++i) {
      if (i) text << " and ";
      text << bounds[i];
    }
  }
  text << " }";
  return text.str();
}

namespace {
/// Wrap a parameter-only ClosedForm result (volume, count -- neither varies
/// across the task space for any access pattern this codebase derives) as a
/// 0-dimensional QuasiPolynomial: `[params] -> { <value> }`.
QuasiPolynomial WrapScalar(ClosedForm const& value, ParamBinding const& known) {
  ParamCollector params(known);
  params.Add(value.FreeSymbols());
  std::ostringstream text;
  if (!params.names().empty()) text << "[" << JoinTuple(params.names()) << "] -> ";
  text << "{ " << value.Substitute(known).ToIslText() << " }";
  return QuasiPolynomial::FromIslText(text.str());
}
}  // namespace

DerivedMetrics ComputeMetrics(CouplingRelation const& C, AccessRelation const& W,
                              AccessRelation const& R,
                              OperatorNode const& producer,
                              OperatorNode const& consumer,
                              ParamBinding const& known) {
  DerivedMetrics metrics;

  // wait(x) = |C(x)|, fanout(y) = |C^-1(y)|: both barvinok counts over the
  // derived relation, per §2 Definition 4. C is already bound to the
  // consumer's own task-space box (DeriveCoupling's `assemble`), so wait is
  // finite without further restriction. fanout additionally needs the
  // producer side bounded -- but only on the reversed map, never folded back
  // into C; see ProducerRangeBoxText for why each direction needs a
  // different regime.
  metrics.wait = C.Card();
  metrics.fanout =
      C.IntersectRange(ProducerTaskSpaceText(C, producer, known)).FanoutCard();

  // volume(y,x) = |W_p(y) ^ R_c(x)|, per tensor axis.  Data-independent of the
  // task space for every access pattern this codebase derives, so it stays a
  // parameter-only value rather than a function read off C.
  ClosedForm volume = ClosedForm::Constant(1);
  for (std::size_t a = 0; a < producer.output.axes.size(); ++a) {
    ClosedForm const& tile = producer.IsTiled(a)
                                 ? producer.tile[a]
                                 : producer.output.axes[a].extent;
    volume = volume * MinExtent(tile, R.index[a].span);
  }
  metrics.volume = WrapScalar(volume, known);

  // count(T_op): the consumer task count, the domain this edge is defined on.
  metrics.count = WrapScalar(consumer.Count(), known);
  (void)W;
  return metrics;
}

std::vector<ClosedForm> ComputeEventShape(OperatorNode const& consumer,
                                          std::vector<std::string> const& occurring) {
  std::vector<std::string> domain = consumer.Coordinates();
  std::vector<ClosedForm> shape;
  std::size_t domain_index = 0;
  for (std::size_t axis = 0; axis < consumer.output.axes.size(); ++axis) {
    if (!consumer.IsTiled(axis)) continue;
    std::string const& name = domain[domain_index++];
    if (std::find(occurring.begin(), occurring.end(), name) != occurring.end())
      shape.push_back(consumer.CoordinateExtent(axis));
  }
  return shape;
}

bool Contains(CouplingRelation const& wide, CouplingRelation const& narrow) {
  return narrow.IsSubset(wide);
}

std::vector<CouplingEdge> CouplingDerivation::Derive(
    OperatorGraph const& graph, ParamBinding const& known) const {
  std::vector<CouplingEdge> edges;
  for (auto const& consumer : graph.nodes) {
    for (std::size_t k = 0; k < consumer.operands.size(); ++k) {
      Operand const& operand = consumer.operands[k];
      if (operand.producer.empty()) continue;
      OperatorNode const* producer = graph.Find(operand.producer);
      if (!producer) continue;

      AccessRelation W = BuildWriteMap(*producer);
      AccessRelation R = BuildReadMap(consumer, k);

      CouplingEdge edge;
      edge.src = TaskSpaceId{producer->name};
      edge.dst = TaskSpaceId{consumer.name};
      CouplingDetail detail;
      edge.C = DeriveCoupling(W, R, *producer, consumer, known, &detail);
      edge.exact = detail.exact;
      edge.guard = detail.guard;
      edge.relaxation = detail.relaxation;
      edge.metrics = ComputeMetrics(edge.C, W, R, *producer, consumer, known);
      edge.event_shape = ComputeEventShape(consumer, detail.occurring);

      CouplingAttributes attributes;
      // The edge is expressed through a declared non-identity layout when
      // either endpoint declares one -- an append writes *into* the laid-out
      // tensor, so the layout is on the consumer's output, not the producer's.
      if (!producer->output.layout_id.empty() ||
          !consumer.output.layout_id.empty() ||
          !operand.tensor.layout_id.empty())
        attributes.relation_kind = RelationKind::kLayoutMediated;
      if (R.data_dependent)
        attributes.relation_kind = RelationKind::kDataDependent;
      bool runtime_extent =
          producer->HasRuntimeTaskSpace() || consumer.HasRuntimeTaskSpace();
      attributes.extent_kind =
          runtime_extent ? ExtentKind::kRuntimeDynamic
          : (AnySymbolicExtent(*producer) || AnySymbolicExtent(consumer))
              ? ExtentKind::kSymbolicStatic
              : ExtentKind::kStaticLiteral;
      attributes.exactness =
          detail.exact ? Exactness::kExact : Exactness::kRelaxed;
      // What the runtime has to hand the kernel, read off the source of the
      // uncertainty rather than off another attribute: an indptr prefix sum
      // for a ragged extent, the index tensor itself for a gather.
      attributes.runtime_requirement =
          R.data_dependent  ? RuntimeRequirement::kTensorValues
          : runtime_extent  ? RuntimeRequirement::kPrefixSum
                            : RuntimeRequirement::kNone;
      attributes.countability =
          R.data_dependent ? Countability::kUncountable
                           : ClassifyCount(edge.metrics.wait);
      edge.attributes = attributes;
      edge.tier = DeriveTier(attributes);
      edge.sync = SyncKind::kGlobal;
      edges.push_back(std::move(edge));
    }
  }
  return edges;
}

}  // namespace tilemega::analysis
