// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingDerivation.h>

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

std::string FreshName(std::string const& wanted,
                      std::vector<std::string> const& taken) {
  std::string name = wanted;
  int suffix = 0;
  while (std::find(taken.begin(), taken.end(), name) != taken.end())
    name = wanted + "_p" + std::to_string(++suffix);
  return name;
}

void CollectCoordinates(AffineExpr const& expr, std::vector<std::string>* out) {
  for (auto const& term : expr.terms)
    if (std::find(out->begin(), out->end(), term.coordinate) == out->end())
      out->push_back(term.coordinate);
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

AffineRelation DeriveCoupling(AccessRelation const& W, AccessRelation const& R,
                              OperatorNode const& producer,
                              OperatorNode const& consumer,
                              CouplingDetail* detail) {
  CouplingDetail local;
  CouplingDetail& info = detail ? *detail : local;
  info = CouplingDetail{};

  if (W.index.size() != R.index.size())
    throw std::invalid_argument("W and R must address the same tensor rank");

  ProducerMap map;
  map.source = TaskSpaceId{producer.name};

  std::vector<std::string> taken = consumer.Coordinates();

  // The relaxation fallback: every producer task on this axis.  Used only when
  // no exact projection rule applies; it widens C, which I2 permits, and the
  // caller raises the tier so nothing downstream reads it as a closed form.
  auto relaxAxis = [&](std::size_t axis, std::string const& why) {
    AffineRange range;
    range.name = FreshName(producer.output.axes[axis].name, taken);
    taken.push_back(range.name);
    range.begin = AffineExpr::Constant(ClosedForm::Constant(0));
    range.extent = producer.CoordinateExtent(axis);
    map.quantified.push_back(range);
    map.coordinates.push_back(AffineExpr::Variable(range.name));
    info.exact = false;
    // One reason per distinct cause: relaxing three axes for the same reason
    // is still one reason in the P3 report.
    if (info.relaxation.find(why) != std::string::npos) return;
    if (!info.relaxation.empty()) info.relaxation += "; ";
    info.relaxation += why;
  };

  if (R.data_dependent) {
    // Tier 3: no affine inverse exists.  Relax the whole producer task space.
    for (std::size_t a = 0; a < producer.output.axes.size(); ++a)
      if (producer.IsTiled(a)) relaxAxis(a, "data-dependent index");
    AffineRelation relation(consumer.Coordinates(), {map});
    return relation;
  }

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

    // A producer axis with a single task contributes the constant coordinate 0.
    // This is where an appended window lands: the whole write is one task, and
    // the consumer coordinates that miss the window are excluded by the guard.
    ClosedForm coordinate_extent = producer.CoordinateExtent(a);
    if (coordinate_extent.IsLiteral(1)) {
      map.coordinates.push_back(
          AffineExpr::Constant(ClosedForm::Constant(0)));
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
      if (count.IsLiteral(1)) {
        map.coordinates.push_back(base);
      } else {
        AffineRange range;
        range.name = FreshName(axis.name, taken);
        taken.push_back(range.name);
        range.begin = base;
        range.extent = count;
        map.quantified.push_back(range);
        map.coordinates.push_back(AffineExpr::Variable(range.name));
      }
      continue;
    }

    if (read.span.IsLiteral(1)) {
      // A single element: floor(base / tile) is exact even though base does not
      // divide the tile.  §2.7 edge 5 is exactly this case.
      AffineExpr projected = shifted;
      projected.divisor = tile;
      map.coordinates.push_back(projected);
      continue;
    }

    relaxAxis(a, "read interval is not tile aligned");
  }

  return AffineRelation(consumer.Coordinates(), {map});
}

DerivedMetrics ComputeMetrics(AffineRelation const& C, AccessRelation const& W,
                              AccessRelation const& R,
                              OperatorNode const& producer,
                              OperatorNode const& consumer) {
  DerivedMetrics metrics;

  // wait(x) = |C(x)|: the product of the quantified extents, summed over the
  // producer maps in the image.
  ClosedForm wait = ClosedForm::Constant(0);
  for (auto const& map : C.Producers()) {
    ClosedForm fiber = ClosedForm::Constant(1);
    for (auto const& range : map.quantified) fiber = fiber * range.extent;
    wait = wait.IsLiteral(0) ? fiber : wait + fiber;
  }
  metrics.wait = wait;

  // fanout(y) = |C^-1(y)|: a consumer coordinate that occurs in C is pinned by
  // y, one that does not is free and contributes its whole range.
  std::vector<std::string> occurring;
  for (auto const& map : C.Producers()) {
    for (auto const& coordinate : map.coordinates)
      CollectCoordinates(coordinate, &occurring);
    for (auto const& range : map.quantified)
      CollectCoordinates(range.begin, &occurring);
  }
  ClosedForm fanout = ClosedForm::Constant(1);
  for (std::size_t i = 0; i < consumer.output.axes.size(); ++i) {
    if (!consumer.IsTiled(i)) continue;
    std::string const& name = consumer.output.axes[i].name;
    if (std::find(occurring.begin(), occurring.end(), name) != occurring.end())
      continue;
    fanout = fanout * consumer.CoordinateExtent(i);
  }
  metrics.fanout = fanout;

  // volume(y,x) = |W_p(y) ^ R_c(x)|, per tensor axis.
  ClosedForm volume = ClosedForm::Constant(1);
  for (std::size_t a = 0; a < producer.output.axes.size(); ++a) {
    ClosedForm const& tile = producer.IsTiled(a)
                                 ? producer.tile[a]
                                 : producer.output.axes[a].extent;
    volume = volume * MinExtent(tile, R.index[a].span);
  }
  metrics.volume = volume;

  // count(T_op): the consumer task count, the domain this edge is defined on.
  metrics.count = consumer.Count();
  (void)W;
  return metrics;
}

std::vector<ClosedForm> ComputeEventShape(AffineRelation const& C,
                                          OperatorNode const& consumer) {
  std::vector<std::string> occurring;
  for (auto const& map : C.Producers()) {
    for (auto const& coordinate : map.coordinates)
      CollectCoordinates(coordinate, &occurring);
    for (auto const& range : map.quantified)
      CollectCoordinates(range.begin, &occurring);
  }
  std::vector<ClosedForm> shape;
  for (std::size_t i = 0; i < consumer.output.axes.size(); ++i) {
    if (!consumer.IsTiled(i)) continue;
    std::string const& name = consumer.output.axes[i].name;
    if (std::find(occurring.begin(), occurring.end(), name) == occurring.end())
      continue;
    shape.push_back(consumer.CoordinateExtent(i));
  }
  return shape;
}

namespace {

/// True when `expr` is the bare variable `name`.
bool IsBareVariable(AffineExpr const& expr, std::string const& name) {
  return expr.terms.size() == 1 && expr.offset.IsLiteral(0) &&
         expr.divisor.IsLiteral(1) && expr.terms.front().coordinate == name &&
         expr.terms.front().coefficient.IsLiteral(1) &&
         expr.terms.front().group.IsLiteral(1);
}

AffineRange const* FindRange(ProducerMap const& map, std::string const& name) {
  for (auto const& range : map.quantified)
    if (range.name == name) return &range;
  return nullptr;
}

/// Position `i` of `map` covers the whole producer axis `i`.
bool CoversAxis(ProducerMap const& map, std::size_t i,
                OperatorNode const& producer) {
  if (i >= map.coordinates.size()) return false;
  AffineExpr const& coordinate = map.coordinates[i];
  if (coordinate.terms.size() != 1) return false;
  AffineRange const* range = FindRange(map, coordinate.terms.front().coordinate);
  if (!range) return false;
  if (!IsBareVariable(coordinate, range->name)) return false;
  if (!range->begin.IsZero()) return false;
  return range->extent.ToString() == producer.CoordinateExtent(i).ToString();
}

bool CoversMap(ProducerMap const& wide, ProducerMap const& narrow,
               OperatorNode const& producer) {
  if (!(wide.source == narrow.source)) return false;
  if (wide.coordinates.size() != narrow.coordinates.size()) return false;
  for (std::size_t i = 0; i < wide.coordinates.size(); ++i) {
    if (CoversAxis(wide, i, producer)) continue;
    if (wide.coordinates[i].ToString() != narrow.coordinates[i].ToString())
      return false;
    // Same expression: the quantified ranges behind it must agree too.
    for (auto const& name : wide.coordinates[i].Coordinates()) {
      AffineRange const* a = FindRange(wide, name);
      AffineRange const* b = FindRange(narrow, name);
      if (!a && !b) continue;
      if (!a || !b) return false;
      if (a->begin.ToString() != b->begin.ToString()) return false;
      if (a->extent.ToString() != b->extent.ToString()) return false;
    }
  }
  return true;
}

}  // namespace

bool Contains(AffineRelation const& wide, AffineRelation const& narrow,
              OperatorNode const& producer) {
  if (narrow.Producers().empty()) return true;
  for (auto const& want : narrow.Producers()) {
    bool covered = false;
    for (auto const& have : wide.Producers())
      if (CoversMap(have, want, producer)) covered = true;
    if (!covered) return false;
  }
  return true;
}

std::vector<CouplingEdge> CouplingDerivation::Derive(
    OperatorGraph const& graph) const {
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
      edge.C = DeriveCoupling(W, R, *producer, consumer, &detail);
      edge.exact = detail.exact;
      edge.guard = detail.guard;
      edge.relaxation = detail.relaxation;
      edge.metrics = ComputeMetrics(edge.C, W, R, *producer, consumer);
      edge.event_shape = ComputeEventShape(edge.C, consumer);

      Tier tier = Tier::kAffine;
      auto raise = [&](Tier candidate) { tier = std::max(tier, candidate); };
      // Tier 1: the edge is expressed through a declared non-identity layout.
      // Either endpoint is enough -- an append writes *into* the laid-out
      // tensor, so the layout is on the consumer's output, not the producer's.
      if (!producer->output.layout_id.empty() ||
          !consumer.output.layout_id.empty() ||
          !operand.tensor.layout_id.empty())
        raise(Tier::kSharedInjectiveLayout);
      if (producer->HasRuntimeTaskSpace() || consumer.HasRuntimeTaskSpace())
        raise(Tier::kStructuredRagged);
      if (!detail.exact) raise(Tier::kStructuredRagged);
      if (R.data_dependent) tier = Tier::kDataDependent;
      edge.tier = tier;
      edge.sync = SyncKind::kGlobal;
      edges.push_back(std::move(edge));
    }
  }
  return edges;
}

}  // namespace tilemega::analysis
