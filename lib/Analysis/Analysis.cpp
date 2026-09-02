// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/CuteLayoutBridge.h>
#include <tilemega/Analysis/EventSynthesis.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TierClassifier.h>

#include "IslUtil.h"

#include <isl/ilp.h>

#include <sstream>
#include <stdexcept>
namespace tilemega::analysis {
LayoutProjection CuteLayoutBridge::Project(
    LayoutDescriptor const& write, LayoutDescriptor const* read) const {
  if (read && !write.layout_id.empty() &&
      write.layout_id == read->layout_id && write.injective && read->injective)
    return {InverseStrategy::kCancelSharedLayout,
            Tier::kSharedInjectiveLayout, true,
            "shared injective layout cancels before inversion"};
  if (write.swizzled || (read && read->swizzled))
    return {InverseStrategy::kRaiseTier, Tier::kDataDependent, false,
            "CuTe swizzle is not a Presburger affine stride map"};
  if (write.dynamic_stride || (read && read->dynamic_stride))
    return {InverseStrategy::kRaiseTier, Tier::kStructuredRagged, false,
            "parameter*coordinate dynamic stride is not Presburger affine"};
  if (!write.injective)
    return {InverseStrategy::kRaiseTier, Tier::kStructuredRagged, false,
            "write layout is not injective"};
  if (write.static_tile_shape)
    return {InverseStrategy::kCuteStaticRightInverse, Tier::kAffine, true,
            "g-specialized static tile admits cute.right_inverse"};
  return {InverseStrategy::kPresburgerRelation, Tier::kAffine, true,
          "symbolic domain extent with constant strides is Presburger affine"};
}
namespace {

/// Symbols that stay free after `known` -- they become isl parameters.
void CollectFree(ClosedForm const& value, ParamBinding const& known,
                 std::vector<std::string>* out) {
  for (auto const& name : value.FreeSymbols()) {
    if (known.Contains(name)) continue;
    if (std::find(out->begin(), out->end(), name) == out->end())
      out->push_back(name);
  }
}

}  // namespace

CouplingRelation CuteLayoutBridge::ToIslMap(LayoutDescriptor const& layout,
                                            ParamBinding const& known) const {
  if (layout.extents.size() != layout.strides.size())
    throw std::invalid_argument(
        "CuTe layout needs one stride per extent (flatten it first)");
  if (layout.extents.empty())
    throw std::invalid_argument("CuTe layout has no modes");
  if (layout.swizzled)
    throw std::domain_error(
        "swizzled layout: a composed layout with a swizzle does not flatten "
        "to an affine stride map, so it has no isl_map (V-F); raise the Tier "
        "instead of approximating it");

  std::vector<std::string> params;
  std::vector<std::string> coordinates;
  std::vector<std::string> bounds;
  std::ostringstream offset;
  for (std::size_t i = 0; i < layout.extents.size(); ++i) {
    std::string coordinate = "i" + std::to_string(i);
    coordinates.push_back(coordinate);

    ClosedForm stride = layout.strides[i].Substitute(known);
    if (!stride.IsConstant())
      throw std::domain_error(
          "dynamic stride '" + layout.strides[i].ToString() +
          "': parameter times coordinate is not Presburger affine, so this "
          "layout has no isl_map; raise the Tier instead of approximating it");
    ClosedForm extent = layout.extents[i].Substitute(known);
    CollectFree(layout.extents[i], known, &params);

    if (i) offset << " + ";
    offset << "(" << stride.ToIslText() << " * " << coordinate << ")";
    bounds.push_back("0 <= " + coordinate);
    bounds.push_back(coordinate + " < " + extent.ToIslText());
  }

  std::ostringstream text;
  if (!params.empty()) {
    text << "[";
    for (std::size_t i = 0; i < params.size(); ++i) {
      if (i) text << ",";
      text << params[i];
    }
    text << "] -> ";
  }
  text << "{ [";
  for (std::size_t i = 0; i < coordinates.size(); ++i) {
    if (i) text << ",";
    text << coordinates[i];
  }
  text << "] -> [o] : o = " << offset.str();
  for (auto const& bound : bounds) text << " and " << bound;
  text << " }";
  return CouplingRelation::FromIslText(text.str());
}

LayoutDescriptor CuteLayoutBridge::FromIslMap(CouplingRelation const& map) const {
  if (map.empty()) throw std::invalid_argument("empty layout map");
  isl_ctx* ctx = SharedIslContext().raw();
  isl_util::Map raw = isl_util::ReadMap(ctx, map.ToString());
  if (isl_map_dim(raw.get(), isl_dim_out) != 1)
    throw std::domain_error("a layout map must have exactly one output (the "
                            "linear offset)");
  isl_bool single = isl_map_is_single_valued(raw.get());
  if (single != isl_bool_true)
    throw std::domain_error("layout map is not single valued, so it is not a "
                            "layout function");

  isl_size rank = isl_map_dim(raw.get(), isl_dim_in);
  // Extents come from the domain box. A solver's chosen layout is concrete,
  // so a non-literal bound is refused rather than guessed (see the header).
  isl_util::Set domain(isl_map_domain(isl_map_copy(raw.get())));
  LayoutDescriptor layout;
  for (int i = 0; i < rank; ++i) {
    isl_util::Val low(isl_set_dim_min_val(isl_set_copy(domain.get()), i));
    isl_util::Val high(isl_set_dim_max_val(isl_set_copy(domain.get()), i));
    if (!low || !high || !isl_val_is_int(low.get()) || !isl_val_is_int(high.get()))
      throw std::domain_error(
          "layout map dimension " + std::to_string(i) +
          " has no literal bounds; a solver result must be concrete");
    if (isl_val_get_num_si(low.get()) != 0)
      throw std::domain_error("layout map dimension " + std::to_string(i) +
                              " does not start at 0");
    layout.extents.push_back(
        ClosedForm::Constant(isl_val_get_num_si(high.get()) + 1));
  }

  // Strides are the affine coefficients of the single output.
  isl_util::PwAff offset(isl_pw_multi_aff_get_pw_aff(
      isl_util::Obj<isl_pw_multi_aff, isl_pw_multi_aff_copy,
                    isl_pw_multi_aff_free>(
          isl_pw_multi_aff_from_map(raw.release()))
          .get(),
      0));
  // Not isl_pw_aff_as_aff: that wants a *total* function, and a layout is
  // defined only on its own box, so it is always partial. What matters is
  // that it has exactly one piece -- one affine stride map, not a case
  // split -- and that piece's isl_aff carries the strides.
  if (isl_pw_aff_n_piece(offset.get()) != 1)
    throw std::domain_error("layout map is piecewise, so it is not one "
                            "affine stride map");
  isl_util::Obj<isl_aff, isl_aff_copy, isl_aff_free> affine;
  auto take_piece = [](isl_set* set, isl_aff* aff, void* user) -> isl_stat {
    *static_cast<isl_util::Obj<isl_aff, isl_aff_copy, isl_aff_free>*>(user) =
        isl_util::Obj<isl_aff, isl_aff_copy, isl_aff_free>(aff);
    isl_set_free(set);
    return isl_stat_ok;
  };
  if (isl_pw_aff_foreach_piece(offset.get(), take_piece, &affine) != isl_stat_ok ||
      !affine)
    throw std::domain_error("could not read the layout map's affine piece");
  for (int i = 0; i < rank; ++i) {
    isl_util::Val stride(
        isl_aff_get_coefficient_val(affine.get(), isl_dim_in, i));
    if (!stride || !isl_val_is_int(stride.get()))
      throw std::domain_error("layout map has a non-integer stride");
    layout.strides.push_back(
        ClosedForm::Constant(isl_val_get_num_si(stride.get())));
  }
  layout.static_tile_shape = true;
  return layout;
}

Tier TierClassifier::Classify(CouplingRelation const& relation) const {
  if (relation.empty()) return Tier::kDataDependent;
  return relation.IsSingleValued() ? Tier::kAffine : Tier::kStructuredRagged;
}
std::vector<EventRequirement> EventSynthesis::Synthesize(
    std::vector<CouplingEdge> const& edges) const {
  std::vector<EventRequirement> result;
  result.reserve(edges.size());
  for (auto const& edge : edges) {
    auto shape = edge.event_shape;
    if (shape.empty() && edge.tier == Tier::kDataDependent)
      shape.push_back(ClosedForm::Constant(1));  // operator-level barrier
    if (shape.empty())
      throw std::invalid_argument("non-Tier-3 coupling has empty image(C_kappa): " +
                                  edge.src.name + " -> " + edge.dst.name);
    result.push_back({edge.src, edge.dst, std::move(shape),
                      edge.metrics.wait, edge.metrics.fanout,
                      edge.metrics.count, edge.tier, edge.sync,
                      edge.exact, edge.guard});
  }
  return result;
}
}  // namespace tilemega::analysis
