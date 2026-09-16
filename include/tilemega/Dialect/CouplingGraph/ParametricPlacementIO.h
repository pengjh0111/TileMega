// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Solver/ParametricPlacement.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
namespace tilemega::dialect {
// Alternative proven template maps are carried by CG, independently of the
// selected finite materialization. Never replace an unrepresentable winner.
inline mlir::DictionaryAttr ParametricPlacementAttr(mlir::MLIRContext* context,
    solver::ParametricPlacement const& plan) {
  mlir::Builder b(context);mlir::NamedAttrList attrs;
  attrs.set("family",b.getStringAttr(plan.family));
  attrs.set("tasks",CouplingMapAttr::get(context,plan.tasks));
  attrs.set("dependencies",CouplingMapAttr::get(context,plan.dependencies));
  attrs.set("pi_sigma",CouplingMapAttr::get(context,plan.pi_sigma));
  attrs.set("rank",CouplingMapAttr::get(context,plan.rank));
  attrs.set("grid_limit",CouplingMapAttr::get(context,plan.grid_limit));
  return attrs.getDictionary(context);
}
inline solver::ParametricPlacement ReadParametricPlacement(mlir::DictionaryAttr attrs) {
  auto get=[&](char const* name) {
    auto attr=attrs.getAs<CouplingMapAttr>(name);
    if (!attr) throw std::invalid_argument(std::string("missing symbolic placement map: ")+name);
    return attr.getMap();
  };
  auto family=attrs.getAs<mlir::StringAttr>("family");
  if (!family)throw std::invalid_argument("missing symbolic placement family");
  return {get("tasks"),get("dependencies"),get("pi_sigma"),get("rank"),get("grid_limit"),family.str()};
}
} // namespace tilemega::dialect
