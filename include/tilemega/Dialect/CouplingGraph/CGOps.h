// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1-5 and §4.3.
#pragma once

#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#define GET_OP_CLASSES
#include "tilemega/Dialect/CouplingGraph/CGOps.h.inc"

namespace tilemega::dialect {

/// Read `kPlacementTableAttr` off a CG module, or leave `*table` empty when the
/// attribute is absent.  False, with `*error` set, for a malformed one; the
/// verifier and codegen both go through here so a table cannot mean two things.
bool ReadPlacementTable(::mlir::ModuleOp module, PlacementTable* table,
                        ::std::string* error);

}  // namespace tilemega::dialect
