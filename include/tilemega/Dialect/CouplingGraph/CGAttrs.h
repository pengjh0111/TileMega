// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 2-5 and invariant I1.
#pragma once

#include <tilemega/Analysis/AffineRelation.h>
#include <tilemega/Analysis/ClosedForm.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>

namespace tilemega::dialect {
mlir::FailureOr<analysis::ClosedForm> parseClosedForm(mlir::AsmParser& parser);
void printClosedForm(mlir::AsmPrinter& printer,
                     analysis::ClosedForm const& value);
}  // namespace tilemega::dialect

#define GET_ATTRDEF_CLASSES
#include "tilemega/Dialect/CouplingGraph/CGAttrs.h.inc"
