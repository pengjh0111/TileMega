// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 2-5 and invariant I1.
#pragma once

#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>

namespace tilemega::dialect {
mlir::FailureOr<analysis::QuasiPolynomial> parseMetric(mlir::AsmParser& parser);
void printMetric(mlir::AsmPrinter& printer, analysis::QuasiPolynomial const& value);
mlir::FailureOr<analysis::CouplingRelation> parseCouplingRelation(
    mlir::AsmParser& parser);
void printCouplingRelation(mlir::AsmPrinter& printer,
                           analysis::CouplingRelation const& value);
}  // namespace tilemega::dialect

#define GET_ATTRDEF_CLASSES
#include "tilemega/Dialect/CouplingGraph/CGAttrs.h.inc"
