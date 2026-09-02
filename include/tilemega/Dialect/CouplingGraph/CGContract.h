// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.3 implementation selection, P4.3 contract verification.
#pragma once

#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Solver/ImplementationContract.h>

#include <mlir/IR/Builders.h>

#include <string>
#include <vector>

namespace tilemega::dialect {

/// One operand's indexing, as `#tilemega.access_map<{operand, coordinates,
/// spans}>`. The same encoding is used on both sides of the §5 check: the
/// task space's derived `index_map` and the implementation's declared
/// `access`, so the verifier compares like with like.
AccessMapAttr emitOperandContract(mlir::Builder& builder,
                                  solver::OperandContract const& operand);
mlir::ArrayAttr emitOperandContracts(
    mlir::Builder& builder, std::vector<solver::OperandContract> const& operands);

/// The task space's own indexing map, evaluated under (theta, g).
bool emitIndexMap(mlir::Builder& builder,
                  std::vector<analysis::AccessRelation> const& derived,
                  analysis::ParamBinding const& theta,
                  analysis::ParamBinding const& g, mlir::ArrayAttr* out,
                  std::string* error);

bool readOperandContracts(mlir::ArrayAttr array,
                          std::vector<solver::OperandContract>* out,
                          std::string* error);

/// The op read back as the solver's own value, so trait and access checking
/// is the same code the solver ran when it emitted the contract.
bool readImplementation(ImplementationOp op,
                        solver::ImplementationContract* out,
                        std::string* error);

/// Build the op from a contract the solver chose.
ImplementationOp emitImplementation(mlir::OpBuilder& builder, mlir::Location loc,
                                    solver::ImplementationContract const& impl);

}  // namespace tilemega::dialect
