// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 2-5 and invariant I1.
include "tilemega/Dialect/CouplingGraph/CGDialect.td"
include "mlir/IR/AttrTypeBase.td"

def CG_TaskKindAttr : CG_Attr<"TaskKind"> {
  let mnemonic = "task_kind";
  let parameters = (ins "::mlir::StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}

def CG_ClosedFormParameter
    : AttrParameter<"::tilemega::analysis::ClosedForm",
                    "a symbolic (theta,g) closed form",
                    "::tilemega::analysis::ClosedForm const &"> {
  let parser = "::tilemega::dialect::parseClosedForm($_parser)";
  let printer = "::tilemega::dialect::printClosedForm($_printer, $_self)";
}

def CG_ClosedFormAttr : CG_Attr<"ClosedForm"> {
  let mnemonic = "closed_form";
  let parameters = (ins CG_ClosedFormParameter:$expression);
  let assemblyFormat = "`<` $expression `>`";
  let extraClassDeclaration = [{
    analysis::ClosedForm getValue() const { return getExpression(); }
  }];
}

// Access maps and coupling maps use DictionaryAttr as their storage key.  The
// dictionary is schema-checked in C++ and is therefore a structured MLIR value,
// not an opaque textual relation.
def CG_AccessMapAttr : CG_Attr<"AccessMap"> {
  let mnemonic = "access_map";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
}

def CG_CouplingMapAttr : CG_Attr<"CouplingMap"> {
  let mnemonic = "coupling_map";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifyStructure(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
    analysis::AffineRelation getRelation() const;
    ClosedFormAttr getFiberCardinality() const;
    ClosedFormAttr getImageCardinality() const;
  }];
}

def CG_TierAttr : CG_Attr<"Tier"> {
  let mnemonic = "tier";
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = "`<` $value `>`";
}

def CG_SyncKindAttr : CG_Attr<"SyncKind"> {
  let mnemonic = "sync";
  let parameters = (ins "::mlir::StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}
