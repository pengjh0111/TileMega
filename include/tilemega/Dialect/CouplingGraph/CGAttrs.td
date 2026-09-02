// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 2-5 and invariant I1.
include "tilemega/Dialect/CouplingGraph/CGDialect.td"
include "mlir/IR/AttrTypeBase.td"

def CG_TaskKindAttr : CG_Attr<"TaskKind"> {
  let mnemonic = "task_kind";
  let parameters = (ins "::mlir::StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}

// wait/fanout/volume/count and an event tensor's extent are all barvinok
// counts over a CouplingRelation (isl_pw_qpolynomial), not ClosedForm AST
// values: they are *solved* quantities, and isl/barvinok is the sole
// solving authority after the Part 3 migration (see docs/DEPENDENCIES.md,
// docs/experiments/P3_ISL/). The mnemonic stays close to the pre-migration
// name to keep the IR text readable; the payload type is what changed.
def CG_MetricParameter
    : AttrParameter<"::tilemega::analysis::QuasiPolynomial",
                    "a piecewise quasi-polynomial (theta) solved from a "
                    "CouplingRelation",
                    "::tilemega::analysis::QuasiPolynomial const &"> {
  let parser = "::tilemega::dialect::parseMetric($_parser)";
  let printer = "::tilemega::dialect::printMetric($_printer, $_self)";
}

def CG_MetricAttr : CG_Attr<"Metric"> {
  let mnemonic = "metric";
  let parameters = (ins CG_MetricParameter:$expression);
  let assemblyFormat = "`<` $expression `>`";
  let extraClassDeclaration = [{
    analysis::QuasiPolynomial getValue() const { return getExpression(); }
  }];
}

// The coupling relation C = W^-1 o R is a genuine isl_map (CouplingRelation,
// see include/tilemega/Analysis/CouplingRelation.h), stored as isl map
// syntax text. It is no longer a schema-checked DictionaryAttr: isl_map
// syntax is itself the schema, parsed and validated by isl at attribute
// parse time (an invalid map fails to parse, which is a stronger guarantee
// than the old hand-rolled DictionaryAttr field check ever gave).
def CG_CouplingRelationParameter
    : AttrParameter<"::tilemega::analysis::CouplingRelation",
                    "C = W^-1 o R as an isl_map",
                    "::tilemega::analysis::CouplingRelation const &"> {
  let parser = "::tilemega::dialect::parseCouplingRelation($_parser)";
  let printer = "::tilemega::dialect::printCouplingRelation($_printer, $_self)";
}

def CG_CouplingMapAttr : CG_Attr<"CouplingMap"> {
  let mnemonic = "coupling_map";
  // Named `map`, not `relation`: CouplingOp's own operand of this attribute
  // type is itself named `relation` (CG_CouplingMapAttr:$relation below), so
  // a caller writes `op.getRelation().getMap()` -- the coupling op's
  // relation *attribute*, then that attribute's isl_map *value* -- instead
  // of the confusing `getRelation().getRelation()` a same-named parameter
  // would produce.
  let parameters = (ins CG_CouplingRelationParameter:$map);
  let assemblyFormat = "`<` $map `>`";
}

def CG_AccessMapAttr : CG_Attr<"AccessMap"> {
  let mnemonic = "access_map";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
}

def CG_TierAttr : CG_Attr<"Tier"> {
  let mnemonic = "tier";
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = "`<` $value `>`";
}

// §2.4's tier is a summary of five independent observations (see
// CouplingDerivation.h). Carrying the five in the IR is what lets the
// verifier recompute the tier instead of trusting it: an edge whose tier
// does not follow from its own attributes is rejected.
def CG_CouplingAttributesAttr : CG_Attr<"CouplingAttributes"> {
  let mnemonic = "coupling_attrs";
  let parameters = (ins "::mlir::StringAttr":$relation_kind,
                        "::mlir::StringAttr":$extent_kind,
                        "::mlir::StringAttr":$exactness,
                        "::mlir::StringAttr":$runtime_requirement,
                        "::mlir::StringAttr":$countability);
  let assemblyFormat = "`<` $relation_kind `,` $extent_kind `,` $exactness `,` "
                       "$runtime_requirement `,` $countability `>`";
}

def CG_SyncKindAttr : CG_Attr<"SyncKind"> {
  let mnemonic = "sync";
  let parameters = (ins "::mlir::StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}
