// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1-5 and §4.3.
include "tilemega/Dialect/CouplingGraph/CGAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def CG_TaskSpaceOp : CG_Op<"task_space", [Symbol]> {
  let summary = "one tiled ATen task space";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       CG_TaskKindAttr:$kind,
                       DictionaryAttr:$granularity,
                       CG_AccessMapAttr:$write_map,
                       I64Attr:$stage,
                       StrAttr:$operator_name);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def CG_EventTensorOp : CG_Op<"event_tensor", [Symbol]> {
  let summary = "event tensor whose extent is image(C_kappa)";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$event_type,
                       CG_MetricAttr:$extent);
  let assemblyFormat = "$sym_name `:` $event_type attr-dict";
  let hasVerifier = 1;
}

def CG_CouplingOp : CG_Op<"coupling", [Symbol]> {
  let summary = "consumer-to-producer structured coupling";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$src,
                       FlatSymbolRefAttr:$dst,
                       CG_AccessMapAttr:$read_map,
                       CG_CouplingMapAttr:$relation,
                       CG_MetricAttr:$wait,
                       CG_MetricAttr:$fanout,
                       CG_MetricAttr:$volume,
                       CG_MetricAttr:$count,
                       CG_TierAttr:$tier,
                       // Optional so that IR written before the five
                       // attributes existed still verifies; when present the
                       // verifier requires `tier` to be exactly what they
                       // imply.
                       OptionalAttr<CG_CouplingAttributesAttr>:$coupling_attrs,
                       CG_SyncKindAttr:$sync_kind,
                       FlatSymbolRefAttr:$event);
  let assemblyFormat = "$sym_name `from` $src `to` $dst attr-dict";
  let hasVerifier = 1;
}

def CG_PlacementOp : CG_Op<"placement", []> {
  let summary = "fixed worker placement for one task space";
  let arguments = (ins FlatSymbolRefAttr:$task,
                       DenseI64ArrayAttr:$map,
                       I64Attr:$cluster);
  let assemblyFormat = "$task `map` `=` $map `cluster` `=` $cluster attr-dict";
  let hasVerifier = 1;
}
