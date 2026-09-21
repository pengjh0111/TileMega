// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1-5 and §4.3.
include "tilemega/Dialect/CouplingGraph/CGAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def CG_TileSpaceOp : CG_Op<"tile_space", [Symbol]> {
  let summary = "one tiled ATen tile space";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       CG_TaskKindAttr:$kind,
                       DictionaryAttr:$granularity,
                       CG_AccessMapAttr:$write_map,
                       I64Attr:$stage,
                       StrAttr:$operator_name,
                       OptionalAttr<StrAttr>:$arithmetic,
                       OptionalAttr<StrAttr>:$semantic,
                       // The derived indexing map, one entry per operand, in
                       // operand order, spans already evaluated under
                       // (theta, g). Optional so IR written before §5 still
                       // verifies; when present it is the truth an
                       // implementation contract is checked against.
                       OptionalAttr<ArrayAttr>:$index_map);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def CG_EventTensorOp : CG_Op<"event_tensor", [Symbol]> {
  let summary = "event tensor whose extent is image(C_kappa)";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$event_type,
                       CG_MetricAttr:$extent,
                       // image(C_kappa) axis by axis. Without it `extent` is
                       // one number no one can check against the shape.
                       OptionalAttr<ArrayAttr>:$dims);
  let assemblyFormat = "$sym_name `:` $event_type attr-dict";
  let hasVerifier = 1;
}

def CG_FusedTileSpaceOp : CG_Op<"fused_task_space", [Symbol]> {
  let summary = "consumer-indexed sequential L-task composition";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       ArrayAttr:$phase_semantics,
                       ArrayAttr:$phase_granularities,
                       DenseI64ArrayAttr:$phase_stages,
                       ArrayAttr:$phase_maps,
                       DictionaryAttr:$reads,
                       DictionaryAttr:$writes,
                       CG_MetricAttr:$task_count);
  let assemblyFormat = "$sym_name attr-dict";
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
                       // "all", "identity" or "window(div,scale,offset,count)"
                       // (Analysis/DependencyForm.h): the per-CTA wait set the
                       // generator narrows to. Optional, and absent means the
                       // I2 relaxation to the producer's whole launch axis.
                       OptionalAttr<StrAttr>:$wait_map,
                       CG_SyncKindAttr:$sync_kind,
                       FlatSymbolRefAttr:$event);
  let assemblyFormat = "$sym_name `from` $src `to` $dst attr-dict";
  let hasVerifier = 1;
}


// R8 BE-7: one Coupling Graph, named, holding the structure ops above. The
// region is a symbol table so a tile space is addressed inside its own graph
// rather than globally.
def CG_GraphOp : CG_Op<"graph", [Symbol, SymbolTable, NoTerminator,
                                 SingleBlock]> {
  let summary = "a complete parameterized Coupling Graph";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}
