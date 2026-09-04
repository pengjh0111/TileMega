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
                       StrAttr:$operator_name,
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

def CG_PlacementOp : CG_Op<"placement", []> {
  let summary = "fixed worker placement for one task space";
  let arguments = (ins FlatSymbolRefAttr:$task,
                       DenseI64ArrayAttr:$map,
                       I64Attr:$cluster);
  let assemblyFormat = "$task `map` `=` $map `cluster` `=` $cluster attr-dict";
  let hasVerifier = 1;
}

def CG_ImplementationOp : CG_Op<"implementation", [Symbol]> {
  let summary = "one backend instantiation bound to a task space";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$task,
                       StrAttr:$backend,
                       DenseI64ArrayAttr:$tile,
                       DenseI64ArrayAttr:$cluster,
                       I64Attr:$stages,
                       I64Attr:$threads,
                       I64Attr:$smem_bytes,
                       // Absent until a tier-3 ptxas log fills it: CUTLASS's
                       // constexpr traits carry no register count, so an
                       // always-present field could only hold a guess.
                       OptionalAttr<I64Attr>:$regs_est,
                       DenseI64ArrayAttr:$alignment,
                       I64Attr:$arch_required,
                       ArrayAttr:$access);
  let assemblyFormat = "$sym_name `for` $task attr-dict";
  let hasVerifier = 1;
}
