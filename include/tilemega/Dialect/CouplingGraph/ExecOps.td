// SPDX-License-Identifier: BSD-3-Clause
// R8 BE-7: the execution half of TileMega IR. Everything here is a decision
// the solver made about a `tmcg.graph`; nothing here describes the graph.
include "tilemega/Dialect/CouplingGraph/CGAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def Exec_PlacementOp : Exec_Op<"placement", []> {
  let summary = "fixed worker placement for one tile space";
  let description = [{
    Carries the solver's Plan for one tile space (skeleton §5.7.1,
    `Plan = (pi, sigma, W, policy, sync, kappa)`).  `sync` and `kappa` ride on
    `tmcg.coupling` and the event tables, so only pi/sigma, the execution
    window and the dispatch policy live here.

    `mode` names how pi and sigma are produced; `params` is that mode's
    parameter vector.  All four plan attributes are optional, and absent means
    the `legacy_grid_stride` form that `map = [0]` has always denoted, so IR
    written before the plan existed still verifies unchanged.

    | mode                 | pi                                                        | params            |
    |----------------------|-----------------------------------------------------------|-------------------|
    | `legacy_grid_stride` | `task % grid`, queues stage-major                          | empty             |
    | `rotate`             | `(task + base[stage]) % grid`, `base` the prefix sum of the active task counts along the stage order, mod grid | empty |
    | `balanced`           | affinity-first greedy under a queue cap                    | empty             |
    | `eft`                | materialized: the explicit (worker, slot) table is the module attribute `tmexec.placement_table`, one object for the whole model at one bound theta | empty |
    | `template`           | closed form selected by `params[0]`: 0 = band tiling, 1 = wavefront (F-93, F-97) | one entry |

    `eft` is the one mode that cannot be evaluated below L2 -- it prices task
    durations -- so it travels already solved, and the verifier requires the
    module table and the mode to appear together in both directions.  The table
    is pinned to the `seq`, `past` and grid it was scheduled for, and the host
    refuses it for any other.

    `params_map` is an alternative to the constant `params` vector: an ISL
    single-valued map from an empty task tuple to the parameter tuple, with
    theta left symbolic. `grid_map` and `resident_limit_map` carry one output
    each over the same theta domain; their product must establish 0 < grid <=
    resident_limit. Constant legacy attributes remain valid without these maps.
    Bound code generation evaluates parameter maps using the module's
    `tmexec.placement_bindings` dictionary. Materialized tables remain one
    module-level CG attribute, avoiding a duplicate full table per task space.

    `window` is W.  The executor implements W = 1 only (strict FIFO); the
    verifier admits any positive W so that EX-E2 changes the executor rather
    than the contract, and codegen rejects W > 1 until then.
  }];
  let arguments = (ins FlatSymbolRefAttr:$task,
                       DenseI64ArrayAttr:$map,
                       I64Attr:$cluster,
                       OptionalAttr<StrAttr>:$mode,
                       OptionalAttr<DenseI64ArrayAttr>:$params,
                       OptionalAttr<I64Attr>:$window,
                       OptionalAttr<StrAttr>:$policy,
                       OptionalAttr<CG_CouplingMapAttr>:$params_map,
                       OptionalAttr<CG_CouplingMapAttr>:$grid_map,
                       OptionalAttr<CG_CouplingMapAttr>:$resident_limit_map);
  let assemblyFormat = "$task `map` `=` $map `cluster` `=` $cluster attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// The §5.7.1 Plan attributes, checked separately from the legacy map.
    ::mlir::LogicalResult verifyPlan();
  }];
}

def Exec_ImplementationOp : Exec_Op<"implementation", [Symbol]> {
  let summary = "one backend instantiation bound to a tile space";
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

// The plan that holds them, named for the graph it was solved on.
def Exec_PlanOp : Exec_Op<"plan", [Symbol, SymbolTable, NoTerminator,
                                   SingleBlock]> {
  let summary = "the execution plan solved for one Coupling Graph";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$graph);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `for` $graph attr-dict-with-keyword $body";
}
