// RUN: tilemega-opt %s -split-input-file -verify-diagnostics
module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{unknown placement mode host_list_schedule}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "host_list_schedule", resident_only = true}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // A named mode is a solver decision, and none of them has an over-resident
  // proof.
  // expected-error@+1 {{placement mode rotate requires resident_only=true}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "rotate"}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{placement mode template takes 1 parameters}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64>, resident_only = true}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{unknown placement template 7}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64: 7>, resident_only = true}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{placement mode rotate takes 0 parameters}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64: 3>, resident_only = true}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{placement window must be positive}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "rotate", resident_only = true, window = 0 : i64}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // jit is reserved for duration-dependent stages and is not implemented.
  // expected-error@+1 {{unknown placement policy jit}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "rotate", policy = "jit", resident_only = true}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{mapping_mode=balanced contradicts the placement mode}}
  tilemega.placement @p map = [0] cluster = 1 {mapping_mode = "balanced", mode = "rotate", resident_only = true}
}

// -----

module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // eft prices task durations; nothing below the solver can recompute it, so
  // the mode without its table names a schedule that cannot be produced.
  // expected-error@+1 {{placement mode eft needs its materialized table}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "eft", params = array<i64>, resident_only = true}
}

// -----

module attributes {tilemega.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 0>, worker = array<i64: 0, 1>}} {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // The other direction: a table a closed form would silently drop.
  // expected-error@+1 {{needs placement mode eft, not rotate}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64>, resident_only = true}
}

// -----

module attributes {tilemega.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 1>, worker = array<i64: 0, 1>}} {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // sigma must be a dense [0, n) per worker or the queue it describes has no
  // total order (§5.7.2).
  // expected-error@+1 {{puts node 1 at slot 1 of a queue of 1}}
  tilemega.placement @p map = [0] cluster = 1 {mode = "eft", params = array<i64>, resident_only = true}
}
