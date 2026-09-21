// RUN: tilemega-opt %s -split-input-file -verify-diagnostics
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{unknown placement mode host_list_schedule}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "host_list_schedule", resident_only = true}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // A named mode is a solver decision, and none of them has an over-resident
  // proof.
  // expected-error@+1 {{placement mode rotate requires resident_only=true}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "rotate"}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{placement mode template takes 1 parameters}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64>, resident_only = true}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{unknown placement template 7}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64: 7>, resident_only = true}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{placement mode rotate takes 0 parameters}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64: 3>, resident_only = true}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{placement window must be positive}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "rotate", resident_only = true, window = 0 : i64}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // jit is reserved for duration-dependent stages and is not implemented.
  // expected-error@+1 {{unknown placement policy jit}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "rotate", policy = "jit", resident_only = true}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{mapping_mode=balanced contradicts the placement mode}}
  tmexec.placement @p map = [0] cluster = 1 {mapping_mode = "balanced", mode = "rotate", resident_only = true}
}

// -----

module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // eft prices task durations; nothing below the solver can recompute it, so
  // the mode without its table names a schedule that cannot be produced.
  // expected-error@+1 {{placement mode eft needs its materialized table}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "eft", params = array<i64>, resident_only = true}
}

// -----

module attributes {tmexec.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 0>, worker = array<i64: 0, 1>}} {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // The other direction: a table a closed form would silently drop.
  // expected-error@+1 {{needs placement mode eft, not rotate}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64>, resident_only = true}
}

// -----

module attributes {tmexec.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 1>, worker = array<i64: 0, 1>}} {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // sigma must be a dense [0, n) per worker or the queue it describes has no
  // total order (§5.7.2).
  // expected-error@+1 {{puts node 1 at slot 1 of a queue of 1}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "eft", params = array<i64>, resident_only = true}
}

// -----
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{params_map must be a single-valued theta function}}
  tmexec.placement @p map = [0] cluster = 1 {mode = "template", params_map = #tmcg.coupling_map<"[S] -> { [] -> [p] : 0 <= p <= 1 and S > 0 }">, resident_only = true}
}

// -----
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{symbolic grid does not prove 0 < grid <= resident_limit}}
  tmexec.placement @p map = [0] cluster = 1 {grid_map = #tmcg.coupling_map<"[S] -> { [] -> [S] : 1 <= S <= 512 }">, resident_limit_map = #tmcg.coupling_map<"[S] -> { [] -> [256] : 1 <= S <= 512 }">, resident_only = true}
}

// -----
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{grid and resident limit theta domains differ}}
  tmexec.placement @p map = [0] cluster = 1 {grid_map = #tmcg.coupling_map<"[S] -> { [] -> [128] : 1 <= S <= 256 }">, resident_limit_map = #tmcg.coupling_map<"[S] -> { [] -> [256] : 1 <= S <= 128 }">, resident_only = true}
}
