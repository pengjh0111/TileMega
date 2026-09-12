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
