// RUN: tilemega-opt %s | tilemega-opt | FileCheck %s
module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // Absent plan attributes are the legacy_grid_stride form map = [0] has
  // always denoted.
  tilemega.placement @p map = [0] cluster = 1
  tilemega.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
  tilemega.placement @p map = [0] cluster = 1 {mapping_mode = "balanced", mode = "balanced", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
  tilemega.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64: 1>, policy = "aot", resident_only = true, window = 1 : i64}
  // eft is materialized, so its (pi, sigma) is a module attribute and the mode
  // may not appear without it.  It gets its own module because the table and a
  // closed-form mode are a contradiction in the same scope.
  module attributes {tilemega.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 0>, worker = array<i64: 0, 1>}} {
    tilemega.task_space @q {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
    tilemega.placement @q map = [0] cluster = 1 {mode = "eft", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
  }
}
// CHECK: tilemega.placement @p map = [0] cluster = 1
// CHECK: tilemega.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
// CHECK: tilemega.placement @p map = [0] cluster = 1 {mapping_mode = "balanced", mode = "balanced", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
// CHECK: tilemega.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64: 1>, policy = "aot", resident_only = true, window = 1 : i64}
// CHECK: module attributes {tilemega.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 0>, worker = array<i64: 0, 1>}}
// CHECK: tilemega.placement @q map = [0] cluster = 1 {mode = "eft", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
