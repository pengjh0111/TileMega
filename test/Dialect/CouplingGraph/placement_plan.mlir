// RUN: tilemega-opt %s | tilemega-opt | FileCheck %s
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // Absent plan attributes are the legacy_grid_stride form map = [0] has
  // always denoted.
  tmexec.placement @p map = [0] cluster = 1
  tmexec.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
  tmexec.placement @p map = [0] cluster = 1 {mapping_mode = "balanced", mode = "balanced", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
  tmexec.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64: 1>, policy = "aot", resident_only = true, window = 1 : i64}
  tmexec.placement @p map = [0] cluster = 1 {mode = "template", params_map = #tmcg.coupling_map<"[S] -> { [] -> [floor(S/129)] : 1 <= S <= 256 }">, grid_map = #tmcg.coupling_map<"[S] -> { [] -> [128] : 1 <= S <= 256 }">, resident_limit_map = #tmcg.coupling_map<"[S] -> { [] -> [256] : 1 <= S <= 256 }">, resident_only = true}
  // eft is materialized, so its (pi, sigma) is a module attribute and the mode
  // may not appear without it.  It gets its own module because the table and a
  // closed-form mode are a contradiction in the same scope.
  module attributes {tmexec.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 0>, worker = array<i64: 0, 1>}} {
    tmcg.tile_space @q {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
    tmexec.placement @q map = [0] cluster = 1 {mode = "eft", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
  }
}
// CHECK: tmexec.placement @p map = [0] cluster = 1
// CHECK: tmexec.placement @p map = [0] cluster = 1 {mode = "rotate", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
// CHECK: tmexec.placement @p map = [0] cluster = 1 {mapping_mode = "balanced", mode = "balanced", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
// CHECK: tmexec.placement @p map = [0] cluster = 1 {mode = "template", params = array<i64: 1>, policy = "aot", resident_only = true, window = 1 : i64}
// CHECK: params_map = #tmcg.coupling_map<
// CHECK: module attributes {tmexec.placement_table = {grid = 2 : i64, past = 3 : i64, seq = 4 : i64, slot = array<i64: 0, 0>, worker = array<i64: 0, 1>}}
// CHECK: tmexec.placement @q map = [0] cluster = 1 {mode = "eft", params = array<i64>, policy = "aot", resident_only = true, window = 1 : i64}
