// RUN: tilemega-opt %s -verify-diagnostics
module {
  tilemega.task_space @p {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  // expected-error@+1 {{resident_only must be true; over-resident proof is unavailable}}
  tilemega.placement @p map = [0] cluster = 1 {resident_only = false}
}
