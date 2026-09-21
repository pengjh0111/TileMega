// RUN: tilemega-opt %s -verify-diagnostics
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  // expected-error@+1 {{resident_only must be true; over-resident proof is unavailable}}
  tmexec.placement @p map = [0] cluster = 1 {resident_only = false}
}
