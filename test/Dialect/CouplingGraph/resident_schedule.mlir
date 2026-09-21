// RUN: tilemega-opt %s | tilemega-opt | FileCheck %s
module {
  tmcg.tile_space @p {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmexec.placement @p map = [0] cluster = 1 {resident_only = true}
}
// CHECK: tmexec.placement @p map = [0] cluster = 1 {resident_only = true}
