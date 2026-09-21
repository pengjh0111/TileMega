// RUN: tilemega-opt %s | tilemega-opt | FileCheck %s
module attributes {tilemega.theta = {S = 4 : i64}, tilemega.g = {Tm = 1 : i64}} {
  tmcg.tile_space @norm {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"rmsnorm">, operator_name = "aten.mul.Tensor", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmcg.tile_space @q {granularity = {Tm = 1 : i64}, kind = #tmcg.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 1 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmcg.event_tensor @e0 : tensor<1xi32> {extent = #tmcg.metric<"{ 1 }">}
  // A triangular relation, so wait(x) is a genuine function of the consumer
  // coordinate: a verifier that collapsed either side to a scalar would
  // accept a constant wait here. For producer j, i ranges from j to 3,
  // hence fanout(j)=4-j; both totals are 10, not four constant fanouts.
  tmcg.coupling @c0 from @norm to @q {count = #tmcg.metric<"{ 4 }">, event = @e0, fanout = #tmcg.metric<"{ [j] -> 4-j : 0 <= j <= 3 }">, read_map = #tmcg.access_map<{kind = "projection"}>, relation = #tmcg.coupling_map<"{ [i] -> [j] : 0 <= j <= i <= 3 }">, sync_kind = #tmcg.sync<"global">, tier = #tmcg.tier<0>, volume = #tmcg.metric<"{ 1 }">, wait = #tmcg.metric<"{ [i] -> i + 1 : 0 <= i <= 3 }">}
  tmexec.placement @norm map = [0] cluster = 1
  tmexec.placement @q map = [0] cluster = 1
}
// CHECK: tmcg.coupling @c0 from @norm to @q
// CHECK-SAME: relation = #tmcg.coupling_map<"{ [i] -> [j] : i <= 3 and 0 <= j <= i }">
// CHECK-SAME: wait = #tmcg.metric<"{ [i] -> (1 + i) : 0 <= i <= 3 }">
