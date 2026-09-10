// RUN: tilemega-opt %s | tilemega-opt | FileCheck %s
module attributes {tilemega.theta = {S = 4 : i64}, tilemega.g = {Tm = 1 : i64}} {
  tilemega.task_space @norm {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"rmsnorm">, operator_name = "aten.mul.Tensor", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.task_space @q {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 1 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.event_tensor @e0 : tensor<1xi32> {extent = #tilemega.metric<"{ 1 }">}
  // A triangular relation, so wait(x) is a genuine function of the consumer
  // coordinate: a verifier that collapsed either side to a scalar would
  // accept a constant wait here. For producer j, i ranges from j to 3,
  // hence fanout(j)=4-j; both totals are 10, not four constant fanouts.
  tilemega.coupling @c0 from @norm to @q {count = #tilemega.metric<"{ 4 }">, event = @e0, fanout = #tilemega.metric<"{ [j] -> 4-j : 0 <= j <= 3 }">, read_map = #tilemega.access_map<{kind = "projection"}>, relation = #tilemega.coupling_map<"{ [i] -> [j] : 0 <= j <= i <= 3 }">, sync_kind = #tilemega.sync<"global">, tier = #tilemega.tier<0>, volume = #tilemega.metric<"{ 1 }">, wait = #tilemega.metric<"{ [i] -> i + 1 : 0 <= i <= 3 }">}
  tilemega.placement @norm map = [0] cluster = 1
  tilemega.placement @q map = [0] cluster = 1
}
// CHECK: tilemega.coupling @c0 from @norm to @q
// CHECK-SAME: relation = #tilemega.coupling_map<"{ [i] -> [j] : i <= 3 and 0 <= j <= i }">
// CHECK-SAME: wait = #tilemega.metric<"{ [i] -> (1 + i) : 0 <= i <= 3 }">
