// RUN: tilemega-opt %s | tilemega-opt | FileCheck %s
module attributes {tilemega.theta = {S = 4 : i64}, tilemega.g = {Tm = 1 : i64}} {
  tilemega.task_space @norm {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"rmsnorm">, operator_name = "aten.mul.Tensor", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.task_space @q {granularity = {Tm = 1 : i64}, kind = #tilemega.task_kind<"gemm">, operator_name = "aten.linear.default", stage = 1 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.event_tensor @e0 : tensor<1xi32> {extent = #tilemega.closed_form<"1">}
  tilemega.coupling @c0 from @norm to @q {count = #tilemega.closed_form<"S">, event = @e0, fanout = #tilemega.closed_form<"1">, read_map = #tilemega.access_map<{kind = "projection"}>, relation = #tilemega.coupling_map<{consumer = ["m", "n"], fiber = #tilemega.closed_form<"1">, image = #tilemega.closed_form<"1">, parameters = ["Tm"], producers = [{coordinates = ["m"], ranges = [], source = "norm"}]}>, sync_kind = #tilemega.sync<"global">, tier = #tilemega.tier<0>, volume = #tilemega.closed_form<"1">, wait = #tilemega.closed_form<"1">}
  tilemega.placement @norm map = [0] cluster = 1
  tilemega.placement @q map = [0] cluster = 1
}
// CHECK: tilemega.coupling @c0 from @norm to @q
// CHECK-SAME: wait = #tilemega.closed_form<"1">
