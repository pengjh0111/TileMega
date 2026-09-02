// RUN: ! tilemega-opt %s 2>&1 | FileCheck %s
// The stored wait is checked against card(C) recomputed from the relation
// itself, as a function of the consumer coordinate. Here the relation's
// fibers have sizes 1..4 and the declared wait is the constant 2, which
// agrees with neither the minimum, the maximum, nor the mean.
module attributes {tilemega.theta = {}, tilemega.g = {}} {
  tilemega.task_space @a {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "aten.add.Tensor", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.task_space @b {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "aten.mul.Tensor", stage = 1 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.event_tensor @e : tensor<1xi32> {extent = #tilemega.metric<"{ 1 }">}
  tilemega.coupling @c from @a to @b {count = #tilemega.metric<"{ 4 }">, event = @e, fanout = #tilemega.metric<"{ 1 }">, read_map = #tilemega.access_map<{kind = "identity"}>, relation = #tilemega.coupling_map<"{ [i] -> [j] : 0 <= j <= i <= 3 }">, sync_kind = #tilemega.sync<"global">, tier = #tilemega.tier<0>, volume = #tilemega.metric<"{ 1 }">, wait = #tilemega.metric<"{ [i] -> 2 : 0 <= i <= 3 }">}
}
// CHECK: wait { [i] -> 2 : 0 <= i <= 3 } does not match the relation's fiber cardinality { [i] -> (1 + i) : 0 <= i <= 3 }
