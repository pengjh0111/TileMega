// RUN: ! tilemega-opt %s 2>&1 | FileCheck %s
module attributes {tilemega.theta = {}, tilemega.g = {}} {
  tilemega.task_space @a {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "aten.add.Tensor", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.task_space @b {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "aten.mul.Tensor", stage = 1 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.event_tensor @e : tensor<1xi32> {extent = #tilemega.metric<"{ 1 }">}
  tilemega.coupling @c from @a to @b {count = #tilemega.metric<"{ 1 }">, event = @e, fanout = #tilemega.metric<"{ 1 }">, read_map = #tilemega.access_map<{kind = "identity"}>, relation = #tilemega.coupling_map<"{ [0] -> [0] }">, sync_kind = #tilemega.sync<"cluster">, tier = #tilemega.tier<3>, volume = #tilemega.metric<"{ 1 }">, wait = #tilemega.metric<"{ 1 }">}
}
// CHECK: Tier 3 coupling cannot use cluster synchronization
