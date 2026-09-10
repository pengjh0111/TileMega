// RUN: ! tilemega-opt %s 2>&1 | FileCheck %s
// A valid wait must not hide a wrong fanout. C contains ten pairs, while
// this stored constant fanout sums to four on the same producer domain.
module attributes {tilemega.theta = {}, tilemega.g = {}} {
  tilemega.task_space @a {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "a", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.task_space @b {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "b", stage = 1 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.event_tensor @e : tensor<1xi32> {extent = #tilemega.metric<"{ 1 }">}
  tilemega.coupling @c from @a to @b {count = #tilemega.metric<"{ 4 }">, event = @e, fanout = #tilemega.metric<"{ [j] -> 1 : 0<=j<=3 }">, read_map = #tilemega.access_map<{kind = "identity"}>, relation = #tilemega.coupling_map<"{ [i] -> [j] : 0 <= j <= i <= 3 }">, sync_kind = #tilemega.sync<"global">, tier = #tilemega.tier<0>, volume = #tilemega.metric<"{ 1 }">, wait = #tilemega.metric<"{ [i] -> i+1 : 0<=i<=3 }">}
}
// CHECK: fanout does not match the inverse relation's fiber cardinality
