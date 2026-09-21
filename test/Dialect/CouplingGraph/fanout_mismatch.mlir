// RUN: ! tilemega-opt %s 2>&1 | FileCheck %s
// A valid wait must not hide a wrong fanout. C contains ten pairs, while
// this stored constant fanout sums to four on the same producer domain.
module attributes {tilemega.theta = {}, tilemega.g = {}} {
  tmcg.tile_space @a {granularity = {}, kind = #tmcg.task_kind<"elementwise">, operator_name = "a", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmcg.tile_space @b {granularity = {}, kind = #tmcg.task_kind<"elementwise">, operator_name = "b", stage = 1 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmcg.event_tensor @e : tensor<1xi32> {extent = #tmcg.metric<"{ 1 }">}
  tmcg.coupling @c from @a to @b {count = #tmcg.metric<"{ 4 }">, event = @e, fanout = #tmcg.metric<"{ [j] -> 1 : 0<=j<=3 }">, read_map = #tmcg.access_map<{kind = "identity"}>, relation = #tmcg.coupling_map<"{ [i] -> [j] : 0 <= j <= i <= 3 }">, sync_kind = #tmcg.sync<"global">, tier = #tmcg.tier<0>, volume = #tmcg.metric<"{ 1 }">, wait = #tmcg.metric<"{ [i] -> i+1 : 0<=i<=3 }">}
}
// CHECK: fanout does not match the inverse relation's fiber cardinality
