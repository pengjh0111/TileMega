// RUN: ! tilemega-opt %s 2>&1 | FileCheck %s
module attributes {tilemega.theta = {}, tilemega.g = {}} {
  tmcg.tile_space @a {granularity = {}, kind = #tmcg.task_kind<"elementwise">, operator_name = "aten.add.Tensor", stage = 0 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmcg.tile_space @b {granularity = {}, kind = #tmcg.task_kind<"elementwise">, operator_name = "aten.mul.Tensor", stage = 1 : i64, write_map = #tmcg.access_map<{kind = "identity"}>}
  tmcg.event_tensor @e : tensor<1xi32> {extent = #tmcg.metric<"{ 1 }">}
  tmcg.coupling @c from @a to @b {count = #tmcg.metric<"{ 1 }">, event = @e, fanout = #tmcg.metric<"{ 1 }">, read_map = #tmcg.access_map<{kind = "identity"}>, relation = #tmcg.coupling_map<"{ [0] -> [0] }">, sync_kind = #tmcg.sync<"cluster">, tier = #tmcg.tier<3>, volume = #tmcg.metric<"{ 1 }">, wait = #tmcg.metric<"{ 1 }">}
}
// CHECK: Tier 3 coupling cannot use cluster synchronization
