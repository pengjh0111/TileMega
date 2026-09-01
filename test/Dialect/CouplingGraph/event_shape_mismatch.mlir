// RUN: ! tilemega-opt %s 2>&1 | FileCheck %s
module attributes {tilemega.theta = {}, tilemega.g = {}} {
  tilemega.task_space @a {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "aten.add.Tensor", stage = 0 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.task_space @b {granularity = {}, kind = #tilemega.task_kind<"elementwise">, operator_name = "aten.mul.Tensor", stage = 1 : i64, write_map = #tilemega.access_map<{kind = "identity"}>}
  tilemega.event_tensor @e : tensor<2xi32> {extent = #tilemega.closed_form<"1">}
  tilemega.coupling @c from @a to @b {count = #tilemega.closed_form<"1">, event = @e, fanout = #tilemega.closed_form<"1">, read_map = #tilemega.access_map<{kind = "identity"}>, relation = #tilemega.coupling_map<{consumer = ["m"], fiber = #tilemega.closed_form<"1">, image = #tilemega.closed_form<"1">, parameters = [], producers = [{coordinates = ["m"], ranges = [], source = "a"}]}>, sync_kind = #tilemega.sync<"global">, tier = #tilemega.tier<0>, volume = #tilemega.closed_form<"1">, wait = #tilemega.closed_form<"1">}
}
// CHECK: event tensor has 2 elements but image(C_kappa) has 1
