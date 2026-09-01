module {
  func.func @dynamic_composition(%arg0: !cute.layout<"?:?">, %arg1: !cute.layout<"4:1">) -> !cute.layout<"4:?"> {
    %0 = cute.composition(%arg0, %arg1) : (!cute.layout<"?:?">, !cute.layout<"4:1">) -> !cute.layout<"4:?">
    return %0 : !cute.layout<"4:?">
  }
  func.func @dynamic_zipped_divide(%arg0: !cute.layout<"(?,8):(?,1)">, %arg1: !cute.shape<"(3,4)">) -> !cute.layout<"((3,4),(?,2)):((?,1),(?,4))"> {
    %0 = cute.zipped_divide(%arg0, %arg1) : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">) -> !cute.layout<"((3,4),(?,2)):((?,1),(?,4))">
    return %0 : !cute.layout<"((3,4),(?,2)):((?,1),(?,4))">
  }
  func.func @dynamic_logical_divide(%arg0: !cute.layout<"(?,8):(?,1)">, %arg1: !cute.shape<"(3,4)">) -> !cute.layout<"((3,?),(4,2)):((?,?),(1,4))"> {
    %0 = cute.logical_divide(%arg0, %arg1) : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">) -> !cute.layout<"((3,?),(4,2)):((?,?),(1,4))">
    return %0 : !cute.layout<"((3,?),(4,2)):((?,?),(1,4))">
  }
  func.func @dynamic_coalesce(%arg0: !cute.layout<"(?):(?)">) -> !cute.layout<"?:?"> {
    %0 = cute.coalesce(%arg0) : (!cute.layout<"(?):(?)">) -> !cute.layout<"?:?">
    return %0 : !cute.layout<"?:?">
  }
  func.func @dynamic_ceil_div(%arg0: !cute.int_tuple<"(?,?)">, %arg1: !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)"> {
    %0 = cute.ceil_div(%arg0, %arg1) : (!cute.int_tuple<"(?,?)">, !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)">
    return %0 : !cute.int_tuple<"(?,?)">
  }
  func.func @dynamic_shape_div(%arg0: !cute.shape<"(?,?)">, %arg1: !cute.shape<"(?,?)">) -> !cute.shape<"(?,?)"> {
    %0 = cute.shape_div(%arg0, %arg1) : (!cute.shape<"(?,?)">, !cute.shape<"(?,?)">) -> !cute.shape<"(?,?)">
    return %0 : !cute.shape<"(?,?)">
  }
}
