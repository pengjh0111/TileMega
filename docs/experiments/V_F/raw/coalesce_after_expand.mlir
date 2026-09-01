module {
}

// -----
module {
  func.func @expand_static(%arg0: !cute.layout<"(4,5):(1,4)">) -> !cute.layout<"20:1"> {
    %0 = cute.static : !cute.layout<"20:1">
    return %0 : !cute.layout<"20:1">
  }
}

// -----
module {
  func.func @expand_layout_dynamic(%arg0: !cute.layout<"(?):(?)">) -> !cute.layout<"?:?"> {
    %0:2 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.layout<"(?):(?)">
    %1 = cute.make_shape(%0#0) : (i32) -> !cute.shape<"?">
    %2 = cute.make_stride(%0#1) : (i32) -> !cute.stride<"?">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"?">, !cute.stride<"?">) -> !cute.layout<"?:?">
    return %3 : !cute.layout<"?:?">
  }
}

// -----
module {
  func.func @expand_composed_layout_dynamic(%arg0: !cute.composed_layout<"S<3,5,4> o 0 o (?):(?)">) -> !cute.composed_layout<"S<3,5,4> o 0 o ?:?"> {
    %0:2 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.composed_layout<"S<3,5,4> o 0 o (?):(?)">
    %1 = cute.make_shape(%0#0) : (i32) -> !cute.shape<"?">
    %2 = cute.make_stride(%0#1) : (i32) -> !cute.stride<"?">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"?">, !cute.stride<"?">) -> !cute.layout<"?:?">
    %4 = cute.make_int_tuple() : () -> !cute.int_tuple<"0">
    %5 = cute.static : !cute.swizzle<"S<3,5,4>">
    %6 = cute.make_composed_layout(%5, %4, %3) : (!cute.swizzle<"S<3,5,4>">, <"0">, <"?:?">) -> <"S<3,5,4> o 0 o ?:?">
    return %6 : !cute.composed_layout<"S<3,5,4> o 0 o ?:?">
  }
}

// -----
module {
  func.func @expand_nested_static(%arg0: !cute.layout<"(3,(4,5)):(8,(1,4))">) -> !cute.layout<"(3,20):(8,1)"> {
    %0 = cute.static : !cute.layout<"(3,20):(8,1)">
    return %0 : !cute.layout<"(3,20):(8,1)">
  }
}

// -----
module {
  func.func @expand_with_profile_static(%arg0: !cute.layout<"(3,(4,5)):(8,(1,4))">, %arg1: !cute.coord<"(1,1)">) -> !cute.layout<"(3,20):(8,1)"> {
    %0 = cute.static : !cute.layout<"(3,20):(8,1)">
    return %0 : !cute.layout<"(3,20):(8,1)">
  }
}

// -----
module {
  func.func @expand_with_profile_dynamic(%arg0: !cute.layout<"(4,?):(1,4)">, %arg1: !cute.coord<"(1,1)">) -> !cute.layout<"(4,?):(1,4)"> {
    %0 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.layout<"(4,?):(1,4)">
    %1 = cute.make_shape(%0) : (i32) -> !cute.shape<"(4,?)">
    %2 = cute.make_stride() : () -> !cute.stride<"(1,4)">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"(4,?)">, !cute.stride<"(1,4)">) -> !cute.layout<"(4,?):(1,4)">
    return %3 : !cute.layout<"(4,?):(1,4)">
  }
}

// -----
module {
  func.func @expand_dyn_middle_blocks_merge(%arg0: !cute.layout<"(4,5,?,3,2):(1,4,?,1,3)">) -> !cute.layout<"(20,?,6):(1,?,1)"> {
    %0:2 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.layout<"(4,5,?,3,2):(1,4,?,1,3)">
    %1 = cute.make_shape(%0#0) : (i32) -> !cute.shape<"(20,?,6)">
    %2 = cute.make_stride(%0#1) : (i32) -> !cute.stride<"(1,?,1)">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"(20,?,6)">, !cute.stride<"(1,?,1)">) -> !cute.layout<"(20,?,6):(1,?,1)">
    return %3 : !cute.layout<"(20,?,6):(1,?,1)">
  }
}
