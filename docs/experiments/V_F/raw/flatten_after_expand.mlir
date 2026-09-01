module {
}

// -----
module {
  func.func @expand_static(%arg0: !cute.layout<"(3,(4,5)):(8,(1,4))">) -> !cute.layout<"(3,4,5):(8,1,4)"> {
    %0 = cute.static : !cute.layout<"(3,4,5):(8,1,4)">
    return %0 : !cute.layout<"(3,4,5):(8,1,4)">
  }
}

// -----
module {
  func.func @expand_composed_static(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (3,(4,5)):(8,(1,4))">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o (3,4,5):(8,1,4)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o (3,4,5):(8,1,4)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o (3,4,5):(8,1,4)">
  }
}

// -----
module {
  func.func @expand_layout_keeps_dynamic(%arg0: !cute.layout<"(?,3):(1,?)">) -> !cute.layout<"(?,3):(1,?)"> {
    %0:2 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.layout<"(?,3):(1,?)">
    %1 = cute.make_shape(%0#0) : (i32) -> !cute.shape<"(?,3)">
    %2 = cute.make_stride(%0#1) : (i32) -> !cute.stride<"(1,?)">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"(?,3)">, !cute.stride<"(1,?)">) -> !cute.layout<"(?,3):(1,?)">
    return %3 : !cute.layout<"(?,3):(1,?)">
  }
}

// -----
module {
  func.func @expand_layout_nested_dynamic(%arg0: !cute.layout<"(3,(?,5)):(?,(1,?))">) -> !cute.layout<"(3,?,5):(?,1,?)"> {
    %0:3 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.layout<"(3,(?,5)):(?,(1,?))">
    %1 = cute.make_shape(%0#0) : (i32) -> !cute.shape<"(3,?,5)">
    %2 = cute.make_stride(%0#1, %0#2) : (i32, i32) -> !cute.stride<"(?,1,?)">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"(3,?,5)">, !cute.stride<"(?,1,?)">) -> !cute.layout<"(3,?,5):(?,1,?)">
    return %3 : !cute.layout<"(3,?,5):(?,1,?)">
  }
}

// -----
module {
  func.func @expand_composed_layout_keeps_dynamic(%arg0: !cute.composed_layout<"S<3,5,4> o 0 o (3,(?,5)):(?,(1,?))">) -> !cute.composed_layout<"S<3,5,4> o 0 o (3,?,5):(?,1,?)"> {
    %0:3 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.composed_layout<"S<3,5,4> o 0 o (3,(?,5)):(?,(1,?))">
    %1 = cute.make_shape(%0#0) : (i32) -> !cute.shape<"(3,?,5)">
    %2 = cute.make_stride(%0#1, %0#2) : (i32, i32) -> !cute.stride<"(?,1,?)">
    %3 = cute.make_layout(%1, %2) : (!cute.shape<"(3,?,5)">, !cute.stride<"(?,1,?)">) -> !cute.layout<"(3,?,5):(?,1,?)">
    %4 = cute.make_int_tuple() : () -> !cute.int_tuple<"0">
    %5 = cute.static : !cute.swizzle<"S<3,5,4>">
    %6 = cute.make_composed_layout(%5, %4, %3) : (!cute.swizzle<"S<3,5,4>">, <"0">, <"(3,?,5):(?,1,?)">) -> <"S<3,5,4> o 0 o (3,?,5):(?,1,?)">
    return %6 : !cute.composed_layout<"S<3,5,4> o 0 o (3,?,5):(?,1,?)">
  }
}

// -----
module {
  func.func @expand_shape_static(%arg0: !cute.shape<"(3,(4,5))">) -> !cute.shape<"(3,4,5)"> {
    %0 = cute.static : !cute.shape<"(3,4,5)">
    return %0 : !cute.shape<"(3,4,5)">
  }
}

// -----
module {
  func.func @expand_shape_dynamic(%arg0: !cute.shape<"(3,(?,5))">) -> !cute.shape<"(3,?,5)"> {
    %0 = cute.get_scalars<{only_dynamic}> (%arg0) : !cute.shape<"(3,(?,5))">
    %1 = cute.make_shape(%0) : (i32) -> !cute.shape<"(3,?,5)">
    return %1 : !cute.shape<"(3,?,5)">
  }
}
