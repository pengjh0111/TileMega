module {
}

// -----
module {
  func.func @no_fold_make_int_tuple(%arg0: i32) -> !cute.int_tuple<"?"> {
    %0 = cute.make_int_tuple(%arg0) : (i32) -> !cute.int_tuple<"?">
    return %0 : !cute.int_tuple<"?">
  }
}

// -----
module {
  func.func @no_fold_make_shape(%arg0: i32) -> !cute.shape<"(?,4)"> {
    %0 = cute.make_shape(%arg0) : (i32) -> !cute.shape<"(?,4)">
    return %0 : !cute.shape<"(?,4)">
  }
}

// -----
module {
  func.func @no_fold_make_stride(%arg0: i32) -> !cute.stride<"?"> {
    %0 = cute.make_stride(%arg0) : (i32) -> !cute.stride<"?">
    return %0 : !cute.stride<"?">
  }
}

// -----
module {
  func.func @no_fold_make_coord(%arg0: i32, %arg1: i32) -> !cute.coord<"(?,?)"> {
    %0 = cute.make_coord(%arg0, %arg1) : (i32, i32) -> !cute.coord<"(?,?)">
    return %0 : !cute.coord<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_make_layout(%arg0: i32) -> !cute.layout<"(?,4):(1,?)"> {
    %0 = cute.make_shape(%arg0) : (i32) -> !cute.shape<"(?,4)">
    %1 = cute.make_stride(%arg0) : (i32) -> !cute.stride<"(1,?)">
    %2 = cute.make_layout(%0, %1) : (!cute.shape<"(?,4)">, !cute.stride<"(1,?)">) -> !cute.layout<"(?,4):(1,?)">
    return %2 : !cute.layout<"(?,4):(1,?)">
  }
}

// -----
module {
  func.func @no_fold_make_tile(%arg0: i32, %arg1: i32) -> !cute.tile<"[(?,3):(1,?)]"> {
    %0 = cute.make_tile(%arg0, %arg1) : (i32, i32) -> !cute.tile<"[(?,3):(1,?)]">
    return %0 : !cute.tile<"[(?,3):(1,?)]">
  }
}

// -----
module {
  func.func @no_fold_make_composed_layout(%arg0: !cute.layout<"(?,5):(1,?)">, %arg1: !cute.int_tuple<"0">, %arg2: !cute.layout<"(2,3):(1,2)">) -> !cute.composed_layout<"(?,5):(1,?) o 0 o (2,3):(1,2)"> {
    %0 = cute.make_composed_layout(%arg0, %arg1, %arg2) : (!cute.layout<"(?,5):(1,?)">, <"0">, <"(2,3):(1,2)">) -> <"(?,5):(1,?) o 0 o (2,3):(1,2)">
    return %0 : !cute.composed_layout<"(?,5):(1,?) o 0 o (2,3):(1,2)">
  }
}

// -----
module {
  func.func @no_fold_make_layout_like(%arg0: !cute.layout<"(?,2):(?,1)">) -> !cute.layout<"(?,2):(2,1)"> {
    %0 = cute.make_layout_like(%arg0) : !cute.layout<"(?,2):(?,1)"> -> !cute.layout<"(?,2):(2,1)">
    return %0 : !cute.layout<"(?,2):(2,1)">
  }
}

// -----
module {
  func.func @no_fold_make_ordered_layout(%arg0: !cute.shape<"(4,3,?,2)">, %arg1: !cute.int_tuple<"(2,1,3,4)">) -> !cute.layout<"(4,3,?,2):(3,1,12,?)"> {
    %0 = cute.make_ordered_layout(%arg0, %arg1) : (!cute.shape<"(4,3,?,2)">, !cute.int_tuple<"(2,1,3,4)">) -> !cute.layout<"(4,3,?,2):(3,1,12,?)">
    return %0 : !cute.layout<"(4,3,?,2):(3,1,12,?)">
  }
}

// -----
module {
  func.func @no_fold_make_identity_layout(%arg0: !cute.shape<"(?,3)">) -> !cute.layout<"(?,3):(1@0,1@1)"> {
    %0 = cute.make_identity_layout(%arg0) : !cute.shape<"(?,3)"> -> !cute.layout<"(?,3):(1@0,1@1)">
    return %0 : !cute.layout<"(?,3):(1@0,1@1)">
  }
}

// -----
module {
  func.func @no_fold_get_shape(%arg0: !cute.layout<"(?,4):(1,?)">) -> !cute.shape<"(?,4)"> {
    %0 = cute.get_shape(%arg0) : !cute.layout<"(?,4):(1,?)"> -> !cute.shape<"(?,4)">
    return %0 : !cute.shape<"(?,4)">
  }
}

// -----
module {
  func.func @no_fold_get_stride(%arg0: !cute.layout<"(?,4):(1,?)">) -> !cute.stride<"(1,?)"> {
    %0 = cute.get_stride(%arg0) : <"(?,4):(1,?)"> -> !cute.stride<"(1,?)">
    return %0 : !cute.stride<"(1,?)">
  }
}

// -----
module {
  func.func @no_fold_composed_get_inner(%arg0: !cute.composed_layout<"(?,4):(1,?) o 0 o (2,4):(1,2)">) -> !cute.layout<"(?,4):(1,?)"> {
    %0 = cute.composed_get_inner(%arg0) : <"(?,4):(1,?) o 0 o (2,4):(1,2)"> -> !cute.layout<"(?,4):(1,?)">
    return %0 : !cute.layout<"(?,4):(1,?)">
  }
}

// -----
module {
  func.func @no_fold_composed_get_offset(%arg0: !cute.composed_layout<"(4,8):(1,4) o ? o (2,4):(1,2)">) -> !cute.int_tuple<"?"> {
    %0 = cute.composed_get_offset(%arg0) : <"(4,8):(1,4) o ? o (2,4):(1,2)"> -> !cute.int_tuple<"?">
    return %0 : !cute.int_tuple<"?">
  }
}

// -----
module {
  func.func @no_fold_composed_get_outer(%arg0: !cute.composed_layout<"(4,8):(1,4) o 0 o (?,4):(1,?)">) -> !cute.layout<"(?,4):(1,?)"> {
    %0 = cute.composed_get_outer(%arg0) : <"(4,8):(1,4) o 0 o (?,4):(1,?)"> -> <"(?,4):(1,?)">
    return %0 : !cute.layout<"(?,4):(1,?)">
  }
}

// -----
module {
  func.func @no_fold_to_int_tuple(%arg0: !cute.shape<"(?,4)">) -> !cute.int_tuple<"(?,4)"> {
    %0 = cute.to_int_tuple(%arg0) : !cute.shape<"(?,4)"> -> !cute.int_tuple<"(?,4)">
    return %0 : !cute.int_tuple<"(?,4)">
  }
}

// -----
module {
  func.func @no_fold_get_leaves(%arg0: !cute.shape<"(?,?)">) -> (!cute.shape<"?">, !cute.shape<"?">) {
    %0:2 = cute.get_leaves(%arg0) : !cute.shape<"(?,?)">
    return %0#0, %0#1 : !cute.shape<"?">, !cute.shape<"?">
  }
}

// -----
module {
  func.func @no_fold_get_layouts_from_tile(%arg0: !cute.tile<"[(?,4):(1,?)]">) -> !cute.layout<"(?,4):(1,?)"> {
    %0 = cute.get_layouts_from_tile(%arg0) : <"[(?,4):(1,?)]">
    return %0 : !cute.layout<"(?,4):(1,?)">
  }
}

// -----
module {
  func.func @no_fold_get(%arg0: !cute.layout<"(?,4):(1,?)">) -> !cute.layout<"?:1"> {
    %0 = cute.get<[0]> (%arg0) : !cute.layout<"(?,4):(1,?)"> -> !cute.layout<"?:1">
    return %0 : !cute.layout<"?:1">
  }
}

// -----
module {
  func.func @no_fold_select(%arg0: !cute.shape<"(4,?,2)">) -> !cute.shape<"(4,?)"> {
    %0 = cute.select<[0, 1]> (%arg0) : !cute.shape<"(4,?,2)"> -> !cute.shape<"(4,?)">
    return %0 : !cute.shape<"(4,?)">
  }
}

// -----
module {
  func.func @no_fold_composition(%arg0: !cute.layout<"?:?">, %arg1: !cute.layout<"4:1">) -> !cute.layout<"4:?"> {
    %0 = cute.composition(%arg0, %arg1) : (!cute.layout<"?:?">, !cute.layout<"4:1">) -> !cute.layout<"4:?">
    return %0 : !cute.layout<"4:?">
  }
}

// -----
module {
  func.func @no_fold_coalesce(%arg0: !cute.layout<"(?):(?)">) -> !cute.layout<"?:?"> {
    %0 = cute.coalesce(%arg0) : (!cute.layout<"(?):(?)">) -> !cute.layout<"?:?">
    return %0 : !cute.layout<"?:?">
  }
}

// -----
module {
  func.func @no_fold_complement(%arg0: !cute.layout<"?:2">, %arg1: !cute.shape<"6">) -> !cute.layout<"(2,?):(1,?)"> {
    %0 = cute.complement(%arg0, %arg1) : (!cute.layout<"?:2">, !cute.shape<"6">) -> !cute.layout<"(2,?):(1,?)">
    return %0 : !cute.layout<"(2,?):(1,?)">
  }
}

// -----
module {
  func.func @no_fold_group_modes(%arg0: !cute.layout<"(?,5,6):(1,?,?)">) -> !cute.layout<"((?,5),6):((1,?),?)"> {
    %0 = cute.group_modes<0, 2> (%arg0) : (!cute.layout<"(?,5,6):(1,?,?)">) -> !cute.layout<"((?,5),6):((1,?),?)">
    return %0 : !cute.layout<"((?,5),6):((1,?),?)">
  }
}

// -----
module {
  func.func @no_fold_recast_layout(%arg0: !cute.layout<"(32,?):(1,?)">) -> !cute.layout<"(8,?):(1,?)"> {
    %0 = cute.recast_layout<32, 8> (%arg0) : !cute.layout<"(32,?):(1,?)"> -> !cute.layout<"(8,?):(1,?)">
    return %0 : !cute.layout<"(8,?):(1,?)">
  }
}

// -----
module {
  func.func @no_fold_slice(%arg0: !cute.layout<"(2,?,4):(1,?,?)">, %arg1: !cute.coord<"(0,_,1)">) -> !cute.layout<"(?):(?)"> {
    %0 = cute.slice(%arg0, %arg1) : !cute.layout<"(2,?,4):(1,?,?)">, !cute.coord<"(0,_,1)">
    return %0 : !cute.layout<"(?):(?)">
  }
}

// -----
module {
  func.func @no_fold_dice(%arg0: !cute.layout<"(2,?,4):(1,?,?)">, %arg1: !cute.coord<"(_,1,_)">) -> !cute.layout<"(?):(?)"> {
    %0 = cute.dice(%arg0, %arg1) : !cute.layout<"(2,?,4):(1,?,?)">, !cute.coord<"(_,1,_)">
    return %0 : !cute.layout<"(?):(?)">
  }
}

// -----
module {
  func.func @no_fold_tuple_add(%arg0: !cute.int_tuple<"(?,?)">, %arg1: !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)"> {
    %0 = cute.tuple_add(%arg0, %arg1) : (!cute.int_tuple<"(?,?)">, !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)">
    return %0 : !cute.int_tuple<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_tuple_sub(%arg0: !cute.int_tuple<"(?,?)">, %arg1: !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)"> {
    %0 = cute.tuple_sub(%arg0, %arg1) : (!cute.int_tuple<"(?,?)">, !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)">
    return %0 : !cute.int_tuple<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_ceil_div(%arg0: !cute.int_tuple<"(?,?)">, %arg1: !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)"> {
    %0 = cute.ceil_div(%arg0, %arg1) : (!cute.int_tuple<"(?,?)">, !cute.int_tuple<"(?,?)">) -> !cute.int_tuple<"(?,?)">
    return %0 : !cute.int_tuple<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_shape_div(%arg0: !cute.shape<"(?,?)">, %arg1: !cute.shape<"(?,?)">) -> !cute.shape<"(?,?)"> {
    %0 = cute.shape_div(%arg0, %arg1) : (!cute.shape<"(?,?)">, !cute.shape<"(?,?)">) -> !cute.shape<"(?,?)">
    return %0 : !cute.shape<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_elem_less(%arg0: !cute.int_tuple<"(?,?)">, %arg1: !cute.int_tuple<"(?,?)">) -> i1 {
    %0 = cute.elem_less(%arg0, %arg1) : (!cute.int_tuple<"(?,?)">, !cute.int_tuple<"(?,?)">) -> i1
    return %0 : i1
  }
}

// -----
module {
  func.func @no_fold_equal(%arg0: !cute.int_tuple<"(?,?)">, %arg1: !cute.int_tuple<"(?,?)">) -> i1 {
    %0 = cute.equal(%arg0, %arg1) : (!cute.int_tuple<"(?,?)">, !cute.int_tuple<"(?,?)">) -> i1
    return %0 : i1
  }
}

// -----
module {
  func.func @no_fold_size(%arg0: !cute.shape<"(4,(16,32),(?,64))">) -> !cute.int_tuple<"?"> {
    %0 = cute.size (%arg0) : (!cute.shape<"(4,(16,32),(?,64))">) -> !cute.int_tuple<"?">
    return %0 : !cute.int_tuple<"?">
  }
}

// -----
module {
  func.func @no_fold_cosize(%arg0: !cute.layout<"(?,?,?):(?,?,?)">) -> !cute.int_tuple<"?"> {
    %0 = cute.cosize (%arg0) : (!cute.layout<"(?,?,?):(?,?,?)">) -> !cute.int_tuple<"?">
    return %0 : !cute.int_tuple<"?">
  }
}

// -----
module {
  func.func @no_fold_tuple_product(%arg0: !cute.shape<"(4,(16,32),(?,64))">) -> !cute.shape<"?"> {
    %0 = cute.tuple_product(%arg0) : (!cute.shape<"(4,(16,32),(?,64))">) -> !cute.shape<"?">
    return %0 : !cute.shape<"?">
  }
}

// -----
module {
  func.func @no_fold_tuple_product_each(%arg0: !cute.shape<"(4,(?,32))">) -> !cute.shape<"(4,?)"> {
    %0 = cute.tuple_product_each(%arg0) : (!cute.shape<"(4,(?,32))">) -> !cute.shape<"(4,?)">
    return %0 : !cute.shape<"(4,?)">
  }
}

// -----
module {
  func.func @no_fold_layout_eval(%arg0: !cute.coord<"(?,?)">, %arg1: !cute.layout<"(4,8):(1,4)">) -> !cute.int_tuple<"?"> {
    %0 = cute.layout_eval(%arg0, %arg1) : (!cute.coord<"(?,?)">, !cute.layout<"(4,8):(1,4)">) -> !cute.int_tuple<"?">
    return %0 : !cute.int_tuple<"?">
  }
}

// -----
module {
  func.func @no_fold_idx2crd(%arg0: !cute.int_tuple<"?">, %arg1: !cute.shape<"(4,8)">) -> !cute.coord<"(?,?)"> {
    %0 = cute.idx2crd(%arg0, %arg1) : (!cute.int_tuple<"?">, !cute.shape<"(4,8)">) -> !cute.coord<"(?,?)">
    return %0 : !cute.coord<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_increment_coord(%arg0: !cute.coord<"(?,?)">, %arg1: !cute.shape<"(4,8)">) -> !cute.coord<"(?,?)"> {
    %0 = cute.increment_coord(%arg0, %arg1) : (!cute.coord<"(?,?)">, !cute.shape<"(4,8)">) -> !cute.coord<"(?,?)">
    return %0 : !cute.coord<"(?,?)">
  }
}

// -----
module {
  func.func @no_fold_append_to_rank(%arg0: !cute.shape<"(4,8)">, %arg1: !cute.shape<"?">) -> !cute.shape<"(4,8,?,?)"> {
    %0 = cute.append_to_rank<4> (%arg0, %arg1) : !cute.shape<"(4,8)">, !cute.shape<"?">
    return %0 : !cute.shape<"(4,8,?,?)">
  }
}

// -----
module {
  func.func @no_fold_prepend_to_rank(%arg0: !cute.shape<"(4,8)">, %arg1: !cute.shape<"?">) -> !cute.shape<"(?,?,4,8)"> {
    %0 = cute.prepend_to_rank<4> (%arg0, %arg1) : !cute.shape<"(4,8)">, !cute.shape<"?">
    return %0 : !cute.shape<"(?,?,4,8)">
  }
}

// -----
module {
}

// -----
module {
  func.func @no_fold_logical_divide(%arg0: !cute.layout<"(?,8):(?,1)">, %arg1: !cute.shape<"(3,4)">) -> !cute.layout<"((3,?),(4,2)):((?,?),(1,4))"> {
    %0 = cute.logical_divide(%arg0, %arg1) : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">) -> !cute.layout<"((3,?),(4,2)):((?,?),(1,4))">
    return %0 : !cute.layout<"((3,?),(4,2)):((?,?),(1,4))">
  }
}

// -----
module {
  func.func @no_fold_zipped_divide(%arg0: !cute.layout<"(?,8):(?,1)">, %arg1: !cute.shape<"(3,4)">) -> !cute.layout<"((3,4),(?,2)):((?,1),(?,4))"> {
    %0 = cute.zipped_divide(%arg0, %arg1) : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">) -> !cute.layout<"((3,4),(?,2)):((?,1),(?,4))">
    return %0 : !cute.layout<"((3,4),(?,2)):((?,1),(?,4))">
  }
}

// -----
module {
  func.func @no_fold_tiled_divide(%arg0: !cute.layout<"(?,8):(?,1)">, %arg1: !cute.shape<"(3,4)">) -> !cute.layout<"((3,4),?,2):((?,1),?,4)"> {
    %0 = cute.tiled_divide(%arg0, %arg1) : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">) -> !cute.layout<"((3,4),?,2):((?,1),?,4)">
    return %0 : !cute.layout<"((3,4),?,2):((?,1),?,4)">
  }
}

// -----
module {
  func.func @no_fold_flat_divide(%arg0: !cute.layout<"(?,8):(?,1)">, %arg1: !cute.shape<"(3,4)">) -> !cute.layout<"(3,4,?,2):(?,1,?,4)"> {
    %0 = cute.flat_divide(%arg0, %arg1) : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">) -> !cute.layout<"(3,4,?,2):(?,1,?,4)">
    return %0 : !cute.layout<"(3,4,?,2):(?,1,?,4)">
  }
}

// -----
module {
  func.func @no_fold_tile_to_shape(%arg0: !cute.layout<"4:1">, %arg1: !cute.shape<"(?,8)">) -> !cute.layout<"((4,?),(1,8)):((1,4),(0,?))"> {
    %0 = cute.tile_to_shape(%arg0, %arg1) : (!cute.layout<"4:1">, !cute.shape<"(?,8)">) -> !cute.layout<"((4,?),(1,8)):((1,4),(0,?))">
    return %0 : !cute.layout<"((4,?),(1,8)):((1,4),(0,?))">
  }
}
