module {
}

// -----
module {
  func.func @fold_composition_layout(%arg0: !cute.layout<"(4,8):(1,4)">, %arg1: !cute.layout<"(2,4):(1,2)">) -> !cute.layout<"(2,4):(1,2)"> {
    %0 = cute.static : !cute.layout<"(2,4):(1,2)">
    return %0 : !cute.layout<"(2,4):(1,2)">
  }
}

// -----
module {
  func.func @fold_composition_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (2,3):(1,2)">, %arg1: !cute.layout<"(2,3):(1,2)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o (2,3):(1,2)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o (2,3):(1,2)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o (2,3):(1,2)">
  }
}

// -----
module {
  func.func @fold_coalesce_layout(%arg0: !cute.layout<"(4,5):(1,4)">) -> !cute.layout<"20:1"> {
    %0 = cute.static : !cute.layout<"20:1">
    return %0 : !cute.layout<"20:1">
  }
}

// -----
module {
  func.func @fold_coalesce_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (4,5):(1,4)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o 20:1"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o 20:1">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o 20:1">
  }
}

// -----
module {
  func.func @fold_complement(%arg0: !cute.layout<"3:2">, %arg1: !cute.shape<"6">) -> !cute.layout<"2:1"> {
    %0 = cute.static : !cute.layout<"2:1">
    return %0 : !cute.layout<"2:1">
  }
}

// -----
module {
  func.func @fold_group_modes_layout(%arg0: !cute.layout<"(4,5,6):(1,4,20)">) -> !cute.layout<"((4,5),6):((1,4),20)"> {
    %0 = cute.static : !cute.layout<"((4,5),6):((1,4),20)">
    return %0 : !cute.layout<"((4,5),6):((1,4),20)">
  }
}

// -----
module {
  func.func @fold_group_modes_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (3,4,5):(1,3,12)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o ((3,4),5):((1,3),12)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o ((3,4),5):((1,3),12)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o ((3,4),5):((1,3),12)">
  }
}

// -----
module {
  func.func @fold_recast_layout_layout(%arg0: !cute.layout<"(32,4):(1,32)">) -> !cute.layout<"(8,4):(1,8)"> {
    %0 = cute.static : !cute.layout<"(8,4):(1,8)">
    return %0 : !cute.layout<"(8,4):(1,8)">
  }
}

// -----
module {
  func.func @fold_recast_layout_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (8,4):(1,8)">) -> !cute.composed_layout<"(2,5):(1,2) o 1 o (4,4):(1,4)"> {
    %0 = cute.static : !cute.composed_layout<"(2,5):(1,2) o 1 o (4,4):(1,4)">
    return %0 : !cute.composed_layout<"(2,5):(1,2) o 1 o (4,4):(1,4)">
  }
}

// -----
module {
  func.func @fold_slice_layout(%arg0: !cute.layout<"(2,3,4):(1,2,6)">, %arg1: !cute.coord<"(0,_,1)">) -> !cute.layout<"(3):(2)"> {
    %0 = cute.static : !cute.layout<"(3):(2)">
    return %0 : !cute.layout<"(3):(2)">
  }
}

// -----
module {
  func.func @fold_slice_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (4,3):(1,4)">, %arg1: !cute.coord<"(0,_)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o (3):(4)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o (3):(4)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o (3):(4)">
  }
}

// -----
module {
  func.func @fold_dice_layout(%arg0: !cute.layout<"(2,3,4):(1,2,6)">, %arg1: !cute.coord<"(_,1,_)">) -> !cute.layout<"(3):(2)"> {
    %0 = cute.static : !cute.layout<"(3):(2)">
    return %0 : !cute.layout<"(3):(2)">
  }
}

// -----
module {
  func.func @fold_dice_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (4,3):(1,4)">, %arg1: !cute.coord<"(_,1)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o (3):(4)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o (3):(4)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o (3):(4)">
  }
}

// -----
module {
  func.func @fold_slice_dynamic_input(%arg0: !cute.layout<"(?,4):(?,1)">, %arg1: !cute.coord<"(0,_)">) -> !cute.layout<"(4):(1)"> {
    %0 = cute.static : !cute.layout<"(4):(1)">
    return %0 : !cute.layout<"(4):(1)">
  }
}

// -----
module {
  func.func @fold_slice_dynamic_input_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (?,4):(?,1)">, %arg1: !cute.coord<"(0,_)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o (4):(1)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o (4):(1)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o (4):(1)">
  }
}

// -----
module {
  func.func @fold_dice_dynamic_input(%arg0: !cute.layout<"(?,4):(?,1)">, %arg1: !cute.coord<"(_,1)">) -> !cute.layout<"(4):(1)"> {
    %0 = cute.static : !cute.layout<"(4):(1)">
    return %0 : !cute.layout<"(4):(1)">
  }
}

// -----
module {
  func.func @fold_dice_dynamic_input_composed_layout(%arg0: !cute.composed_layout<"(4,5):(1,4) o 2 o (?,4):(?,1)">, %arg1: !cute.coord<"(_,1)">) -> !cute.composed_layout<"(4,5):(1,4) o 2 o (4):(1)"> {
    %0 = cute.static : !cute.composed_layout<"(4,5):(1,4) o 2 o (4):(1)">
    return %0 : !cute.composed_layout<"(4,5):(1,4) o 2 o (4):(1)">
  }
}

// -----
module {
  func.func @fold_right_inverse(%arg0: !cute.layout<"(4,3):(1,4)">) -> !cute.layout<"12:1"> {
    %0 = cute.static : !cute.layout<"12:1">
    return %0 : !cute.layout<"12:1">
  }
}

// -----
module {
  func.func @fold_left_inverse(%arg0: !cute.layout<"(4,3):(3,1)">) -> !cute.layout<"(3,4):(4,1)"> {
    %0 = cute.static : !cute.layout<"(3,4):(4,1)">
    return %0 : !cute.layout<"(3,4):(4,1)">
  }
}
