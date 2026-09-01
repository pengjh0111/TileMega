module {
}

// -----
module {
  func.func @fold_logical_product_layout() -> !cute.layout<"((3,4),(2,5)):((4,1),(12,24))"> {
    %0 = cute.static : !cute.layout<"((3,4),(2,5)):((4,1),(12,24))">
    return %0 : !cute.layout<"((3,4),(2,5)):((4,1),(12,24))">
  }
}

// -----
module {
  func.func @fold_logical_product_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,4):(4,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,5)):((4,1),(12,24))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,5)):((4,1),(12,24))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,5)):((4,1),(12,24))">
  }
}

// -----
module {
  func.func @fold_zipped_product_layout() -> !cute.layout<"((3,4),(2,5)):((4,1),(12,24))"> {
    %0 = cute.static : !cute.layout<"((3,4),(2,5)):((4,1),(12,24))">
    return %0 : !cute.layout<"((3,4),(2,5)):((4,1),(12,24))">
  }
}

// -----
module {
  func.func @fold_zipped_product_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,4):(4,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,5)):((4,1),(12,24))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,5)):((4,1),(12,24))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,5)):((4,1),(12,24))">
  }
}

// -----
module {
  func.func @fold_tiled_product_layout() -> !cute.layout<"((3,4),2,5):((4,1),12,24)"> {
    %0 = cute.static : !cute.layout<"((3,4),2,5):((4,1),12,24)">
    return %0 : !cute.layout<"((3,4),2,5):((4,1),12,24)">
  }
}

// -----
module {
  func.func @fold_tiled_product_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,4):(4,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),2,5):((4,1),12,24)"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),2,5):((4,1),12,24)">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),2,5):((4,1),12,24)">
  }
}

// -----
module {
  func.func @fold_flat_product_layout() -> !cute.layout<"(3,4,2,5):(4,1,12,24)"> {
    %0 = cute.static : !cute.layout<"(3,4,2,5):(4,1,12,24)">
    return %0 : !cute.layout<"(3,4,2,5):(4,1,12,24)">
  }
}

// -----
module {
  func.func @fold_flat_product_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,4):(4,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o (3,4,2,5):(4,1,12,24)"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o (3,4,2,5):(4,1,12,24)">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o (3,4,2,5):(4,1,12,24)">
  }
}

// -----
module {
  func.func @fold_raked_product_layout() -> !cute.layout<"((2,3),(5,4)):((12,4),(24,1))"> {
    %0 = cute.static : !cute.layout<"((2,3),(5,4)):((12,4),(24,1))">
    return %0 : !cute.layout<"((2,3),(5,4)):((12,4),(24,1))">
  }
}

// -----
module {
  func.func @fold_raked_product_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,4):(4,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((2,3),(5,4)):((12,4),(24,1))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((2,3),(5,4)):((12,4),(24,1))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((2,3),(5,4)):((12,4),(24,1))">
  }
}

// -----
module {
  func.func @fold_blocked_product_layout() -> !cute.layout<"((3,2),(4,5)):((4,12),(1,24))"> {
    %0 = cute.static : !cute.layout<"((3,2),(4,5)):((4,12),(1,24))">
    return %0 : !cute.layout<"((3,2),(4,5)):((4,12),(1,24))">
  }
}

// -----
module {
  func.func @fold_blocked_product_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,4):(4,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(4,5)):((4,12),(1,24))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(4,5)):((4,12),(1,24))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(4,5)):((4,12),(1,24))">
  }
}

// -----
module {
  func.func @fold_logical_divide_layout() -> !cute.layout<"((3,2),(4,2)):((8,24),(1,4))"> {
    %0 = cute.static : !cute.layout<"((3,2),(4,2)):((8,24),(1,4))">
    return %0 : !cute.layout<"((3,2),(4,2)):((8,24),(1,4))">
  }
}

// -----
module {
  func.func @fold_logical_divide_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (6,8):(8,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(4,2)):((8,24),(1,4))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(4,2)):((8,24),(1,4))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(4,2)):((8,24),(1,4))">
  }
}

// -----
module {
  func.func @fold_zipped_divide_layout() -> !cute.layout<"((3,4),(2,2)):((8,1),(24,4))"> {
    %0 = cute.static : !cute.layout<"((3,4),(2,2)):((8,1),(24,4))">
    return %0 : !cute.layout<"((3,4),(2,2)):((8,1),(24,4))">
  }
}

// -----
module {
  func.func @fold_zipped_divide_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (6,8):(8,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,2)):((8,1),(24,4))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,2)):((8,1),(24,4))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),(2,2)):((8,1),(24,4))">
  }
}

// -----
module {
  func.func @fold_tiled_divide_layout() -> !cute.layout<"((3,4),2,2):((8,1),24,4)"> {
    %0 = cute.static : !cute.layout<"((3,4),2,2):((8,1),24,4)">
    return %0 : !cute.layout<"((3,4),2,2):((8,1),24,4)">
  }
}

// -----
module {
  func.func @fold_tiled_divide_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (6,8):(8,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),2,2):((8,1),24,4)"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),2,2):((8,1),24,4)">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,4),2,2):((8,1),24,4)">
  }
}

// -----
module {
  func.func @fold_flat_divide_layout() -> !cute.layout<"(3,4,2,2):(8,1,24,4)"> {
    %0 = cute.static : !cute.layout<"(3,4,2,2):(8,1,24,4)">
    return %0 : !cute.layout<"(3,4,2,2):(8,1,24,4)">
  }
}

// -----
module {
  func.func @fold_flat_divide_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (6,8):(8,1)">) -> !cute.composed_layout<"S<3,4,3> o 0 o (3,4,2,2):(8,1,24,4)"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o (3,4,2,2):(8,1,24,4)">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o (3,4,2,2):(8,1,24,4)">
  }
}

// -----
module {
  func.func @fold_tile_to_shape_layout() -> !cute.layout<"((3,2),(2,4)):((1,6),(3,12))"> {
    %0 = cute.static : !cute.layout<"((3,2),(2,4)):((1,6),(3,12))">
    return %0 : !cute.layout<"((3,2),(2,4)):((1,6),(3,12))">
  }
}

// -----
module {
  func.func @fold_tile_to_shape_composed_layout(%arg0: !cute.composed_layout<"S<3,4,3> o 0 o (3,2):(1,3)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(2,4)):((1,6),(3,12))"> {
    %0 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(2,4)):((1,6),(3,12))">
    return %0 : !cute.composed_layout<"S<3,4,3> o 0 o ((3,2),(2,4)):((1,6),(3,12))">
  }
}
