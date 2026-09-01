// V-F minimal symbolic layout probe. Run with:
//   cute-opt -cute-fold-static dynamic_layout_probe.mlir
// The dynamic operations should remain in the result instead of being folded
// to an invalid static layout.

module {
  func.func @dynamic_composition(%outer: !cute.layout<"?:?">,
                                 %inner: !cute.layout<"4:1">)
      -> !cute.layout<"4:?"> {
    %result = cute.composition(%outer, %inner)
      : (!cute.layout<"?:?">, !cute.layout<"4:1">) -> !cute.layout<"4:?">
    return %result : !cute.layout<"4:?">
  }

  func.func @dynamic_zipped_divide(%layout: !cute.layout<"(?,8):(?,1)">,
                                   %tiler: !cute.shape<"(3,4)">)
      -> !cute.layout<"((3,4),(?,2)):((?,1),(?,4))"> {
    %result = cute.zipped_divide(%layout, %tiler)
      : (!cute.layout<"(?,8):(?,1)">, !cute.shape<"(3,4)">)
        -> !cute.layout<"((3,4),(?,2)):((?,1),(?,4))">
    return %result : !cute.layout<"((3,4),(?,2)):((?,1),(?,4))">
  }
}
