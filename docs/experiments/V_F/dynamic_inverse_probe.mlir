// Dynamic inverse is intentionally illegal in the pinned CuTe dialect.

func.func @dynamic_right_inverse(%src: !cute.layout<"(?,3):(1,4)">) {
  // expected-error@+2 {{expects a static-shape input layout, but got '!cute.layout<"(?,3):(1,4)">'}}
  // expected-error@+1 {{'cute.right_inverse' op failed to infer returned types}}
  %result = cute.right_inverse(%src)
    : (!cute.layout<"(?,3):(1,4)">) -> !cute.layout<"12:1">
  return
}

// -----

func.func @dynamic_left_inverse(%src: !cute.layout<"(?,3):(3,1)">) {
  // expected-error@+2 {{expects a static-shape input layout, but got '!cute.layout<"(?,3):(3,1)">'}}
  // expected-error@+1 {{'cute.left_inverse' op failed to infer returned types}}
  %result = cute.left_inverse(%src)
    : (!cute.layout<"(?,3):(3,1)">) -> !cute.layout<"(3,4):(4,1)">
  return
}
