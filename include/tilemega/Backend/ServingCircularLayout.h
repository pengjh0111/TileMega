// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/layout_composed.hpp>

namespace tilemega::backend {
// A named offset keeps recast's upcast/downcast lookup independent of the
// order in which CuTe's layout and numeric headers were included.
struct ServingCircularIndex {
  int value=0;
  CUTE_HOST_DEVICE constexpr operator int() const {return value;}
  CUTE_HOST_DEVICE constexpr friend auto operator+(ServingCircularIndex a,ServingCircularIndex b) {
    return ServingCircularIndex{a.value+b.value};
  }
  template<class Index>
  CUTE_HOST_DEVICE constexpr friend auto operator+(ServingCircularIndex a,Index const& b) {
    return ServingCircularIndex{a.value+int(b)};
  }
  template<class Index>
  CUTE_HOST_DEVICE constexpr friend auto operator+(Index const& a,ServingCircularIndex b) {
    return ServingCircularIndex{int(a)+b.value};
  }
};
template<int Scale>
CUTE_HOST_DEVICE constexpr auto upcast(ServingCircularIndex index) {
  return ServingCircularIndex{index.value/Scale};
}
template<int Scale>
CUTE_HOST_DEVICE constexpr auto downcast(ServingCircularIndex index) {
  return ServingCircularIndex{index.value*Scale};
}
// Page groups can straddle the ring's final slot. Preserve the swizzled
// matrix coordinates while wrapping each ldmatrix row address independently.
template<int Elements>
struct ServingCircularOffset {
  static_assert(Elements>0);
  template<class Index>
  CUTE_HOST_DEVICE constexpr auto operator()(Index const& index) const {
    return int(index)%Elements;
  }
};
template<int Scale,int Elements>
CUTE_HOST_DEVICE constexpr auto upcast(ServingCircularOffset<Elements>) {
  static_assert(Elements%Scale==0);
  return ServingCircularOffset<Elements/Scale>{};
}
template<int Scale,int Elements>
CUTE_HOST_DEVICE constexpr auto downcast(ServingCircularOffset<Elements>) {
  return ServingCircularOffset<Elements*Scale>{};
}
} // namespace tilemega::backend
