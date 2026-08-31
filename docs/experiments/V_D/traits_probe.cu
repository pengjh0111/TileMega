// V-D: compile-time-only backend query. This translation unit declares no
// __global__ function and therefore generates no kernel code.
#include "traits_common.cuh"
#include <cstdio>

template <int Id>
void print_candidate() {
  constexpr int m = 64 + 32 * (Id / 10);
  constexpr int n = 64 + 32 * (Id % 10);
  using C = v_d::Collective<m, n>;
  constexpr auto bytes = sizeof(typename C::SharedStorage);
  constexpr auto mma_threads = cute::size(typename C::TiledMma{});
  constexpr bool legal_for_sm89_budget = bytes <= 101376;
  std::printf("candidate=%d m=%d n=%d k=32 stages=3 legal=%d smem=%zu mma_size=%d\n",
              Id, m, n, static_cast<int>(legal_for_sm89_budget), bytes,
              static_cast<int>(mma_threads));
}

template <int... Id>
void print_all(std::integer_sequence<int, Id...>) {
  (print_candidate<Id>(), ...);
}

int main() {
  print_all(std::make_integer_sequence<int, 100>{});
  return 0;
}
