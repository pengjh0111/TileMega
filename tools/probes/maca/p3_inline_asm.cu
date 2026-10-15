// P-3: when does PTX inline asm fail under cucc? Select one case with -DCASE=n.
//   1: asm in a called __device__ function
//   2: asm in an uncalled non-template __device__ function
//   3: asm in an uninstantiated function template
//   4: asm in an instantiated function template
//   5: asm guarded by `if constexpr (false)` in an instantiated template
#include <cuda_runtime.h>
__device__ int Used(int x) {
#if CASE == 1
  asm volatile("trap;");
#endif
  return x + 1;
}
#if CASE == 2
__device__ void Unused() { asm volatile("trap;"); }
#endif
template <class T> __device__ T Tpl(T x) {
#if CASE == 3 || CASE == 4
  asm volatile("trap;");
#endif
#if CASE == 5
  if constexpr (sizeof(T) == 1) asm volatile("trap;");
#endif
  return x;
}
__global__ void k(int* o) {
  o[0] = Used(o[0]);
#if CASE == 4 || CASE == 5
  o[1] = Tpl<int>(o[1]);
#endif
}
int main() { return 0; }
