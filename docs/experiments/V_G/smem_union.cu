// V-G: shared-memory liveness across a heterogeneous dispatch switch.
#include <cuda_runtime.h>

template <int KiB, int Op>
struct Task {
  struct Storage { unsigned char bytes[KiB * 1024]; };
  __device__ static int apply(int x) {
    if constexpr (Op == 0) return x + KiB;
    if constexpr (Op == 1) return x * 3 + KiB;
    return (x ^ (KiB * 17)) + Op;
  }
};

using T1 = Task<1, 0>; using T2 = Task<2, 1>; using T3 = Task<3, 2>;
using T4 = Task<4, 0>; using T5 = Task<5, 1>; using T6 = Task<6, 2>;
using T7 = Task<7, 0>; using T8 = Task<8, 1>;

template <int N> union UnionStorage;
template <> union UnionStorage<1> { T1::Storage t1; };
template <> union UnionStorage<2> { T2::Storage head; UnionStorage<1> tail; };
template <> union UnionStorage<3> { T3::Storage head; UnionStorage<2> tail; };
template <> union UnionStorage<4> { T4::Storage head; UnionStorage<3> tail; };
template <> union UnionStorage<5> { T5::Storage head; UnionStorage<4> tail; };
template <> union UnionStorage<6> { T6::Storage head; UnionStorage<5> tail; };
template <> union UnionStorage<7> { T7::Storage head; UnionStorage<6> tail; };
template <> union UnionStorage<8> { T8::Storage head; UnionStorage<7> tail; };

template <int N>
__device__ void dispatch_body(int kind, UnionStorage<N>& storage, int* out) {
  auto* bytes = reinterpret_cast<unsigned char*>(&storage);
  int index = (threadIdx.x * 257 + blockIdx.x) % sizeof(storage);
  bytes[index] = static_cast<unsigned char>(kind + threadIdx.x);
  int value = kind;
  switch (kind % N) {
    case 0: value = T1::apply(value); break;
    case 1: value = T2::apply(value); break;
    case 2: value = T3::apply(value); break;
    case 3: value = T4::apply(value); break;
    case 4: value = T5::apply(value); break;
    case 5: value = T6::apply(value); break;
    case 6: value = T7::apply(value); break;
    default: value = T8::apply(value); break;
  }
  if (threadIdx.x == 0) out[blockIdx.x] = value + bytes[index];
}

#define DEFINE_DISPATCH(N) \
extern "C" __global__ __launch_bounds__(256) void dispatch_##N(int kind, int* out) { \
  __shared__ UnionStorage<N> storage; dispatch_body(kind, storage, out); \
}
DEFINE_DISPATCH(2)
DEFINE_DISPATCH(3)
DEFINE_DISPATCH(5)
DEFINE_DISPATCH(8)

#define DEFINE_SINGLE(N) \
extern "C" __global__ __launch_bounds__(256) void single_##N(int* out) { \
  __shared__ T##N::Storage storage; \
  int i = (threadIdx.x * 257 + blockIdx.x) % sizeof(storage); \
  storage.bytes[i] = threadIdx.x; \
  if (threadIdx.x == 0) out[blockIdx.x] = storage.bytes[i]; \
}
DEFINE_SINGLE(1) DEFINE_SINGLE(2) DEFINE_SINGLE(3) DEFINE_SINGLE(4)
DEFINE_SINGLE(5) DEFINE_SINGLE(6) DEFINE_SINGLE(7) DEFINE_SINGLE(8)

extern "C" __global__ __launch_bounds__(256) void nested_dispatch(int outer, int inner, int* out) {
  __shared__ UnionStorage<8> storage;
  int kind = outer ? (inner ? 7 : 5) : (inner ? 3 : 1);
  dispatch_body(kind, storage, out);
}

extern "C" __global__ __launch_bounds__(256) void loop_carried_union(int kind, int* out) {
  __shared__ UnionStorage<8> storage;
  unsigned char* selected = reinterpret_cast<unsigned char*>(&storage);
  int limit = sizeof(storage);
  for (int i = threadIdx.x; i < limit; i += blockDim.x) selected[i] = kind;
  __syncthreads();
  int value = 0;
  for (int i = kind; i < limit; i += 1024) value += selected[i];
  if (threadIdx.x == 0) out[blockIdx.x] = value;
}

extern "C" __global__ __launch_bounds__(256) void loop_carried_separate(int kind, int* out) {
  __shared__ T1::Storage s1; __shared__ T2::Storage s2;
  __shared__ T3::Storage s3; __shared__ T4::Storage s4;
  __shared__ T5::Storage s5; __shared__ T6::Storage s6;
  __shared__ T7::Storage s7; __shared__ T8::Storage s8;
  unsigned char* selected = s1.bytes; int limit = sizeof(s1);
  switch (kind & 7) {
    case 0: selected=s1.bytes; limit=sizeof(s1); break;
    case 1: selected=s2.bytes; limit=sizeof(s2); break;
    case 2: selected=s3.bytes; limit=sizeof(s3); break;
    case 3: selected=s4.bytes; limit=sizeof(s4); break;
    case 4: selected=s5.bytes; limit=sizeof(s5); break;
    case 5: selected=s6.bytes; limit=sizeof(s6); break;
    case 6: selected=s7.bytes; limit=sizeof(s7); break;
    default:selected=s8.bytes; limit=sizeof(s8); break;
  }
  for (int i = threadIdx.x; i < limit; i += blockDim.x) selected[i] = kind;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = selected[kind % limit];
}
