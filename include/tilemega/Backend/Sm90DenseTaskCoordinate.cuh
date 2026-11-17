// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseOperands.cuh>
namespace tilemega::backend {
struct Sm90DenseCyclicTasks {
  __device__ static int Count(int total) {return total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));}
  __device__ static int Index(int local) {return int(blockIdx.x)+local*int(gridDim.x);}
};
// The experimental queue uses the unused end of the operand header. Rings,
// operand banks and every body payload retain their original addresses.
struct Sm90DenseTaskContext {
  int const* coordinates;
  int coordinate,count;
};
static_assert(Sm90DenseOperandHeader::kEnd+sizeof(Sm90DenseTaskContext)<=1024);
struct Sm90DenseExplicitTasks {
  __device__ static Sm90DenseTaskContext& Context() {
    extern __shared__ __align__(1024) char tilemega_task_shared[];
    return *reinterpret_cast<Sm90DenseTaskContext*>(tilemega_task_shared+Sm90DenseOperandHeader::kEnd);
  }
  __device__ static int Count(int) {return Context().count;}
  __device__ static int Index(int local) {
    auto const& c=Context();return c.coordinates?c.coordinates[local]:c.coordinate;
  }
};
} // namespace tilemega::backend
