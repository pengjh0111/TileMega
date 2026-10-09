// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_DN_VECTOR_SUMS
#define TILEMEGA_DN_VECTOR_SUMS 0
#endif
#ifndef TILEMEGA_SWIGLU_U
#define TILEMEGA_SWIGLU_U 16
#endif
namespace tilemega::backend {
__device__ inline float DeferredNormBlock(float const* sums,int row,int width,int block) {
#if TILEMEGA_DN_VECTOR_SUMS
  auto* p=sums+row*((width+7)/8)+4*block;
  // Reproduce xor-1 then xor-2 of four eight-element producer sums.
  return (p[0]+p[1])+(p[2]+p[3]);
#else
  return sums[row*(width/32)+block];
#endif
}
}
