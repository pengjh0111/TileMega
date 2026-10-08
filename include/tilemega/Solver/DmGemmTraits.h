// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>

namespace tilemega::solver {

constexpr bool DmServingBF16ShapeLegal(int m,int n,int k,int stages) {
  return (m==16 || m==32 || m==64 || m==128) &&
      (n==16 || n==32 || n==64 || n==128 || n==256) &&
      (k==16 || k==32 || k==64 || k==128) && m*n<=16384 && stages>=2;
}
constexpr int DmServingBF16SmemBytes(int m,int n,int k,int stages) {
  int mainloop=2*stages*k*(m+n);
  int partials=4*(m==16 && n==16?2:1)*m*n;
  int epilogue=4*m*n+8*m;
  return mainloop>partials?(mainloop>epilogue?mainloop:epilogue):
      (partials>epilogue?partials:epilogue);
}
inline BackendTraits DmServingBF16Traits(int m,int n,int k,int stages) {
  auto result=ServingBF16Traits(m,n,k,stages);
  result.shape_legal=DmServingBF16ShapeLegal(m,n,k,stages);
  result.smem_bytes=DmServingBF16SmemBytes(m,n,k,stages);
  return result;
}

} // namespace tilemega::solver
