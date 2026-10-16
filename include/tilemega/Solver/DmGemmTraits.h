// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>

namespace tilemega::solver {

/// xcore1000 DM tiles (Backend/Mx/ServingDmGemmMx.h): register-direct MMA
/// mainloop over 16-row atoms split across two waves, 32-K chunks; the only
/// shared memory is the FP32 tile plus epilogue scratch.
constexpr bool DmServingBF16ShapeLegal(int m,int n,int k,int stages) {
  return (m==16 || m==32 || m==64) && (n==32 || n==64 || n==128) &&
      (k==32 || k==64 || k==128) && m*n<=8192 && stages>=2;
}
constexpr int DmServingBF16SmemBytes(int m,int n,int k,int stages) {
  (void)k;(void)stages;
  return 4*m*n+8*m;
}
constexpr int DmServingPageActivationBytes(int m,int n,int k,int small_stages=2) {
  return 2*m*k*((n==16 || k<64)?small_stages:4);
}
constexpr int DmServingPageScratchBytes(int m,int n) {
  int partials=4*(m==16 && n==16?2:1)*m*n;
  int statistics=4*m*n+8*m;
  return partials>statistics?partials:statistics;
}
constexpr int DmServingPageWorkspaceBytes(int m,int n,int k,int small_stages=2) {
  int activation=DmServingPageActivationBytes(m,n,k,small_stages);
  int scratch=DmServingPageScratchBytes(m,n);
  return activation>scratch?activation:scratch;
}
inline BackendTraits DmServingBF16Traits(int m,int n,int k,int stages) {
  auto result=ServingBF16Traits(m,n,k,stages);
  result.shape_legal=DmServingBF16ShapeLegal(m,n,k,stages);
  result.smem_bytes=DmServingBF16SmemBytes(m,n,k,stages);
  return result;
}

} // namespace tilemega::solver
