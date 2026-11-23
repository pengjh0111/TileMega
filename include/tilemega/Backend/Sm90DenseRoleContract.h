// SPDX-License-Identifier: BSD-3-Clause
#pragma once
namespace tilemega::backend {
struct Sm90DenseRoleAbi {
  static constexpr int kThreads=160,kConsumers=128,kProducers=32;
};
template<int Threads,int Y=1,int Z=1>
struct Sm90DenseRoleLaunch:Sm90DenseRoleAbi {
  static_assert((Threads==128 || Threads==Sm90DenseRoleAbi::kThreads || Threads==192) && Y==1 && Z==1,
      "native dense role ABI requires 128, 160 or 192 threads along x");
  static constexpr int kThreads=Threads;
  static constexpr int kY=Y,kZ=Z;
};
}  // namespace tilemega::backend
