// SPDX-License-Identifier: BSD-3-Clause
#pragma once
namespace tilemega::backend {
struct Sm90DenseRoleAbi {
  static constexpr int kThreads=160,kConsumers=128,kProducers=32;
};
template<int Threads,int Y=1,int Z=1>
struct Sm90DenseRoleLaunch:Sm90DenseRoleAbi {
  static_assert(Threads==kThreads && Y==1 && Z==1,
      "native dense role ABI requires a 160x1x1 thread block");
  static constexpr int kY=Y,kZ=Z;
};
}  // namespace tilemega::backend
