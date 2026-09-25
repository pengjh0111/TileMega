// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingGemm.h>

using tilemega::backend::ServingGemmConfig;
using tilemega::arch::Sm80;
using tilemega::arch::Sm89;
using tilemega::arch::Sm90;
using tilemega::arch::Sm100;
using tilemega::arch::Sm120;
using tilemega::solver::ServingBF16ShapeLegal;

static_assert(ServingBF16ShapeLegal(16, 128, 64, 2));
static_assert(!ServingBF16ShapeLegal(8, 128, 64, 2));
static_assert(!ServingBF16ShapeLegal(16, 16, 64, 2));
static_assert(!ServingBF16ShapeLegal(16, 128, 32, 2));
static_assert(!ServingBF16ShapeLegal(128, 256, 64, 2));

template <class Arch>
constexpr bool CheckArch() {
  using Config = ServingGemmConfig<Arch, 16, 128, 64, 2>;
  return Config::kThreads == 128 &&
         Config::kSharedBytes >= sizeof(typename Config::Mainloop::SharedStorage);
}
static_assert(CheckArch<Sm80>());
static_assert(CheckArch<Sm89>());
static_assert(CheckArch<Sm90>());
static_assert(CheckArch<Sm100>());
static_assert(CheckArch<Sm120>());

using M16K128 = ServingGemmConfig<Sm89, 16, 128, 128, 2>;
using M32 = ServingGemmConfig<Sm89, 32, 128, 64, 3>;
using M64 = ServingGemmConfig<Sm89, 64, 128, 64, 3>;
using M128 = ServingGemmConfig<Sm89, 128, 128, 64, 2>;
static_assert(M16K128::kSharedBytes >= sizeof(typename M16K128::Mainloop::SharedStorage));
static_assert(M32::kSharedBytes >= sizeof(typename M32::Mainloop::SharedStorage));
static_assert(M64::kSharedBytes >= sizeof(typename M64::Mainloop::SharedStorage));
static_assert(M128::kSharedBytes >= sizeof(typename M128::Mainloop::SharedStorage));

int main() { return 0; }
