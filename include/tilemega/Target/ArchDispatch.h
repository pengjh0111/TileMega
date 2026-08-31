// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- the single compile-time architecture dispatch point.
// Skeleton refs: §3.4, §5.3, §8.7.
#pragma once

#include <string_view>

#include <cutlass/arch/arch.h>

namespace tilemega::arch {

using Sm80 = cutlass::arch::Sm80;
using Sm89 = cutlass::arch::Sm89;
using Sm90 = cutlass::arch::Sm90;
using Sm120 = cutlass::arch::Sm120;

/// Compile-time capability switches. Code generation queries these switches;
/// it never infers a feature from an architecture version comparison.
template <class Arch>
struct Caps {
  static constexpr bool kCluster = false;
  static constexpr bool kTma = false;
  static constexpr bool kWarpSpecialized = false;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kCpAsync = false;
  static constexpr bool kMbarrier = false;
  static constexpr int kMaxClusterSize = 1;
  static constexpr char const* kCollective = "unsupported";
};

template <>
struct Caps<Sm80> {
  static constexpr bool kCluster = false;
  static constexpr bool kTma = false;
  static constexpr bool kWarpSpecialized = false;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr int kMaxClusterSize = 1;
  static constexpr char const* kCollective = "cp.async multistage";
};

template <>
struct Caps<Sm89> : Caps<Sm80> {};

template <>
struct Caps<Sm90> {
  static constexpr bool kCluster = true;
  static constexpr bool kTma = true;
  static constexpr bool kWarpSpecialized = true;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr int kMaxClusterSize = 8;
  static constexpr char const* kCollective = "TMA warp-specialized";
};

// Blackwell GeForce (sm_120) is deliberately not derived from a numerically
// earlier architecture: it has TMA/cluster support but no tcgen05/TMEM.
template <>
struct Caps<Sm120> {
  static constexpr bool kCluster = true;
  static constexpr bool kTma = true;
  static constexpr bool kWarpSpecialized = true;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr int kMaxClusterSize = 8;
  static constexpr char const* kCollective = "TMA warp-specialized (SM120 MMA)";
};

/// Host-side projection of the same exact-tag capability table, used by
/// TargetSpec::Probe without duplicating architecture policy.
struct RuntimeCaps {
  bool cluster;
  bool tma;
  bool warp_specialized;
  bool tcgen05;
  bool cp_async;
  bool mbarrier;
  int max_cluster_size;
  char const* collective;
};

template <class Arch>
constexpr RuntimeCaps RuntimeCapsFor() {
  return {Caps<Arch>::kCluster, Caps<Arch>::kTma,
          Caps<Arch>::kWarpSpecialized, Caps<Arch>::kTcgen05,
          Caps<Arch>::kCpAsync, Caps<Arch>::kMbarrier,
          Caps<Arch>::kMaxClusterSize, Caps<Arch>::kCollective};
}

inline constexpr RuntimeCaps RuntimeCapsForTag(std::string_view tag) {
  return tag == "sm_80"  ? RuntimeCapsFor<Sm80>()
       : tag == "sm_89"  ? RuntimeCapsFor<Sm89>()
       : tag == "sm_90"  ? RuntimeCapsFor<Sm90>()
       : tag == "sm_120" ? RuntimeCapsFor<Sm120>()
                           : RuntimeCapsFor<void>();
}

// The only __CUDA_ARCH__ selection in include/ or lib/. Exact matches avoid
// accidentally treating sm_120 as an sm_100 tcgen05 target.
#if defined(__CUDA_ARCH__)
#  if __CUDA_ARCH__ == 1200
using CurrentArch = Sm120;
#  elif __CUDA_ARCH__ == 900
using CurrentArch = Sm90;
#  elif __CUDA_ARCH__ == 890
using CurrentArch = Sm89;
#  elif __CUDA_ARCH__ == 800
using CurrentArch = Sm80;
#  endif
#endif

}  // namespace tilemega::arch
