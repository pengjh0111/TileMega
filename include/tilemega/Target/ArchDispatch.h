// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- the single compile-time architecture dispatch point.
// Skeleton refs: §3.4, §5.3, §8.7.
#pragma once

#include <string_view>

#include <cutlass/arch/arch.h>

namespace tilemega::arch {

using Sm80 = cutlass::arch::Sm80;
using Sm86 = cutlass::arch::Sm86;
using Sm89 = cutlass::arch::Sm89;
using Sm90 = cutlass::arch::Sm90;
using Sm100 = cutlass::arch::Sm100;
using Sm120 = cutlass::arch::Sm120;

/// Compile-time capability switches. Code generation queries these switches;
/// it never infers a feature from an architecture version comparison.
template <class Arch>
struct Caps {
  static constexpr bool kBf16TensorCore = false;
  static constexpr bool kCluster = false;
  static constexpr bool kTma = false;
  static constexpr bool kWarpSpecialized = false;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kL15 = false;
  static constexpr bool kNet = false;
  static constexpr bool kCpAsync = false;
  static constexpr bool kMbarrier = false;
  /// R8 BE-2: CUTLASS ships a BF16 tensor-op `CollectiveBuilder` for this
  /// architecture. A capability of the *toolchain*, measured rather than
  /// assumed -- `docs/experiments/BACKEND/be2_collective/` records the probe
  /// that instantiates the builder for each arch, including the two that
  /// refuse. Business code asks this; it never asks which CUTLASS this is.
  static constexpr bool kBf16CollectiveBuilder = false;
  static constexpr int kMaxClusterSize = 1;
  static constexpr char const* kCollective = "unsupported";
};

template <>
struct Caps<Sm80> {
  static constexpr bool kBf16TensorCore = true;
  static constexpr bool kCluster = false;
  static constexpr bool kTma = false;
  static constexpr bool kWarpSpecialized = false;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kL15 = false;
  static constexpr bool kNet = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr bool kBf16CollectiveBuilder = false;
  static constexpr int kMaxClusterSize = 1;
  static constexpr char const* kCollective = "cp.async multistage";
};

template <>
struct Caps<Sm86> : Caps<Sm80> {};

template <>
struct Caps<Sm89> : Caps<Sm80> {};

template <>
struct Caps<Sm90> {
  static constexpr bool kBf16TensorCore = true;
  static constexpr bool kCluster = true;
  static constexpr bool kTma = true;
  static constexpr bool kWarpSpecialized = true;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kL15 = false;
  static constexpr bool kNet = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr bool kBf16CollectiveBuilder = true;
  static constexpr int kMaxClusterSize = 8;
  static constexpr char const* kCollective = "TMA warp-specialized";
};

// Blackwell datacenter (sm_100) is the only target with tcgen05, and so the
// only one where the tmem and L1.5 lanes of the cost model exist at all.
// Nothing here has been measured: TileMega has never run on one, which is
// exactly why the lanes carry a status instead of a number.
template <>
struct Caps<Sm100> {
  static constexpr bool kBf16TensorCore = true;
  static constexpr bool kCluster = true;
  static constexpr bool kTma = true;
  static constexpr bool kWarpSpecialized = true;
  static constexpr bool kTcgen05 = true;
  static constexpr bool kL15 = true;
  static constexpr bool kNet = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr bool kBf16CollectiveBuilder = true;
  static constexpr int kMaxClusterSize = 8;
  static constexpr char const* kCollective = "TMA warp-specialized (tcgen05)";
};

// Blackwell GeForce (sm_120) is deliberately not derived from a numerically
// earlier architecture: it has TMA/cluster support but no tcgen05/TMEM.
template <>
struct Caps<Sm120> {
  static constexpr bool kBf16TensorCore = true;
  static constexpr bool kCluster = true;
  static constexpr bool kTma = true;
  static constexpr bool kWarpSpecialized = true;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kL15 = false;
  static constexpr bool kNet = false;
  static constexpr bool kCpAsync = true;
  static constexpr bool kMbarrier = true;
  static constexpr bool kBf16CollectiveBuilder = false;
  static constexpr int kMaxClusterSize = 8;
  static constexpr char const* kCollective = "TMA warp-specialized (SM120 MMA)";
};

/// Host-side projection of the same exact-tag capability table, used by
/// TargetSpec::Probe without duplicating architecture policy.
struct RuntimeCaps {
  bool bf16_tensor_core;
  bool cluster;
  bool tma;
  bool warp_specialized;
  bool tcgen05;
  bool l1_5;
  bool net;
  bool cp_async;
  bool mbarrier;
  int max_cluster_size;
  char const* collective;
};

template <class Arch>
constexpr RuntimeCaps RuntimeCapsFor() {
  return {Caps<Arch>::kBf16TensorCore, Caps<Arch>::kCluster, Caps<Arch>::kTma,
          Caps<Arch>::kWarpSpecialized, Caps<Arch>::kTcgen05,
          Caps<Arch>::kL15, Caps<Arch>::kNet,
          Caps<Arch>::kCpAsync, Caps<Arch>::kMbarrier,
          Caps<Arch>::kMaxClusterSize, Caps<Arch>::kCollective};
}

inline constexpr RuntimeCaps RuntimeCapsForTag(std::string_view tag) {
  return tag == "sm_80"  ? RuntimeCapsFor<Sm80>()
       : tag == "sm_86"  ? RuntimeCapsFor<Sm86>()
       : tag == "sm_89"  ? RuntimeCapsFor<Sm89>()
       : tag == "sm_90"  ? RuntimeCapsFor<Sm90>()
       : tag == "sm_100" ? RuntimeCapsFor<Sm100>()
       : tag == "sm_120" ? RuntimeCapsFor<Sm120>()
                           : RuntimeCapsFor<void>();
}

/// The arch tag codegen spells into the generated source, and the
/// `major*100 + minor*10` identifier that travels with it. One numbering with
/// `__CUDA_ARCH__` below, so a generated source and the device pass compare
/// the same integer instead of two spellings of the same thing.
template <class Arch>
struct ArchId {
  static constexpr int kValue = 0;
  static constexpr char const* kTag = "unsupported";
};
template <> struct ArchId<Sm80>  { static constexpr int kValue = 800;  static constexpr char const* kTag = "sm_80"; };
template <> struct ArchId<Sm86>  { static constexpr int kValue = 860;  static constexpr char const* kTag = "sm_86"; };
template <> struct ArchId<Sm89>  { static constexpr int kValue = 890;  static constexpr char const* kTag = "sm_89"; };
template <> struct ArchId<Sm90>  { static constexpr int kValue = 900;  static constexpr char const* kTag = "sm_90"; };
template <> struct ArchId<Sm100> { static constexpr int kValue = 1000; static constexpr char const* kTag = "sm_100"; };
template <> struct ArchId<Sm120> { static constexpr int kValue = 1200; static constexpr char const* kTag = "sm_120"; };

/// The inverse: the generated source carries the identifier, and the TaskBody
/// instantiation needs the type back. `void` for an identifier no build of
/// this compiler knows, which every caller turns into a hard failure rather
/// than a silent fallback.
template <int Id> struct ArchFromId { using type = void; };
template <> struct ArchFromId<800>  { using type = Sm80; };
template <> struct ArchFromId<860>  { using type = Sm86; };
template <> struct ArchFromId<890>  { using type = Sm89; };
template <> struct ArchFromId<900>  { using type = Sm90; };
template <> struct ArchFromId<1000> { using type = Sm100; };
template <> struct ArchFromId<1200> { using type = Sm120; };

/// Host-side tag -> identifier, for codegen and for the runtime check.
inline int ArchIdForTag(std::string_view tag) {
  return tag == "sm_80"  ? ArchId<Sm80>::kValue
       : tag == "sm_86"  ? ArchId<Sm86>::kValue
       : tag == "sm_89"  ? ArchId<Sm89>::kValue
       : tag == "sm_90"  ? ArchId<Sm90>::kValue
       : tag == "sm_100" ? ArchId<Sm100>::kValue
       : tag == "sm_120" ? ArchId<Sm120>::kValue
                         : 0;
}

// The only __CUDA_ARCH__ selection in include/ or lib/. Exact matches avoid
// accidentally treating sm_120 as an sm_100 tcgen05 target.
#if defined(__CUDA_ARCH__)
#  if __CUDA_ARCH__ == 1200
using CurrentArch = Sm120;
#  elif __CUDA_ARCH__ == 1000
using CurrentArch = Sm100;
#  elif __CUDA_ARCH__ == 900
using CurrentArch = Sm90;
#  elif __CUDA_ARCH__ == 890
using CurrentArch = Sm89;
#  elif __CUDA_ARCH__ == 860
using CurrentArch = Sm86;
#  elif __CUDA_ARCH__ == 800
using CurrentArch = Sm80;
#  endif
#else
// The host pass has no architecture of its own, yet nvcc still parses every
// __device__ body in it.  `void` keeps those bodies well-formed while claiming
// nothing: Caps<void> is the primary template, with every capability off.
using CurrentArch = void;
#endif

/// True only inside the device pass.  A capability assertion about
/// `CurrentArch` is meaningful only there, and this is the one file allowed to
/// know how to tell the two passes apart.
inline constexpr bool kDevicePass =
#if defined(__CUDA_ARCH__)
    true;
#else
    false;
#endif

}  // namespace tilemega::arch
