// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- the single compile-time architecture dispatch point.
// Skeleton refs: §3.4, §5.3, §8.7.
//
// This branch targets MetaX C550 only (XCORE1000 family, ISA xcore1002). The
// NVIDIA architectures and their CUTLASS tags are not part of this build.
#pragma once

#include <string_view>

namespace tilemega::arch {

/// MetaX XCORE1000 (C550). Tag `xcore1000`, identifier 7000.
struct Xcore1000 {};

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
  static constexpr bool kMbarrierTryWait = false;
  static constexpr bool kMbarrierTx = false;
  static constexpr bool kBulkCopy = false;
  static constexpr bool kBulkPrefetch = false;
  static constexpr bool kPdl = false;
  static constexpr bool kBf16CollectiveBuilder = false;
  static constexpr int kMaxClusterSize = 1;
  static constexpr int kWarpSize = 32;
  static constexpr int kMaxThreadsPerBlock = 0;
  static constexpr char const* kCollective = "unsupported";
};

// Values verified by tools/probes/maca (docs/results/MX1_probe.md). The NVIDIA
// async features (cp.async PTX, mbarrier, TMA, bulk, PDL, cluster) do not
// exist; MetaX async copies go through Target/Maca/DevicePrimitives.h.
template <>
struct Caps<Xcore1000> {
  static constexpr bool kBf16TensorCore = true;
  static constexpr bool kCluster = false;
  static constexpr bool kTma = false;
  static constexpr bool kWarpSpecialized = false;
  static constexpr bool kTcgen05 = false;
  static constexpr bool kL15 = false;
  static constexpr bool kNet = false;
  static constexpr bool kCpAsync = false;
  static constexpr bool kMbarrier = false;
  static constexpr bool kMbarrierTryWait = false;
  static constexpr bool kMbarrierTx = false;
  static constexpr bool kBulkCopy = false;
  static constexpr bool kBulkPrefetch = false;
  static constexpr bool kPdl = false;
  static constexpr bool kBf16CollectiveBuilder = false;
  static constexpr int kMaxClusterSize = 1;
  static constexpr int kWarpSize = 64;
  static constexpr int kMaxThreadsPerBlock = 1024;
  static constexpr char const* kCollective = "MetaX wave64 MMA";
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
  bool mbarrier_try_wait;
  bool mbarrier_tx;
  bool bulk_copy;
  bool bulk_prefetch;
  bool pdl;
};

template <class Arch>
constexpr RuntimeCaps RuntimeCapsFor() {
  return {Caps<Arch>::kBf16TensorCore, Caps<Arch>::kCluster, Caps<Arch>::kTma,
          Caps<Arch>::kWarpSpecialized, Caps<Arch>::kTcgen05,
          Caps<Arch>::kL15, Caps<Arch>::kNet,
          Caps<Arch>::kCpAsync, Caps<Arch>::kMbarrier,
          Caps<Arch>::kMaxClusterSize, Caps<Arch>::kCollective,
          Caps<Arch>::kMbarrierTryWait, Caps<Arch>::kMbarrierTx, Caps<Arch>::kBulkCopy, Caps<Arch>::kBulkPrefetch, Caps<Arch>::kPdl};
}

inline constexpr RuntimeCaps RuntimeCapsForTag(std::string_view tag) {
  return tag == "xcore1000" ? RuntimeCapsFor<Xcore1000>() : RuntimeCapsFor<void>();
}

/// The arch tag codegen spells into the generated source and the identifier
/// that travels with it (`TILEMEGA_ARCH_ID`).
template <class Arch>
struct ArchId {
  static constexpr int kValue = 0;
  static constexpr char const* kTag = "unsupported";
};
template <> struct ArchId<Xcore1000> { static constexpr int kValue = 7000; static constexpr char const* kTag = "xcore1000"; };

/// The inverse: `void` for an identifier this build does not know, which every
/// caller turns into a hard failure rather than a silent fallback.
template <int Id> struct ArchFromId { using type = void; };
template <> struct ArchFromId<7000> { using type = Xcore1000; };

inline int ArchIdForTag(std::string_view tag) {
  return tag == "xcore1000" ? ArchId<Xcore1000>::kValue : 0;
}

// The only device-architecture selection in include/ or lib/. cucc defines
// __CUDA_ARCH__ (as 800) next to __MACA_ARCH__, so __MACA_ARCH__ is tested
// first and a pass with only __CUDA_ARCH__ comes from an NVIDIA compiler.
#if defined(__MACA_ARCH__)
#  if __MACA_ARCH__ == 1002
using CurrentArch = Xcore1000;
#  else
#    error "TileMega xcore1000 build: unsupported __MACA_ARCH__ (expected 1002, C550)"
#  endif
#elif defined(__CUDA_ARCH__)
#  error "TileMega on this branch targets MetaX only; NVIDIA device passes are not supported"
#else
// The host pass has no architecture of its own, yet the compiler still parses
// every __device__ body in it. Caps<void> keeps every capability off.
using CurrentArch = void;
#endif

/// True only inside the device pass.
inline constexpr bool kDevicePass =
#if defined(__MACA_ARCH__)
    true;
#else
    false;
#endif

}  // namespace tilemega::arch
