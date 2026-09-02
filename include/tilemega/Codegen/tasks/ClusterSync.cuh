// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- cluster-scoped synchronization, §5.5's second lowering path.
//
// Policy vs availability. Which path a stage takes is decided by
// `arch::Caps<Arch>::kCluster` at every use site, never by an architecture
// comparison and never by a preprocessor test. The one `#if` in this file is
// not a second policy switch: cooperative_groups *declares* `cluster_group`
// only when the device pass targets sm_90+, so on sm_89 the names fail
// lookup rather than compiling to nothing, and no `if constexpr` can discard
// a name that was never declared. `ClusterSync<Arch>` static_asserts that the
// two agree, so a target whose capability table claims clusters can never
// silently fall through to the single-CTA stubs.
#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <tilemega/Target/ArchDispatch.h>

#include <cstdint>

namespace tilemega::codegen {

#if defined(_CG_HAS_CLUSTER_GROUP)
inline constexpr bool kClusterGroupAvailable = true;
#else
inline constexpr bool kClusterGroupAvailable = false;
#endif

/// One epoch counter per CTA, held in that CTA's own shared memory. A
/// consumer in the same cluster reads its producer's counter through
/// distributed shared memory, so the release never leaves the GPC -- that is
/// the entire reason the cluster path exists next to the global one.
struct ClusterEvent {
  unsigned long long epoch;
};

template <class Arch>
struct ClusterSync {
  static constexpr bool kEnabled = arch::Caps<Arch>::kCluster;
  static constexpr int kMaxSize = arch::Caps<Arch>::kMaxClusterSize;
  static_assert(!kEnabled || kClusterGroupAvailable,
                "the capability table claims clusters but this device pass "
                "has no cooperative_groups cluster_group");

  __device__ static unsigned Rank() {
    if constexpr (kEnabled) {
#if defined(_CG_HAS_CLUSTER_GROUP)
      return cooperative_groups::this_cluster().block_rank();
#endif
    }
    return 0u;
  }

  __device__ static unsigned Size() {
    if constexpr (kEnabled) {
#if defined(_CG_HAS_CLUSTER_GROUP)
      return cooperative_groups::this_cluster().num_blocks();
#endif
    }
    return 1u;
  }

  /// Cluster-wide barrier. On a target without clusters the cluster is one
  /// CTA, so the CTA barrier *is* the cluster barrier -- the stub is the
  /// degenerate case, not a no-op that quietly drops an ordering edge.
  __device__ static void Sync() {
    if constexpr (kEnabled) {
#if defined(_CG_HAS_CLUSTER_GROUP)
      cooperative_groups::this_cluster().sync();
      return;
#endif
    }
    __syncthreads();
  }

  /// The DSMEM address of `self` as it lives in CTA `rank` of this cluster.
  __device__ static ClusterEvent* Peer(ClusterEvent* self, unsigned rank) {
    if constexpr (kEnabled) {
#if defined(_CG_HAS_CLUSTER_GROUP)
      return cooperative_groups::this_cluster().map_shared_rank(
          self, static_cast<int>(rank));
#endif
    }
    return rank == 0u ? self : nullptr;
  }

  /// Point-to-point release (§8.5): every write this CTA published must be
  /// visible before the epoch a peer polls. The fence is cluster-scoped and
  /// not device-scoped precisely because the consumer is another CTA of the
  /// same cluster; the CTA barrier is the writers converging, not a cluster
  /// barrier -- `Publish`/`WaitPeer` are the pairwise form, and a CTA that
  /// needs the whole cluster calls `Sync` instead.
  __device__ static void Publish(ClusterEvent* self, unsigned long long value) {
    __threadfence_block();
    __syncthreads();
    if (threadIdx.x == 0) {
      if constexpr (kEnabled) {
        asm volatile("fence.acq_rel.cluster;" ::: "memory");
      }
      // Monotone (§8.2): never reset between iterations, so a late CTA of
      // iteration i can never be read as an early one of iteration i+1.
      *reinterpret_cast<volatile unsigned long long*>(&self->epoch) = value;
    }
  }

  /// §8.1/§8.3: one polling thread, backoff, monotone compare.
  __device__ static void WaitPeer(ClusterEvent* self, unsigned rank,
                                  unsigned long long need) {
    if (threadIdx.x == 0) {
      ClusterEvent* peer = Peer(self, rank);
      while (*reinterpret_cast<volatile unsigned long long*>(&peer->epoch) < need)
        __nanosleep(64);
      if constexpr (kEnabled) {
        asm volatile("fence.acq_rel.cluster;" ::: "memory");
      }
    }
    __syncthreads();
  }

  /// Two-level stage barrier: the cluster closes over itself in hardware,
  /// then one CTA per cluster carries the arrival to the global counter. The
  /// global half is unchanged from `GridBarrier` -- clusters change who pays
  /// the L2 round trip, not what the barrier means. `Publish`/`WaitPeer` are
  /// deliberately *not* used here: `cluster.sync()` already orders every CTA
  /// of the cluster, so a DSMEM epoch handshake on top of it would be dead
  /// weight. They are the pairwise path, for a consumer that waits on one
  /// named producer rather than on its whole cluster.
  __device__ static void StageBarrier(unsigned long long* arrivals,
                                      unsigned long long* epoch,
                                      unsigned long long iteration,
                                      unsigned clusters) {
    __threadfence_block();
    Sync();
    if (clusters <= 1u) return;
    unsigned long long const needed =
        static_cast<unsigned long long>(clusters) * (iteration + 1ull);
    if (Rank() == 0u) {
      __threadfence();
      if (threadIdx.x == 0) {
        unsigned long long ticket = atomicAdd(arrivals, 1ull);
        if (ticket + 1ull == needed) {
          __threadfence();
          atomicExch(epoch, iteration + 1ull);
        } else {
          while (atomicAdd(epoch, 0ull) < iteration + 1ull) __nanosleep(64);
        }
      }
      __syncthreads();
    }
    // The cluster's rank 0 held the global wait; the rest of the cluster
    // learns about it here, and only then is the whole grid ordered.
    Sync();
    __threadfence();
  }
};

}  // namespace tilemega::codegen
