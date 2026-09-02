#!/usr/bin/env bash
# TileMega Part 7 -- cluster / DSMEM validation. Requires sm_90+ hardware.
#
# Self-contained: this script carries its own CUDA sources and needs nothing
# from the build tree, only the repository headers, nvcc and a GPU whose
# TargetSpec probe reports caps.cluster == true. It hard-fails on any other
# machine rather than silently measuring the single-CTA fallback.
#
#   bash docs/experiments/CLUSTER/run_on_h100.sh            # 200 fresh runs
#   RUNS=500 bash docs/experiments/CLUSTER/run_on_h100.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=${TILEMEGA_SOURCE_DIR:-$(cd "$here/../../.." && pwd)}
runs=${RUNS:-200}
out="$here/raw"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

nvcc=${CUDACXX:-$(command -v nvcc || true)}
if [[ -z "$nvcc" ]]; then
  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [[ -x "$candidate" ]] && { nvcc=$candidate; break; }
  done
fi
[[ -x "$nvcc" ]] || { echo "nvcc not found" >&2; exit 2; }
mkdir -p "$out"

inc=(-I"$root/include" -I"$root/third_party/cutlass/include")

# ---------------------------------------------------------------- self-check
cat > "$work/probe.cu" <<'CPP'
#include <tilemega/Target/TargetSpec.h>
#include <cstdio>
int main() {
  tilemega::TargetSpec target = tilemega::TargetSpec::Probe();
  std::printf("%s %d %d %d\n", target.arch_tag.c_str(),
              target.caps.cluster ? 1 : 0, target.res.max_cluster_size,
              target.res.num_sms);
  return target.caps.cluster ? 0 : 3;
}
CPP
# TargetSpec.cpp is compiled as CUDA (-x cu): CUTLASS's host/device macros
# expand to __forceinline__ as soon as nvcc defines __NVCC__, so the host
# compiler cannot be handed the file directly.
"$nvcc" -std=c++17 -O2 -Wno-deprecated-gpu-targets "${inc[@]}" \
    -x cu "$work/probe.cu" "$root/lib/Target/TargetSpec.cpp" -o "$work/probe" \
    2> "$out/probe_build.txt" || { cat "$out/probe_build.txt" >&2; exit 2; }
read -r arch cluster max_cluster sms < <("$work/probe" || true)
if [[ "${cluster:-0}" != "1" ]]; then
  echo "FAIL: TargetSpec::Probe() reports caps.cluster == false on ${arch:-unknown}." >&2
  echo "      This experiment needs sm_90+; refusing to measure the" >&2
  echo "      single-CTA fallback and call it a cluster result." >&2
  exit 3
fi
case "$arch" in
  sm_90) arch_type=Sm90 ;;
  sm_120) arch_type=Sm120 ;;
  *) echo "FAIL: $arch has no cluster capability entry" >&2; exit 3 ;;
esac
printf 'arch\t%s\ncaps.cluster\t%s\nmax_cluster_size\t%s\nnum_sms\t%s\nruns\t%s\n' \
    "$arch" "$cluster" "$max_cluster" "$sms" "$runs" > "$out/target.txt"
echo "== $arch, cluster=$cluster, max_cluster_size=$max_cluster, ${sms} SMs, $runs runs"

# ------------------------------------------------------------------ the test
cat > "$work/cluster_test.cu" <<'CPP'
// Three checks, each against the shipped ClusterSync<Arch>:
//   A DSMEM visibility  -- every rank reads every peer's shared word
//   B pairwise ordering -- Publish/WaitPeer must order a payload, not just
//                          the epoch word (a missing release fence shows up
//                          here, probabilistically, which is why the caller
//                          runs this many fresh processes)
//   C two-level barrier -- StageBarrier's monotone counters (skeleton §8.2)
#include <tilemega/Codegen/tasks/ClusterSync.cuh>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tmc = tilemega::codegen;
using Arch = tilemega::arch::TILEMEGA_TEST_ARCH_TYPE;
using CS = tmc::ClusterSync<Arch>;

constexpr int kThreads = 128;
constexpr int kPayload = 64;

struct Block {
  tmc::ClusterEvent event;
  unsigned long long word;
  unsigned long long payload[kPayload];
};

__global__ void dsmem_visibility(unsigned long long* mismatches) {
  __shared__ Block block;
  if (threadIdx.x == 0) {
    block.word = 1000ull * (CS::Rank() + 1ull) + 7ull;
    block.event.epoch = 0ull;
  }
  __syncthreads();
  CS::Sync();
  unsigned const size = CS::Size();
  if (threadIdx.x == 0) {
    unsigned long long bad = 0;
    for (unsigned rank = 0; rank < size; ++rank) {
      Block* peer = reinterpret_cast<Block*>(
          CS::Peer(&block.event, rank));
      if (peer->word != 1000ull * (rank + 1ull) + 7ull) ++bad;
    }
    if (bad) atomicAdd(mismatches, bad);
  }
  CS::Sync();
}

__global__ void pairwise_ordering(unsigned long long* mismatches,
                                  unsigned long long iterations) {
  __shared__ Block block;
  if (threadIdx.x == 0) block.event.epoch = 0ull;
  __syncthreads();
  unsigned const size = CS::Size();
  unsigned const peer_rank = (CS::Rank() + 1u) % size;
  for (unsigned long long it = 0; it < iterations; ++it) {
    for (int i = threadIdx.x; i < kPayload; i += blockDim.x)
      block.payload[i] = it * 31ull + static_cast<unsigned long long>(i);
    CS::Publish(&block.event, it + 1ull);
    CS::WaitPeer(&block.event, peer_rank, it + 1ull);
    if (threadIdx.x == 0) {
      Block* peer = reinterpret_cast<Block*>(CS::Peer(&block.event, peer_rank));
      unsigned long long bad = 0;
      for (int i = 0; i < kPayload; ++i)
        if (peer->payload[i] != it * 31ull + static_cast<unsigned long long>(i))
          ++bad;
      if (bad) atomicAdd(mismatches, bad);
    }
    CS::Sync();
  }
}

/// One counter per stage. After StageBarrier every CTA of the grid must see
/// the stage complete, so a CTA that runs ahead is caught by a count that is
/// short of the grid -- the same monotone-counter shape as §8.2.
__global__ void two_level_barrier(unsigned long long* counters,
                                  unsigned long long* arrivals,
                                  unsigned long long* epochs,
                                  unsigned long long* mismatches,
                                  unsigned stages, unsigned clusters) {
  __shared__ Block block;
  if (threadIdx.x == 0) block.event.epoch = 0ull;
  __syncthreads();
  for (unsigned stage = 0; stage < stages; ++stage) {
    if (threadIdx.x == 0) atomicAdd(&counters[stage], 1ull);
    CS::StageBarrier(&arrivals[stage], &epochs[stage], stage, clusters);
    if (threadIdx.x == 0 && counters[stage] != gridDim.x)
      atomicAdd(mismatches, 1ull);
  }
}

int main(int argc, char** argv) {
  int size = argc > 1 ? std::atoi(argv[1]) : 2;
  int clusters = argc > 2 ? std::atoi(argv[2]) : 8;
  unsigned iterations = argc > 3 ? std::atoi(argv[3]) : 32u;
  int grid = size * clusters;

  unsigned long long *mismatches, *counters, *arrivals, *epochs;
  cudaMallocManaged(&mismatches, sizeof(unsigned long long));
  cudaMallocManaged(&counters, 64 * sizeof(unsigned long long));
  cudaMallocManaged(&arrivals, 64 * sizeof(unsigned long long));
  cudaMallocManaged(&epochs, 64 * sizeof(unsigned long long));
  *mismatches = 0;
  for (int i = 0; i < 64; ++i) counters[i] = arrivals[i] = epochs[i] = 0;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim = {static_cast<unsigned>(size), 1, 1};
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = 0;
  config.attrs = attribute;
  config.numAttrs = 1;

  cudaError_t status = cudaLaunchKernelEx(&config, dsmem_visibility, mismatches);
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    status = cudaLaunchKernelEx(&config, pairwise_ordering, mismatches,
                                static_cast<unsigned long long>(iterations));
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaLaunchKernelEx(&config, two_level_barrier, counters, arrivals,
                                epochs, mismatches, 16u,
                                static_cast<unsigned>(clusters));
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    std::printf("LAUNCHFAIL %s\n", cudaGetErrorString(status));
    return 2;
  }
  std::printf("%s mismatches=%llu size=%d clusters=%d\n",
              *mismatches ? "MISMATCH" : "PASS", *mismatches, size, clusters);
  return *mismatches ? 1 : 0;
}
CPP

"$nvcc" -std=c++17 -O2 -arch="$arch" -DTILEMEGA_TEST_ARCH_TYPE="$arch_type" \
    "${inc[@]}" "$work/cluster_test.cu" -o "$work/cluster_test" \
    2> "$out/build.txt"
echo "built: $arch / $arch_type"

# ------------------------------------------------------------- fresh-process
: > "$out/runs.tsv"
printf 'cluster_size\trun\tstatus\n' >> "$out/runs.tsv"
printf 'cluster_size\tpass\ttotal\n' > "$out/summary.tsv"
overall=0
for size in 2 4 8; do
  if (( size > max_cluster )); then
    printf '%d\tskipped_max_cluster_size_%s\n' "$size" "$max_cluster" \
        >> "$out/summary.tsv"
    continue
  fi
  pass=0
  for ((run = 1; run <= runs; ++run)); do
    if line=$("$work/cluster_test" "$size" 8 32 2>&1); then
      status=PASS; pass=$((pass + 1))
    else
      status="FAIL:${line}"
      overall=1
    fi
    printf '%d\t%d\t%s\n' "$size" "$run" "$status" >> "$out/runs.tsv"
  done
  printf '%d\t%d\t%d\n' "$size" "$pass" "$runs" >> "$out/summary.tsv"
  echo "cluster size $size: $pass/$runs"
done

cat "$out/summary.tsv"
exit $overall
