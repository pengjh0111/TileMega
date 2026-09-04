#!/usr/bin/env bash
# TileMega Part 7 -- cluster / DSMEM validation. Requires sm_90+ hardware.
#
# Self-contained: this script carries its own CUDA sources and needs nothing
# from the build tree, only the repository headers, nvcc and a GPU whose
# TargetSpec probe reports caps.cluster == true. It hard-fails on any other
# machine rather than silently measuring the single-CTA fallback.
#
#   bash docs/experiments/CLUSTER/run_on_cluster_gpu.sh            # 200 fresh runs
#   RUNS=500 bash docs/experiments/CLUSTER/run_on_cluster_gpu.sh
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
    -x cu "$work/probe.cu" "$root/lib/Target/TargetSpec.cpp" \
    "$root/lib/Support/Json.cpp" -o "$work/probe" \
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

# ------------------------------------------------------------ the megakernel
# The primitives above are validated in isolation.  This arm is the thing the
# rest of the repository actually runs: the generated L1/L2 megakernel, built
# with TILEMEGA_GENERATED_CLUSTER_DIM so its stage barrier is
# `ClusterSync::StageBarrier` and its launch is `cudaLaunchKernelEx`.
#
# It needs the build tree, which the sections above deliberately do not, so it
# announces a skip rather than failing when the artifacts are absent.  What it
# will not do is run without checking: a "cluster" arm whose cubin contains no
# `barrier.cluster` is a flat kernel wearing a label, and that is a hard fail.
# Both accepted models, so a cluster result is a property of the mechanism and
# not of one graph.
mega_models=(
  "gqa2|$root/docs/experiments/E2E_GEN/raw/generated_e2e.cu|$root/docs/experiments/E2E/fixture"
  "mha4|$root/docs/experiments/P3_GENERALIZATION/raw/generated.cu|$root/docs/experiments/P3_GENERALIZATION/raw/fixture"
)
mega_lib="$root/build-phase12/libtilemega.a"
: > "$out/megakernel.tsv"
printf 'model\tcluster_dim\tbarrier_cluster_asm\tpass\ttotal\thash\tl1_ms\tl2_ms\n' \
    >> "$out/megakernel.tsv"
have_all=1
for entry in "${mega_models[@]}"; do
  IFS='|' read -r _ src fixture <<<"$entry"
  [[ -e "$src" && -d "$fixture" ]] || have_all=0
done
if [[ ! -e "$mega_lib" || "$have_all" != 1 ]]; then
  echo "SKIPPED megakernel arm: build-phase12/libtilemega.a, a generated" \
       "source or a fixture is missing" | tee -a "$out/megakernel.tsv"
else
  mega_inc=("${inc[@]}" -I"$root/third_party/cutlass/tools/util/include"
            -I"$root/third_party/cutlass/test")
  mega_runs=${MEGA_RUNS:-50}
  med() { sort -n | awk '{v[NR]=$1} END {if(!NR){print "nan";exit}
      printf "%.6f\n",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
  for entry in "${mega_models[@]}"; do
    IFS='|' read -r model mega_src mega_fixture <<<"$entry"
    # The flat control's output hash is the reference every clustered arm has
    # to reproduce; §7 P5.2 asks for bitwise identity, not for a tolerance.
    ref_hash=
    for dim in 1 2 4 8; do
      if (( dim > max_cluster )); then
        printf '%s\t%d\tskipped_max_cluster_size_%s\t-\t-\t-\t-\t-\n' \
            "$model" "$dim" "$max_cluster" >> "$out/megakernel.tsv"
        continue
      fi
      "$nvcc" "$mega_src" -std=c++17 -O2 -arch="$arch" \
          -DTILEMEGA_GENERATED_CLUSTER_DIM="$dim" "${mega_inc[@]}" \
          "$mega_lib" -L"${CUDA_HOME:-/usr/local/cuda}/lib64" -lcudart \
          -o "$work/mega_${model}_$dim" 2>> "$out/build.txt"
      # `UCGABAR_ARV`/`UCGABAR_WAIT` is how ptxas spells `barrier.cluster` in
      # SASS (CGA = cooperative grid array).  Verified by cross-compiling this
      # very source on an sm_89 box: 0 at dim 1, 8 at dim 2 and dim 8, on both
      # sm_90 and sm_120.  Grepping for a plausible-looking `BAR.CLUSTER`
      # instead would have failed every arm on a machine where everything was
      # correct.
      asm=$(cuobjdump -sass "$work/mega_${model}_$dim" | grep -c 'UCGABAR' || true)
      # dim 1 is the control and must contain no cluster barrier at all; every
      # other arm must contain one, or it is not measuring what it claims to.
      if (( dim == 1 && asm != 0 )); then
        echo "FAIL: $model flat control emitted a cluster barrier" >&2; exit 4
      fi
      if (( dim > 1 && asm == 0 )); then
        echo "FAIL: $model cluster dim $dim emitted no cluster barrier; the" \
             "arm would measure the flat kernel under a cluster label" >&2
        exit 4
      fi
      pass=0; l1s=; l2s=; hashes=
      for ((run = 1; run <= mega_runs; ++run)); do
        if line=$("$work/mega_${model}_$dim" "$mega_fixture" 2>&1); then
          grep -q '^RESULT status=PASS' <<<"$line" && pass=$((pass + 1))
          t=$(grep -o 'l1_ms=[0-9.]*' <<<"$line" | head -1 | cut -d= -f2)
          u=$(grep -o 'l2_ms=[0-9.]*' <<<"$line" | head -1 | cut -d= -f2)
          h=$(grep -o 'l1=[0-9a-f]*' <<<"$line" | head -1 | cut -d= -f2)
          l1s+="$t"$'\n'; l2s+="$u"$'\n'; hashes+="$h"$'\n'
        fi
      done
      hash=$(printf '%s' "$hashes" | sort -u | grep -v '^$' | paste -sd, -)
      [[ -z "$ref_hash" ]] && ref_hash="$hash"
      if [[ "$hash" != "$ref_hash" ]]; then
        echo "FAIL: $model dim $dim output hash $hash != flat control $ref_hash" >&2
        overall=1
      fi
      printf '%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n' "$model" "$dim" "$asm" "$pass" \
          "$mega_runs" "${hash:-none}" "$(printf '%s' "$l1s" | med)" \
          "$(printf '%s' "$l2s" | med)" >> "$out/megakernel.tsv"
      (( pass == mega_runs )) || overall=1
    done
  done
  cat "$out/megakernel.tsv"
fi

exit $overall
