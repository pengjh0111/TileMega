#!/usr/bin/env bash
# TileMega P4.1 -- per-target calibration. Runs on the machine being measured.
#
# Self-contained: compiles the calibration sources directly with nvcc, so it
# needs only the repository, CUDA and a GPU. It does not need MLIR, isl or a
# configured build tree.
#
# The profile it writes describes the GPU it ran on and nothing else. The
# script refuses to write configs/targets/sm_XX.json for an architecture the
# machine does not have, so an uncalibrated profile stays uncalibrated rather
# than being filled with numbers from a different device.
#
#   bash docs/experiments/CALIB/run.sh                 # calibrate this machine
#   EXPECT=sm_80  bash docs/experiments/CALIB/run.sh   # A100
#   EXPECT=sm_90  bash docs/experiments/CALIB/run.sh   # H100
#   EXPECT=sm_120 bash docs/experiments/CALIB/run.sh   # Blackwell
#   REPEATS=101 DEVICE=1 bash docs/experiments/CALIB/run.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=${TILEMEGA_SOURCE_DIR:-$(cd "$here/../../.." && pwd)}
device=${DEVICE:-0}
repeats=${REPEATS:-41}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

nvcc=${CUDACXX:-$(command -v nvcc || true)}
if [[ -z "$nvcc" ]]; then
  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [[ -x "$candidate" ]] && { nvcc=$candidate; break; }
  done
fi
[[ -x "$nvcc" ]] || { echo "nvcc not found" >&2; exit 2; }

inc=(-I"$root/include"
     -I"$root/third_party/cutlass/include"
     -I"$root/third_party/cutlass/tools/util/include")

# -arch=native: calibration measures the machine it runs on, so it is compiled
# for the device present rather than for any project-wide default.
"$nvcc" -std=c++17 -O2 -arch=native --expt-relaxed-constexpr \
  -Wno-deprecated-gpu-targets "${inc[@]}" -x cu \
  "$root/tools/tilemega-calibrate.cpp" \
  "$root/lib/Target/Calibration.cu" \
  "$root/lib/Target/GemmCalibration.cu" \
  "$root/lib/Target/TargetSpec.cpp" \
  "$root/lib/Support/Json.cpp" \
  -o "$work/calibrate"

# ---------------------------------------------------------------- self-check
# Probe first, on a run that skips the GEMM fits, and read the architecture out
# of the profile itself rather than out of nvidia-smi: the profile is what the
# solver loads, so the tag written there is the one that has to match.
"$work/calibrate" --device "$device" --repeats 1 --skip-streamk --quiet \
  --out "$work/probe.json" >/dev/null 2>&1
arch=$(sed -n 's/.*"arch_tag": *"\([^"]*\)".*/\1/p' "$work/probe.json" | head -1)
[[ -n "$arch" ]] || { echo "could not read arch_tag from the probe" >&2; exit 3; }

expect=${EXPECT:-$arch}
if [[ "$arch" != "$expect" ]]; then
  echo "this machine is $arch, not $expect; refusing to write" \
       "configs/targets/$expect.json" >&2
  echo "run this script on $expect hardware -- $expect stays uncalibrated" >&2
  exit 3
fi
case "$arch" in
  sm_80|sm_89|sm_90|sm_120) ;;
  *) echo "no profile is tracked for $arch" >&2; exit 3 ;;
esac

# ---------------------------------------------------------------- calibrate
out="$root/configs/targets/$arch.json"
log="$here/raw/$arch.log"
mkdir -p "$here/raw"
"$work/calibrate" --device "$device" --repeats "$repeats" --out "$out" \
  2>&1 | tee "$log"

grep -q '"calibrated": true' "$out" || {
  echo "the run did not complete a full calibration; $out left as written" >&2
  exit 4
}
echo
echo "wrote $out"
echo "log   $log"

# cluster.sync() is still uncalibrated on every target: measuring it needs a
# cluster launch, which lands with the Part 7 cluster support. The profile
# records cluster_sync_calibrated=false rather than a plausible number.
