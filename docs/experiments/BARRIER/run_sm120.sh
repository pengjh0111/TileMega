#!/usr/bin/env bash
# R8: the role-granularity release litmus on sm_120.
#
# NOT RUN ON THIS MACHINE. The development host is a 4090 (sm_89). SELF_CHECK=1
# was run here and covers the capability checks, the disk budget and the build
# of the litmus itself; the rounds need the target device and were not run.
#
# Why sm_120 is worth its own run: it has clusters, so a producer role and a
# consumer role can sit in different CTAs of one cluster and publish through
# distributed shared memory -- a scope sm_89 cannot express at all. On sm_89
# the `nobarrier` control never failed (BARRIER/README.md), which is why §8.5
# was left unchanged; a part with a different publication scope is the next
# place that experiment can actually fail.
#
# Artifacts, under --out (default $PWD/sm120):
#   raw/<arm>_g<grid>_e<elements>/r*.log, r*.json   one pair per fresh process
#   raw/litmus.tsv                                   cell -> rounds, passing
#   status.txt
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$PWD/sm120}"
ROUNDS="${ROUNDS:-50}"
SELF_CHECK="${SELF_CHECK:-0}"
NEED_MIB=4096
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

trap 'echo "FAILED at line $LINENO" >>"${OUT:-/tmp}/status.txt"' ERR

for name in $(env | sed -n 's/^\(TILEMEGA_[A-Za-z0-9_]*\)=.*/\1/p'); do
  echo "refusing inherited $name; unset it and re-run" >&2
  exit 2
done

mkdir -p "$OUT"
: >"$OUT/status.txt"

free_mib=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
echo "DISK NEED_MIB=$NEED_MIB FREE_MIB=$free_mib" | tee -a "$OUT/status.txt"
[ "$free_mib" -ge "$NEED_MIB" ] || { echo "disk budget" >&2; exit 3; }

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
echo "COMPUTE_CAP sm_$cap" | tee -a "$OUT/status.txt"
if [ "$SELF_CHECK" = "1" ]; then
  echo "SELF_CHECK=1: building for this host, not running the matrix" | tee -a "$OUT/status.txt"
  arch="sm_$cap"
elif [ "$cap" != "120" ]; then
  echo "this script is for sm_120; found sm_$cap" >&2
  exit 4
else
  arch="sm_120"
fi

"$NVCC" -std=c++17 -O2 -arch="$arch" "$REPO/docs/experiments/BARRIER/litmus.cu" \
  -o "$OUT/litmus" >"$OUT/build.log" 2>&1
echo "BUILD litmus for $arch" | tee -a "$OUT/status.txt"

if [ "$SELF_CHECK" = "1" ]; then
  echo "SELF_CHECK litmus built, matrix not run" | tee -a "$OUT/status.txt"
  echo "COMPLETE" | tee -a "$OUT/status.txt"
  exit 0
fi

python3 "$REPO/docs/experiments/BARRIER/run_litmus.py" --binary "$OUT/litmus" \
  --out "$OUT/raw" --rounds "$ROUNDS" | tee -a "$OUT/status.txt"

echo "COMPLETE" | tee -a "$OUT/status.txt"
