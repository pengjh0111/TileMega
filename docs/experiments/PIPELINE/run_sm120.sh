#!/usr/bin/env bash
# R7 B1 paging and cross-task pipelining ablation on sm_120.
#
# NOT RUN ON THIS MACHINE. The development host is a 4090 (sm_89). SELF_CHECK=1
# was run here and covers the guards, the disk and capability checks and the
# projection inventory; the builds, the occupancy probe and the rounds all need
# the target device and were not run. Run this on the target.
#
# Why sm_120 is worth its own run: the prefetch page is dynamic shared memory
# appended after the `TaskSmem` union on the L2 worker kernel only, so it is
# paid in residency. sm_120's shared capacity per SM is not sm_89's, so the
# occupancy trade-off the page buys can come out the other way round, and the
# three arms can rank differently for that reason alone.
#
# The Plan is NOT re-solved here, unlike `E2E_REAL/run_sm120.sh`. There the
# Plan is the object under test; here the object is the mechanism, and the
# ablation is arm against arm inside one cell, all three sharing FORK6's
# sm_89 selection. That keeps the three arms comparable to each other and to
# the sm_89 run -- and it means the selection itself is not optimal for
# sm_120. `occupancy.py` reports whether each arm still keeps the residency
# its cell was selected at; if it does not, the arms are still comparable but
# the cell is off its design point, and the line in `status.txt` says so.
#
# Artifacts, under --out (default $PWD/sm120):
#   <cell>/specs.json                 the three arms, macro for macro
#   <cell>/build/<arm>.{log,json}     the nvcc command, both hashes and the head
#   <cell>/correctness/<arm>/r*.log   one file pair per fresh process
#   <cell>/e2e/<arm>/r*.log           the rotated timing rounds
#   occupancy.tsv occupancy_cells.tsv occupancy_arms.tsv
#   e2e.tsv                           the paired ratios with their intervals
#   status.txt                        one line per completed step
set -euo pipefail
trap 'echo "FAILED at line $LINENO" >>"${OUT:-/tmp}/status.txt"' ERR

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$PWD/sm120}"
ROUNDS="${ROUNDS:-50}"
E2E_ROUNDS="${E2E_ROUNDS:-25}"
JOBS="${JOBS:-4}"
SELF_CHECK="${SELF_CHECK:-0}"
NEED_MIB=32768

# An inherited TILEMEGA_* would silently change what is compiled or run, and
# would not appear in any artifact. Refuse rather than guess.
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
  echo "SELF_CHECK=1: not compiling for sm_120 and not running" | tee -a "$OUT/status.txt"
elif [ "$cap" != "120" ]; then
  echo "this script is for sm_120; found sm_$cap" >&2
  exit 4
fi
ARCH="sm_${cap}"

# The projected sources live in the repository and are architecture free --
# `run.py` reads them from there whatever --raw says -- so only the cubins and
# the rounds below belong to the target machine.
for cell in gqa2_s4 gqa2_s128 mha4_s4 mha4_s128 real_s4 real_s128; do
  src="$REPO/docs/experiments/PIPELINE/raw/$cell/src"
  [ -d "$src" ] || { echo "missing projection $src; run regen.py first" >&2; exit 5; }
done
echo "SOURCES six projections present" | tee -a "$OUT/status.txt"

if [ "$SELF_CHECK" = "1" ]; then
  echo "SELF_CHECK projections checked, nothing built" | tee -a "$OUT/status.txt"
  echo "COMPLETE" | tee -a "$OUT/status.txt"
  exit 0
fi

echo "BUILD three arms, six cells" | tee -a "$OUT/status.txt"
python3 "$REPO/docs/experiments/PIPELINE/run.py" build \
  --raw "$OUT" --arch "$ARCH" --jobs "$JOBS"

echo "OCCUPANCY on this device" | tee -a "$OUT/status.txt"
PIPELINE_ARCH="$ARCH" PIPELINE_OUT="$OUT" PIPELINE_WORK="$OUT/occ" \
  python3 "$REPO/docs/experiments/PIPELINE/occupancy.py" | tee -a "$OUT/status.txt"

echo "CORRECTNESS rounds=$ROUNDS" | tee -a "$OUT/status.txt"
python3 "$REPO/docs/experiments/PIPELINE/run.py" correctness \
  --raw "$OUT" --arch "$ARCH" --rounds "$ROUNDS"

echo "E2E rounds=$E2E_ROUNDS" | tee -a "$OUT/status.txt"
python3 "$REPO/docs/experiments/PIPELINE/run.py" e2e \
  --raw "$OUT" --arch "$ARCH" --rounds "$E2E_ROUNDS"
PIPELINE_OUT="$OUT" python3 "$REPO/docs/experiments/PIPELINE/e2e.py" | tee -a "$OUT/status.txt"

echo "COMPLETE" | tee -a "$OUT/status.txt"
