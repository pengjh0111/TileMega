#!/usr/bin/env bash
# R8: the reworked backend on sm_120 -- correctness and the three levels.
#
# NOT RUN ON THIS MACHINE. The development host is a 4090 (sm_89). SELF_CHECK=1
# was run here and covers the capability checks, the disk budget and the
# presence of every input; the solves, the builds and the rounds all need the
# target device and were not run. Run this on the target.
#
# Why sm_120 is worth its own run: `Caps<Sm120>` has `tma`, `cluster` and
# `warp_specialized`, but CUTLASS 4.8's sm_120 builder refuses BF16 (see
# porting.md), so the GEMM takes the cp.async multistage collective there --
# the same code path as sm_89 on a very different machine. That makes it the
# one part where this round's selection logic can be checked end to end
# without the role work BE-5 did not deliver.
#
# Artifacts, under --out (default $PWD/sm120):
#   <cell>/auto.cu, solve.log, solve.json   the Plan, solved on the target
#   <cell>/build/default.{log,json}         nvcc command, hashes, head
#   <cell>/correctness/r*.log, r*.json      one file pair per fresh process
#   <cell>/timing/r*.log                    the three-level rounds
#   models.tsv, occupancy.tsv, status.txt
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$PWD/sm120}"
ROUNDS="${ROUNDS:-50}"
TIMING_ROUNDS="${TIMING_ROUNDS:-5}"
SELF_CHECK="${SELF_CHECK:-0}"
NEED_MIB=32768

trap 'echo "FAILED at line $LINENO" >>"${OUT:-/tmp}/status.txt"' ERR

# An inherited TILEMEGA_* would silently change what is compiled or run and
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
  echo "SELF_CHECK=1: not solving, not compiling for sm_120 and not running" | tee -a "$OUT/status.txt"
elif [ "$cap" != "120" ]; then
  echo "this script is for sm_120; found sm_$cap" >&2
  exit 4
fi

# The target description the Plan is solved against must name this device;
# the harness refuses a binary whose Plan names another (R8 BE-1).
TARGET="${TARGET:-$REPO/configs/targets/sm_120.json}"
if [ ! -f "$TARGET" ]; then
  echo "missing target description $TARGET; write one with tilemega-calibrate" \
    | tee -a "$OUT/status.txt"
  [ "$SELF_CHECK" = "1" ] || exit 5
fi
echo "TARGET $TARGET" | tee -a "$OUT/status.txt"

for model in gqa2 mha4; do
  src="$REPO/docs/experiments/SEQSCAN/raw/export/$model.json"
  [ -f "$src" ] || { echo "missing export $src" >&2; exit 6; }
done
echo "SOURCES two reference exports present" | tee -a "$OUT/status.txt"

if [ "$SELF_CHECK" = "1" ]; then
  echo "SELF_CHECK inputs checked, nothing solved or built" | tee -a "$OUT/status.txt"
  echo "COMPLETE" | tee -a "$OUT/status.txt"
  exit 0
fi

echo "SOLVE four cells on this device's target" | tee -a "$OUT/status.txt"
TILEMEGA_TARGET="$TARGET" python3 "$REPO/docs/experiments/BACKEND/run_models.py" solve --out "$OUT"

echo "CORRECTNESS rounds=$ROUNDS" | tee -a "$OUT/status.txt"
python3 "$REPO/docs/experiments/BACKEND/run_models.py" correctness --out "$OUT" \
  --rounds "$ROUNDS" --arch sm_120

echo "TIMING rounds=$TIMING_ROUNDS" | tee -a "$OUT/status.txt"
python3 "$REPO/docs/experiments/BACKEND/run_models.py" timing --out "$OUT" \
  --timing-rounds "$TIMING_ROUNDS" --arch sm_120
python3 "$REPO/docs/experiments/BACKEND/run_models.py" report --out "$OUT" | tee -a "$OUT/status.txt"
python3 "$REPO/docs/experiments/BACKEND/occupancy.py" --models "$OUT" \
  --out "$OUT/occupancy.tsv" | tee -a "$OUT/status.txt"

echo "COMPLETE" | tee -a "$OUT/status.txt"
