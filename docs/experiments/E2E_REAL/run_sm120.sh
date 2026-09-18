#!/usr/bin/env bash
# R7 real-model end-to-end on sm_120.
#
# NOT RUN ON THIS MACHINE. The development host is a 4090 (sm_89); everything
# below was self-checked here with SELF_CHECK=1, which exercises every step
# except the sm_120 compilation and the runs themselves. Run it on the target.
#
# The Plan is re-solved on the target machine and never carried across: a
# materialized Plan binds the resident grid, which is a whole-kernel property of
# that machine's occupancy, so a Plan solved on sm_89 is not a Plan for sm_120.
#
# Artifacts, under --out (default $PWD/sm120):
#   <model>/exported_program.pt2      the export, produced here
#   <model>/fixture/                  inputs, weights and the CPU golden
#   <model>/auto.cu                   the generated megakernel
#   <model>/auto.mlir                 the CG dump the Plan was written back into
#   <model>/solve.{log,json}          the solver invocation and its choices
#   <model>/build/selected.{log,json} the nvcc command, both hashes and the head
#   <model>/correctness/r*.{log,json} one file pair per fresh process
#   status.txt                        one line per completed step
set -euo pipefail
trap 'echo "FAILED at line $LINENO" >>"${OUT:-/tmp}/status.txt"' ERR

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$PWD/sm120}"
ROUNDS="${ROUNDS:-50}"
SEQ="${SEQ:-4}"
PAST="${PAST:-3}"
CAPACITY="${CAPACITY:-12}"
SELF_CHECK="${SELF_CHECK:-0}"
NEED_MIB=65536

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

for model in llama qwen3; do
  case "$model" in
    llama) config="$REPO/docs/experiments/MODELS/sources/llama_config_public_copy.json" ;;
    qwen3) config="$REPO/docs/experiments/MODELS/sources/qwen_config.json" ;;
  esac
  root="$OUT/$model"
  echo "EXPORT $model" | tee -a "$OUT/status.txt"
  python3 "$REPO/docs/experiments/MODELS2/export_full.py" \
    --out "$root" --config "$config" --seq "$SEQ" --past "$PAST"

  if [ "$SELF_CHECK" = "1" ]; then
    echo "SELF_CHECK $model export only" | tee -a "$OUT/status.txt"
    continue
  fi

  echo "SOLVE $model (on this machine, not carried over)" | tee -a "$OUT/status.txt"
  python3 "$REPO/docs/experiments/E2E_REAL/run_e2e.py" \
    --root "$root" --arch "$ARCH" --seq "$SEQ" --past "$PAST" \
    --capacity "$CAPACITY" --rounds "$ROUNDS"
  echo "DONE $model rounds=$ROUNDS" | tee -a "$OUT/status.txt"
done
echo "COMPLETE" | tee -a "$OUT/status.txt"
