#!/usr/bin/env bash
# MX3-6 final measurement driver: every cell solved by one frozen tree (its
# own compiler, TREE/build/tools/tilemega), decode chosen by measurement over
# the attention bodies the target holds (--attention-impl auto), prefill by
# prediction; then, as each cell's two plans land, TPOT (2 fresh processes)
# and C-1/C-2. Raw output goes to runs/mx3/$OUT in this checkout.
#
#   TREE=/path/to/frozen/worktree OUT=final2 scripts/mx3_final_run.sh solve <gpu> cell...
#   OUT=final2 scripts/mx3_final_run.sh measure <gpu> cell...
# A cell is model:batch, e.g. qwen3_0_6b:1. `solve` times its decode
# candidates on <gpu> and queries devices there; `measure` waits for each
# cell's plans and measures on <gpu>.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-final2}
mode=$1; gpu=$2; shift 2
log=$ROOT/runs/mx3/${OUT}_$mode.log
mkdir -p $ROOT/runs/mx3
case $mode in
solve)
  : "${TREE:?frozen worktree}"
  cd $TREE
  for cell in "$@"; do
    id=${cell%:*}; b=${cell#*:}
    for p in decode prefill; do
      out=$ROOT/runs/mx3/$OUT/$id/B$b/$p
      grep -q '^rc=0' $out/compile.log 2>/dev/null && continue
      rm -rf $out
      CELLS=$ROOT/runs/mx3/$OUT SOLVE_BIN=$TREE/build/tools/tilemega TARGET=${TARGET:-configs/targets/xcore1000.json} \
        JOBS=${JOBS:-6} CELL_TIMEOUT=14400 QGPU=$gpu MGPU=$gpu SEARCH_BUDGET_MS=${SEARCH_BUDGET_MS:-600000} \
        bash scripts/mx_serving_cell.sh solve $id $b $p >> $log 2>&1
      echo "$(date -Is) solved $id B$b $p rc=$?" >> $log
    done
  done
  echo "SOLVE LANE DONE: $*" >> $log ;;
measure)
  cd $ROOT
  pending=1
  while [ $pending -gt 0 ]; do
    pending=0
    for cell in "$@"; do
      id=${cell%:*}; b=${cell#*:}
      C=runs/mx3/$OUT/$id/B$b
      [ -f $C/.measured ] && continue
      case $id in qwen*) pr=configs/prompts/qwen_isl64.json;; *) pr=configs/prompts/llama3_isl64.json;; esac
      ok=1
      for p in decode prefill; do
        grep -q '^rc=0' $C/$p/compile.log 2>/dev/null || { grep -qE '^rc=[1-9]' $C/$p/compile.log 2>/dev/null || pending=1; ok=0; }
      done
      [ $ok = 1 ] || continue
      for round in 1 2; do
        CELLS=runs/mx3/$OUT bash scripts/mx_serving_cell.sh measure $id $b /root/models/${id}_rand $pr $gpu $round >> $log 2>&1
      done
      CELLS=runs/mx3/$OUT bash scripts/mx_serving_cell.sh check $id $b /root/models/${id}_rand $pr $gpu >> $log 2>&1
      touch $C/.measured
      echo "$(date -Is) measured $id B$b" >> $log
    done
    [ $pending -gt 0 ] && sleep 120
  done
  echo "MEASURE DONE: $*" >> $log ;;
*) echo "usage: $0 solve|measure <gpu> cell..." >&2; exit 2 ;;
esac
