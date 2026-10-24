#!/usr/bin/env bash
# MX-4 final (MX4 prompt §9) of the 20 cells from the committed pins, through
# scripts/mx_serving_cell.sh into $CELLS (runs/mx4/final):
#   mx4_final_run.sh build   <gpu> <id:b>...   prefill from configs/pins/mx4/<id>_B<b>_prefill.json,
#                                                 then decode (+ watchdog twin) from <id>_B<b>.json
#   mx4_final_run.sh check   <gpu> <id:b>...   C-1 and C-2 (scripts/mx_check_modes.sh)
#   mx4_final_run.sh measure <gpu> <round> <id:b>...   one fresh process per cell and round
# Device queries and measurements run on <gpu> through the guard.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd $ROOT
export CELLS=${CELLS:-runs/mx4/final}
phase=$1; gpu=$2; shift 2
round=""; [ $phase = measure ] && { round=$1; shift; }
for cell in "$@"; do
  IFS=: read id b <<< "$cell"
  rand=/root/models/${id}_rand
  case $id in qwen*) prompts=configs/prompts/qwen_isl64.json;; *) prompts=configs/prompts/llama3_isl64.json;; esac
  case $phase in
    build)
      SELECTION=predicted PIN=configs/pins/mx4/${id}_B${b}_prefill.json QGPU=$gpu MGPU=$gpu \
        bash scripts/mx_serving_cell.sh solve $id $b prefill
      SELECTION=predicted PIN=configs/pins/mx4/${id}_B${b}.json QGPU=$gpu MGPU=$gpu \
        bash scripts/mx_serving_cell.sh solve $id $b decode
      echo "$(date +%H:%M) build $id B$b prefill $(grep '^rc=' $CELLS/$id/B$b/prefill/compile.log | tail -1)" \
           "decode $(grep '^rc=' $CELLS/$id/B$b/decode/compile.log | tail -1)";;
    check)
      bash scripts/mx_serving_cell.sh check $id $b $rand $prompts $gpu > /dev/null 2>&1
      C=$CELLS/$id/B$b
      echo "$(date +%H:%M) check $id B$b C-2 $(python3 -c "import json;d=json.load(open('$C/c2/mode_check.json'));print(d.get('mismatches'))" 2>&1 | tail -1)" \
           "C-1 $(python3 -c "import json;print(json.load(open('$C/c1/hf_check.json'))['pass'])" 2>&1 | tail -1)";;
    measure)
      bash scripts/mx_serving_cell.sh measure $id $b $rand $prompts $gpu $round
      echo "$(date +%H:%M) measure r$round $id B$b $([ -f $CELLS/$id/B$b/measure/round$round/.done ] && echo done || echo FAILED)";;
  esac
done
