#!/usr/bin/env bash
set -eu
[ "$#" -ge 3 ] || { echo 'usage: mx_tp_cell.sh export|build|check|measure <tp> <batch>' >&2; exit 2; }
command=$1; tp=$2; batch=$3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
case "$tp" in 2|4|8) ;; *) echo 'only TP2, TP4 and TP8 are enabled' >&2; exit 2;; esac
case "$batch" in 1|2|4|8|16) ;; *) echo 'batch must be 1,2,4,8,16' >&2; exit 2;; esac
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
MODEL=${MODEL:-/root/models/qwen3_1_7b_rand}
EXPORT_ROOT=${EXPORT_ROOT:-runs/mx7/export}
CELLS=${CELLS:-runs/mx7/tm}
export_dir=$EXPORT_ROOT/qwen3_1_7b_tp${tp}_decode
cell=$CELLS/tp$tp/B$batch
case "$command" in
export)
  mkdir -p "$export_dir"
  [ ! -e "$export_dir/bridge.json" ] || { echo 'export already exists; preserving it'; exit 0; }
  env CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= python -m tilemega.serving.export \
    --config "$MODEL/config.json" --model qwen3 --phase decode --capacity 1088 --prompt-len 64 \
    --tp "$tp" --out "$export_dir" > "$export_dir/export.log" 2>&1
  env CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= python -m tilemega.export_bridge \
    "$export_dir/exported_program.pt2" --out "$export_dir/bridge.json" >> "$export_dir/export.log" 2>&1
  ;;
build)
  : "${PIN:?PIN must name a development or solver pin}"
  python - "$export_dir/manifest.json" "$tp" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get('tp')!=int(sys.argv[2]):
 raise SystemExit('export manifest TP differs from --serving-tp')
PY
  mkdir -p "$cell"
  [ ! -e "$cell/plan.so" ] || { echo 'build requires a fresh cell directory' >&2; exit 2; }
  env CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= "${SOLVE_BIN:-build/tools/tilemega}" compile \
    "$export_dir/bridge.json" "$cell/plan.so" --serving decode --emit serving --frontend decoder \
    --batch "$batch" --prompt-len 64 --capacity 1088 --past-range 0:1086 --serving-tp "$tp" \
    --runtime-target configs/targets/xcore1000.json --evaluate-configs "$PIN" --selection predicted \
    --watchdog "${WATCHDOG:-0}" > "$cell/compile.log" 2>&1
  ;;
check|measure)
  : "${GROUP:?GROUP must be the C0-approved TP group}"
  devices=$(python - "$tp" <<'PY'
import sys
print(','.join(map(str, range(int(sys.argv[1])))))
PY
)
  [ "$(awk -F, '{print NF}' <<< "$GROUP")" -eq "$tp" ] || { echo 'TP group size mismatch' >&2; exit 2; }
  out=${OUT:-$cell/$command/round${ROUND:-1}}
  mkdir -p "$out"
  if [ "$command" = check ]; then
    reference=(); [ -z "${REFERENCE:-}" ] || reference=(--reference "$REFERENCE")
    NEEDS_MIB=12288 scripts/mx_gpu_group_job.sh "$GROUP" 900 "$out/guard" \
      python -m tilemega.serving.check_tp --model "$MODEL" --decode-so "$cell/plan.so" \
      --prompt-ids configs/prompts/qwen_isl64.json --batch "$batch" --devices "$devices" \
      --steps "${STEPS:-1024}" --out "$out/check" "${reference[@]}" > "$out/run.log" 2>&1
  else
    mkdir -p "$out"
    NEEDS_MIB=12288 scripts/mx_gpu_group_job.sh "$GROUP" 900 "$out/guard" \
      python -m tilemega.serving.measure_tp --model "$MODEL" --decode-so "$cell/plan.so" \
      --prompt-ids configs/prompts/qwen_isl64.json --batch "$batch" --devices "$devices" \
      --max-new-tokens 1024 --warmup 1 --repeats 2 --policy configs/bench/measurement_policy_c550.json \
      --out "$out/measure" > "$out/run.log" 2>&1
  fi
  ;;
*) echo 'unknown command' >&2; exit 2;;
esac
