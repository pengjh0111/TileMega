#!/usr/bin/env bash
set -eu
[ "$#" -eq 3 ] || { echo 'usage: mx_tp_baselines.sh <group> <tp> <model>' >&2; exit 2; }
group=$1; tp=$2; model=$3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
case "$tp" in 2|4) ;; *) echo 'TP must be 2 or 4' >&2; exit 2;; esac
[ "$(awk -F, '{print NF}' <<< "$group")" -eq "$tp" ] || { echo 'group size differs from TP' >&2; exit 2; }
OUT=${OUT:-$ROOT/runs/mx7/baselines}
for round in ${ROUNDS:-1 2}; do
  for batch in ${BATCHES:-1 2 4 8 16}; do
    for system in ${SYSTEMS:-vllm sglang torchcompile}; do
      d=$OUT/tp$tp/$system/$model/B$batch/round$round
      [ ! -f "$d/.done" ] || continue
      [ ! -f "$d/.failed" ] || continue
      mkdir -p "$d"
      args=(--model "/root/models/$model" --prompt-ids configs/prompts/qwen_isl64.json
            --batch "$batch" --out "$d" --max-tokens 1024 --warmup 1 --repeats 2
            --policy configs/bench/measurement_policy_c550.json)
      case "$system" in
        vllm) command=(env VLLM_CACHE_ROOT=/root/deps/vllm_src/vllm_cache /root/venv_vllm/bin/python
                       python/tilemega/serving/vllm_baseline.py --tp "$tp");;
        sglang) command=(python python/tilemega/serving/sglang_baseline.py --tp "$tp");;
        torchcompile) command=(torchrun --standalone --nproc-per-node "$tp"
                              python/tilemega/serving/torchcompile_tp_baseline.py);;
        *) echo "unknown system $system" >&2; exit 2;;
      esac
      python experiments/mx7/baseline_identity.py --out "$d" -- "${command[@]}" "${args[@]}"
      set +e
      NEEDS_MIB=12288 scripts/mx_gpu_group_job.sh "$group" 900 "$d" "${command[@]}" "${args[@]}" > "$d/run.log" 2>&1
      rc=$?
      set -e
      printf '%s\n' "$rc" > "$d/exit_code.txt"
      if [ "$rc" -eq 0 ]; then touch "$d/.done"; else touch "$d/.failed"; fi
    done
  done
done
