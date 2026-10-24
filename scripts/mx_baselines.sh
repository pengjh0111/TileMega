#!/usr/bin/env bash
# MX-4 baselines on the C550: the six LLMs with vLLM, SGLang and torch.compile
# (R16 M-1 drivers, python/tilemega/serving/*_baseline.py), the five DNNs with
# torch.compile (python -m tilemega.dnn.baseline); their other baseline,
# TensorRT, has no build for the C550 (NVIDIA-only; MACA 3.8.2 ships none).
# Protocol of MX4 prompt §9:
# 2 rounds of fresh processes, systems interleaved within a round, warm-up 1 +
# timed 2; LLMs B in {1,2,4,8,16}, ISL/OSL 64/1024; DNNs B in {1,2,4,8,16}.
# Every job runs through the GPU guard (scripts/mx_gpu_job.sh) and is a timed
# job of at most 15 minutes, so an LLM job is one (model, system, batch).
# Resumable: a job with a .done marker is skipped.
#   mx_baselines.sh llm <gpu> <model>...   (model = directory under /root/models)
#   mx_baselines.sh dnn <gpu> <model>...   (resnet18 mbv1 mbv2 bert nafnet)
set -u
kind=$1; GPU=$2; shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-$ROOT/runs/mx4/base}
POLICY=$ROOT/configs/bench/measurement_policy_c550.json
ROUNDS=${ROUNDS:-"1 2"}
cd $ROOT
for round in $ROUNDS; do
  for model in "$@"; do
    if [ $kind = llm ]; then
      case $model in llama*) ids=configs/prompts/llama3_isl64.json;; *) ids=configs/prompts/qwen_isl64.json;; esac
      systems=${SYSTEMS:-"vllm sglang torchcompile"}; batches=${BATCHES:-"1 2 4 8 16"}
    else
      systems=torchcompile; batches=all
    fi
    for system in $systems; do for batch in $batches; do
      dir=$OUT/$kind/$system/$model/B$batch/round$round
      [ -f $dir/.done ] && continue
      mkdir -p $dir
      extra=""; hidden=1024
      case $kind:$system in
        llm:vllm) cmd="env VLLM_CACHE_ROOT=/root/deps/vllm_src/vllm_cache /root/venv_vllm/bin/python python/tilemega/serving/vllm_baseline.py";;
        llm:sglang) cmd="python python/tilemega/serving/sglang_baseline.py";;
        llm:torchcompile) cmd="python python/tilemega/serving/torchcompile_baseline.py"
                          case $model in qwen3_30b*) extra="--experts-implementation grouped_mm";; esac;;
        dnn:torchcompile) cmd="python -m tilemega.dnn.baseline";;
      esac
      # The MoE checkpoint leaves several GiB of allocator slack unattributed
      # (MX-1): the hidden-memory budget of a dense model would flag it.
      case $model in *30b*) hidden=12288;; esac
      echo "$(date -Is) start round=$round kind=$kind model=$model system=$system batch=$batch" | tee -a $OUT/sweep.log
      if [ $kind = llm ]; then
        args="--model /root/models/$model --prompt-ids $ids --batch $batch --max-tokens 1024 --warmup 1 --repeats 2 --policy $POLICY --out $dir $extra"
      else
        args="--model $model --batch all --warmup 1 --repeats 2 --out $dir"
      fi
      NEEDS_MIB=${NEEDS_MIB:-40960} TILEMEGA_GUARD_MAX_HIDDEN_MIB=$hidden \
        $ROOT/scripts/mx_gpu_job.sh $GPU ${BENCH_TIMEOUT:-900} $dir/guard env -u ALL_PROXY -u all_proxy \
        PYTHONPATH=$ROOT/python $cmd $args > $dir/run.log 2>&1
      rc=$?
      echo "$(date -Is) end round=$round kind=$kind model=$model system=$system batch=$batch rc=$rc" | tee -a $OUT/sweep.log
      [ $rc = 0 ] && touch $dir/.done
      # Compile caches grow per model; keep the disk tidy between jobs.
      rm -rf /tmp/torchinductor_root/*/ 2>/dev/null
    done; done
  done
done
