#!/usr/bin/env bash
# Usage: mx_check_modes.sh <model_dir> <prefill.so> <decode.so> <batch> <steps> <out> [prompts]
#
# C-1 (against HuggingFace tokens) and C-2 (four-mode equivalence). Was
# /root/deps/c1c2.sh outside the tree through MX-1 and MX-2 (MX3-0 item 1).
set -e
M=$1; P=$2; D=$3; B=$4; S=$5; O=$6; PR=${7:-configs/prompts/qwen_isl64.json}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
python -m tilemega.serving.check_modes --model "$M" --prefill-so "$P" --decode-so "$D" \
  --prompt-ids "$PR" --out "$O/c2" --batch "$B" --steps "$S" | grep -v '^E2E'
HF_EXTRA=()
if [[ -n ${MX_MOE_HF_EXPERTS:-} ]]; then
  HF_EXTRA+=(--experts-implementation "$MX_MOE_HF_EXPERTS")
fi
python -m tilemega.serving.hf_check --model "$M" --prompt-ids "$PR" "${HF_EXTRA[@]}" \
  --generated "$O/c2/tokens_L1_separate.json" --out "$O/c1/hf_check.json" --skip-free-greedy |
  python3 -c "import sys,json; d=json.loads(sys.stdin.readline()); print('C1', {k:d[k] for k in ('pass','gap_le_0_5_ratio','max_gap','gap_zero_ratio')})"
