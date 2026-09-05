#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-phase12}"
out="${OUT_TARGET:-${repo}/configs/targets/sm_89.json}"
cmake --build "${build}" --target tilemega-calibrate --parallel "$(nproc)"
"${build}/tools/tilemega-calibrate" --dtype bf16 --base "${out}" \
  --repeats 41 --out "${out}"
python3 - "${out}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
b=p['calibration_by_dtype']['bf16']
assert p['calibration']['calibrated'], 'FP32 profile was not preserved'
assert b['calibrated'], 'BF16 run failed the idle-machine acceptance guard'
assert b['pipelines']['tc_bf16_gflops'] > 0
assert len(b['streamk']) == 6
print('BF16 calibration accepted:', b['device'], b['measured_at'])
PY
echo "Run ../SEQSCAN/run.sh for the two-model 1500-process correctness matrix."
echo "Run ../ORACLE/run.sh for BF16 rank validation and lane ablation."
