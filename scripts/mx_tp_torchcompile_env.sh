#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_dir=${1:-/root/venv_mx7_tc}
python -m venv --system-site-packages "$env_dir"
"$env_dir/bin/python" -m pip install --no-deps -r "$ROOT/experiments/mx7/requirements_torchcompile_tp.txt"
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" - <<'PY'
import accelerate, torch, transformers
print(dict(accelerate=accelerate.__version__,torch=torch.__version__,transformers=transformers.__version__))
PY
