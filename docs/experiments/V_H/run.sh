#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
mkdir -p "${raw}"
export PYTHONPATH="${repo}/.venv/site${PYTHONPATH:+:${PYTHONPATH}}"

python3 - <<'PY' > "${raw}/environment.txt"
import importlib.util
import platform
print(f"python={platform.python_version()}")
for name in ("torch", "sympy"):
    print(f"{name}={'present' if importlib.util.find_spec(name) else 'missing'}")
PY

if ! python3 -c 'import torch, sympy' >/dev/null 2>&1; then
  echo "BLOCKED: CPU torch and/or sympy is not installed" \
    | tee "${raw}/run_status.txt"
  exit 77
fi

python3 "${here}/export_probe.py" --out "${raw}" | tee "${raw}/stdout.json"
echo PASS > "${raw}/run_status.txt"
