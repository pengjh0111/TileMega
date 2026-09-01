#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
raw="${here}/raw"
mkdir -p "${raw}"

python3 - <<'PY' > "${raw}/environment.txt"
import importlib.util
import platform
print(f"python={platform.python_version()}")
for name in ("torch", "transformers", "sympy"):
    print(f"{name}={'present' if importlib.util.find_spec(name) else 'missing'}")
PY

if ! python3 -c 'import torch, transformers' >/dev/null 2>&1; then
  echo "BLOCKED: torch and/or transformers is not installed" \
    | tee "${raw}/run_status.txt"
  exit 77
fi

python3 "${here}/export_probe.py" > "${raw}/export_report.json"
echo PASS > "${raw}/run_status.txt"
