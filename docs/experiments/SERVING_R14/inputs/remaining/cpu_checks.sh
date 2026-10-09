#!/usr/bin/env bash
set -eu
cd /root/TileMega
out=docs/experiments/SERVING_R14/raw/remaining_definition
trap 'rc=$?; if [ "$rc" -eq 0 ]; then touch "$out/cpu.done"; else touch "$out/cpu.failed"; fi' EXIT
export PYTHONPATH=/root/TileMega/python
task_python=/root/venvs/tilemega-torch213-cu126/bin/python
"$task_python" docs/experiments/SERVING_R14/test_remaining_tests.py
"$task_python" docs/experiments/SERVING_R14/test_framework.py
"$task_python" docs/experiments/SERVING_R14/test_phase_d.py
"$task_python" docs/experiments/SERVING_R14/test_phase_d_final.py
"$task_python" -m py_compile docs/experiments/SERVING_R14/remaining_tests.py docs/experiments/SERVING_R14/make_remaining_tests.py docs/experiments/SERVING_R14/status.py docs/experiments/SERVING_R14/phase_d_r14.py
"$task_python" python/tilemega/fingerprint.py --check build-phase12/tools/tilemega
