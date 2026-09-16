#!/usr/bin/env bash
# R5 phase diagnostic. Written/self-checked on 4090; NOT RUN on sm_120.
# Parameters: SELF_CHECK=1 (no GPU execution), OUT_DIR (fresh directory),
# NEED_MIB=131072. Dependencies: see ../JOINT/run_sm120.sh.
# Re-exports local inputs; writes target.json, phase/{build,correctness,
# measure,trace,analysis.tsv}, fork.txt, pipeline.log and status.txt.
# The local 32 ns timer and local FORK5 may differ from sm_89 results.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-${here}/raw_sm120}"
export PHASE_ONLY=1
exec bash "${here}/../JOINT/run_sm120.sh"
