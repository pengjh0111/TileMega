#!/usr/bin/env bash
# R6 COSTMODEL runner: written/self-checked on 4090, NOT RUN on sm_120.
# Parameters/artifact schema: ../JOINT2/sm120_common.sh (full documentation).
# SELF_CHECK=1 performs guard/parser/sm_120 compile checks without GPU launch.
# OUT_DIR, NEED_MIB=131072, SEARCH_CAPACITY=18, LOCAL_R5_ROOT are supported.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-${here}/raw_sm120}"
exec bash "${here}/../JOINT2/sm120_common.sh" cost
