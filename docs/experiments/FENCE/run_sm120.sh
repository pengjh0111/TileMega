#!/usr/bin/env bash
# R4 device-fence five-arm pricing. Written on sm_89; NOT RUN on sm_120.
# Controls contain no materialized worker table: their host derives resident
# geometry and solves legacy/rotate placement on the target machine.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*) echo 'refusing sm_89 outputs' >&2; exit 2;; esac
label=fence
NEED_MIB=4096
python_sources=("${here}/run.py")
source "${repo}/docs/experiments/SYNC_V3/sm120_common.sh"
if [[ "${SELF_CHECK:-0}" == 1 ]]; then
  r4_init
  exit 0
fi
r4_init
python3 "${here}/run.py" build --arch sm_120 --raw "${raw}"
python3 "${here}/run.py" measure --arch sm_120 --raw "${raw}"
python3 "${here}/run.py" summarize --arch sm_120 --raw "${raw}" > "${raw}/summary.txt"
echo PASS > "${raw}/status.txt"
