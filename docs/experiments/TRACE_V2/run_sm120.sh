#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# 本脚本尚未在 sm_120 上运行。
#
# EX-D1 on Blackwell.  The development machine is sm_89, so everything below is
# written from the sm_89 runner and checked only by its own CPU-side self-check
# (SELF_CHECK=1, which touches no GPU).  Whoever runs it first on sm_120 should
# expect the %globaltimer tick to be measured again rather than assumed: the
# 1024 ns figure in resolution.tsv is an sm_89 measurement.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_sm120"
status="${raw}/status.txt"

fail() { mkdir -p "${raw}"; echo FAIL > "${status}"; }
trap fail ERR

reject_inherited() {
  # An inherited TILEMEGA_* override would silently change what is measured and
  # the log would still look like a clean run.
  local leaked=()
  while IFS='=' read -r name _; do
    case "${name}" in
      TILEMEGA_TRACE_V2_OUT) ;;
      TILEMEGA_*) leaked+=("${name}") ;;
    esac
  done < <(env)
  if ((${#leaked[@]})); then
    echo "refusing to run with inherited overrides: ${leaked[*]}" >&2
    return 1
  fi
  return 0
}

compute_cap_ok() { [[ "$1" == "12.0" ]]; }

if [[ -n "${SELF_CHECK:-}" ]]; then
  # CPU-side only: the two decisions this script makes, exercised without a GPU.
  compute_cap_ok 12.0 || { echo "self-check: 12.0 rejected" >&2; exit 1; }
  ! compute_cap_ok 8.9 || { echo "self-check: 8.9 accepted" >&2; exit 1; }
  ! compute_cap_ok "12.0 " || { echo "self-check: padded cap accepted" >&2; exit 1; }
  (unset "${!TILEMEGA_@}"; reject_inherited) \
    || { echo "self-check: clean environment rejected" >&2; exit 1; }
  ! (export TILEMEGA_PLACEMENT=5; reject_inherited 2>/dev/null) \
    || { echo "self-check: leaked override accepted" >&2; exit 1; }
  [[ -x "${here}/run.sh" ]] || { echo "self-check: run.sh missing" >&2; exit 1; }
  bash -n "${here}/run.sh"
  echo "SELF_CHECK trace_v2 sm120 ok"
  exit 0
fi

mkdir -p "${raw}"
reject_inherited
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
if ! compute_cap_ok "${cap}"; then
  echo "compute_cap=${cap}, this script is for 12.0 only" >&2
  echo FAIL > "${status}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# The tick is device specific, so it is measured here before anything reads it.
bash "${here}/run_resolution.sh"
RUNS="${RUNS:-25}" CORRECTNESS_RUNS="${CORRECTNESS_RUNS:-50}" bash "${here}/run.sh"
cp -r "${here}/raw/dump" "${raw}/dump"
cp "${here}/analysis.tsv" "${here}/analysis.md" "${raw}/" 2>/dev/null || true
echo PASS > "${status}"
