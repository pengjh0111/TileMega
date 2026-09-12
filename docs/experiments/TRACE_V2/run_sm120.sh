#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# 本脚本已于 2026-09-12 在 sm_120 (RTX 5090, cc 12.0) 上运行过一次，
# 结果见 docs/experiments/sm120_round_one_20260912.md。
#
# EX-D1 on Blackwell.  The development machine is sm_89, so everything below is
# written from the sm_89 runner and checked only by its own CPU-side self-check
# (SELF_CHECK=1, which touches no GPU).  The tick is measured by the run rather
# than assumed: resolution.tsv's 1024 ns is an sm_89 figure and the Blackwell
# run measured 32 ns.
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

# run.sh writes into this experiment's own raw/ tree and its top-level tsv
# files, which is where the sm_89 round-one evidence lives.  The 2026-09-12
# Blackwell run overwrote that evidence in place, so everything a run changes is
# copied under raw_sm120/insitu/ and the sm_89 files are put back.
harvest() {
  local entry st rel dest restore=()
  while IFS= read -r -d '' entry; do
    st="${entry:0:2}"; rel="${entry:3}"
    # Binaries and exported weights are build products, not evidence: copying
    # them bloats the tree and removing them throws away work the run needs.
    # Scripts are skipped so a local edit to one is never reverted below.
    case "${rel}" in
      *"/raw_sm120/"*|*/bin/*|*/realwidth/model_*|*.sh|*.py) continue ;;
    esac
    dest="${raw}/insitu/${rel#docs/experiments/TRACE_V2/}"
    mkdir -p "$(dirname "${dest}")"
    cp -p "${repo}/${rel}" "${dest}" 2>/dev/null || true
    if [[ "${st}" == "??" ]]; then rm -f "${repo}/${rel}"; else restore+=("${rel}"); fi
  done < <(git -C "${repo}" status --porcelain=v1 -z -uall -- "docs/experiments/TRACE_V2")
  # Only the files this run actually touched are restored, so a blanket checkout
  # cannot undo an unrelated local change elsewhere in the experiment.
  if ((${#restore[@]})); then
    git -C "${repo}" checkout -- "${restore[@]}"
  fi
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
  git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "self-check: ${repo} is not a git work tree" >&2; exit 1; }
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
harvest
echo PASS > "${status}"
