#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S2 on Blackwell: the plan the solver chose, measured against mode 5, mode 0
# and L1, plus the 50-process correctness arm.
#
# Control sources remain frozen. EFT tables are solved again from sm_120
# control geometry: their worker/slot arrays are bound to a resident grid.
#
# Written on the 4090 and checked only by its own CPU-side self-check
# (SELF_CHECK=1, which touches no GPU); it has never run on a Blackwell part.
#
# It writes under raw_sm120/ by pointing run.sh's RAW_DIR there, so no committed
# sm_89 file is an output of this script: round one's Blackwell run overwrote
# 2291 of them in place.
#
# Disk is checked first and hard, before anything is compiled, because that is
# how round one's Blackwell session ended (F-142): the real-width seq=128 export
# filled the filesystem and took the rest of the session with it.  The estimates
# below are measured on sm_89 and listed per item so a short machine can be told
# what to drop.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*)
  echo 'refusing to overwrite sm_89 outputs' >&2; exit 2;; esac
status="${raw}/status.txt"
realwidth="${REALWIDTH:-0}"

# Measured on sm_89: 24 arm binaries and their dumps came to 53 MiB, the
# decomposition probes roughly double the binaries, and a real-width work tree
# is 3.3 GiB per seq.  The reference budget is rounded well up because the
# failure it guards against is a run that dies two hours in.
run_mib="${RUN_MIB:-4096}"
realwidth_mib="${REALWIDTH_MIB:-7168}"

fail() { mkdir -p "${raw}"; echo FAIL > "${status}"; }
trap fail ERR

reject_inherited() {
  local leaked=()
  while IFS='=' read -r name _; do
    case "${name}" in TILEMEGA_*) leaked+=("${name}") ;; esac
  done < <(env)
  if ((${#leaked[@]})); then
    echo "refusing to run with inherited overrides: ${leaked[*]}" >&2
    return 1
  fi
  return 0
}

compute_cap_ok() { [[ "$1" == "12.0" ]]; }

free_mib() { df -Pm "$1" | awk 'NR==2 {print $4}'; }

check_disk() {
  local path="$1" need="$2" have
  have="$(free_mib "${path}")"
  if ((have < need)); then
    echo "only ${have} MiB free under ${path}; this run needs about ${need} MiB" >&2
    echo "  reference cells ${run_mib} MiB; real-width adds ${realwidth_mib} MiB" >&2
    echo "  REALWIDTH=0 drops the real-width arm and its budget" >&2
    return 1
  fi
  echo "disk ok: ${have} MiB free under ${path}, need about ${need} MiB"
  return 0
}

# The fixtures are not committed -- 64 MiB for the four reference cells and
# 3.3 GiB per real-width seq -- so they are checked by hand rather than
# regenerated here: exporting them is a PyTorch job whose failure mode is the
# one this script exists to avoid.
check_fixtures() {
  local missing=() model seq
  for model in gqa2 mha4; do
    for seq in 4 128; do
      [[ -d "${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3" ]] \
        || missing+=("SEQSCAN/raw/fixture/${model}_s${seq}_p3")
    done
  done
  if [[ "${realwidth}" == 1 ]]; then
    for seq in 4 128; do
      [[ -d "${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s${seq}/export/fixture" ]] \
        || missing+=("REALMODEL/raw/work/r2sim_s${seq}/export/fixture")
    done
  fi
  if ((${#missing[@]})); then
    echo "missing fixtures:" >&2
    printf '  docs/experiments/%s\n' "${missing[@]}" >&2
    echo "produce them with docs/experiments/SEQSCAN/run.sh (64 MiB of fixture) and" >&2
    echo "docs/experiments/REALMODEL/run.sh (3.3 GiB per seq), then run this again" >&2
    return 1
  fi
  return 0
}

if [[ -n "${SELF_CHECK:-}" ]]; then
  compute_cap_ok 12.0 || { echo "self-check: 12.0 rejected" >&2; exit 1; }
  ! compute_cap_ok 8.9 || { echo "self-check: 8.9 accepted" >&2; exit 1; }
  ! compute_cap_ok "" || { echo "self-check: empty cap accepted" >&2; exit 1; }
  (unset "${!TILEMEGA_@}"; reject_inherited) \
    || { echo "self-check: clean environment rejected" >&2; exit 1; }
  ! (export TILEMEGA_PLACEMENT=5; reject_inherited 2>/dev/null) \
    || { echo "self-check: leaked override accepted" >&2; exit 1; }
  check_disk "${here}" "${run_mib}" >/dev/null || { echo "self-check: disk check failed" >&2; exit 1; }
  (! check_disk "${here}" $((1 << 30)) 2>/dev/null) \
    || { echo "self-check: an impossible requirement was accepted" >&2; exit 1; }
  (realwidth=1
   REPO_MISSING="$(mktemp -d)"
   repo="${REPO_MISSING}"
   ! check_fixtures 2>/dev/null) \
    || { echo "self-check: missing fixtures accepted" >&2; exit 1; }
  bash -n "${here}/run.sh"
  python3 -c "import ast,sys;[ast.parse(open(p).read()) for p in sys.argv[1:]]" \
    "${here}/summarize.py" "${here}/verify.py" "${here}/prepare_sm120.py"
  # run.sh must honour RAW_DIR, or this script would write over the sm_89 tree.
  # -F: in a basic regexp GNU grep does not match a literal `${`.
  grep -Fq 'raw="${RAW_DIR:-' "${here}/run.sh" \
    || { echo "self-check: run.sh does not honour RAW_DIR" >&2; exit 1; }
  [[ -s "${here}/prepare_sm120.py" ]] \
    || { echo "self-check: sm120 plan preparation missing" >&2; exit 1; }
  python3 "${here}/test_prepare_sm120.py"
  echo "SELF_CHECK place_eft sm120 ok"
  exit 0
fi

mkdir -p "${raw}"
reject_inherited
check_disk "${here}" $((run_mib + (realwidth == 1 ? realwidth_mib : 0)))
check_fixtures
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
if ! compute_cap_ok "${cap}"; then
  echo "compute_cap=${cap}, this script is for 12.0 only" >&2
  echo FAIL > "${status}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 "${here}/prepare_sm120.py" --out "${raw}" --realwidth "${realwidth}"

RAW_DIR="${raw}" SKIP_GENERATE=1 REALWIDTH="${realwidth}" bash "${here}/run.sh"
echo PASS > "${status}"
