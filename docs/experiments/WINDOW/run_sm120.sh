#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-E2 on Blackwell: the slot window at W in {1, 2, 4}.
#
# The window is the one §1 lever whose value does not depend on the per-hop
# constant.  E3-0's backoff has nothing to take on sm_120 -- the hop sits at a
# ~400 ns floor with a backoff term near 32 ns, against 910 of 1235 ns on sm_89
# -- but head-of-line time is idle time on the critical path whatever a hop
# costs, so W is measured here on its own terms rather than assumed to transfer.
#
# Unlike CHAIN, this script needs no plan re-solve.  A window plan carries mode
# 0 and a window; it has no (worker, slot) arrays, so nothing in it is bound to
# a resident grid and H9's re-solve requirement does not apply.  `plan_window.py`
# stamps the same source on either machine and the placement stays the legacy
# grid stride the reference sources already use.
#
# Each arm is recomputed here from its own build: the W arms, the `today`
# reference E2-b pairs against, the H4 negative control that must fail, and the
# refusal probe.  Nothing is carried from the 4090.
#
# Written on the 4090 and checked only by its own CPU-side self-check
# (SELF_CHECK=1, which touches no GPU); it has never run on a Blackwell part.
#
# It writes under raw_sm120/ by pointing run.sh's RAW_DIR there, so no committed
# sm_89 file is an output of this script.
#
# Disk is checked first and hard, before anything is compiled, because that is
# how round one's Blackwell session ended (F-142).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*)
  echo 'refusing to overwrite sm_89 outputs' >&2; exit 2;; esac
status="${raw}/status.txt"
windows="${WINDOWS:-1 2 4}"

# Measured on sm_89: fifteen arm binaries at about 2.2 MiB each (three windows
# x two models x plain and traced, plus `today`, the negative control and the
# refusal probe), the per-run logs are small, and a trace dump is a row per slot
# for twelve cells.  Rounded well up, because the failure this guards against is
# a run that dies two hours in.
run_mib="${RUN_MIB:-2048}"

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
    echo "  fifteen arm binaries plus logs and twelve trace dumps" >&2
    echo "  WINDOWS='1 2' drops the W = 4 arm and about a third of the budget" >&2
    return 1
  fi
  echo "disk ok: ${have} MiB free under ${path}, need about ${need} MiB"
  return 0
}

# The fixtures are not committed, so they are checked by hand rather than
# regenerated here: exporting them is a PyTorch job whose failure mode is the
# one this script exists to avoid.  E2-a is the whole SEQSCAN matrix, so every
# seq in it is required, not just the two reference points.
check_fixtures() {
  local missing=() model seq past
  for model in gqa2 mha4; do
    for seq in 1 4 128 512 2048; do
      for past in 0 3 512; do
        [[ -d "${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p${past}" ]] \
          || missing+=("SEQSCAN/raw/fixture/${model}_s${seq}_p${past}")
      done
    done
  done
  if ((${#missing[@]})); then
    echo "missing fixtures:" >&2
    printf '  docs/experiments/%s\n' "${missing[@]}" >&2
    echo "produce them with docs/experiments/SEQSCAN/run.sh (64 MiB of fixture)," >&2
    echo "then run this again" >&2
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
  ! (export TILEMEGA_SLOT_WINDOW=4; reject_inherited 2>/dev/null) \
    || { echo "self-check: leaked override accepted" >&2; exit 1; }
  check_disk "${here}" "${run_mib}" >/dev/null || { echo "self-check: disk check failed" >&2; exit 1; }
  (! check_disk "${here}" $((1 << 30)) 2>/dev/null) \
    || { echo "self-check: an impossible requirement was accepted" >&2; exit 1; }
  (REPO_MISSING="$(mktemp -d)"
   repo="${REPO_MISSING}"
   ! check_fixtures 2>/dev/null) \
    || { echo "self-check: missing fixtures accepted" >&2; exit 1; }
  bash -n "${here}/run.sh"
  python3 -c "import ast,sys;[ast.parse(open(p).read()) for p in sys.argv[1:]]" \
    "${here}/plan_window.py" "${here}/summarize.py"
  # run.sh must honour RAW_DIR, or this script would write over the sm_89 tree.
  # -F: in a basic regexp GNU grep does not match a literal `${`.
  grep -Fq 'raw="${RAW_DIR:-' "${here}/run.sh" \
    || { echo "self-check: run.sh does not honour RAW_DIR" >&2; exit 1; }
  # The stamp must stay inert at W = 1, or E2-b compares two different programs
  # rather than two windows.  Checked here on source, since SELF_CHECK is CPU
  # only: the SASS half of the same claim is run.sh's `identity` phase.
  python3 "${here}/plan_window.py" "${repo}/docs/experiments/SEQSCAN/raw/src/gqa2.cu" \
    1 "$(mktemp)" >/dev/null \
    || { echo "self-check: the W = 1 stamp failed" >&2; exit 1; }
  # A source that already carries a plan must be refused, not appended to.
  chain="${repo}/docs/experiments/CHAIN/raw/plan/gqa2_s4_chain.cu"
  if [[ -f "${chain}" ]]; then
    (! python3 "${here}/plan_window.py" "${chain}" 2 "$(mktemp)" 2>/dev/null) \
      || { echo "self-check: a plan-carrying source was rewritten" >&2; exit 1; }
  fi
  echo "SELF_CHECK window sm120 ok"
  exit 0
fi

mkdir -p "${raw}"
reject_inherited
check_disk "${here}" "${run_mib}"
check_fixtures
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
if ! compute_cap_ok "${cap}"; then
  echo "compute_cap=${cap}, this script is for 12.0 only" >&2
  echo FAIL > "${status}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RAW_DIR="${raw}" WINDOWS="${windows}" bash "${here}/run.sh"
echo PASS > "${status}"
