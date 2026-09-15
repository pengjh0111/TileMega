#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S2r on Blackwell: every candidate remeasured against the ceiling under each
# mechanism configuration (R3 §7.2, §7.3), so the round's ablation can be
# recomputed on the second architecture rather than inferred from the 4090's.
#
# The configurations are run in full -- a, b, d and w -- because §11 asks each
# mechanism's contribution to be independently recomputable here: `a` is the
# baseline with no flags, `b` the protocol at W=1, `d` protocol plus window, and
# `w` the window alone, which is what makes the additivity row checkable.
# Configuration C of §7.2 is not a set of flags but the (b-flags, chain) cell of
# the same table, so it needs no arm of its own.
#
# What differs from sm_89 is which lever is expected to pay.  The backoff term
# is worth about 32 ns of a ~400 ns hop here against 910 of 1235 ns there, so
# configuration b should do much less on this part, and the chain candidate --
# fewer hops on the critical path -- is the term that can still fall.  That
# contrast is the reason this script exists rather than a port of the 4090 run.
#
# The wait policy is read from the sm_120 target, not inherited: the two
# architectures calibrate differently (spin_iters 64 against 0), and run.sh
# defaults its tag to sm_89, so ARCH_TAG is passed explicitly below.  Compiling
# a Blackwell part with the 4090's constant is exactly what H9 forbids.
#
# Plans are solved again from sm_120 control geometry: both tools bind
# (worker, slot) to the resident grid they were solved against, and the
# committed dumps are a 4090's at grid 256 over 128 SMs.  Round one's copied
# EFT tables were refused by the host's plan guard for this (H9);
# `prepare_sm120.py` takes its own probe, writes its own manifest, and hands
# run.sh a populated plan directory with SKIP_GENERATE=1.
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
# filled the filesystem and took the rest of the session with it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*)
  echo 'refusing to overwrite sm_89 outputs' >&2; exit 2;; esac
status="${raw}/status.txt"
realwidth="${REALWIDTH:-1}"
configs="${CONFIGS:-a b d w}"
# Not chosen here: the window WINDOW/run_sm120.sh measured best on this part.
# Passed in so the two scripts cannot disagree about W.
window="${WINDOW:-2}"

# Measured on sm_89: four configurations over six candidates and four reference
# cells is 96 arm binaries at about 2.2 MiB, plus the trace dumps their own
# ceilings are recomputed from; real-width arms are about 6 MiB each and carry
# the larger dumps.  Both figures are rounded well up, because the failure they
# guard against is a run that dies two hours in.  The real-width *fixtures* are
# 3.3 GiB per seq but are inputs here, checked rather than regenerated.
run_mib="${RUN_MIB:-6144}"
realwidth_mib="${REALWIDTH_MIB:-8192}"

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
    echo "  CONFIGS='a b' drops the window arms and roughly halves the binaries" >&2
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
    "${here}/verify.py" "${here}/prepare_sm120.py" "${here}/test_prepare_sm120.py"
  # run.sh must honour RAW_DIR, or this script would write over the sm_89 tree.
  # -F: in a basic regexp GNU grep does not match a literal `${`.
  grep -Fq 'raw="${RAW_DIR:-' "${here}/run.sh" \
    || { echo "self-check: run.sh does not honour RAW_DIR" >&2; exit 1; }
  # ...and ARCH_TAG, or configurations b and d would be built on a Blackwell
  # part with the 4090's calibrated spin budget.
  grep -Fq 'arch_tag="${ARCH_TAG:-' "${here}/run.sh" \
    || { echo "self-check: run.sh does not honour ARCH_TAG" >&2; exit 1; }
  grep -Fq 'SKIP_GENERATE' "${here}/run.sh" \
    || { echo "self-check: run.sh cannot be told to keep the local plans" >&2; exit 1; }
  [[ -s "${here}/prepare_sm120.py" ]] \
    || { echo "self-check: sm120 plan preparation missing" >&2; exit 1; }
  [[ -f "${repo}/configs/targets/sm_120.json" ]] \
    || { echo "self-check: sm_120 target profile missing" >&2; exit 1; }
  (cd "${here}" && python3 test_prepare_sm120.py)
  echo "SELF_CHECK place_eft2 sm120 ok"
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

RAW_DIR="${raw}" SKIP_GENERATE=1 ARCH_TAG=sm_120 REALWIDTH="${realwidth}" \
  CONFIGS="${configs}" WINDOW="${window}" bash "${here}/run.sh"
echo PASS > "${status}"
