#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S2c on Blackwell: the critical chain placed on one worker, measured against
# mode 5 and mode 0, plus the 50-process correctness arm.
#
# This is the round's most important sm_120 item.  The other lever on protocol
# cost is the backoff constant, and on sm_120 there is nothing there to take:
# the hop sits at a ~400 ns floor with a backoff term of about 32 ns, against
# 910 of 1235 ns on sm_89 (F-145 and its sm_120 companion).  Fewer hops on the
# critical path is the only term left that falls on that part, so what this
# script measures is the whole of the sm_120 story for §1's three levers.
#
# Control sources remain frozen.  Chain plans are solved again from sm_120
# control geometry: the emitted (worker, slot) arrays are bound to a resident
# grid, and the committed dumps are a 4090's at grid 256 over 128 SMs.  Round
# one's copied EFT tables were refused by the host's plan guard for exactly this
# (H9); `prepare_sm120.py` takes its own probe and writes its own manifest.
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
realwidth="${REALWIDTH:-0}"

# Measured on sm_89: a reference arm binary is 2.2 MiB and there are twelve of
# them plus six control probes, the per-run logs are small, and a real-width arm
# binary is about 6 MiB.  Both figures are rounded well up, because the failure
# they guard against is a run that dies two hours in.  The real-width *fixtures*
# are 3.3 GiB per seq but are inputs here, checked rather than regenerated.
run_mib="${RUN_MIB:-2048}"
realwidth_mib="${REALWIDTH_MIB:-3072}"

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

# The fixtures are not committed, so they are checked by hand rather than
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
    "${here}/summarize.py" "${here}/prepare_sm120.py"
  # run.sh must honour RAW_DIR, or this script would write over the sm_89 tree.
  # -F: in a basic regexp GNU grep does not match a literal `${`.
  grep -Fq 'raw="${RAW_DIR:-' "${here}/run.sh" \
    || { echo "self-check: run.sh does not honour RAW_DIR" >&2; exit 1; }
  # ... and must treat the manifest as an input under SKIP_GENERATE, or it would
  # replace the one prepare_sm120.py wrote with a manifest naming 4090 dumps.
  grep -Fq '} > "${raw}/manifest.tsv"
  "${build}/tools/tilemega-place-chain"' "${here}/run.sh" \
    || { echo "self-check: run.sh rewrites the manifest outside SKIP_GENERATE" >&2; exit 1; }
  [[ -s "${here}/prepare_sm120.py" ]] \
    || { echo "self-check: sm120 plan preparation missing" >&2; exit 1; }
  (cd "${here}" && python3 test_prepare_sm120.py)
  echo "SELF_CHECK chain sm120 ok"
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
