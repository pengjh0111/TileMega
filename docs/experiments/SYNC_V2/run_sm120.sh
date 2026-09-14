#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-E3 on Blackwell: every step of the sync protocol work, recomputed there.
#
# H9 requires each mechanism's ablation to be independently recomputable on the
# target, so this drives the four sm_89 runners rather than summarising them:
# the backoff calibration (which is the step whose answer is expected to differ
# most), the barrier reduction, the single-member publish, and the release-fence
# litmus.  Each writes under raw_sm120/ through the OUT_DIR its runner already
# honours, so no committed sm_89 table is an output of this script.
#
# What sm_120 is expected to say about step 0: nothing that sm_89 said.  Its hop
# sits near a 400 ns floor with a backoff term worth about 32 ns, against 910 of
# 1235 ns on sm_89, so the graded policy has almost nothing to reclaim and the
# calibration is expected to pick a different point -- possibly pure spin.  That
# is the measurement this script exists to take; the value in
# configs/targets/sm_120.json today is the status quo, not a calibration.
#
# Written on a 4090 and checked only by its own CPU-side self-check
# (SELF_CHECK=1, which touches no GPU); it has never run on a Blackwell part.
#
# Disk is checked first and hard, before anything is compiled, because that is
# how round one's Blackwell session ended (F-142).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
# run_backoff.sh defaults its OUT_DIR to $here and writes backoff.tsv and
# friends straight into the committed tree, so the guard covers $here itself
# and not only the raw_* directories.
case "${raw}" in
  "${here}"|"${here}/raw_barrier"*|"${here}/raw_solo"*|"${here}/raw_litmus"*|"${here}/raw_notify"*)
    echo 'refusing to overwrite sm_89 outputs' >&2; exit 2;;
esac
status="${raw}/status.txt"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"

# Measured on sm_89: run_barrier.sh builds twenty binaries at about 2.2 MiB and
# eight full SASS dumps at about 7 MiB, and it runs twice here (once per
# switch); the litmus and backoff binaries are small.  Rounded well up, because
# the failure this guards against is a run that dies two hours in.
run_mib="${RUN_MIB:-4096}"

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
    echo "  twenty arm binaries and eight SASS dumps per switch, two switches" >&2
    return 1
  fi
  echo "disk ok: ${have} MiB free under ${path}, need about ${need} MiB"
  return 0
}

# The library and the fixtures are inputs, checked rather than built: producing
# them is a separate job whose failure mode is the one this script avoids.
check_prereqs() {
  local missing=() model seq past
  [[ -f "${build}/libtilemega.a" ]] || missing+=("${build}/libtilemega.a")
  for model in gqa2 mha4; do
    [[ -f "${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu" ]] \
      || missing+=("SEQSCAN/raw/src/${model}.cu")
    for seq in 1 4 128 512 2048; do
      for past in 0 3 512; do
        [[ -d "${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p${past}" ]] \
          || missing+=("SEQSCAN/raw/fixture/${model}_s${seq}_p${past}")
      done
    done
  done
  if ((${#missing[@]})); then
    echo "missing prerequisites:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "build the library with cmake, and the fixtures with" >&2
    echo "docs/experiments/SEQSCAN/run.sh, then run this again" >&2
    return 1
  fi
  # run_barrier.sh exits when rg is absent, and an absent rg once produced four
  # PASS reports carrying no barrier data at all.
  command -v rg >/dev/null || { echo 'sass_report.sh needs ripgrep on PATH' >&2; return 1; }
  return 0
}

if [[ -n "${SELF_CHECK:-}" ]]; then
  compute_cap_ok 12.0 || { echo "self-check: 12.0 rejected" >&2; exit 1; }
  ! compute_cap_ok 8.9 || { echo "self-check: 8.9 accepted" >&2; exit 1; }
  ! compute_cap_ok "" || { echo "self-check: empty cap accepted" >&2; exit 1; }
  (unset "${!TILEMEGA_@}"; reject_inherited) \
    || { echo "self-check: clean environment rejected" >&2; exit 1; }
  ! (export TILEMEGA_EVENT_SOLO=1; reject_inherited 2>/dev/null) \
    || { echo "self-check: leaked override accepted" >&2; exit 1; }
  check_disk "${here}" "${run_mib}" >/dev/null \
    || { echo "self-check: disk check failed" >&2; exit 1; }
  (! check_disk "${here}" $((1 << 30)) 2>/dev/null) \
    || { echo "self-check: an impossible requirement was accepted" >&2; exit 1; }
  (repo="$(mktemp -d)"; build="${repo}/none"; ! check_prereqs 2>/dev/null) \
    || { echo "self-check: missing prerequisites accepted" >&2; exit 1; }
  bash -n "${here}/run_backoff.sh"
  bash -n "${here}/run_barrier.sh"
  bash -n "${here}/run_litmus.sh"
  python3 -c "import ast,sys;[ast.parse(open(p).read()) for p in sys.argv[1:]]" \
    "${here}/backoff_fit.py" "${here}/summarize_barrier.py" "${here}/notify_share.py"
  # Each runner must honour the knob this script steers it with, or the sm_89
  # tree is what gets overwritten.  -F: GNU grep will not match a literal `${`
  # in a basic regexp.
  grep -Fq 'out=${OUT_DIR:-$here}' "${here}/run_backoff.sh" \
    || { echo "self-check: run_backoff.sh does not honour OUT_DIR" >&2; exit 1; }
  grep -Fq 'raw="${OUT_DIR:-' "${here}/run_barrier.sh" \
    || { echo "self-check: run_barrier.sh does not honour OUT_DIR" >&2; exit 1; }
  grep -Fq 'switch="${SWITCH:-' "${here}/run_barrier.sh" \
    || { echo "self-check: run_barrier.sh does not honour SWITCH" >&2; exit 1; }
  grep -Fq 'raw="${OUT_DIR:-' "${here}/run_litmus.sh" \
    || { echo "self-check: run_litmus.sh does not honour OUT_DIR" >&2; exit 1; }
  grep -Fq 'arch="${ARCH:-' "${here}/run_litmus.sh" \
    || { echo "self-check: run_litmus.sh does not honour ARCH" >&2; exit 1; }
  echo "SELF_CHECK sync_v2 sm120 ok"
  exit 0
fi

mkdir -p "${raw}"
reject_inherited
check_disk "${here}" "${run_mib}"
check_prereqs
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
if ! compute_cap_ok "${cap}"; then
  echo "compute_cap=${cap}, this script is for 12.0 only" >&2
  echo FAIL > "${status}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Step 0 first: it is the step whose sm_89 answer is least likely to transfer,
# and the policy it fits is what the other steps then run under.
OUT_DIR="${raw}/backoff" BIN="${raw}/bin/backoff" NVCC="${nvcc}" \
  bash "${here}/run_backoff.sh"

for pair in "TILEMEGA_BARRIER_V2:barrier" "TILEMEGA_EVENT_SOLO:solo"; do
  switch="${pair%%:*}"; tag="${pair##*:}"
  SWITCH="${switch}" TAG="${tag}" OUT_DIR="${raw}/${tag}" BUILD_DIR="${build}" \
    CUDACXX="${nvcc}" CUOBJDUMP="${cuobjdump}" \
    bash "${here}/run_barrier.sh"
  python3 "${here}/summarize_barrier.py" "${raw}/${tag}" \
    > "${raw}/${tag}/summary.txt"
done

ARCH=sm_120 OUT_DIR="${raw}/litmus" CUDACXX="${nvcc}" CUOBJDUMP="${cuobjdump}" \
  bash "${here}/run_litmus.sh"

echo PASS > "${status}"
