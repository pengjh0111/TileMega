#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# 本脚本已于 2026-09-12 在 sm_120 (RTX 5090, cc 12.0) 上运行过一次，
# 结果见 docs/experiments/sm120_round_one_20260912.md。
#
# EX-D2 on Blackwell.  Written on an sm_89 machine and checked only by its own
# CPU-side self-check (SELF_CHECK=1, no GPU touched).  The fork rule is applied
# by fork.py exactly as it is here, so a Blackwell run produces its own FORK
# line rather than inheriting the sm_89 one.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_sm120"
status="${raw}/status.txt"

fail() { mkdir -p "${raw}"; echo FAIL > "${status}"; }
trap fail ERR

reject_inherited() {
  local leaked=()
  while IFS='=' read -r name _; do
    case "${name}" in
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
    dest="${raw}/insitu/${rel#docs/experiments/PLACE_ROTATE/}"
    mkdir -p "$(dirname "${dest}")"
    cp -p "${repo}/${rel}" "${dest}" 2>/dev/null || true
    if [[ "${st}" == "??" ]]; then rm -f "${repo}/${rel}"; else restore+=("${rel}"); fi
  done < <(git -C "${repo}" status --porcelain=v1 -z -uall -- "docs/experiments/PLACE_ROTATE")
  # Only the files this run actually touched are restored, so a blanket checkout
  # cannot undo an unrelated local change elsewhere in the experiment.
  if ((${#restore[@]})); then
    git -C "${repo}" checkout -- "${restore[@]}"
  fi
}

compute_cap_ok() { [[ "$1" == "12.0" ]]; }

if [[ -n "${SELF_CHECK:-}" ]]; then
  compute_cap_ok 12.0 || { echo "self-check: 12.0 rejected" >&2; exit 1; }
  ! compute_cap_ok 8.9 || { echo "self-check: 8.9 accepted" >&2; exit 1; }
  ! compute_cap_ok "" || { echo "self-check: empty cap accepted" >&2; exit 1; }
  (unset "${!TILEMEGA_@}"; reject_inherited) \
    || { echo "self-check: clean environment rejected" >&2; exit 1; }
  ! (export TILEMEGA_SCHEDULE_POLICY=stage; reject_inherited 2>/dev/null) \
    || { echo "self-check: leaked override accepted" >&2; exit 1; }
  git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "self-check: ${repo} is not a git work tree" >&2; exit 1; }
  bash -n "${here}/run.sh"
  python3 -c "import ast,sys;[ast.parse(open(p).read()) for p in sys.argv[1:]]" \
    "${here}/summarize.py" "${here}/fork.py" "${here}/headroom.py"
  # The rule table itself, on canned input, so a Blackwell run cannot be the
  # first time fork.py is exercised.
  tmp="$(mktemp -d)"
  emit() { printf 'ALL\tALL\t%s\t100\t-\t-\t%s\t%s\t%s\t1e-3\tcells=4\n' "$@"; }
  { emit neither 0.9400 0.9200 0.9600; emit full 0.9950 0.9800 1.0100; } > "${tmp}/one.tsv"
  { emit neither 0.9400 0.9200 0.9600; emit full 0.9700 0.9600 0.9800; } > "${tmp}/two.tsv"
  { emit neither 0.9900 0.9700 1.0100; emit full 0.9950 0.9800 1.0100; } > "${tmp}/three.tsv"
  for expected in 1 2 3; do
    case "${expected}" in 1) file=one;; 2) file=two;; 3) file=three;; esac
    got="$(python3 "${here}/fork.py" "${tmp}/${file}.tsv" | sed -n 's/^FORK rule=\([0-9]\).*/\1/p')"
    [[ "${got}" == "${expected}" ]] || {
      echo "self-check: fork.py gave rule=${got}, expected ${expected}" >&2
      rm -rf "${tmp}"; exit 1; }
  done
  rm -rf "${tmp}"
  echo "SELF_CHECK place_rotate sm120 ok"
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

RUNS="${RUNS:-25}" CORRECTNESS_RUNS="${CORRECTNESS_RUNS:-50}" bash "${here}/run.sh"
bash "${here}/realwidth.sh" || echo "realwidth arm failed; see ${here}/raw/realwidth" >&2
cp "${here}/raw/summary.tsv" "${here}/raw/fork.txt" "${raw}/" 2>/dev/null || true
harvest
echo PASS > "${status}"
