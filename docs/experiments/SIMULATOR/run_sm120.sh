#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S1 §5.2 on Blackwell: the polling-contention sweep, for the clock rather
# than for the architecture.  %globaltimer ticks every 1024 ns on sm_89 and
# every 32 ns on sm_120 (F-137, F-138), so the sm_89 curve cannot tell a 200 ns
# hop from a 900 ns one -- every cell below one tick reads as one tick.  The
# same sweep on sm_120 resolves 32 ns, which is the only reason to run it.
#
# Written on the 4090 and checked only by its own CPU-side self-check
# (SELF_CHECK=1, which touches no GPU); it has never run on a Blackwell part.
# Everything it produces lands under raw_sm120/ and no path here is one run.sh
# writes: round one's Blackwell run overwrote 2291 committed sm_89 files in
# place, and this script is arranged so that cannot happen again.
#
# The sweep sets (N up to 256 consumers, R up to 64 rows) are compiled into
# contention.cu, so both parts sweep the same grid.  `guard` is in clock64()
# cycles rather than nanoseconds, so the default carries over unchanged.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_sm120"
status="${raw}/status.txt"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
rounds="${ROUNDS:-4096}"
guard="${GUARD:-20000}"
# Two fatbins and two sweeps of a few hundred kilobytes each.  The margin is
# there because round one's Blackwell session died of a full disk in an
# unrelated export (F-142), and a run that fails halfway leaves no curve.
need_mib="${NEED_MIB:-256}"

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

# Free space on the filesystem that will hold the outputs, in MiB.
free_mib() { df -Pm "$1" | awk 'NR==2 {print $4}'; }

check_disk() {
  local have
  have="$(free_mib "$1")"
  if ((have < need_mib)); then
    echo "only ${have} MiB free under $1; this run needs about ${need_mib} MiB" >&2
    return 1
  fi
  echo "disk ok: ${have} MiB free under $1, need about ${need_mib} MiB"
  return 0
}

if [[ -n "${SELF_CHECK:-}" ]]; then
  compute_cap_ok 12.0 || { echo "self-check: 12.0 rejected" >&2; exit 1; }
  ! compute_cap_ok 8.9 || { echo "self-check: 8.9 accepted" >&2; exit 1; }
  ! compute_cap_ok "" || { echo "self-check: empty cap accepted" >&2; exit 1; }
  (unset "${!TILEMEGA_@}"; reject_inherited) \
    || { echo "self-check: clean environment rejected" >&2; exit 1; }
  ! (export TILEMEGA_EVENT_LOAD_POLL=1; reject_inherited 2>/dev/null) \
    || { echo "self-check: leaked override accepted" >&2; exit 1; }
  check_disk "${here}" >/dev/null || { echo "self-check: disk check failed" >&2; exit 1; }
  (need_mib=$((1 << 40)); ! check_disk "${here}" 2>/dev/null) \
    || { echo "self-check: an impossible requirement was accepted" >&2; exit 1; }
  python3 -c "import ast,sys;[ast.parse(open(p).read()) for p in sys.argv[1:]]" \
    "${here}/hop_fit.py"
  # The Blackwell run must not be the first time the source is put through a
  # Blackwell compile, so build it here, cubin only, and throw the object away.
  tmp="$(mktemp -d)"
  "${nvcc}" -std=c++17 -O2 -arch=sm_120 -I "${repo}/include" \
    -c "${here}/contention.cu" -o "${tmp}/contention.o"
  "${nvcc}" -std=c++17 -O2 -arch=sm_120 -DTILEMEGA_EVENT_LOAD_POLL=1 \
    -I "${repo}/include" -c "${here}/contention.cu" -o "${tmp}/contention_load.o"
  # And hop_fit.py on the committed sm_89 sweeps, writing to the scratch copy,
  # so the fit is exercised without touching hop_ns.tsv.
  python3 "${here}/hop_fit.py" "${here}/contention.tsv" "${here}/contention_load.tsv" \
    "${tmp}/hop_ns.tsv" > "${tmp}/hop_fit.txt"
  diff -q "${tmp}/hop_ns.tsv" "${here}/hop_ns.tsv" \
    || { echo "self-check: refitting the sm_89 sweeps did not reproduce hop_ns.tsv" >&2
         rm -rf "${tmp}"; exit 1; }
  rm -rf "${tmp}"
  echo "SELF_CHECK simulator sm120 ok"
  exit 0
fi

mkdir -p "${raw}/bin"
reject_inherited
check_disk "${here}"
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
if ! compute_cap_ok "${cap}"; then
  echo "compute_cap=${cap}, this script is for 12.0 only" >&2
  echo FAIL > "${status}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${nvcc}" -std=c++17 -O2 -arch=sm_120 -I "${repo}/include" \
  -o "${raw}/bin/contention_rmw" "${here}/contention.cu"
"${nvcc}" -std=c++17 -O2 -arch=sm_120 -DTILEMEGA_EVENT_LOAD_POLL=1 \
  -I "${repo}/include" -o "${raw}/bin/contention_load" "${here}/contention.cu"

"${raw}/bin/contention_rmw"  "${rounds}" "${guard}" "${raw}/contention.tsv"
"${raw}/bin/contention_load" "${rounds}" "${guard}" "${raw}/contention_load.tsv"

python3 "${here}/hop_fit.py" "${raw}/contention.tsv" "${raw}/contention_load.tsv" \
  "${raw}/hop_ns.tsv" | tee "${raw}/hop_fit.txt"

( cd "${repo}" && sha256sum \
    "docs/experiments/SIMULATOR/raw_sm120/contention.tsv" \
    "docs/experiments/SIMULATOR/raw_sm120/contention_load.tsv" \
    "docs/experiments/SIMULATOR/raw_sm120/hop_ns.tsv" ) > "${raw}/sha256.txt"
cat "${raw}/sha256.txt"
echo PASS > "${status}"
