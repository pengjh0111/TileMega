#!/usr/bin/env bash
# Shared guards for R4 runners. Written on sm_89; never run on sm_120 here.
set -euo pipefail

r4_cap_ok() { [[ "$1" == '12.0' ]]; }
r4_reject_inherited() {
  local name
  while IFS='=' read -r name _; do
    if [[ "${name}" == TILEMEGA_* ]]; then
      echo "refusing inherited ${name}" >&2
      return 1
    fi
  done < <(env)
}
r4_disk() {
  local free
  free="$(df -Pm "$1" | awk 'NR==2 {print $4}')"
  echo "DISK NEED_MIB=$2 FREE_MIB=${free}"
  [[ "${free}" -ge "$2" ]]
}
r4_fail() { mkdir -p "${raw}"; echo FAIL > "${raw}/status.txt"; }
r4_init() {
  trap r4_fail ERR
  if [[ "${SELF_CHECK:-0}" == 1 ]]; then
    r4_cap_ok 12.0
    ! r4_cap_ok 8.9
    ! r4_cap_ok ''
    (unset "${!TILEMEGA_@}"; r4_reject_inherited)
    ! (export TILEMEGA_FAKE_OVERRIDE=1; r4_reject_inherited 2>/dev/null)
    r4_disk "${here}" "${NEED_MIB}"
    ! r4_disk "${here}" 1099511627776
    python3 -c 'import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]' "${python_sources[@]}"
    bash -n "${BASH_SOURCE[1]}"
    echo "SELF_CHECK ${label} PASS; GPU untouched; sm_120 untested"
    return 0
  fi
  mkdir -p "${raw}"
  echo RUNNING > "${raw}/status.txt"
  r4_reject_inherited
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
  r4_cap_ok "${cap}" || { echo "requires compute_cap=12.0, got ${cap}" >&2; return 1; }
  r4_disk "${raw}" "${NEED_MIB}"
}
