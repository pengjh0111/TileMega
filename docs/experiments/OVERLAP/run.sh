#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Fifty fresh-process traces of the queue property a stage loop cannot expose.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
runs="${RUNS:-50}"
raw="${here}/raw"
mkdir -p "${raw}"

printf 'model\trun\ttask_starts\tearly\tearly_pct\tpairs\ttransitions\n' \
  > "${raw}/trace.tsv"
for model in gqa2 mha4; do
  bin="${repo}/docs/experiments/TASKQUEUE/raw/bin/${model}"
  if [[ "${model}" == gqa2 ]]; then
    fixture="${repo}/docs/experiments/ORACLE/raw_bf16/fixture/gqa2"
  else
    fixture="${repo}/docs/experiments/ORACLE/raw_bf16/export/mha4/fixture"
  fi
  [[ -x "${bin}" && -f "${fixture}/manifest.json" ]] || {
    echo 'run TASKQUEUE/run.sh first' >&2; exit 77;
  }
  for run in $(seq 1 "${runs}"); do
    log="${raw}/${model}_${run}.txt"
    TILEMEGA_TASK_TRACE=1 timeout 30s "${bin}" "${fixture}" > "${log}"
    grep -q '^RESULT status=PASS' "${log}"
    row="$(grep -m1 '^E2E_OVERLAP' "${log}")"
    field() { sed -n "s/.* $1=\\([^ ]*\\).*/\\1/p" <<< "${row}"; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${model}" "${run}" "$(field task_starts)" \
      "$(field cross_stage_early)" "$(field cross_stage_early_pct)" \
      "$(field overlap_pairs)" "$(field queue_stage_transitions)" \
      >> "${raw}/trace.tsv"
  done
done

awk -F'\t' '
  NR == 1 { next }
  { n[$1]++; early[$1]+=$4; pct[$1]+=$5; pairs[$1]+=$6; transitions[$1]+=$7;
    if (!(($1) in min) || $4 < min[$1]) min[$1]=$4;
    if (!(($1) in max) || $4 > max[$1]) max[$1]=$4 }
  END {
    for (m in n) printf "%s\t%d\t%.3f\t%.4f\t%d\t%d\t%.3f\t%.3f\n",
      m,n[m],early[m]/n[m],pct[m]/n[m],min[m],max[m],pairs[m]/n[m],transitions[m]/n[m]
  }' "${raw}/trace.tsv" | sort > "${raw}/summary.body"
{
  printf 'model\tprocesses\tmean_early\tmean_early_pct\tmin_early\tmax_early\tmean_pairs\tmean_transitions\n'
  cat "${raw}/summary.body"
} > "${raw}/summary.tsv"
cat "${raw}/summary.tsv"
