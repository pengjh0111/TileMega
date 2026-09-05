#!/usr/bin/env bash
# Pair the BF16 oracle winner against the historical control in 25 interleaved
# fresh-process rounds, then emit bootstrap CI and Wilcoxon statistics.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
raw="${RAW_DIR:-${here}/raw_bf16}"
runs="${PAIR_RUNS:-25}"
[[ "${runs}" == 25 ]] || { echo 'PAIR_RUNS must be 25 for the committed protocol' >&2; exit 2; }
mkdir -p "${raw}/paired_logs"
for model in gqa2 mha4; do
  table="${raw}/final_${model}.tsv"
  [[ -s "${table}" ]] || { echo "missing ${table}" >&2; exit 2; }
  best="$(awk -F'\t' 'NR>1 && $9=="25/25" {print $6"\t"$1"x"$2"x"$3"s"$4"k"$5}' \
            "${table}" | sort -n | awk 'NR==1{print $2}')"
  [[ -n "${best}" ]] || { echo "no correct finalist for ${model}" >&2; exit 2; }
  control='128x128x16s3k1'
  printf 'round\tarm\ttag\tl05_ms\tl1_ms\tl2_ms\tstatus\n' > "${raw}/paired_${model}.tsv"
  for round in $(seq 1 "${runs}"); do
    order=(control best); (( round % 2 == 0 )) && order=(best control)
    for arm in "${order[@]}"; do
      tag="${control}"; [[ "${arm}" == best ]] && tag="${best}"
      bin="${raw}/bin/${model}_${tag}"
      fixture="${raw}/fixture/gqa2"
      [[ "${model}" == mha4 ]] && fixture="${raw}/export/mha4/fixture"
      log="${raw}/paired_logs/${model}_r${round}_${arm}.txt"
      timeout 120 "${bin}" "${fixture}" > "${log}"
      grep -q '^RESULT status=PASS' "${log}" || {
        echo "${model} ${tag} failed paired round ${round}" >&2; exit 1;
      }
      l05="$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' "${log}")"
      l1="$(sed -n 's/.* l1_ms=\([0-9.]*\).*/\1/p' "${log}")"
      l2="$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' "${log}")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\tPASS\n' \
        "${round}" "${arm}" "${tag}" "${l05}" "${l1}" "${l2}" \
        >> "${raw}/paired_${model}.tsv"
    done
  done
done
python3 "${here}/summarize_pair.py" "${raw}" > "${raw}/paired_stats.tsv"
cat "${raw}/paired_stats.tsv"
