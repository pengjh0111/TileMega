#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# T1.3-B. Execute manually on the cluster-capable target. No 5090 result is
# implied by shipping this script. Requires the SEQSCAN exports and fixtures.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
out="${OUT_DIR:-${here}/raw_notify_sm120}"
python="${PYTHON:-python3}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
mkdir -p "${out}"
for model in gqa2 mha4; do
  test -f "${repo}/docs/experiments/SEQSCAN/raw/export/${model}.json"
  for seq in 4 128; do
    test -d "${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
  done
done
printf 'cluster_dim\tmodel\tvariant\tl2_ucgabar_instructions\n' > "${out}/sass.tsv"
for dim in 1 2 4 8; do
  # Match storage and residency in treatment and control.
  "${python}" "${repo}/docs/experiments/L2_ATTRIB/run_t1.py" \
    --out "${out}/dim${dim}" --arch sm_120 --cluster-dim "${dim}" \
    --cluster-reserve --variants load_lines,cluster --kappa "${KAPPA:-4}" \
    --seqs 4,128 --runs 25 --correctness-runs 50 --phases build,correctness,attrib
  "${python}" "${repo}/docs/experiments/L2_ATTRIB/summarize_t1.py" \
    "${out}/dim${dim}/attrib.tsv" --baseline load_lines --out "${out}/dim${dim}/report"
  for model in gqa2 mha4; do
    for variant in load_lines cluster; do
      count="$("${cuobjdump}" --dump-sass "${out}/dim${dim}/bin/${model}_${variant}_full" |
        awk '/Function :/ {active=($0 ~ /tilemega_l2_kernel/)}
             active && /UCGABAR/ {n++} END {print n+0}')"
      printf '%s\t%s\t%s\t%s\n' "${dim}" "${model}" "${variant}" "${count}" >> "${out}/sass.tsv"
    done
  done
done
# Missing or failing full runs abort above.
printf 'cluster_dim\tmodel\tvariant\tseq\tpass\tprocesses\n' > "${out}/correctness.tsv"
for dim in 1 2 4 8; do
  awk -F '\t' -v dim="${dim}" 'NR>1 {k=$1 FS $2 FS $4; n[k]++; ok[k]+=$7}
    END {for(k in n) print dim "\t" k "\t" ok[k] "\t" n[k]}' \
    "${out}/dim${dim}/correctness.tsv" >> "${out}/correctness.tsv"
done
printf 'cluster_dim\tmodel\tseq\tvariant\tmetric\trounds\tmedian\tci_low\tci_high\n' > "${out}/notify.tsv"
for dim in 1 2 4 8; do
  awk -F '\t' -v dim="${dim}" 'NR>1 && $4=="notify_ms" {print dim "\t" $0}' \
    "${out}/dim${dim}/report/summary.tsv" >> "${out}/notify.tsv"
done
printf 'PASS\n' > "${out}/status.txt"
