#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# EX-S2c: the feedback re-extraction sweep, re-run against the unified arbiter.
#
# `raw/feedback_sweep.tsv` is kept as it was measured and is F-154's evidence:
# there the loop handed back the pass's own estimate, and the two models
# disagreed -- mha4 s128 went 39 -> 41 hops and 530126 -> 532494 ns.  F-155
# reports the sweep after `ChainRequest::evaluate` made one simulation both rank
# and score, so it is a different measurement of a different code state and gets
# its own file rather than overwriting the one a committed finding cites.
#
# Host tool only: no device is opened, so this is safe to run beside GPU work.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${RAW_DIR:-${here}/raw}"
build="${BUILD_DIR:-${repo}/build-portable}"
out="${raw}/feedback_sweep_arbiter.tsv"
rounds="${ROUNDS:-0 1 2 4}"
tool="${build}/tools/tilemega-place-chain"
[[ -x "${tool}" ]] || { echo "FAIL: build ${tool} first" >&2; exit 1; }
[[ -f "${raw}/manifest.tsv" ]] || { echo "FAIL: no ${raw}/manifest.tsv" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${raw}/log"

# predicted.tsv: model=1 seq=2 candidate=4 makespan=6 max_queue_ns=15
# spine=16 cp_hops=17 cp_same_worker=18 cp_queue=19 cp_len=20.  Column 21 is
# eval_us, wall clock, so identity is only ever checked through column 20.
pick() {
  awk -F'\t' -v fb="$2" 'NR>1 && $4=="'"$3"'" {
    printf "%s\t%s_s%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
      fb, $1, $2, $4, $17, $18, $19, $20, $6, $15, $16 }' "$1"
}

{
  echo "# EX-S2c: feedback rounds under the unified arbiter (ChainRequest::evaluate)."
  echo "# fb=0 is the committed default and reproduces raw/predicted.tsv through col20."
  printf 'fb\tcell\tcandidate\tcp_hops\tcp_same_worker\tcp_queue\tcp_len\tmakespan_ns\tmax_queue_ns\tspine_length_ns\n'
} > "${out}"

for fb in ${rounds}; do
  dest="${work}/fb${fb}"
  mkdir -p "${dest}/plan"
  env TILEMEGA_CHAIN_FEEDBACK="${fb}" "${tool}" "${repo}" "${raw}/manifest.tsv" \
    "${dest}" 2>> "${raw}/log/feedback_sweep.log"
  pick "${dest}/predicted.tsv" "${fb}" chain >> "${out}"
  if [[ "${fb}" == 0 ]]; then
    if diff -q <(cut -f1-20 "${raw}/predicted.tsv") \
               <(cut -f1-20 "${dest}/predicted.tsv") > /dev/null; then
      echo "IDENTITY fb=0 matches raw/predicted.tsv through col20"
    else
      echo "FAIL: fb=0 does not reproduce raw/predicted.tsv" >&2; exit 1
    fi
  fi
done
# rotate is the gate's reference arm and carries no feedback, so it is listed once.
pick "${work}/fb0/predicted.tsv" - rotate >> "${out}"
echo "wrote ${out} ($(($(grep -vc '^#' "${out}") - 1)) rows)"
