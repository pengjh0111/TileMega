#!/usr/bin/env bash
# B0 SIMT-side exposed wait on sm_120.
#
# NOT RUN ON THIS MACHINE. The development host is a 4090 (sm_89); everything
# below was self-checked here with SELF_CHECK=1, which exercises every step
# except the sm_120 compilation and the runs themselves. Run it on the target.
#
# FORK7 is architecture specific and the sm_89 line does not carry over. The
# globaltimer tick and the SM clock differ, and so does the balance the line
# measures: the wait share is dominated by the GEMM K-loop, whose exposed
# cp.async wait depends on the memory system, not on the schedule. Re-derive
# the line on the target; do not quote the sm_89 one for sm_120.
#
# Artifacts, under --out (default $PWD/sm120):
#   raw/<cell>/build/selected_phase.{log,json}  nvcc command, hashes, head
#   raw/<cell>/correctness/selected/r*.{log,json}  one pair per fresh process
#   raw/<cell>/phase/selected_r*/dump/          one traced process per round
#   raw/{analysis,cells,segments}.tsv, raw/fork7.txt
#   status.txt                                  one line per completed step
set -euo pipefail
trap 'echo "FAILED at line $LINENO" >>"${OUT:-/tmp}/status.txt"' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
OUT="${OUT:-$PWD/sm120}"
ARCH="${ARCH:-sm_120}"
ROUNDS="${ROUNDS:-50}"
DUMPS="${DUMPS:-9}"
SELF_CHECK="${SELF_CHECK:-0}"
mkdir -p "${OUT}"
status="${OUT}/status.txt"

# The cells and configurations are FORK6's, read back from its frozen specs.
for cell in gqa2_s4 gqa2_s128 mha4_s4 mha4_s128; do
  spec="${REPO}/docs/experiments/COSTMODEL/raw_kloop/${cell}/specs.json"
  [[ -f "${spec}" ]] || { echo "missing ${spec}" >&2; exit 1; }
  python3 -c "import json,sys,pathlib
s=json.load(open(sys.argv[1]))['selected']
p=pathlib.Path(s['source'])
sys.exit(0 if p.is_file() else f'missing source {p}')" "${spec}"
done
echo "specs resolved" >>"${status}"

if [[ "${SELF_CHECK}" == 1 ]]; then
  python3 -c "import ast,sys
for f in ('run.py','analyze.py'): ast.parse(open(sys.argv[1]+'/'+f).read())" "${HERE}"
  echo 'SELF_CHECK PASS; sm_120 not run' | tee -a "${status}"
  exit 0
fi

python3 "${HERE}/run.py" build --raw "${OUT}/raw" --arch "${ARCH}"
echo "build ${ARCH}" >>"${status}"
python3 "${HERE}/run.py" correctness --raw "${OUT}/raw" --rounds "${ROUNDS}"
echo "correctness ${ROUNDS}/cell" >>"${status}"
python3 "${HERE}/run.py" dump --raw "${OUT}/raw" --rounds "${DUMPS}"
echo "dump ${DUMPS}/cell" >>"${status}"
python3 "${HERE}/analyze.py" --raw "${OUT}/raw" | tee -a "${status}"
