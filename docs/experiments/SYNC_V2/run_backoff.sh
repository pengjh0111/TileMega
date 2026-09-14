#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-E3 step 0 calibration, sm_89 (RTX 4090): the wait-policy hop sweep and the
# spin-interference pair.  Neither changes a repository default; the values they
# produce are written into configs/targets/ by hand, with this output as their
# provenance.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
here=$repo/docs/experiments/SYNC_V2
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
rounds=${ROUNDS:-4096}
guard=${GUARD:-20000}
spin_rounds=${SPIN_ROUNDS:-41}
spin_iters=${SPIN_ITERS:-20000}
bin=${BIN:-/tmp/sync-v2}
# Outputs move with OUT_DIR so the sm_120 runner cannot overwrite the
# committed sm_89 tables; sources are always read from $here.
out=${OUT_DIR:-$here}

if env | grep -q '^TILEMEGA_'; then
  echo "refusing to run with TILEMEGA_* inherited from the environment:" >&2
  env | grep '^TILEMEGA_' >&2
  exit 2
fi

mkdir -p "$bin" "$out"
"$nvcc" -std=c++17 -O2 -arch=native -I "$repo/include" \
  -o "$bin/backoff" "$here/backoff.cu"
"$nvcc" -std=c++17 -O2 -arch=native -I "$repo/include" \
  -o "$bin/spin_interference" "$here/spin_interference.cu"

# Three passes with the arm order rotated.  The first pass showed the
# spin-first arms fast in their last five cells and slow in their first two;
# rotating the order is what separates a policy effect from a warm-up or clock
# transient, and the fit reports the per-rotation spread either way.
for rot in 0 1 2; do
  "$bin/backoff" "$rounds" "$guard" "$out/backoff_rot${rot}.tsv" "$rot"
done
python3 - "$out" <<'MERGE'
import sys
here = sys.argv[1]
out = open(here + "/backoff.tsv", "w")
for i, rot in enumerate((0, 1, 2)):
    with open("%s/backoff_rot%d.tsv" % (here, rot)) as f:
        head = f.readline()
        if i == 0:
            out.write(head)
        out.writelines(f)
out.close()
MERGE
"$bin/spin_interference" "$spin_rounds" "$spin_iters" "$out/spin_interference.tsv"

python3 "$here/backoff_fit.py" "$out/backoff.tsv" "$out/spin_interference.tsv" \
  "$out/backoff_policy.tsv" | tee "$out/backoff_fit.txt"

cd "$repo"
rel=$(realpath --relative-to="$repo" "$out")
sha256sum "$rel/backoff.tsv" \
          "$rel/spin_interference.tsv" \
          "$rel/backoff_policy.tsv" > "$out/sha256.txt"
cat "$out/sha256.txt"
