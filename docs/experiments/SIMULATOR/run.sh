#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S1 §5.2: the polling-contention calibration, sm_89 (RTX 4090).
#
# Builds the microbenchmark twice -- the default RMW poll and, for comparison
# only, TILEMEGA_EVENT_LOAD_POLL=1 -- runs both full-length sweeps, and fits
# hop_ns(N, R).  The repository default is not changed by either arm.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
here=$repo/docs/experiments/SIMULATOR
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
rounds=${ROUNDS:-4096}
guard=${GUARD:-20000}
bin=${BIN:-/tmp/simulator-contention}

if env | grep -q '^TILEMEGA_'; then
  echo "refusing to run with TILEMEGA_* inherited from the environment:" >&2
  env | grep '^TILEMEGA_' >&2
  exit 2
fi

mkdir -p "$bin"
"$nvcc" -std=c++17 -O2 -arch=native -I "$repo/include" \
  -o "$bin/contention_rmw" "$here/contention.cu"
"$nvcc" -std=c++17 -O2 -arch=native -DTILEMEGA_EVENT_LOAD_POLL=1 \
  -I "$repo/include" -o "$bin/contention_load" "$here/contention.cu"

"$bin/contention_rmw"  "$rounds" "$guard" "$here/contention.tsv"
"$bin/contention_load" "$rounds" "$guard" "$here/contention_load.tsv"

python3 "$here/hop_fit.py" "$here/contention.tsv" "$here/contention_load.tsv" \
  "$here/hop_ns.tsv" | tee "$here/hop_fit.txt"

cd "$repo"
sha256sum docs/experiments/SIMULATOR/contention.tsv \
          docs/experiments/SIMULATOR/contention_load.tsv \
          docs/experiments/SIMULATOR/hop_ns.tsv > "$here/sha256.txt"
cat "$here/sha256.txt"
