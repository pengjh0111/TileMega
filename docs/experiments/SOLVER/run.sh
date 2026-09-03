#!/usr/bin/env bash
# TileMega P4.5 -- chain DP over the calibrated cost model.
#
# Needs no GPU. The candidate set, the occupancy inputs and the measured
# latencies the solution is ranked against are the same committed artifacts
# COST_MODEL/run.sh uses, so run that first (or let this script do it) to
# regenerate the register tables.
#
#   bash docs/experiments/SOLVER/run.sh
#   BUILD=build-portable bash docs/experiments/SOLVER/run.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=${TILEMEGA_SOURCE_DIR:-$(cd "$here/../../.." && pwd)}
build=${BUILD:-build-portable}
raw="$here/raw"
cost_raw="$root/docs/experiments/COST_MODEL/raw"
mkdir -p "$raw"

if [[ ! -s "$cost_raw/registers_gqa2.tsv" || ! -s "$cost_raw/registers_mha4.tsv" ]]; then
  bash "$root/docs/experiments/COST_MODEL/run.sh"
fi

if [[ ! -x "$root/$build/tools/tilemega-solve" ]]; then
  cmake --build "$root/$build" --target tilemega-solve -j "$(nproc)"
fi

# Exits non-zero when acceptance (a) fails, so this doubles as the regression
# check: the uniform-g solution must stay inside the measured top 3%.
"$root/$build/tools/tilemega-solve" --repo "$root" --cost-dir "$cost_raw" \
  --out "$raw" | tee "$raw/summary.txt"
