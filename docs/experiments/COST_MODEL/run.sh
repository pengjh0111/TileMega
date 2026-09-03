#!/usr/bin/env bash
# TileMega P4.4 -- cost-model validation against the ORACLE measurements.
#
# Needs no GPU: every number it consumes is already committed. The measured
# latencies come from ORACLE/raw/screen_*.tsv, the occupancy inputs from the
# ptxas logs those runs were compiled with, the model description from the
# generated .cu the same runs were built from, and the hardware constants from
# configs/targets/sm_89.json.
#
#   bash docs/experiments/COST_MODEL/run.sh
#   BUILD=build-portable bash docs/experiments/COST_MODEL/run.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=${TILEMEGA_SOURCE_DIR:-$(cd "$here/../../.." && pwd)}
build=${BUILD:-build-portable}
raw="$here/raw"
mkdir -p "$raw"

# ------------------------------------------------------- occupancy inputs
# Registers are a tier-3 quantity: they do not follow from the tile shape (a
# 16x256 tile spans 122..248 across stage counts), so they are read back out
# of the compiler logs rather than predicted. Max over entry points, because
# the megakernel's occupancy is set by its most expensive one. Split-K does
# not move them -- 216 shapes, none with two distinct values across split
# factors -- so the shape alone keys the table.
for model in gqa2 mha4; do
  {
    echo -e "# shape\tmax_registers  (max over entry points in ORACLE/raw/log/*.ptxas)"
    for f in "$root"/docs/experiments/ORACLE/raw/log/"$model"_*.ptxas; do
      base=${f##*/}; base=${base%.ptxas}; base=${base#${model}_}
      regs=$( { grep -o 'Used [0-9]* registers' "$f" || true; } |
             awk '{print $2}' | sort -rn | awk 'NR==1')
      # A log with no register line is a shape that failed to compile; it has
      # no measured latency either, so it is dropped rather than defaulted.
      if [[ -n "$regs" ]]; then echo -e "${base%%k*}\t$regs"; fi
    done | sort -u -k1,1 -k2,2rn | awk '!seen[$1]++'
  } > "$raw/registers_$model.tsv"
done

# ------------------------------------------------------------------ build
if [[ ! -x "$root/$build/tools/tilemega-costmodel" ]]; then
  cmake --build "$root/$build" --target tilemega-costmodel -j "$(nproc)"
fi

"$root/$build/tools/tilemega-costmodel" --repo "$root" --out "$raw" |
  tee "$raw/summary.txt"
