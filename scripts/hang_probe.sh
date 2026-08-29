#!/usr/bin/env bash
# ==============================================================================
# Hang forensics (skeleton P0.3)
#
# Criterion for telling deadlock from livelock (established by R2-A): sample the
# PCs of one process several times.
#   PCs byte-for-byte identical across a gap > 20s  ->  deadlock
#   PCs changing                                    ->  livelock
# This is exactly how R2-A determined that 57 of 170 blocks were stuck at the
# very same PC.
#
# Usage: hang_probe.sh <pid> [samples=3] [gap seconds=25]
# ==============================================================================
set -uo pipefail
PID="${1:?usage: hang_probe.sh <pid> [samples] [gap-seconds]}"
N="${2:-3}"
GAP="${3:-25}"
OUT="${TILEMEGA_ARTIFACT_DIR:-/tmp}/hang_probe_$PID"
mkdir -p "$OUT"

kill -0 "$PID" 2>/dev/null || { echo "error: no such process $PID" >&2; exit 1; }
command -v cuda-gdb >/dev/null || { echo "error: cuda-gdb not found" >&2; exit 1; }

for ((i=1; i<=N; i++)); do
  echo "=== sample $i/$N (t=$(date +%T)) ==="
  cuda-gdb -p "$PID" -batch \
    -ex "info cuda threads" \
    -ex "info cuda blocks" > "$OUT/sample_$i.txt" 2>&1
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw \
    --format=csv,noheader >> "$OUT/sample_$i.txt" 2>/dev/null
  grep -oP '0x[0-9a-f]+' "$OUT/sample_$i.txt" | sort -u > "$OUT/pcs_$i.txt"
  echo "  distinct PCs: $(wc -l < "$OUT/pcs_$i.txt")"
  [[ $i -lt $N ]] && sleep "$GAP"
done

echo
echo "=== verdict ==="
if [[ $N -ge 2 ]] && diff -q "$OUT/pcs_1.txt" "$OUT/pcs_$N.txt" >/dev/null 2>&1; then
  echo "  DEADLOCK: PC set identical across ${GAP}s x $((N-1)); not one instruction advanced."
else
  echo "  LIVELOCK or still progressing: the PC set is changing."
  diff "$OUT/pcs_1.txt" "$OUT/pcs_$N.txt" | head -10
fi
echo "  samples: $OUT"
