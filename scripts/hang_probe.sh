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
PID="${1:?用法: hang_probe.sh <pid> [次数] [间隔秒]}"
N="${2:-3}"
GAP="${3:-25}"
OUT="${TILEMEGA_ARTIFACT_DIR:-/tmp}/hang_probe_$PID"
mkdir -p "$OUT"

kill -0 "$PID" 2>/dev/null || { echo "错误: 进程 $PID 不存在" >&2; exit 1; }
command -v cuda-gdb >/dev/null || { echo "错误: 找不到 cuda-gdb" >&2; exit 1; }

for ((i=1; i<=N; i++)); do
  echo "=== 采样 $i/$N (t=$(date +%T)) ==="
  cuda-gdb -p "$PID" -batch \
    -ex "info cuda threads" \
    -ex "info cuda blocks" > "$OUT/sample_$i.txt" 2>&1
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw \
    --format=csv,noheader >> "$OUT/sample_$i.txt" 2>/dev/null
  grep -oP '0x[0-9a-f]+' "$OUT/sample_$i.txt" | sort -u > "$OUT/pcs_$i.txt"
  echo "  不同 PC 数: $(wc -l < "$OUT/pcs_$i.txt")"
  [[ $i -lt $N ]] && sleep "$GAP"
done

echo
echo "=== 判定 ==="
if [[ $N -ge 2 ]] && diff -q "$OUT/pcs_1.txt" "$OUT/pcs_$N.txt" >/dev/null 2>&1; then
  echo "  死锁：${GAP}s×$((N-1)) 间隔内 PC 集合完全一致，一条指令未前进。"
else
  echo "  活锁或仍在推进：PC 集合在变化。"
  diff "$OUT/pcs_1.txt" "$OUT/pcs_$N.txt" | head -10
fi
echo "  采样数据: $OUT"
