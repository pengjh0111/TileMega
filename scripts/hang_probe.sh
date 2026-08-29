#!/usr/bin/env bash
# ==============================================================================
# 挂起现场分析（骨架 P0.3）
#
# 区分死锁与活锁的判据（R2-A 建立）：对同一个进程多次采样 PC，
#   PC 逐字节不变 且 间隔 > 20s  →  死锁
#   PC 在变                      →  活锁
# R2-A 正是靠这个判定 170 个 block 里 57 个卡在完全相同的 PC。
#
# 用法: hang_probe.sh <pid> [采样次数=3] [采样间隔秒=25]
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
