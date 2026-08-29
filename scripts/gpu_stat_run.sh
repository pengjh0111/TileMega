#!/usr/bin/env bash
# ==============================================================================
# Statistical GPU test runner (skeleton P0.3)
#
# Three disciplines distilled from the R1/R2 lessons, all built in here:
#
#  1. A single success is not evidence. R2's grid scan showed the hang rate is a
#     probabilistic quantity (39/50 vs 48/50 at grid=30), so one run proves
#     nothing. N defaults to 50.
#
#  2. Hangs must time out, count as failures, and leave their artifacts behind.
#     A GPU test without a timeout wedges the whole CI pipeline.
#
#  3. EXCLUSIVE GPU ACCESS. The hang is a timing-sensitive race (R1-E5's central
#     finding: warm-up could "rescue" grid=80). If another process is loading the
#     GPU or the CPU at the same time, the measured hang rate is meaningless.
#     This serialises through flock and checks that the machine is idle before
#     starting.
#
# Emits CSV so results can be compared across versions and variants.
#
# Usage:
#   gpu_stat_run.sh --label <name> --runs 50 --timeout 6 -- <command> [args...]
# ==============================================================================
set -uo pipefail

LOCK_FILE="${TILEMEGA_GPU_LOCK:-/tmp/tilemega-gpu.lock}"
RUNS=50
TIMEOUT_S=6
LABEL="unnamed"
CSV=""
SUCCESS_PATTERN=""
ALLOW_BUSY=0

usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)   LABEL="$2"; shift 2;;
    --runs)    RUNS="$2"; shift 2;;
    --timeout) TIMEOUT_S="$2"; shift 2;;
    --csv)     CSV="$2"; shift 2;;
    --expect)  SUCCESS_PATTERN="$2"; shift 2;;
    --allow-busy) ALLOW_BUSY=1; shift;;
    -h|--help) usage 0;;
    --) shift; break;;
    *) echo "未知选项: $1" >&2; usage 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "错误: 缺少要执行的命令" >&2; usage 2; }

# --- machine idle check ----------------------------------------------------
if [[ $ALLOW_BUSY -eq 0 ]]; then
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  load1=$(cut -d' ' -f1 /proc/loadavg)
  ncpu=$(nproc)
  busy=0
  [[ -n "$gpu_util" && "$gpu_util" -gt 10 ]] && { echo "警告: GPU 利用率 ${gpu_util}%，非空闲" >&2; busy=1; }
  awk -v l="$load1" -v n="$ncpu" 'BEGIN{exit !(l > n*0.5)}' && {
    echo "警告: CPU 负载 $load1 (核数 $ncpu)，非空闲" >&2; busy=1; }
  if [[ $busy -eq 1 ]]; then
    cat >&2 <<'EOF'
错误: 机器不空闲，拒绝执行。
  挂起是对时序敏感的竞态（R1-E5：预热可以"救活" grid=80），在有负载的机器上
  测出的挂起率不可用于任何结论。等机器空闲，或明确用 --allow-busy 并在报告里
  注明该数据是在有负载条件下采集的。
EOF
    exit 3
  fi
fi

exec 9>"$LOCK_FILE"
flock 9 || { echo "错误: 拿不到 GPU 锁 $LOCK_FILE" >&2; exit 3; }

ARTIFACT_DIR="${TILEMEGA_ARTIFACT_DIR:-/tmp/tilemega-gpu-$LABEL-$$}"
mkdir -p "$ARTIFACT_DIR"

ok=0; hang=0; err=0
echo "=== $LABEL : $RUNS 次，单次超时 ${TIMEOUT_S}s ==="
for ((i=1; i<=RUNS; i++)); do
  out="$ARTIFACT_DIR/run_$i.out"
  timeout --signal=KILL "$TIMEOUT_S" "$@" > "$out" 2>&1
  rc=$?
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    hang=$((hang+1)); status=HANG
  elif [[ $rc -ne 0 ]]; then
    err=$((err+1)); status="ERR(rc=$rc)"
  elif [[ -n "$SUCCESS_PATTERN" ]] && ! grep -q "$SUCCESS_PATTERN" "$out"; then
    err=$((err+1)); status="ERR(输出不含 '$SUCCESS_PATTERN')"
  else
    ok=$((ok+1)); status=OK
    rm -f "$out"   # discard successful output; keep only failure artifacts
  fi
  printf "\r  %3d/%d  ok=%d hang=%d err=%d  最近=%s        " \
         "$i" "$RUNS" "$ok" "$hang" "$err" "$status"
done
echo

hang_rate=$(awk -v h="$hang" -v n="$RUNS" 'BEGIN{printf "%.1f", 100.0*h/n}')
echo "=== $LABEL 结果: ok=$ok/$RUNS  挂起=$hang/$RUNS (${hang_rate}%)  错误=$err/$RUNS ==="
[[ $((hang+err)) -gt 0 ]] && echo "    失败现场保留在: $ARTIFACT_DIR"

if [[ -n "$CSV" ]]; then
  [[ -f "$CSV" ]] || echo "label,runs,ok,hang,err,hang_rate_pct,timeout_s,timestamp,tileiras,driver" > "$CSV"
  echo "$LABEL,$RUNS,$ok,$hang,$err,$hang_rate,$TIMEOUT_S,$(date -Is),${TILEMEGA_TILEIRAS:-default},$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1)" >> "$CSV"
fi

# Any hang is a failure: a hang must never be waved away as "flaky".
[[ $hang -eq 0 && $err -eq 0 ]]
