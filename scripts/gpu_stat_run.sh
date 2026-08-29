#!/usr/bin/env bash
# ==============================================================================
# 统计化 GPU 测试执行器（骨架 P0.3）
#
# 三条从 R1/R2 教训里提炼的纪律，全部内建在这里：
#
#  1. 单次成功不构成证据。R2 的 grid-scan 证明挂起率是个概率量
#     （grid=30 时 39/50 vs 48/50），跑一次什么也说明不了。默认 N=50。
#
#  2. 挂起必须超时并记为失败，且保留现场。没有 timeout 的 GPU 测试会把
#     整条 CI 挂死。
#
#  3. **GPU 独占**。挂起是对时序敏感的竞态（R1-E5 的核心发现：预热能
#     "救活" grid=80）。若另一个进程同时压 GPU 或 CPU，测出来的挂起率
#     没有意义。这里用 flock 串行化，并在开跑前检查机器是否空闲。
#
# 输出 CSV，便于跨版本/跨变体对比。
#
# 用法:
#   gpu_stat_run.sh --label <名字> --runs 50 --timeout 6 -- <命令> [参数...]
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

# --- 机器空闲检查 ----------------------------------------------------------
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
    rm -f "$out"   # 成功的输出不留，只保留失败现场
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

# 有挂起就是失败：挂起绝不能被当成"偶发"放过。
[[ $hang -eq 0 && $err -eq 0 ]]
