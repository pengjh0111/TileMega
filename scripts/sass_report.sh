#!/usr/bin/env bash
# ==============================================================================
# SASS 结构报告（骨架 P0.3）
#
# 自动提取 R1/R2 调查中反复要人工去数的几个量：
#   BAR.SYNC.DEFER_BLOCKING  自旋循环内的硬件屏障 —— 假设 H2 的核心对象
#   CCTL.IVALL               缓存失效 —— R2-B 修复的衡量指标
#   循环回边                  自旋循环的位置与体积
#   CGA 相关指令              R1 意外发现 3：无条件生成，疑似与死锁相关
#
# 用法: sass_report.sh <cubin> [更多 cubin...]
# ==============================================================================
set -uo pipefail
[[ $# -gt 0 ]] || { echo "用法: $0 <cubin> [cubin...]" >&2; exit 2; }

for cubin in "$@"; do
  echo "══════ $cubin ══════"
  sass="${cubin%.cubin}.sass"
  cuobjdump -sass "$cubin" > "$sass" 2>/dev/null || { echo "  反汇编失败"; continue; }

  printf "  %-28s %s\n" "指令总数"                "$(grep -c '/\*[0-9a-f]\{4\}\*/' "$sass")"
  printf "  %-28s %s\n" "BAR.SYNC.DEFER_BLOCKING" "$(grep -c 'BAR\.SYNC\.DEFER_BLOCKING' "$sass")"
  printf "  %-28s %s\n" "CCTL.IVALL"              "$(grep -c 'CCTL\.IVALL' "$sass")"
  printf "  %-28s %s\n" "CGA (CGAERRBAR/CgaSize)" "$(grep -cE 'CGAERRBAR|SR_CgaSize' "$sass")"
  cuobjdump --dump-resource-usage "$cubin" 2>/dev/null | grep -oP 'REG:\K[0-9]+' | head -1 | \
    xargs -I{} printf "  %-28s %s\n" "REG/thread" {}
  cuobjdump -elf "$cubin" 2>/dev/null | grep -A2 EIATTR_REQNTID | grep -oP 'Value:\s+\K0x[0-9a-f]+' | head -1 | \
    xargs -I{} sh -c 'printf "  %-28s %s (=%d)\n" "REQNTID (blockDim)" "$1" "$1"' _ {}

  echo "  ── 循环回边（自旋循环候选）──"
  python3 - "$sass" <<'PY'
import re, sys
insns = []
for line in open(sys.argv[1], errors='ignore'):
    m = re.search(r'/\*([0-9a-f]{4})\*/\s+(.*?);', line)
    if m:
        insns.append((int(m.group(1), 16), m.group(2).strip()))
by_off = dict(insns)
found = False
for off, txt in insns:
    if 'BRA' not in txt:
        continue
    t = re.search(r'BRA\s+(?:[^;]*?)0x([0-9a-f]+)', txt)
    if not t:
        continue
    tgt = int(t.group(1), 16)
    if tgt < off and off - tgt < 2048:
        body = [by_off.get(o, '') for o in range(tgt, off + 16, 16)]
        nbar = sum('BAR.SYNC' in b for b in body)
        found = True
        print(f"    0x{off:04x} -> 0x{tgt:04x}  体积 {off-tgt}B  "
              f"循环体内 BAR.SYNC = {nbar}"
              + ("   <-- 每次轮询全部线程会合" if nbar else ""))
if not found:
    print("    （未发现回边）")
PY
  echo
done
