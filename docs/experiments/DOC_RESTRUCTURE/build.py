#!/usr/bin/env python3
"""Rebuild the v2.1 documents deterministically from base 059f8532 + prompt blocks.

Usage: TILEMEGA_RESTRUCTURE_PROMPT=<prompt.md> build.py <mode>   mode = migrate | full
  migrate: skeleton with only S-2 (TOC annotations), S-3, S-4(a), S-15 applied
  full:    every S-edit, plus STATUS/TODO/archive/FINDINGS/PROPOSED
"""
import os, re, subprocess, sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
BASE = '059f8532'
PROMPT = os.environ.get('TILEMEGA_RESTRUCTURE_PROMPT', '/root/Prompt/TileMega_v2.1_restructure_prompt.md')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
os.chdir(REPO)


def show(path, rev=BASE):
    return subprocess.run(['git', 'show', f'{rev}:{path}'], capture_output=True,
                          text=True, check=True).stdout


def write(path, lines):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    text = '\n'.join(lines)
    if not text.endswith('\n'):
        text += '\n'
    open(path, 'w', encoding='utf-8').write(text)


# ---------------------------------------------------------------- prompt blocks
plines = open(PROMPT, encoding='utf-8').read().split('\n')
blocks, cur = [], None
for l in plines:
    if l.strip().startswith('~~~~'):
        if cur is None:
            cur = []
        else:
            blocks.append(cur)
            cur = None
    elif cur is not None:
        cur.append(l)


def blk(key):
    hits = [b for b in blocks if key in '\n'.join(b)]
    assert len(hits) == 1, (key, len(hits))
    b = hits[0][:]
    while b and not b[0].strip():
        b.pop(0)
    while b and not b[-1].strip():
        b.pop()
    return b


DOCMAP = blk('设计与契约 → ')
assert len(DOCMAP) == 1
DOCMAP = DOCMAP[0]

# P1 values (K23 inputs; verified against L2_ATTRIB/result.md by the audit)
K23 = {'gqa2/4': (.372736, .029760, .058112, .039456, -.011104),
       'gqa2/128': (.524288, .031744, .067584, .039936, -.009248),
       'mha4/4': (.739360, .053248, .102400, .073952, -.015360),
       'mha4/128': (1.140576, .053312, .171008, .076960, -.026368)}
ceil = {k: 100 * (b + abs(l)) / L1 for k, (L1, w, n, b, l) in K23.items()}
ratio = {k: (w + n) / b for k, (L1, w, n, b, l) in K23.items()}
order = ['gqa2/4', 'gqa2/128', 'mha4/4', 'mha4/128']
CEIL_TXT = ', '.join(f'{ceil[k]:.1f}%' for k in order)
RATIO_TXT = ', '.join(f'{ratio[k]:.1f}×' for k in order)
CEIL_RANGE = f'{min(ceil.values()):.1f}–{max(ceil.values()):.1f}%'

S1 = [DOCMAP if '（此处放第 5 节的文档地图块正文）' in l else l
      for l in blk('（此处放第 5 节的文档地图块正文）')]
S3 = blk('实现状态已迁至 [`docs/STATUS.md`]')
S4B = blk('**Place 的执行语义（v2.1）**')
S5 = blk('L1 对 Place 的消费以 §5.7 为契约')
S6A = blk('### 4.4.1 L1 目标')
S6B = blk('### 4.4.2 L2 目标')
S7 = blk('**L2 执行下的 Label（v2.1）**')
S8 = blk('(d) 中的"静态调度表"')
S9 = blk('### 5.3.1 分相 ABI')
S10 = blk('**当前实现（as-built，v2.1 核实）**')
S11 = blk('### 5.5.1 当前协议')
S12 = blk('每一行的当前落地程度见')
S13 = [l.replace('<按 P1 计算的最小值>–<最大值>%', CEIL_RANGE)
       for l in blk('## 5.7 执行模型与 Plan 契约')]
S14 = blk('文档布局（v2.1）')
S15 = blk('待办已迁至 [`docs/TODO.md`]')
S16A = blk('⚠️ v2.1 待验证（不改变本规则）')
S16B = blk('## 8.10 等待提升')
S17A = blk('| R11 |')
S17B = blk('跨 stage 连续轮询放置能否抬高')
S18A = blk('JIT/AOT 双队列（stated）')
S18B = blk('静态队列构建（stated）')
S18C = blk('| MPK（v2） |')
S19 = [l.replace('<YYYY-MM-DD>', '2026-09') for l in blk('| <YYYY-MM-DD> | v2.1 |')]
ST_INTRO = blk('> 本文件是实现状态的唯一权威来源')
ST152 = [l.replace('⚠️ inferred（若写入 F-131 则引用之）', '⚠️ inferred（权重字节未核实）') for l in blk('### 1.5.2 v2.1 状态修订')]
assert any('权重字节未核实' in l for l in ST152)
ST153 = blk('### 1.5.3 被执行器结构混淆')
ST155 = blk('### 1.5.5 求解结果到代码的落地程度')
TD_INTRO = blk('> 本文件是待办的唯一权威来源')
TD_COND = blk('> **v2.1 新增常设条件**')
EX = blk('## 1. 主线：执行模型与执行感知求解（EX，v2.1）')
AR_HDR = [l.replace('<hash>', BASE).replace('<YYYY-MM-DD>', '2026-09-12')
          for l in blk('# TileMega v2.0 分阶段 TODO（归档）')]
F126 = '\n'.join(blk('## F-126 — Default L2 ownership'))
F126 = re.sub(r'<P1 values for gqa2/4,\s*gqa2/128, mha4/4, mha4/128>',
              CEIL_TXT + ' for gqa2/4, gqa2/128, mha4/4 and mha4/128', F126)
F126 = F126.replace('<P1 ratios>', RATIO_TXT)
assert '<P1' not in F126, 'placeholder left in F-126'
FANN = blk('⚠️ Read with F-126:')
PROP = blk('"TaskBody ABI: CTA-to-task ownership map" is resolved')
for name, b in [('S13', S13), ('S19', S19), ('AR_HDR', AR_HDR)]:
    assert not any(re.search(r'<[^<>=\-]{2,40}>', l) for l in b), name

# ---------------------------------------------------------------- base
base = show('TileMega_skeleton.md').split('\n')
L = lambda n: base[n - 1]


def expect(n, pred, what):
    assert pred(L(n)), f'anchor {n} ({what}) mismatch: {L(n)!r}'


expect(10, lambda s: s.startswith('> **维护约定**'), 'maint')
expect(96, lambda s: s == '## 1.5 当前状态', '1.5')
expect(347, lambda s: s.strip().endswith('`docs/experiments/COARSEN/result.md`。'), '1.5 end')
expect(479, lambda s: s.startswith('**操作的实际接入状态**'), '2.3 table start')
expect(488, lambda s: s.startswith('| **Fuse** |'), '2.3 table end')
expect(590, lambda s: s.startswith('| L5 运行时 |'), '2.6 table end')
expect(864, lambda s: s == '## 4.4 求解流程', '4.4')
expect(1036, lambda s: s == '## 4.5 Label：通信归属', '4.5')
expect(1067, lambda s: s.startswith('## 4.6'), '4.6')
expect(1089, lambda s: '(d) Megakernel 主体' in s, '5.1 (d)')
expect(1095, lambda s: s == '```', '5.1 close')
expect(1166, lambda s: s == '## 5.4 Megakernel 骨架', '5.4')
expect(1232, lambda s: s == '## 5.5 同步的三条 lowering 路径', '5.5')
expect(1261, lambda s: s == '## 5.6 求解结果到代码的映射', '5.6')
expect(1272, lambda s: s.startswith('| 区间划分 |'), '5.6 table end')
expect(1343, lambda s: s == '```', '6 tree close')
expect(1354, lambda s: s == '# 7. 分阶段 TODO', '7')
expect(2183, lambda s: s == '- [ ] 符号化的贡献（vs bucketing）', '7 end')
expect(2248, lambda s: s.startswith('norm/RoPE 一类小 tile'), '8.5 end')
expect(2293, lambda s: s == 'split-K 用 `k_begin` / `k_count`。', '8.9 end')
expect(2312, lambda s: s.startswith('| R10 |'), 'R10')
expect(2325, lambda s: '「task 类型数 3 / 5 / 10」' in s, '9.2 end')
expect(2365, lambda s: s.startswith('| 资源预算参考 |'), 'A.2 end')
expect(2375, lambda s: s.startswith('| split-K 事件范例 |'), 'A.3 end')
expect(2426, lambda s: s.startswith('| Mirage |'), 'B.3 end')
expect(2434, lambda s: s.startswith('| 2026-08 | v2.0 |'), 'changelog end')

toc15 = [i for i, s in enumerate(base[:40]) if '(#15-当前状态)' in s]
toc7 = [i for i, s in enumerate(base[:40]) if '(#7-分阶段-todo)' in s]
toc5 = [i for i, s in enumerate(base[:40]) if '(#5-lowering-路径)' in s]
assert len(toc15) == len(toc7) == len(toc5) == 1, (toc15, toc7, toc5)

# ---------------------------------------------------------------- skeleton
ops = []  # (base_line, kind, payload); kind in after/before/replace(end)


def after(n, lines): ops.append((n, 'after', lines))


def before(n, lines): ops.append((n, 'before', lines))


def replace(a, b, lines): ops.append((a, 'replace', (b, lines)))


full = MODE == 'full'
if full:
    after(2434, S19)
    after(2426, S18C)
    after(2375, S18B)
    after(2365, S18A)
    after(2325, S17B)
    after(2312, S17A)
    after(2293, [''] + S16B)
    after(2248, [''] + S16A)
replace(1354, 2183, S15)
if full:
    after(1343, [''] + S14)
    after(1272, [''] + S12 + [''] + S13)
    before(1261, S11 + [''])
    before(1232, S10 + [''])
    before(1166, S9 + [''])
    after(1095, [''] + S8)
    before(1067, S7 + [''])
    before(1036, S6B + [''])
    after(864, [''] + S6A)
    after(590, [''] + S5)
pointer = '各操作的实际接入状态见 [`docs/STATUS.md`](docs/STATUS.md) §1.5.4（v2.1 迁出）。'
replace(479, 488, (S4B + [''] if full else []) + [pointer])
replace(96, 347, S3)

sk = base[:]
for n, kind, payload in sorted(ops, key=lambda o: o[0], reverse=True):
    i = n - 1
    if kind == 'after':
        sk[i + 1:i + 1] = payload
    elif kind == 'before':
        sk[i:i] = payload
    else:
        end, lines = payload
        sk[i:end] = lines

# TOC (indices < 96 unaffected by later ops)
sk[toc15[0]] = sk[toc15[0]] + '（⚠️ v2.1：已迁至 docs/STATUS.md）'
sk[toc7[0]] = sk[toc7[0]] + '（⚠️ v2.1：已迁至 docs/TODO.md）'
if full:
    sk.insert(toc5[0] + 1, '  - [5.7 执行模型与 Plan 契约](#57-执行模型与-plan-契约)')
    j = [i for i, s in enumerate(sk[:20]) if s.startswith('> **维护约定**')][0]
    sk[j + 1:j + 1] = S1
write('TileMega_skeleton.md', sk)

# ---------------------------------------------------------------- archive
sec7 = base[1353:2183]
write('docs/archive/TODO_v2.0.md', AR_HDR + [''] + sec7)

if not full:
    print('migrate mode: skeleton + archive written')
    sys.exit(0)

# ---------------------------------------------------------------- STATUS
ANN = lambda t: f' （⚠️ v2.1：{t}）'


def annotate_row(row, text):
    r = row.rstrip()
    assert r.endswith('|'), row
    return r[:-1].rstrip() + ANN(text) + ' |'


st15 = base[95:347]
hit = {'solver': 0, 'exec': 0}
for i, row in enumerate(st15):
    if row.startswith('| L2 Solver |'):
        st15[i] = annotate_row(row, '目标函数为 L1，Place 仅为 stage 置换，见 §1.5.2 G1/G8')
        hit['solver'] += 1
    elif row.startswith('| L2 执行模型 |'):
        st15[i] = annotate_row(row, '正确性成立；执行能力受限——归属与 L1 相同、队列 stage-major、W=1，见 §1.5.2 G2/G3')
        hit['exec'] += 1
assert hit == {'solver': 1, 'exec': 1}, hit
st23 = base[478:488]
for i, row in enumerate(st23):
    if row.startswith('| **Place** | ✅ 是 |'):
        st23[i] = row.replace('| **Place** | ✅ 是 |',
                              '| **Place** | ✅ 是（⚠️ v2.1：降级为部分——只有 stage 置换与 host 映射，slot 未被消费，见 §1.5.2） |', 1)
    elif row.startswith('| **Label** |'):
        st23[i] = annotate_row(row, 'L2 执行下 Label 是 Place 的子决策，见 skeleton §5.7.1')
assert sum('v2.1' in r for r in st23) == 2
status = (['# TileMega 实现状态', '', DOCMAP, '>'] + ST_INTRO + [''] + st15 + [''] + ST152 +
          [''] + ST153 + ['', '### 1.5.4 CG 操作的实际接入状态（原 skeleton §2.3，v2.0 原文）', ''] +
          st23 + [''] + ST155)
write('docs/STATUS.md', status)

# ---------------------------------------------------------------- TODO
pre = []
k = 1  # sec7[0] is heading; find first '> ' line
while not sec7[k].startswith('> '):
    k += 1
while k < len(sec7) and sec7[k].startswith('>'):
    pre.append(sec7[k])
    k += 1

P5_START = 2044 - 1354  # index in sec7
P5_END = 2127 - 1354    # exclusive (### P5.1 heading)
assert sec7[P5_START].startswith('**进入条件（Phase 5 不得绕过）**')
assert sec7[P5_END].startswith('### P5.1')

MAP = [('P3.5', '- [~] cluster 路径：', '  → 承接：EX-E3（第 6 步）、EX-S2（cluster 共置）'),
       ('P4.4', 'T_bubble', '  → 承接：EX-S1（在模拟器中表达流水气泡与共驻）'),
       ('P4.4', 'regime 判别', '  → 承接：EX-S4'),
       ('P4.7', '消融：Label 开 / 关', '  → 承接：EX-E3（第 6 步）、EX-S2；需 caps.cluster 目标'),
       ('P4.8', '分层 DAG 上的 list scheduling', '  → 承接：EX-S2'),
       ('P4.8', '掩盖同步延迟', '  → 承接：EX-E2、EX-E4、EX-S2'),
       ('P5.1', '代价函数以', '  → 协同：EX-S5（放置的参数化）'),
       ('P6.2', '混合 batch regime 对比', '  → 承接：EX-V1'),
       ('P6.2', '端到端对比', '  → 承接：EX-V1'),
       ('P6.3', 'L2 → L3', '  → 承接：EX-V1（逐机制消融）')]
map_hits = {m[:2]: 0 for m in MAP}
P48_NOTE = '> v2.1：Round 5 的 balanced 映射（B2）负结果由 EX-S2 承接；P4.8 的 Round 5 更新段落与"已作废的 pre-queue Place 实验"原文见 archive。'

indent = lambda s: len(s) - len(s.lstrip(' '))
groups = []  # [phase, psec, [lines]]
phase = psec = None
i = 1
special_p61 = None
while i < len(sec7):
    s = sec7[i]
    if i == P5_START:
        groups.append([phase, '__P5COND__', sec7[P5_START:P5_END]])
        i = P5_END
        continue
    if s.startswith('## Phase'):
        phase, psec = s[3:].strip(), None
        i += 1
        continue
    if s.startswith('### '):
        psec = s[4:].strip()
        if psec.startswith('P6.1'):
            j = i + 1
            while j < len(sec7) and not sec7[j].startswith('#'):
                j += 1
            groups.append([phase, psec, [x for x in sec7[i + 1:j]]])
            i = j
        else:
            i += 1
        continue
    m = re.match(r'^(\s*)- \[( |~|!)\]', s)
    if m:
        ind = indent(s)
        j = i + 1
        while j < len(sec7):
            t = sec7[j]
            if t.startswith('#') or j == P5_START:
                break
            if not t.strip():
                q = j + 1
                while q < len(sec7) and not sec7[q].strip():
                    q += 1
                if q < len(sec7) and indent(sec7[q]) > ind and not sec7[q].startswith('#'):
                    j = q
                    continue
                break
            if indent(t) <= ind:
                break
            j += 1
        item = sec7[i:j]
        for (ps, key, line) in MAP:
            if psec and psec.startswith(ps) and key in item[0]:
                item = item + [line]
                map_hits[(ps, key)] += 1
        groups.append([phase, psec, item])
        i = j
        continue
    i += 1
assert all(v == 1 for v in map_hits.values()), map_hits

todo2 = ['## 2. v2.0 延续项（原文迁入）', '',
         '> 本节为 v2.0 §7 中状态为 `[ ]`、`[~]`、`[!]` 的条目原文副本（含 Phase 5 进入条件与 P6.1 说明），按原 Phase 与小节分组；以"→ 承接 / → 协同"开头的行与 P4.8 的注记为 v2.1 追加。完整原文见 `docs/archive/TODO_v2.0.md`。', '']
last_phase = last_psec = None
for ph, ps, lines in groups:
    if ph != last_phase:
        todo2 += [f'### {ph}', '']
        last_phase, last_psec = ph, None
    if ps == '__P5COND__':
        todo2 += ['#### Phase 5 进入条件（v2.0 原文）', ''] + lines + ['']
        last_psec = ps
        continue
    if ps != last_psec:
        todo2 += [f'#### {ps}', '']
        if ps and ps.startswith('P4.8'):
            todo2 += [P48_NOTE, '']
        last_psec = ps
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    todo2 += lines + ['']

ledger = open('docs/experiments/ROUND5_LEDGER.md', encoding='utf-8').read().split('\n')
R5 = ['A12.1', 'B1.2', 'B1.4', 'B1.5', 'B2.2', 'B2.5', 'B2.6', 'B3.1', 'B3.2', 'B4.1']
REL = {'B2.5': 'EX-S2', 'B2.2': 'EX-S5', 'B1.5': '待 sm_120', 'B2.6': '待 sm_120',
       'A12.1': '独立（isl 引用审计）', 'B4.1': '独立（isl 引用审计）'}
todo2 += ['### Round 5 未关闭条目（指针；状态以 docs/experiments/ROUND5_LEDGER.md 为准）', '',
          '| ID | 实现状态 | 验证状态/剩余项 | 关联 |', '|---|---|---|---|']
for rid in R5:
    rows = [r for r in ledger if r.startswith(f'| {rid} |')]
    assert len(rows) == 1, rid
    cells = [c.strip() for c in rows[0].strip().strip('|').split('|')]
    rel = REL.get(rid) or ('独立（融合）' if rid.startswith('B1') else '独立（符号 DP）')
    todo2.append(f'| {rid} | {cells[2]} | {cells[3]} | {rel} |')
todo2.append('')

todo3 = ['## 3. 已完成条目索引（v2.0，原文见 archive）', '']
dropped = []
psec = None
counts, paths = {}, {}
order_ps = []
for s in sec7:
    if s.startswith('### '):
        psec = s[4:].strip()
        order_ps.append(psec)
        counts[psec], paths[psec] = 0, []
        continue
    if psec is None:
        continue
    if re.match(r'^\s*- \[x\]', s):
        counts[psec] += 1
    if re.match(r'^\s*- \[-\]', s):
        first = re.sub(r'^\s*- \[-\]\s*', '', s)
        dropped.append((psec, re.split(r'(?<=[。；])', first)[0][:120]))
    for p in re.findall(r'docs/experiments/[A-Za-z0-9_.*/-]*[A-Za-z0-9_/*]', s):
        if p not in paths[psec]:
            paths[psec].append(p)
for ps in order_ps:
    if counts[ps]:
        ev = '、'.join(f'`{p}`' for p in paths[ps]) if paths[ps] else '见 archive 原文'
        todo3.append(f'- {ps} — [x] {counts[ps]} 项 — 证据：{ev}')
todo3 += ['', '已放弃（`[-]`）条目：', '']
todo3 += [f'- {ps}：{t}' for ps, t in dropped] or ['- （无）']

todo = (['# TileMega 待办', '', DOCMAP, '>'] + TD_INTRO + ['', '## 0. 约定', ''] + pre + [''] +
        TD_COND + [''] + EX + [''] + todo2 + todo3)
write('docs/TODO.md', todo)

# ---------------------------------------------------------------- FINDINGS
fd = show('docs/FINDINGS.md').split('\n')
for fid in ['F-118', 'F-86', 'F-82']:
    h = [i for i, s in enumerate(fd) if re.match(rf'^## {fid}\b', s)]
    assert len(h) == 1, fid
    nxt = [i for i in range(h[0] + 1, len(fd)) if fd[i].startswith('## F-')]
    end = nxt[0] if nxt else len(fd)
    k = end
    while k > h[0] and not fd[k - 1].strip():
        k -= 1
    fd[k:k] = [''] + FANN
while fd and not fd[-1].strip():
    fd.pop()
fd += [''] + F126.split('\n')
write('docs/FINDINGS.md', fd)

# ---------------------------------------------------------------- PROPOSED
pp = show('docs/PROPOSED_SKELETON_CHANGES.md').split('\n')
a = [i for i, s in enumerate(pp) if s.startswith('## TaskBody ABI: CTA-to-task ownership map')][0]
b = [i for i, s in enumerate(pp) if s.startswith('## Stabilize export-bridge schema')][0]
r = [i for i, s in enumerate(pp) if s.startswith('## Resolved and removed from this file')][0]
assert r < a < b
body = pp[:a] + pp[b:]
k = a
while k > r and not body[k - 1].strip():
    k -= 1
body[k:k] = [''] + PROP
write('docs/PROPOSED_SKELETON_CHANGES.md', body)

print('full mode: all documents written')
print('P1 ceilings', CEIL_TXT, '| ratios', RATIO_TXT, '| range', CEIL_RANGE)
print('open-item groups', len(groups), '| mapped', sum(map_hits.values()))
