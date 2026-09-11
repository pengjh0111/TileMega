#!/usr/bin/env python3
"""v2.1 document-restructure audit (A1-A7).

Run from anywhere inside the repository:
    python3 docs/experiments/DOC_RESTRUCTURE/audit.py
Outputs *.tsv and audit_summary.txt next to this script. Exit status is
non-zero when a hard check (A1, A3, A4, A5, A7) fails. A2 and A6 are reports.
"""
import glob, os, re, subprocess, sys
from pathlib import Path

BASE = '059f8532509fcf21c9cc96e078903cc1edc13a5a'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.chdir(ROOT)
DOCS = ['TileMega_skeleton.md', 'docs/STATUS.md', 'docs/TODO.md', 'docs/archive/TODO_v2.0.md']
CODE_DIRS = ['lib', 'include', 'tools', 'test', 'python', 'cmake', 'scripts', 'third_party']
PLANNED_DIRS = {'TRACE_V2', 'PLACE_ROTATE', 'PLAN_CONTRACT', 'WINDOW', 'SYNC_V2', 'PREFETCH',
                'SIMULATOR', 'PLACE_EFT'}
ANN = re.compile(r'\s*（⚠️ v2\.1[^）]*）')
summary, hard_fail = [], False


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout


def base_text(path):
    return git('show', f'{BASE}:{path}')


def read(path):
    return Path(path).read_text(encoding='utf-8')


def line_of(ref):
    path, n = ref.rsplit(':', 1)
    return read(path).split('\n')[int(n) - 1]


def tsv(name, header, rows):
    with open(HERE / name, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(str(x).replace('\t', ' ').replace('\n', ' ') for x in r) + '\n')


def verdict(tag, ok, detail, hard=True):
    global hard_fail
    summary.append(f"{tag}\t{'PASS' if ok else ('FAIL' if hard else 'REPORT')}\t{detail}")
    if hard and not ok:
        hard_fail = True


# ------------------------------------------------------------------ A1 line preservation
old = base_text('TileMega_skeleton.md').split('\n')
new_lines, norm_lines = set(), set()
for d in DOCS:
    for l in read(d).split('\n'):
        new_lines.add(l)
        norm_lines.add(ANN.sub('', l))
allow = {}
if (HERE / 'allowlist.tsv').exists():
    for r in read(HERE / 'allowlist.tsv').split('\n')[1:]:
        if r.strip():
            n, _, why = r.split('\t', 2)
            allow[int(n)] = why
rows, missing = [], 0
for i, l in enumerate(old, 1):
    if not l.strip() or l in new_lines:
        continue
    if l in norm_lines:
        rows.append((i, 'annotated', l))
        continue
    rows.append((i, 'ALLOWLISTED: ' + allow[i] if i in allow else 'MISSING', l))
    missing += i not in allow
tsv('audit_lines.tsv', ['base_line', 'state', 'text'], rows)
if not (HERE / 'allowlist.tsv').exists():
    tsv('allowlist.tsv', ['base_line', 'text', 'reason'], [])
n_nonempty = sum(1 for l in old if l.strip())
verdict('A1', missing == 0, f'{n_nonempty} non-empty base lines; missing {missing}; '
        f'annotated {sum(1 for r in rows if r[1] == "annotated")}')

# ------------------------------------------------------------------ A2 new numbers (report)
corpus = base_text('TileMega_skeleton.md') + base_text('docs/FINDINGS.md')
for p in ['docs/experiments/L2_ATTRIB/result.md', 'docs/experiments/E2E_L2/result.md',
          'docs/experiments/OVERLAP/result.md', 'docs/experiments/PLACE/round5_balanced_result.md',
          'docs/experiments/TASKQUEUE/result.md', 'docs/experiments/ROUND5_LEDGER.md',
          'docs/experiments/COARSEN/result.md']:
    corpus += read(p) if Path(p).exists() else ''
P1 = {'13.6', '9.4', '12.1', '9.1', '2.2', '2.5', '2.1', '2.9'}
STATED = {'2512.22219', '2604.13327', '2026-06-10', '1.09', '1.06', '1.07', '0.83', '0.82',
          '0.85', '0.89'}
base_set = set(old) | set(base_text('docs/FINDINGS.md').split('\n')) | \
    set(base_text('docs/PROPOSED_SKELETON_CHANGES.md').split('\n'))
arch_set = set(read('docs/archive/TODO_v2.0.md').split('\n'))
ident = re.compile(r'^#+\s*[\d.]+|§[\d.()a-z]+|\bF-\d+|\bEX-[A-Z]\d*|\b[GKMRP]\d+(\.\d+)*\b|\bL-[a-f]\b|'
                   r'\bsm_\d+\b|\bv\d(\.\d)?\b|\bPhase \d\b|\b[A-B]\d+(\.\d+)?\b|`[^`]*`|'
                   r'\bW\s*[=≥<>]\s*\d+|\bI\d\b|\bT\d\b|\bS-\d+|\bC\d\b|\bL[0-5](\.5)?\b|\b\d+[×x]\d+\b')
num = re.compile(r'(?<![\w.])[-−]?\d+(?:\.\d+)?%?(?![\w])')
R5 = ['A12.1', 'B1.2', 'B1.4', 'B1.5', 'B2.2', 'B2.5', 'B2.6', 'B3.1', 'B3.2', 'B4.1']
arows = []
for d in DOCS[:3] + ['docs/FINDINGS.md', 'docs/PROPOSED_SKELETON_CHANGES.md']:
    for i, l in enumerate(read(d).split('\n'), 1):
        if l in base_set or ANN.sub('', l) in base_set or l in arch_set:
            continue
        ledger_row = d == 'docs/TODO.md' and any(l.startswith(f'| {x} |') for x in R5)
        for tok in num.findall(ident.sub(' ', l)):
            t = tok.rstrip('%').replace('−', '-').lstrip('-')
            if ledger_row:
                src = 'ROUND5_LEDGER (copied cells)'
            elif t in P1 and (tok.endswith('%') or '×' in l):
                src = 'P1 formula'
            elif t in corpus:
                src = 'base docs / cited results'
            elif t in STATED:
                src = 'stated (paper)'
            else:
                src = 'UNSOURCED'
            arows.append((d, i, tok, src))
tsv('audit_numbers.tsv', ['file', 'line', 'token', 'source'], arows)
uns = [r for r in arows if r[3] == 'UNSOURCED']
verdict('A2', not uns, f'{len(arows)} numeric tokens in new lines; unsourced {len(uns)} '
        f'(review audit_numbers.tsv)', hard=False)

# ------------------------------------------------------------------ A3 references
findings = read('docs/FINDINGS.md')
f_ids = set(re.findall(r'^## (F-\d+)\b', findings, re.M))
todo = read('docs/TODO.md')
ex_defined = set(re.findall(r'^#### (EX-[A-Z]\d+)\b', todo, re.M))


def slug(h):
    h = h.strip().lower()
    h = re.sub(r'[^\w\- \u4e00-\u9fff]', '', h)
    return h.replace(' ', '-')


rrows, bad = [], 0
for d in DOCS[:3] + ['docs/FINDINGS.md']:
    text = read(d)
    heads = {slug(m) for m in re.findall(r'^#{1,6} (.+)$', text, re.M)}
    newpart = text if d != 'docs/FINDINGS.md' else text[text.find('## F-126'):]
    new_only = '\n'.join(l for l in newpart.split('\n') if l not in base_set and ANN.sub('', l) not in base_set
                         and l not in arch_set)
    for p in sorted(set(re.findall(r'docs/experiments/[A-Za-z0-9_.*/-]*[A-Za-z0-9_/*]', newpart))):
        parts = p.split('/')
        top = parts[2] if len(parts) > 2 else ''
        exists = bool(glob.glob(p))
        state = 'ok' if exists else ('planned' if top in PLANNED_DIRS else
                                     ('pre-existing (base text)' if p not in new_only or
                                      any(p in l for l in base_set) else 'MISSING'))
        rrows.append((d, 'path', p, state))
        bad += state == 'MISSING'
    for f in sorted(set(re.findall(r'\bF-\d+\b', newpart))):
        rrows.append((d, 'finding', f, 'ok' if f in f_ids else 'MISSING'))
        bad += f not in f_ids
    for e in sorted(set(re.findall(r'\bEX-[A-Z]\d+\b', newpart))):
        rrows.append((d, 'ex', e, 'ok' if e in ex_defined else 'UNDEFINED'))
        bad += e not in ex_defined
    for a in sorted(set(re.findall(r'\]\(#([^)]+)\)', text))):
        rrows.append((d, 'anchor', a, 'ok' if a in heads else 'UNRESOLVED'))
        bad += a not in heads
    for link in sorted(set(re.findall(r'\]\(((?:\.\./|docs/)[^)#]+)\)', text))):
        ok = (Path(d).parent / link).exists()
        rrows.append((d, 'link', link, 'ok' if ok else 'BROKEN'))
        bad += not ok
tsv('audit_refs.tsv', ['file', 'kind', 'ref', 'state'], rrows)
verdict('A3', bad == 0, f'{len(rrows)} references; unresolved {bad}')

# ------------------------------------------------------------------ A4 identifiers
sec7 = '\n'.join(old[1353:2183])
pids = sorted(set(re.findall(r'^### (P\d\.\d+)\b', sec7, re.M)))
arch = read('docs/archive/TODO_v2.0.md')
irows, bad = [], 0
for p in pids:
    ok = p in todo or p in arch
    irows.append(('P', p, 'ok' if ok else 'MISSING'))
    bad += not ok
for e in sorted(set(re.findall(r'\bEX-[A-Z]\d+\b', todo))):
    n = len(re.findall(rf'^#### {e}\b', todo, re.M))
    irows.append(('EX heading count', e, n))
    bad += n != 1
tsv('audit_ids.tsv', ['kind', 'id', 'state'], irows)
verdict('A4', bad == 0, f'{len(pids)} P ids, {len(ex_defined)} EX headings; problems {bad}')

# ------------------------------------------------------------------ A5 zero code change
changed = [l for l in git('diff', '--name-only', BASE, '--').split('\n') if l]
changed += [l for l in git('ls-files', '--others', '--exclude-standard').split('\n') if l]
code = [c for c in changed if c.split('/')[0] in CODE_DIRS or
        c in ('CMakeLists.txt', 'CLAUDE.md', 'AGENTS.md', 'README.md')]
exp = [c for c in changed if c.startswith('docs/experiments/') and
       not c.startswith('docs/experiments/DOC_RESTRUCTURE/')]
verdict('A5', not code and not exp, f'code changes {code or 0}; other experiment changes {exp or 0}')

# ------------------------------------------------------------------ A6 dangling section refs (report)
nums = set()
for h in re.findall(r'^#{1,6} (.+)$', read('TileMega_skeleton.md'), re.M):
    m = re.match(r'(?:附录 )?([0-9A-B]+(?:\.[0-9]+)*)\.?\s', h)
    if m:
        nums.add(m.group(1).rstrip('.'))
drows = []
for d in ['lib', 'include', 'tools', 'test', 'python']:
    for f in sorted(glob.glob(f'{d}/**/*', recursive=True)):
        if not os.path.isfile(f) or not f.endswith(('.cpp', '.h', '.cuh', '.cu', '.py', '.td', '.txt', '.mlir')):
            continue
        for i, l in enumerate(open(f, encoding='utf-8', errors='replace'), 1):
            for ref in re.findall(r'§(\d+(?:\.\d+)*)(\([a-z]\))?', l):
                if ref[0] not in nums:
                    drows.append((f, i, '§' + ref[0] + ref[1], 'unknown'))
tsv('audit_dangling_refs.tsv', ['file', 'line', 'reference', 'mapping'], drows)
verdict('A6', True, f'{len(drows)} code references to sections absent from the skeleton '
        f'(pre-existing; report only)', hard=False)

# ------------------------------------------------------------------ A7 facts
MH = 'include/tilemega/Codegen/tasks/ModelHarness.cuh'


def grep(pattern_path, pat):
    out = []
    for p in sorted(glob.glob(pattern_path, recursive=True)):
        if os.path.isfile(p):
            for i, l in enumerate(open(p, encoding='utf-8', errors='replace'), 1):
                if re.search(pat, l):
                    out.append(f'{p}:{i}')
    return out


def body(path, sig):
    lines = read(path).split('\n')
    for i, l in enumerate(lines):
        if re.search(sig, l):
            depth, started, out = 0, False, []
            for j in range(i, len(lines)):
                out.append((j + 1, lines[j]))
                depth += lines[j].count('{') - lines[j].count('}')
                started = started or '{' in lines[j]
                if started and depth <= 0:
                    return out
    return []


def pos(b, pat):
    for n, l in b:
        if re.search(pat, l):
            return n
    return None


facts = []


def fact(k, ok, ev):
    facts.append((k, '✅' if ok else '❌', ev))


e1 = grep(MH, r'task_owner\[stage\]\[task\]\s*=\s*physical_worker\[task\s*%\s*grid\]')
e1b = grep('include/tilemega/Codegen/tasks/Placement.cuh',
           r'TILEMEGA_PLACEMENT\s*==\s*0\s*\|\|\s*TILEMEGA_PLACEMENT\s*==\s*4')
fact('K1', bool(e1 and e1b), '; '.join(e1 + e1b))
e2 = grep('include/tilemega/Codegen/tasks/*.h', r'PlacedBlock\(\).*gridDim\.x')
fact('K2', any('GemmStageTaskBody' in x for x in e2) and any('GemmStageTaskBody' not in x for x in e2),
     '; '.join(e2[:6]))
e3 = grep(MH, r'for \(std::uint32_t stage : model\.stage_order\)') + grep(MH, r'owned\[worker\]\[stage\]')
fact('K3', len(e3) >= 2, '; '.join(e3))
e4w, e4s = grep(MH, r'placed_tasks\.worker'), grep(MH, r'placed_tasks\.slot')
fact('K4', bool(e4w) and not e4s, f'worker uses {"; ".join(e4w)}; slot reads {len(e4s)}')
tp_files = [f for d in ['lib', 'include', 'tools', 'test'] for f in glob.glob(f'{d}/**/*.*', recursive=True)
            if os.path.isfile(f) and re.search(r'TaskPlacement|BalanceTaskPlacement', read(f))]
e5 = [x for f in tp_files for x in grep(f, r'\.slot\b')]
reads5 = [x for x in e5 if not re.search(r'out\.slot(\.resize\(|\[)', line_of(x))]
fact('K5', bool(e5) and not reads5, f'hits {"; ".join(e5)}; non-write hits {reads5}')
e6 = grep('lib/Frontend/Frontend.cpp', r'getDenseI64ArrayAttr\(\{0\}\)') + \
    grep('lib/Frontend/Frontend.cpp', r'"cluster".*getI64IntegerAttr\(1\)')
e6s = grep('lib/Solver/*.cpp', r'PlacementOp|tilemega\.placement')
fact('K6', len(e6) >= 2 and not e6s, f'{"; ".join(e6)}; solver writes {len(e6s)}')
b7 = body('lib/Codegen/Codegen.cpp', r'BuildVariantSchedule\(')
e7 = grep('include/tilemega/Codegen/tasks/ModelRuntime.h', r'struct ScheduleStageDesc')
fact('K7', bool(pos(b7, r'ListScheduler')) and bool(e7),
     f'lib/Codegen/Codegen.cpp:{pos(b7, r"ListScheduler")}; {"; ".join(e7)}')
e8 = grep('lib/Solver/ListScheduler.cpp', r'level|height')
fact('K8', bool(e8), 'comparator lines ' + '; '.join(e8[:6]))
b9 = body(MH, r'tilemega_l2_kernel\(')
p9 = [pos(b9, r'WaitTaskDependencies\('), pos(b9, r'RunTask\('), pos(b9, r'NotifyTask\(')]
fact('K9', None not in p9 and p9 == sorted(p9), f'{MH} wait/run/notify at lines {p9}')
b10, b10a = body(MH, r'void NotifyTask\('), body(MH, r'void ArriveEvent\(')
p10 = [pos(b10, r'__threadfence'), pos(b10, r'__syncthreads'), pos(b10, r'ArriveEvent\(')]
fact('K10', None not in p10 and p10 == sorted(p10) and bool(pos(b10a, r'atomicAdd\(')),
     f'NotifyTask fence/barrier/arrive at {p10}; ArriveEvent atomicAdd at {pos(b10a, r"atomicAdd")}')
e11 = grep('include/tilemega/Codegen/tasks/EventSync.cuh', r'define TILEMEGA_EVENT_LOAD_POLL 0') + \
    grep('include/tilemega/Codegen/tasks/EventSync.cuh', r'atomicAdd\(event,\s*0ull\)')
fact('K11', len(e11) >= 2, '; '.join(e11))
bw = body(MH, r'void WaitTaskDependencies\(')
c12 = [sum('__syncthreads()' in l for _, l in b) for b in (bw, b9, b10)]
fact('K12', sum(c12) == 5, f'barriers in WaitTaskDependencies / L2 kernel / NotifyTask = {c12}')
e13 = grep(MH, r'seen\[worker\]') + grep(MH, r'per_group\s*==\s*1\s*&&\s*owner\s*==\s*worker')
fact('K13', len(e13) >= 2, '; '.join(e13))
e14 = grep('include/tilemega/Codegen/tasks/ModelRuntime.h', r'struct TaskTrace') + \
    grep(MH, r'atomicAdd\(.*trace_sequence')
fact('K14', len(e14) >= 2, '; '.join(e14))
e15 = grep('lib/Solver/ChainDP.cpp', r'l2_events') + grep('lib/Solver/ChainDP.cpp', r'\+\s*barrier')
fact('K15', len(e15) >= 2, '; '.join(e15))
e16 = grep('lib/Solver/CostModel.cpp', r'CostModel::EventNs|CostModel::Evaluate')
fact('K16', len(e16) >= 2, '; '.join(e16))
e17 = grep('lib/Solver/BalancedPlacement.cpp', r'affinity\[w\]\s*>\s*affinity\[chosen\]')
fact('K17', bool(e17), '; '.join(e17))
e18 = [x for d in ['lib', 'include', 'tools'] for x in grep(f'{d}/**/*.*', r'kLastTaskOfStage')]
r18 = [x for x in e18 if re.search(r'&\s*(\w+::)*kLastTaskOfStage\b', line_of(x)) and '|' not in line_of(x)]
fact('K18', bool(e18) and not r18, f'hits {"; ".join(e18)}; reads {r18}')
e19 = [x for d in ['lib', 'include', 'tools', 'test'] for x in grep(f'{d}/**/*.*', r'#include.*GeneratedLlamaRuntime')]
fact('K19', not e19, f'includes {e19}')
b20 = body('include/tilemega/Codegen/tasks/GemmStageTaskBody.h', r'RunLogicalTask\(|static.*RunTask\(')
fact('K20', bool(pos(b20, r'mainloop\(')) and bool(pos(b20, r'epilogue\(')),
     f'mainloop at {pos(b20, "mainloop[(]")}, epilogue at {pos(b20, "epilogue[(]")}')
e21 = grep('include/tilemega/Solver/RuntimeProjection.h', r'stage-major')
fact('K21', bool(e21), '; '.join(e21))
e22 = grep(MH, r'define TILEMEGA_EVENT_KAPPA 1')
fact('K22', bool(e22), '; '.join(e22))
la = read('docs/experiments/L2_ATTRIB/result.md')
vals = ['.372736', '.029760', '.058112', '.039456', '.011104', '.524288', '.031744', '.067584',
        '.039936', '.009248', '.739360', '.053248', '.102400', '.073952', '.015360', '1.140576',
        '.053312', '.171008', '.076960', '.026368']
miss23 = [v for v in vals if v not in la]
fact('K23', not miss23, f'all 20 medians found in docs/experiments/L2_ATTRIB/result.md; missing {miss23}')
e2e = read('docs/experiments/E2E_L2/result.md')
m24 = [v for v in ['1.081', '1.113', '0.694', '0.322', '1.171', '1.163'] if v not in e2e]
fact('K24', not m24, f'missing {m24}')
r5 = read('docs/experiments/PLACE/round5_balanced_result.md')
bf = base_text('docs/FINDINGS.md')
m25 = [v for v in ['1.639908', '3.087011', '1.396662', '4.002253', '212', '24576'] if v not in r5]
fact('K25', not m25 and '128 threads' in bf, f'missing {m25}; F-90 128 threads {"128 threads" in bf}')
m26 = [v for v in ['1.000000', '0.998552'] if v not in bf] + \
    [v for v in ['34.36', '40.18'] if v not in read('docs/experiments/OVERLAP/result.md')]
fact('K26', not m26, f'missing {m26}')
e27 = grep('include/tilemega/Codegen/tasks/TaskBase.h', r'TaskOwnership') + grep(MH, r'Ownership')
fact('K27', len(e27) >= 2, '; '.join(e27[:4]))
lastf = re.findall(r'^## F-(\d+)\b', bf, re.M)[-1]
fact('K28', lastf == '125', f'last base finding F-{lastf}')
tree = '\n'.join(old[1277:1343])
fact('K29', '├── tasks/' in tree and not Path('include/tilemega/tasks').exists()
     and Path('include/tilemega/Codegen/tasks').exists(),
     'base §6 tree lists include/tilemega/tasks/; actual include/tilemega/Codegen/tasks')
K = {'gqa2/4': (.372736, .029760, .058112, .039456, -.011104),
     'gqa2/128': (.524288, .031744, .067584, .039936, -.009248),
     'mha4/4': (.739360, .053248, .102400, .073952, -.015360),
     'mha4/128': (1.140576, .053312, .171008, .076960, -.026368)}
p1 = '; '.join(f'{k}: ceiling {100 * (b + abs(l)) / L1:.1f}% / ratio {(w + n) / b:.1f}x'
               for k, (L1, w, n, b, l) in K.items())
fact('P1', not miss23, p1)
tsv('facts.tsv', ['id', 'status', 'evidence'], facts)
bad = [f[0] for f in facts if f[1] != '✅']
verdict('A7', not bad, f'{len(facts)} facts; failing {bad}')

(HERE / 'audit_summary.txt').write_text(
    f'base\t{BASE}\nhead\t{git("rev-parse", "HEAD").strip()}\n' + '\n'.join(summary) + '\n',
    encoding='utf-8')
print('\n'.join(summary))
sys.exit(1 if hard_fail else 0)
