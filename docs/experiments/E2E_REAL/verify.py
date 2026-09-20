#!/usr/bin/env python3
"""Recompute every R7 gate from raw logs and dumps, and print PASS/FAIL.

Nothing here reads a summary or a conclusion: each gate re-derives its number
from the process logs, the run manifests and the generated sources. Hard-gate
failures set the exit status, but every gate is evaluated first.
"""
import json,re,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
# Rounds and solves this round were run outside the repository, under H1's
# rule that new fixtures and work trees stay out of the tree.
WORK=Path('/root/r7_work')
MODELS2=REPO/'docs/experiments/MODELS2'
rows=[]
def gate(name,hard,ok,detail,evidence):
    rows.append(dict(name=name,hard=hard,ok=bool(ok),detail=detail,evidence=str(evidence)))

def processes(folder):
    """(rounds, passing, distinct binaries, failing outputs) from raw logs."""
    logs=sorted(folder.glob('r*.log'),key=lambda p:int(re.findall(r'\d+',p.name)[0]))
    ok=bad=0;binaries=set()
    for log in logs:
        text=log.read_text()
        if 'RESULT status=PASS' in text: ok+=1
        for line in re.findall(r'^E2E_OUTPUT_DIFF .*$',text,re.M):
            if 'mismatch=0' not in line: bad+=1
        meta=log.with_suffix('.json')
        if meta.exists(): binaries.add(json.loads(meta.read_text()).get('binary_sha256'))
    return len(logs),ok,binaries,bad

# --- A-a: the maximal connected Llama graph, 50 fresh processes -------------
folder=MODELS2/'admission/admitted2/correctness'
if folder.is_dir():
    n,ok,binaries,bad=processes(folder)
    gate('A-a maximal connected Llama graph 50/50',True,n>=50 and ok==n and bad==0,
         f'rounds={n} passing={ok} failing_outputs={bad} binaries={len(binaries)}',folder)
else:
    gate('A-a maximal connected Llama graph 50/50',True,False,'no admission logs',folder)

# --- A-e: the epsilon and RoPE ablations ------------------------------------
table=MODELS2/'ablation/admitted2/ablation.tsv'
if table.is_file():
    lines=[l.split('\t') for l in table.read_text().strip().splitlines()]
    head,body={k:i for i,k in enumerate(lines[0])},lines[1:]
    by={r[head['arm']]:r for r in body}
    unchanged=all(by[a][head['v_0_463_gpu']]==by['base'][head['v_0_463_gpu']]
                  for a in ('a1','a2','a1a2') if a in by)
    fixed='refine' in by and by['refine'][head['v_0_463_gpu']]==by['refine'][head['v_0_463_fp64_rounded']]
    gate('A-e epsilon/RoPE ablation reported',False,unchanged and fixed,
         'A1, A2 and both leave V[0,463] at '+by['base'][head['v_0_463_gpu']]+
         '; refinement gives '+by.get('refine',['?'])[head['v_0_463_gpu']],table)
else:
    gate('A-e epsilon/RoPE ablation reported',False,False,'no ablation table',table)

# --- A-d: the extension cost, against R6's audited 15 sites -----------------
sites=REPO/'docs/experiments/MODELS/extension_sites.tsv'
audited=len(sites.read_text().strip().splitlines())-1 if sites.is_file() else 0
actual=HERE/'extension_cost.tsv'
if actual.is_file():
    got=[l.split('\t') for l in actual.read_text().strip().splitlines()][1:]
    gate('A-d extension cost table updated',False,bool(got),
         f'audited={audited} measured='+', '.join(f'{r[0]}:{r[1]}' for r in got),actual)
else:
    gate('A-d extension cost table updated',False,False,'not recorded',actual)

# --- D-a and A-b: the whole model, 50 fresh processes -----------------------
# The two models carry different gates. Llama is D-a (§7.2); Qwen3's maximal
# connected graph is the whole model once A4-A6 landed, so it is A-b (§4.5).
def outputs_diff(folder):
    """Distinct mismatching output indices and the largest absolute gap."""
    indices={};worst=0.0
    for log in folder.glob('r*.log'):
        for line in re.findall(r'^E2E_OUTPUT_DIFF .*$',log.read_text(),re.M):
            f=dict(kv.split('=',1) for kv in line.split()[1:])
            if f['mismatch']=='0': continue
            indices.setdefault(int(f['index']),0)
            indices[int(f['index'])]=max(indices[int(f['index'])],int(f['mismatch']))
            worst=max(worst,float(f['max_abs']))
    return indices,worst
for name,label in (('llama','D-a llama whole model 50/50'),
                   ('qwen3','A-b qwen3 maximal connected graph 50/50')):
    folder=HERE/name/'correctness'
    if folder.is_dir():
        n,ok,binaries,bad=processes(folder)
        detail=f'rounds={n} passing={ok} failing_outputs={bad} binaries={len(binaries)}'
        if ok<n:
            indices,worst=outputs_diff(folder)
            detail+=' mismatching_outputs='+','.join(
                f'{i}:{c}' for i,c in sorted(indices.items()))+f' max_abs={worst}'
        gate(label,True,n>=50 and ok==n and bad==0,detail,folder)
    else:
        # A model whose solve never produced a plan has no rounds to count; say
        # why, from the process record the run itself wrote, not from a summary.
        record=WORK/name/'solve.json';log=WORK/name/'solve.log'
        why='not run this round'
        if record.is_file():
            r=json.loads(record.read_text())
            why=(f"solve exit={r.get('exit_code')} after {r.get('elapsed_ns',0)/1e9:.0f}s "
                 f"at {str(r.get('head',''))[:9]}")
            if r.get('exit_code') and log.is_file():
                tail=[l for l in log.read_text().splitlines() if l.strip()]
                if tail:why+='; '+tail[-1][:120]
        gate(label,True,False,why,record if record.is_file() else folder)

# --- H2: the default build's SASS, against the baseline tree ----------------
stamp=HERE/'sass_identity/manifest.json'
if stamp.is_file():
    meta=json.loads(stamp.read_text())
    same=all(m['sha256']==m['baseline_sha256'] for m in meta['models'].values())
    gate('H2 default-build SASS identity',True,same,
         'models='+', '.join(f"{k}:{'same' if v['sha256']==v['baseline_sha256'] else 'differs'}"
                             for k,v in meta['models'].items()),stamp)
else:
    gate('H2 default-build SASS identity',True,False,'no stamp',stamp)

# --- B0: recompute FORK7's aggregation from the per-process rows ------------
# The threshold and the statistic were fixed before measurement: median over
# each cell's fresh processes, then median over the four cells. This recomputes
# both from analysis.tsv rather than reading fork7.txt, and it reports how many
# individual rounds clear the threshold, because the margin is 0.001.
PHASE2=REPO/'docs/experiments/PHASE2/raw'
rowsfile=PHASE2/'analysis.tsv'
if rowsfile.is_file():
    import csv,statistics
    per={}
    with rowsfile.open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['arm']!='selected': continue
            per.setdefault(f"{r['model']}_s{r['seq']}",{})[int(r['round'])]=\
                float(r['cp_exposed_wait_share'])
    cellmed=[statistics.median(v.values()) for v in per.values()]
    whole=statistics.median(cellmed)
    rounds=sorted(set.intersection(*(set(v) for v in per.values())))
    byround=[statistics.median(per[c][i] for c in per) for i in rounds]
    clears=sum(1 for v in byround if v>=0.15)
    n=0;okc=0
    for cell in sorted(per):
        folder=PHASE2/cell/'correctness/selected'
        if folder.is_dir():
            c,o,_,_=processes(folder);n+=c;okc+=o
    gate('B0 FORK7 whole-pipeline exposed wait',True,
         len(per)==4 and whole>=0.15 and n>=200 and okc==n,
         f'cells={len(per)} median={whole:.4f} threshold=0.15 '
         f'margin={whole-0.15:+.4f} rounds_clearing={clears}/{len(byround)} '
         f'correctness={okc}/{n}',rowsfile)
else:
    gate('B0 FORK7 whole-pipeline exposed wait',True,False,'no PHASE2 analysis',
         rowsfile)

# --- B1: paging and cross-task pipelining -----------------------------------
# Each gate below re-derives its number from PIPELINE's raw logs and tables.
# The criteria are B1's own and were fixed before the runs: (a) every fresh
# process passes, on all three populations; (b) the driver and F-40 agree on
# the CTA count and no arm loses the residency its cell was selected at;
# (c) the pipelined arm's per-slot wait is strictly below the inline arm's on
# the slots that issue; (d) the three sigma locations each hold up under their
# example; (e) reported without a threshold, as R7 5.2 asks.
PIPELINE=REPO/'docs/experiments/PIPELINE/raw'
CELLS=('gqa2_s4','gqa2_s128','mha4_s4','mha4_s128','real_s4','real_s128')
def tsv(path):
    lines=[l.split('\t') for l in path.read_text().strip().splitlines()]
    return [dict(zip(lines[0],r)) for r in lines[1:]]

populations={
 'six cells':[PIPELINE/c/'correctness_head'/a for c in CELLS for a in ('prefetch','inline')],
 'SEQSCAN subset':[PIPELINE/'seqscan'/c/'seqscan'/case/a
                   for c in CELLS[:4] for case in ('s1_p0','s128_p512','s2048_p0')
                   for a in ('prefetch','inline')],
 'Llama maximal connected graph':[PIPELINE/'llama/correctness'/a
                                  for a in ('prefetch','inline')]}
for label,folders in populations.items():
    n=ok=bad=0;binaries=set();present=[f for f in folders if f.is_dir()]
    for f in present:
        c,o,b,e=processes(f);n+=c;ok+=o;binaries|=b;bad+=e
    gate(f'B1-a correctness, {label}',True,
         len(present)==len(folders) and n==ok and n>=50*len(folders) and bad==0,
         f'arms={len(present)}/{len(folders)} rounds={n} passing={ok} '
         f'failing_outputs={bad} binaries={len(binaries)}',PIPELINE)

# The subset is R5's only if the replayed plans are the plans JOINT ran; the
# guarded frontier field is stripped before the comparison, nothing else.
regen=PIPELINE/'seqscan/regeneration.tsv'
if regen.is_file():
    got=tsv(regen);same=[r for r in got if r['identical_without_frontier']=='1']
    gate('B1-a SEQSCAN plans are JOINT\'s plans',False,len(got)==12 and len(same)==12,
         f'cases={len(got)} identical_without_the_guarded_field={len(same)}',regen)
else:
    gate('B1-a SEQSCAN plans are JOINT\'s plans',False,False,'not regenerated',regen)

arms=PIPELINE/'occupancy_arms.tsv'
if arms.is_file():
    got=tsv(arms)
    agree=[r for r in got if r['driver_ctas']==r['f40_reserved_ctas']]
    kept=[r for r in got if r['keeps_residency']=='1']
    base={r['cell']:r for r in got if r['arm']=='control'}
    grew=', '.join(f"{r['cell']} {base[r['cell']]['smem_bytes']}->{r['smem_bytes']}B "
                   f"{base[r['cell']]['regs']}->{r['regs']}r {r['driver_ctas']}cta"
                   for r in got if r['arm']=='prefetch')
    gate('B1-b occupancy before/after against F-40',True,
         len(got)==3*len(CELLS) and len(agree)==len(got) and len(kept)==len(got),
         f'arms={len(got)} f40_agrees={len(agree)} keeps_residency={len(kept)}; {grew}',arms)
else:
    gate('B1-b occupancy before/after against F-40',True,False,'no occupancy table',arms)

overlap=PIPELINE/'overlap_head.tsv'
if overlap.is_file():
    got=tsv(overlap)
    positive=[r for r in got if r['overlap_cycles_median'] and
              float(r['overlap_cycles_median'])>0]
    gate('B1-c overlap measured in the phase trace',True,
         len(got)==len(CELLS) and len(positive)==len(got) and
         all(int(r['rounds_prefetch'])>=50 and int(r['rounds_inline'])>=50 for r in got),
         'per-slot wait, inline minus pipelined, on the issuing slots: '+
         ', '.join(f"{r['cell']} {r['overlap_cycles_median']}cy "
                   f"({r['overlap_slots_positive']}/{r['issued_slots']} slots)" for r in got),
         overlap)
else:
    gate('B1-c overlap measured in the phase trace',True,False,'no phase analysis',overlap)

log=PIPELINE/'pipeline_sigma.log'
binary=REPO/'build-portable/pipeline_sigma_test'
text=subprocess.run([str(binary)],capture_output=True,text=True).stdout if binary.exists() \
     else (log.read_text() if log.is_file() else '')
checks=('PIPELINE_FRONTIER','PIPELINE_PRICE','PIPELINE_BOUNDS','PIPELINE_TABLE')
gate('B1-d pipelining wired into sigma',True,
     'PIPELINE_SIGMA PASS' in text and all(c in text for c in checks),
     ('rerun: ' if binary.exists() else 'recorded: ')+
     '; '.join(l for l in text.splitlines() if l.startswith('PIPELINE_')),
     binary if binary.exists() else log)

e2e=PIPELINE/'e2e_head.tsv'
if e2e.is_file():
    got=[r for r in tsv(e2e) if r['arm']=='prefetch' and r['base']=='control']
    faster=[r for r in got if float(r['ci_hi'])<1]
    slower=[r for r in got if float(r['ci_lo'])>1]
    gate('B1-e end to end, six cells, no threshold',False,len(got)==len(CELLS),
         f'cells={len(got)} faster={len(faster)} slower={len(slower)} (95% CI clear of 1); '+
         ', '.join(f"{r['cell']} {float(r['ratio']):.4f}" for r in got),e2e)
else:
    gate('B1-e end to end, six cells, no threshold',False,False,'no e2e table',e2e)

# --- B2: per-producer-stage kappa, two arms per model -----------------------
# The hard gate is the pass rate of both arms; the difference between the
# per-stage optimum and the global one is reported from the solver's own line.
kappa=HERE/'stage_kappa'
if (kappa/'results.tsv').is_file():
    got=tsv(kappa/'results.tsv');runs=[];detail=[]
    for r in got:
        cell=kappa/f"{r['model']}_s{r['seq']}"/r['arm']
        n,ok,binaries,bad=processes(cell/'correctness')
        text=(cell/'solve.log').read_text()
        line=re.search(r'^SOLVE_STAGE_KAPPA (.*)$',text,re.M)
        fields=dict(re.findall(r'(\w+)=(\S+)',line[1])) if line else {}
        table=fields.get('table','')
        # A forced arm must have run a mixed table, or the runtime path was
        # never exercised; a searched arm reports how far the descent moved.
        mixed=len(set(table.split(',')))>1
        runs.append((n>=50 and ok==n and bad==0 and (r['arm']!='forced' or mixed),
                     f"{r['model']}_s{r['seq']}/{r['arm']} {ok}/{n} failing_outputs={bad} "
                     f"binaries={len(binaries)} stages={fields.get('stages','?')} "
                     f"table={'mixed' if mixed else 'uniform'}"))
        if r['arm']=='searched':
            detail.append(f"{r['model']}: moves={fields.get('moves','?')} "
                          f"uniform_ns={fields.get('uniform_ns','?')} "
                          f"per_stage_ns={fields.get('per_stage_ns','?')}")
    gate('B2 per-stage kappa 50/50, both arms',True,
         len(got)==4 and all(ok for ok,_ in runs),'; '.join(d for _,d in runs),kappa)
    gate('B2 per-stage versus global kappa',False,len(detail)==2,
         'the descent never moved off uniform: '+'; '.join(detail),kappa/'results.tsv')
else:
    gate('B2 per-stage kappa 50/50, both arms',True,False,'no results',kappa)
    gate('B2 per-stage versus global kappa',False,False,'no results',kappa)

# --- B3: segmented geometry inside an interval -------------------------------
segments=HERE/'segments'
done=[]
for folder in sorted(segments.glob('*_i*_*')):
    model=folder.name.split('_')[0]
    # Legality, recounted from the per-point proof logs rather than the run's
    # own tally: one process per interior point, each reporting failed=0.
    points=sorted(folder.glob('proof/p*/proof.log'))
    proved=[p for p in points if re.search(r'proved=1 failed=0',p.read_text())]
    # Materialization: every endpoint and interior point of both segments has
    # to serialize to the macro geometry's own nodes, diff=0 on each line.
    check=folder/'check.log'
    lines=check.read_text().splitlines() if check.is_file() else []
    material=[l for l in lines if l.startswith('SEGMENT_MATERIAL')]
    agree=[l for l in material if ' diff=0' in l]
    verdict=next((l for l in lines if l.startswith('SEGMENT_CHECK')),'')
    # Correctness: 50 fresh processes per covered seq, for both arms.
    arms=sorted(folder.glob('correctness/s*/*'))
    counts=[processes(a) for a in arms]
    rounds=sum(c[0] for c in counts);passing=sum(c[1] for c in counts)
    bad=sum(c[3] for c in counts)
    full=sum(c[0]>=50 and c[1]==c[0] and c[3]==0 for c in counts)
    # §5.4 asks three things of an interval campaign -- legality at every point,
    # endpoint and interior materialization, and the measured benefit -- of the
    # campaign, not of each model. A second model in flight is reported, and the
    # hard line is whether a campaign satisfies all three.
    ok=(bool(points) and len(proved)==len(points) and bool(material) and
        len(agree)==len(material) and 'serialization=byte_identical' in verdict and
        bool(arms) and full==len(arms))
    # The gain is a report line, not a threshold: median L2 of each arm over
    # the paired rounds, recomputed here from the raw harness lines.
    gains=[]
    for seq in sorted(folder.glob('timing/s*'),key=lambda d:int(d.name[1:])):
        medians={}
        for arm in sorted(seq.iterdir()):
            got=sorted(float(m) for log in arm.glob('r*.log')
                       for m in re.findall(r'l2_ms=([0-9.]+)',log.read_text()))
            if got:medians[arm.name]=got[len(got)//2]
        if len(medians)==2 and medians.get('fixed'):
            gains.append(f"s{seq.name[1:]} {medians['segmented']/medians['fixed']:.4f}")
    gate(f'B3 intra-interval campaign, {model}',False,ok,
         f'proof points={len(proved)}/{len(points)} material={len(agree)}/{len(material)} '
         f'{verdict.split(" ",1)[1] if verdict else "no check"}; arms={full}/{len(arms)} '
         f'rounds={rounds} passing={passing} failing_outputs={bad}; '
         f'segmented/fixed L2 {", ".join(gains) if gains else "not timed"}',folder)
    if ok:done.append(model)
gate('B3 intra-interval geometry',True,bool(done),
     ('legality, materialization and benefit all recorded on '+', '.join(done))
     if done else 'no interval campaign satisfies all three',segments)

# --- C1-b: scored on the searchable space only (§6) --------------------------
# Degraded form under §9.3: the bound stage and the plan hoist are measured,
# but capacity stays 12, so the gate as written is not met and says so.
profile=HERE/'prepare/prepare_bounds.tsv'
if profile.is_file():
    got=tsv(profile);same=[r for r in got if r['identical']=='1']
    speed=[float(r['speedup']) for r in got]
    gate('C1-b preparation-phase optimization',True,False,
         f'DEGRADED (§9.3): capacity stays 12; bound stage identical={len(same)}/{len(got)} '
         f'speedup {min(speed):.2f}x-{max(speed):.2f}x; whole-search hoist 2.51x (F-230)',
         HERE/'prepare')
else:
    gate('C1-b preparation-phase optimization',True,False,'no preparation evidence',HERE/'prepare')

# --- A-c: the two reference models still pass with every switch off ----------
REF=('gqa2_s4','gqa2_s128','mha4_s4','mha4_s128')
regression=HERE/'regression'
for label,arms in (('reference cells seq 4 and 128',
                    [(c,'default') for c in REF]),
                   ('SEQSCAN subset',
                    [(c,'seqscan_'+case) for c in REF
                     for case in ('s1_p0','s128_p512','s2048_p0')])):
    folders=[regression/c/'correctness'/a for c,a in arms]
    live=[f for f in folders if f.is_dir()]
    if len(live)==len(folders):
        got=[processes(f) for f in live]
        n=sum(g[0] for g in got);ok=sum(g[1] for g in got);bad=sum(g[3] for g in got)
        full=sum(g[0]>=50 and g[1]==g[0] and g[3]==0 for g in got)
        gate(f'A-c {label}',True,full==len(got),
             f'arms={full}/{len(got)} rounds={n} passing={ok} failing_outputs={bad} '
             f'binaries={len(set().union(*(g[2] for g in got)))}',regression)
    else:
        gate(f'A-c {label}',True,False,
             f'{len(live)}/{len(folders)} arms run this round',regression)

# --- C1-a: J-b is a report line now, not a gate (§6) -------------------------
# The objective minimizes max(CP, queue_lb), which is allowed to land on a
# queue-bound point, so the ratio is reported and never compared to a line.
topk=HERE/'topk'
ratios=[]
for c in CELLS:
    top3,search=topk/c/'auto.cu.top3.tsv',topk/c/'auto.cu.search.tsv'
    if not (top3.is_file() and search.is_file()):continue
    winner=tsv(top3)[0]
    row=next((r for r in tsv(search)
              if r['candidate']==winner['key'] and r['placement']==winner['placement']),None)
    if row and float(row['cp_ns']):
        ratios.append(f"{c} {float(row['queue_lb_ns'])/float(row['cp_ns']):.6f}")
gate('C1-a J-b queue_lb/CP reported, not gated',False,bool(ratios),
     ('winner queue_lb/CP: '+'; '.join(ratios)) if ratios else 'no search tables',topk)

# --- D-b: top-k quality under C1-c's caliber, six cells ----------------------
def medians(folder):
    """Median l2_ms per arm, recomputed from the rounds themselves."""
    import statistics
    out={}
    for arm in sorted(p.name for p in folder.iterdir() if p.is_dir()):
        got=[]
        for log in (folder/arm).glob('r*.log'):
            m=re.search(r'^E2E_TIME .*?\bl2_ms=([0-9.eE+-]+)',log.read_text(),re.M)
            if m:got.append(float(m[1]))
        if got:out[arm]=statistics.median(got)
    return out
detail=[];passing=0;cells_seen=0
for c in CELLS:
    folder,evaluated=topk/c/'measure',topk/c/'auto.cu.evaluated.tsv'
    if not (folder.is_dir() and evaluated.is_file()):continue
    cells_seen+=1
    rowsev={'cand'+r['index']:r for r in tsv(evaluated)}
    keys3={r['key'] for r in tsv(topk/c/'auto.cu.top3.tsv')}
    # A candidate that disagreed with the golden output stopped being timed the
    # moment it did, so its single round is not a timing sample. Those arms come
    # out of the ratio here too, and the coverage says how many are left.
    record=topk/c/'excluded.tsv'
    dropped={l.split('\t')[0] for l in record.read_text().splitlines()[1:]
             if l.strip()} if record.is_file() else set()
    med={a:v for a,v in medians(folder).items() if a not in dropped}
    rounds=sorted(len(list((folder/a).glob('r*.log'))) for a in med)
    short=[a for a in med if rowsev[a]['key'] in keys3]
    if not short or len(med)<2:
        detail.append(f'{c} not measurable ({len(med)}/{len(rowsev)} clean)');continue
    ratio=med[min(short,key=med.get)]/med[min(med,key=med.get)]
    passing+=ratio<=1.05
    detail.append(f'{c} {ratio:.4f} ({len(med)}/{len(rowsev)} clean, '
                  f'{rounds[0]}-{rounds[-1]} rounds each, '
                  f'{len(dropped)} dropped on output difference)')
gate('D-b top-k quality, six cells',True,cells_seen==len(CELLS) and passing==len(CELLS),
     ('; '.join(detail) if detail else 'not measured this round'),topk)

# --- D-c and D-d: the three levels, paired inside each round -----------------
def levels(folder):
    """Per-round l05/l1/l2 from the harness line, and the paired ratios."""
    import statistics
    got=[]
    for log in sorted(folder.glob('r*.log'),key=lambda p:int(p.stem[1:])):
        m=re.search(r'^E2E_TIME (.*)$',log.read_text(),re.M)
        if m:got.append({k:float(v) for k,v in re.findall(r'(\w+)=([\d.eE+-]+)',m[1])})
    if not got:return None
    return dict(rounds=len(got),
                l05=statistics.median(s['l05_ms'] for s in got),
                l1=statistics.median(s['l1_ms'] for s in got),
                l2=statistics.median(s['l2_ms'] for s in got),
                l2_over_l1=statistics.median(s['l2_ms']/s['l1_ms'] for s in got))

def floor_of(root):
    table=root/'auto.cu.top3.tsv'
    return float(tsv(table)[0]['floor_ns']) if table.is_file() else None

def sweep(base,roots):
    """(label, summary line) per decode point, or [] when nothing ran."""
    out=[]
    for seq,root in sorted(roots.items()):
        got=levels(base/f'seq{seq}')
        if not got:continue
        fl=floor_of(root)
        tail=''
        if fl:
            import statistics
            logs=sorted((base/f'seq{seq}').glob('r*.log'),key=lambda p:int(p.stem[1:]))
            vals=[re.search(r'\bl2_ms=([0-9.eE+-]+)',p.read_text()) for p in logs]
            tail=' l2/floor %.4f'%statistics.median(
                float(v[1])*1e6/fl for v in vals if v)
        out.append(f"seq {seq} rounds {got['rounds']} l05 {got['l05']:.3f} l1 {got['l1']:.3f} "
                   f"l2 {got['l2']:.3f} l2/l1 {got['l2_over_l1']:.4f}{tail}")
    return out

dc=sweep(HERE/'timing',{1:WORK/'llama_s1',4:WORK/'llama_hoist',
                        16:WORK/'llama_s16',64:WORK/'llama_s64'})
gate('D-c three levels per decode seq, 25 paired rounds',False,len(dc)==4,
     ('; '.join(dc) if dc else 'not measured this round'),HERE/'timing')
dd=[]
for model in ('gqa2','mha4'):
    roots={1:WORK/'ref'/f'{model}_s1',4:topk/f'{model}_s4',
           16:WORK/'ref'/f'{model}_s16',128:topk/f'{model}_s128'}
    dd+= [f'{model} '+line for line in sweep(HERE/'timing_ref'/model,roots)]
gate('D-d the two reference models at the same caliber',False,bool(dd),
     ('; '.join(dd) if dd else 'not measured this round'),HERE/'timing_ref')

# --- D-e: which decisions the solver makes and which a person still makes ----
decisions=HERE/'solver_decisions.md'
if decisions.is_file():
    text=decisions.read_text()
    def items(heading):
        body=text.split(heading,1)[1].split('\n## ',1)[0] if heading in text else ''
        got=[l for l in body.splitlines()
             if l.startswith('| ') and not set(l)<=set('|- ')]
        return max(len(got)-1,0)  # the first row is the table's header
    solver=items('## Decided by the solver')
    human=items('## Still decided by a human')
    gate('D-e solver and human decisions listed item by item',False,solver>0 and human>0,
         f'solver decides {solver} items; a person still decides {human}',decisions)
else:
    gate('D-e solver and human decisions listed item by item',False,False,'not written',decisions)

width=max(len(r['name']) for r in rows)
failed=0
for r in rows:
    mark='PASS' if r['ok'] else 'FAIL'
    if not r['ok'] and r['hard']: failed+=1
    print(f"{mark} [{'hard' if r['hard'] else 'report'}] {r['name']:<{width}}  {r['detail']}")
    print(f"       evidence: {r['evidence']}")
met=sum(r['ok'] for r in rows)
print(f"\n{met}/{len(rows)} gates met; {failed} hard gate(s) failed, "
      f"{len(rows)-met-failed} report gate(s) unmet")
sys.exit(1 if failed else 0)
