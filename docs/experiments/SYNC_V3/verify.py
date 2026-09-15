#!/usr/bin/env python3
"""R4 stop-state audit. No verdict is read from an experiment summary.

Implemented checks cover the recovered task DAG, fresh prerequisite litmus,
fresh PTX and default SASS. Unperformed mechanisms remain hard failures; this
is not an acceptance implementation for C2/C3/D that have not been built.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'docs/experiments/TRACE_V2'))
import analyze
import rebuild_r4

results = []


def gate(name, kind, ok, detail, evidence):
    results.append((name,kind,bool(ok),detail,str(evidence)))
    print(f'{name}\t{kind}\t{"PASS" if ok else "FAIL"}\t{detail}\n  evidence: {evidence}',flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sass_check():
    out = HERE/'sass_identity'
    try:
        m = json.loads((out/'manifest.json').read_text())
        if m['base'] != 'ee905036d2ec0c9dc880df604097981552423e53':
            raise ValueError('SASS baseline is not the R4 baseline')
        live = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        fresh = live == m['tested_head']
        if not fresh:
            parent = subprocess.check_output(['git','rev-parse','HEAD^'],cwd=REPO,text=True).strip()
            changed = subprocess.check_output(['git','diff','--name-only','HEAD^','HEAD'],cwd=REPO,text=True).splitlines()
            fresh = parent == m['tested_head'] and all(p.startswith('docs/experiments/SYNC_V3/sass_identity/') for p in changed)
        for p, digest in m['inputs'].items():
            if sha(REPO/p) != digest:
                raise ValueError(f'compiled input changed: {p}')
        sizes = []
        for model in ('gqa2','mha4'):
            base, tested = out/f'{model}_base.sass', out/f'{model}_head.sass'
            if not base.is_file() or not tested.is_file() or not base.read_bytes():
                raise ValueError('missing SASS')
            if base.read_bytes() != tested.read_bytes() or (out/f'{model}.diff').read_bytes():
                raise ValueError(f'{model}: nonempty SASS diff')
            if sha(tested) != m['models'][model]['head_sha256']:
                raise ValueError(f'{model}: artifact digest mismatch')
            if any(n not in tested.read_text() for n in ('tilemega_l1_kernel','tilemega_l2_kernel')):
                raise ValueError('missing kernel')
            sizes.append(f'{model}={tested.stat().st_size} bytes')
        gate('H2','hard',fresh,'; '.join(sizes)+f'; tested={m["tested_head"]} live={live}',out)
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as e:
        gate('H2','hard',False,str(e),out)


def litmus_counts(raw):
    counts = {}
    for path in sorted((raw/'log').glob('*_a*_g*_t*.txt')):
        m = re.fullmatch(r'(per_writer|thread0_fence|no_barrier|no_fence)_a([01])_g(\d+)_t(\d+)\.txt',path.name)
        if not m:
            continue
        arm, acquire, grid, tile = m.groups()
        statuses = re.findall(r'^RESULT status=(\S+)',path.read_text(),re.M)
        counts[(arm,int(acquire),int(grid),int(tile))] = dict(
            runs=len(statuses),pass_count=statuses.count('pass'),
            mismatch=statuses.count('MISMATCH'),other=sum(x not in ('pass','MISMATCH') for x in statuses))
    return counts


def check_litmus():
    raw = HERE/'litmus_recheck'
    counts = litmus_counts(raw)
    complete = True
    for arm in ('per_writer','thread0_fence','no_barrier','no_fence'):
        for acquire in (0,1):
            for grid in (64,128,256):
                for tile in (1024,4096,16384):
                    r=counts.get((arm,acquire,grid,tile),{})
                    complete &= r.get('runs',0)==50 and r.get('other',1)==0
                    if arm in ('per_writer','thread0_fence'):
                        complete &= r.get('pass_count',0)==50
    passed=0
    detail=[]
    for grid in (64,128,256):
        for tile in (1024,4096):
            def get(arm,key):
                return counts.get((arm,0,grid,tile),{}).get(key,0)
            a,b=get('per_writer','pass_count'),get('thread0_fence','pass_count')
            nf,nb=get('no_fence','mismatch'),get('no_barrier','mismatch')
            ok=a==b==50 and nf>0 and nb>0
            passed+=ok
            detail.append(f'g{grid}/t{tile}: writer={a}/50 candidate={b}/50 no_fence={nf}/50 no_barrier={nb}/50')
    gate('C1-litmus','hard',complete and passed==6,
         f'raw_cells={len(counts)}/72 sensitive_cells={passed}/6; '+'; '.join(detail),raw/'log')


def check_premise():
    p=HERE/'premise_audit'
    try:
        meta=json.loads((p/'compile.json').read_text())
        text=(p/'gqa2_b.ptx').read_text()
        if meta['exit_code'] or sha(p/'gqa2_b.ptx')!=meta['outputs']['gqa2_b.ptx']:
            raise ValueError('PTX compilation/evidence failed')
        text=text[text.index('.visible .entry _ZN8tilemega7codegen18tilemega_l2_kernel'):]
        text=text[:text.index('.visible .entry',1)]
        publish=text[text.rindex('membar.gl;'):]
        ops=re.findall(r'^\s*((?:atom|red)\.[^;]+;)',publish,re.M)
        ok=len(ops)==2 and all('.release.' in op or '.acq_rel.' in op for op in ops)
        gate('P1-release-semantics','stop',ok,'; '.join(op.strip() for op in ops),p/'gqa2_b.ptx')
    except (OSError,KeyError,ValueError) as e:
        gate('P1-release-semantics','stop',False,str(e),p)


def main():
    sass_check()
    analyzed=[]
    try:
        for item in rebuild_r4.inventory():
            r=analyze.analyze(REPO/item['dump'],REPO/item['source'],item['window'])
            r.update(item);analyzed.append(r)
        primary=[r for r in analyzed if r['suite']=='placement']
        a=sum(r['measured_l2_ms']*1e6>=max(r['cp_corrected_ns'],r['queue_lb_ns']) for r in primary)
        b=sum(min(r['cp_split_'+n+'_ns'] for n in ('task','wait','prerun_barrier','publish','gap'))>=0
              and abs(sum(r['cp_split_'+n+'_ns'] for n in ('task','wait','prerun_barrier','publish','gap'))-r['cp_reconstructed_ns'])<=.01*r['cp_reconstructed_ns']
              and r['cp_split_wait_ns']<=r['kernel_span_ns'] for r in primary)
        gate('A-a','hard',a==32,f'{a}/32 floors <= measured',REPO/'docs/experiments/PLACE_EFT2/raw/final')
        gate('A-b','hard',b==32,f'{b}/32 split closes, nonnegative, wait <= span',REPO/'docs/experiments/PLACE_EFT2/raw/final')
        pair=[r for r in primary if r['config']=='a' and r['model']=='gqa2' and r['seq']=='4']
        c=len(pair)==2 and len({(r['cp_corrected_nodes'],r['cp_corrected_task_ns']) for r in pair})==1
        gate('A-c','hard',c,'; '.join(f'{r["candidate"]}: {r["cp_corrected_nodes"]} nodes {r["cp_corrected_task_ns"]} ns' for r in pair),REPO/'docs/experiments/PLACE_EFT2/raw/final')
        detail='; '.join(f'{r["model"]}/s{r["seq"]}/{r["config"]}={r["cp_corrected_ns"]} ns' for r in primary if r['candidate']=='rotate')
        gate('A-d','report',True,detail,REPO/'docs/experiments/PLACE_EFT2/raw/final')
        windows=[r for r in analyzed if r['suite']=='window']
        gate('A-window','report',len(windows)==12,'; '.join(f'{r["model"]}/s{r["seq"]}/{r["config"]}: cp={r["cp_corrected_ns"]} hol={r["hol_reclaimable_ns"]}' for r in windows),REPO/'docs/experiments/WINDOW/raw/final')
    except (OSError,ValueError,KeyError) as e:
        for name in ('A-a','A-b','A-c'):
            gate(name,'hard',False,str(e),REPO/'docs/experiments/TRACE_V2/analyze.py')
    check_premise()
    check_litmus()
    # Record the actual violating commit, including its parent, instead of
    # laundering a revert into a claim that the original H4 order held.
    early=subprocess.check_output(['git','show','-s','--format=%H %P','66ae000e'],cwd=REPO,text=True).strip()
    gate('B-c/H4','hard',False,'C1 was committed without B measurements: '+early, 'git show 66ae000e')
    hard_pending={'B-a':'docs/experiments/FENCE', 'C1-correctness':'docs/experiments/SYNC_V3',
                  'C2-deadlock':'docs/experiments/SYNC_V3','C2-correctness':'docs/experiments/SYNC_V3',
                  'C3a-correctness':'docs/experiments/SYNC_V3','C3b-degeneracy':'docs/experiments/SYNC_V3',
                  'D-a':'docs/experiments/CHAIN2','D-b':'docs/experiments/CHAIN2'}
    for name,path in hard_pending.items():
        gate(name,'hard',False,'NOT RUN: stopped on the contradicted R4 premise; no acceptance evidence',REPO/path)
    for name in ('B-b','C-five-arm/MEMBAR','C3a-window-gain','D-c','D-d','E-targets','ablation'):
        gate(name,'report',False,'NOT COMPLETE at the prerequisite stop',HERE)
    gate('wait+notify<=barrier','research',False,'NOT MEASURED this round; no 25-round protocol matrix',HERE)
    hard=[r for r in results if r[1]=='hard']
    print(f'\n{sum(r[2] for r in hard)}/{len(hard)} hard gates pass; protocol work stopped',flush=True)
    return int(any(not r[2] for r in results if r[1] in ('hard','stop')))


if __name__=='__main__':
    raise SystemExit(main())
