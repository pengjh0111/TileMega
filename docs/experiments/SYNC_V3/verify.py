#!/usr/bin/env python3
"""Recompute all R4 gates from raw process logs, traces and instruction dumps."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO/'docs/experiments/TRACE_V2'))
import analyze
import rebuild_r4
import audit_raw
import metrics
import trace_dependency
import verify_window_probe

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
        gate('H2','hard',fresh,'; '.join(sizes)+'; input digests match; tested HEAD or artifact-only direct child='+str(fresh),out)
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


def checked(name, kind, function, evidence):
    try:
        detail=function()
        gate(name,kind,True,detail,evidence)
    except (OSError,KeyError,ValueError,ArithmeticError,StopIteration,AssertionError,subprocess.CalledProcessError) as e:
        gate(name,kind,False,str(e),evidence)


def check_litmus():
    checked('C1-litmus','hard',audit_raw.litmus,HERE/'litmus_v3/scan')


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


def cluster_degeneracy():
    out=HERE/'sass_identity'
    manifest=json.loads((out/'manifest.json').read_text())
    for m in ('gqa2','mha4'):
        head,disabled=out/f'{m}_head.sass',out/f'{m}_cluster_false.sass'
        if head.read_bytes()!=disabled.read_bytes() or (out/f'{m}_cluster_false.diff').read_bytes():
            raise ValueError(f'{m}: cluster=false changes SASS')
        if sha(disabled)!=manifest['models'][m]['cluster_false_sha256']:
            raise ValueError(f'{m}: stale fallback digest')
    return '2/2 complete sm_89 SASS files identical with cluster-scope switch enabled; freshness checked by H2'


def cluster_compile():
    out=HERE/'cluster_compile'
    for p,digest in json.loads((out/'inputs.json').read_text()).items():
        if sha(REPO/p)!=digest:raise ValueError(f'compile input changed: {p}')
    op='atom.acq_rel.cluster.shared::cluster.add.u64'
    if op not in (out/'probe_sm_120.ptx').read_text() or op in (out/'probe_sm_89.ptx').read_text():
        raise ValueError('cluster scope PTX does not match target capability')
    if any(c['exit_code'] for c in json.loads((out/'commands.json').read_text())):
        raise ValueError('cluster compilation failed')
    return 'sm_89 and sm_120 compile; cluster atomic present only at sm_120; sm_120 execution NOT RUN'


def membar():
    details=[]
    for m in ('gqa2','mha4'):
        counts=[]
        for name,raw in (('baseline',REPO/'docs/experiments/FENCE/raw'),('c1',HERE/'c1')):
            text=(raw/'sass'/f'{m}_p0_full.sass').read_text()
            chunk=next(c for c in text.split('Function : ')[1:] if 'tilemega_l2_kernel' in c.splitlines()[0])
            counts.append(len(re.findall(r'\bMEMBAR\.SC\.GPU',chunk)))
            if not (raw/'sass'/f'{m}_p0_full.report.txt').is_file():raise ValueError('missing sass_report evidence')
        if counts!=[2,2]:raise ValueError(f'{m}: unexpected static MEMBAR counts {counts}')
        details.append(f'{m}: static L2 MEMBAR.SC.GPU 2 -> 2; publishing release executes on 128 -> 1 threads/task')
    source=(REPO/'include/tilemega/Codegen/tasks/ModelHarness.cuh').read_text()
    if '#if TILEMEGA_RELEASE_AFTER_BARRIER && !TILEMEGA_UNSAFE_NO_NOTIFY_FENCE' not in source:
        raise ValueError('single-thread conditional release is missing')
    return '; '.join(details)+'; prompt static-count expectation corrected: predication changes dynamic participation, not static instruction sites'


def chain_comparison():
    import random
    import statistics
    rng=random.Random(167)
    raw=REPO/'docs/experiments/CHAIN2/final'
    detail=audit_raw.chain_pairs(raw)
    values=[]
    for m in ('gqa2','mha4','real'):
        for seq in (4,128):
            cell=raw/'paired'/f'{m}_s{seq}'
            reference=[metrics.timing(cell/'rotate'/f'r{i}.log')['l2_ms'] for i in range(25)]
            for arm in ('rotate','chain','chain_control','original'):
                current=[metrics.timing(cell/arm/f'r{i}.log')['l2_ms'] for i in range(25)]
                ratios=[a/b for a,b in zip(current,reference)]
                boot=sorted(statistics.median(rng.choices(ratios,k=25)) for _ in range(10000))
                ratio,lo,hi=statistics.median(ratios),boot[249],boot[9749]
                median=statistics.median(current)
                values.append(f'{m}/s{seq}/{arm}: L2={median:.6f} ms, ratio={ratio:.5f} [{lo:.5f},{hi:.5f}]')
    return detail+'; '+'; '.join(values)


def matrix_measurements():
    import statistics
    raw=HERE/'ablation'
    detail=audit_raw.matrix(raw)
    for c in metrics.CONFIGS:
        for m in ('gqa2','mha4'):
            for seq in (4,128):
                for p in (0,5):
                    cell=raw/c/'paired'/f'{m}_s{seq}_p{p}'
                    samples=metrics.samples(cell)
                    fields=('l2_ms','nofence_ms','nowait_ms','neither_ms','l1nosync_ms',
                            'fence_ms','wait_ms','notify_ms','barrier_ms','protocol_over_barrier','l2_over_l1')
                    values=' '.join(f'{k}={statistics.median(r[k] for r in samples):.6f}' for k in fields)
                    print(f'ABLATION {c}/{m}/s{seq}/p{p} {values} evidence={cell}',flush=True)
    return detail


def instruction_counts():
    details=[]
    for c in metrics.CONFIGS:
        raw=REPO/'docs/experiments/FENCE/raw' if c=='baseline' else HERE/c
        for m in ('gqa2','mha4'):
            for p in (0,5):
                text=(raw/'sass'/f'{m}_p{p}_full.sass').read_text()
                chunk=next(c for c in text.split('Function : ')[1:] if 'tilemega_l2_kernel' in c.splitlines()[0])
                membar=len(re.findall(r'\bMEMBAR\.SC\.GPU',chunk))
                barriers=len(re.findall(r'\bBAR\.SYNC',chunk))
                if not (raw/'sass'/f'{m}_p{p}_full.report.txt').is_file():raise ValueError('missing BARRIERS report')
                details.append(f'{c}/{m}/p{p}: MEMBAR.SC.GPU={membar}, BAR.SYNC={barriers}')
    return '; '.join(details)


def local_probe_barriers():
    out=HERE/'local_probe_sass'
    details=[]
    for r in json.loads((out/'commands.json').read_text()):
        c,m=r['configuration'],r['model']
        p=out/f'{c}_{m}_neither.sass'
        build=json.loads((HERE/c/'log'/f'{m}_p0_neither.build.json').read_text())
        if r['exit_code'] or sha(p)!=r['sass_sha256'] or r['binary_sha256']!=build['binary_sha256']:
            raise ValueError(f'{p}: disassembly/build mismatch')
        chunk=next(x for x in p.read_text().split('Function : ')[1:] if 'tilemega_l2_kernel' in x.splitlines()[0])
        count=len(re.findall(r'\bBAR\.SYNC',chunk))
        if count!=({'window2':6,'local2':8}[c]):raise ValueError(f'{p}: unexpected barrier sites')
        details.append(f'{c}/{m}/neither: BAR.SYNC={count}')
    if len(details)!=4:raise ValueError('missing local neither-arm disassembly')
    return '; '.join(details)+'; local completion convergence remains in neither; full/neither changes reported separately'


def fence_pricing():
    details=[]
    for m in ('gqa2','mha4'):
        for seq in (4,128):
            for p in (0,5):
                samples=metrics.samples(REPO/'docs/experiments/FENCE/raw/paired'/f'{m}_s{seq}_p{p}')
                med=metrics.interval([r['fence_ms'] for r in samples])[0]*1000
                notify=metrics.interval([r['fence_ms']/r['notify_ms'] for r in samples])[0]
                protocol=metrics.interval([r['fence_ms']/r['protocol_ms'] for r in samples])[0]
                details.append(f'{m}/s{seq}/p{p}: fence={med:.3f} us, fence/notify={notify:.4f}, fence/protocol={protocol:.4f}')
    return '; '.join(details)


def window_gain():
    details=[]
    for m in ('gqa2','mha4'):
        for seq in (4,128):
            for p in (0,5):
                for w in (2,4):
                    cell=f'{m}_s{seq}_p{p}'
                    off=metrics.samples(HERE/'ablation'/f'window{w}'/'paired'/cell)
                    on=metrics.samples(HERE/'ablation'/f'local{w}'/'paired'/cell)
                    fifo=metrics.samples(HERE/'ablation'/'c2'/'paired'/cell)
                    for config in (f'window{w}',f'local{w}'):
                        for i in range(25):
                            text=(HERE/'ablation'/config/'paired'/cell/'full'/f'r{i}.log').read_text()
                            line=re.findall(r'^E2E_RESOURCE (.+)$',text,re.M)
                            if len(line)!=1:raise ValueError(f'{config}/{cell}/r{i}: missing resource record')
                            resource=dict(re.findall(r'(\w+)=([^ ]+)',line[0]))
                            expected={'grid':'256','ctas_per_sm':'2','task_smem':'24576',
                                'reg':'220' if config.startswith('local') else '218',
                                'static_smem':'16' if config.startswith('local') else '0'}
                            if any(resource.get(k)!=v for k,v in expected.items()):
                                raise ValueError(f'{config}/{cell}/r{i}: resource comparison changed')
                    ratio,lo,hi=metrics.interval([a['l2_ms']/b['l2_ms'] for a,b in zip(on,off)])
                    net,nlo,nhi=metrics.interval([a['l2_ms']/b['l2_ms'] for a,b in zip(on,fifo)])
                    details.append(f'{cell}/W{w}: smem/control={ratio:.5f} [{lo:.5f},{hi:.5f}], smem/C2(W1)={net:.5f} [{nlo:.5f},{nhi:.5f}]')
    return '; '.join(details)+'; all window/shared full logs retain grid=256, 2 CTA/SM and TaskSmem=24576 B; registers 218 -> 220, static shared 0 -> 16 B'


def self_checks():
    for name in ('fence','sync','chain'):
        text=(HERE/'self_check'/f'{name}.log').read_text()
        if 'SELF_CHECK ' not in text or ' PASS;' not in text:raise ValueError(f'{name}: self-check failed')
        meta=json.loads((HERE/'self_check'/f'{name}.json').read_text())
        if meta['exit_code'] or meta['env']!={'SELF_CHECK':'1'}:
            raise ValueError(f'{name}: self-check process failed')
        for path,digest in meta['inputs'].items():
            if sha(REPO/path)!=digest:raise ValueError(f'{name}: self-check source changed: {path}')
    return '3/3 runners SELF_CHECK=1 on sm_89; sm_120 has not been run'


def research_gate():
    import statistics
    try:
        scores={}
        values={}
        for c in metrics.CONFIGS:
            values[c]={}
            for m in ('gqa2','mha4'):
                for seq in (4,128):
                    values[c][m,seq]=metrics.samples(HERE/'ablation'/c/'paired'/f'{m}_s{seq}_p0')
            scores[c]=statistics.geometric_mean(statistics.median(r['l2_ms'] for r in v) for v in values[c].values())
        achieved={}
        protocol_scores={}
        for c in metrics.CONFIGS:
            medians=[statistics.median(r['protocol_over_barrier'] for r in v) for v in values[c].values()]
            achieved[c]=sum(r<=1 for r in medians)
            protocol_scores[c]=statistics.geometric_mean(medians)
            print(f'RESEARCH_CONFIGURATION {c}: achieved={achieved[c]}/4 protocol_ratio_geomean={protocol_scores[c]:.6f} full_L2_geomean_ms={scores[c]:.6f}',flush=True)
        best=min(metrics.CONFIGS,key=lambda c:(-achieved[c],protocol_scores[c],scores[c]))
        initial={('gqa2',4):2.23,('gqa2',128):2.49,('mha4',4):2.10,('mha4',128):2.91}
        details=[];passed=0
        for cell,v in values[best].items():
            ratio,lo,hi=metrics.interval([r['protocol_over_barrier'] for r in v])
            closed=(initial[cell]-ratio)/(initial[cell]-1)
            passed+=ratio<=1
            details.append(f'{cell[0]}/s{cell[1]}={ratio:.4f} [{lo:.4f},{hi:.4f}], historical excess gap closed={closed:.2%}')
        gate('wait+notify<=barrier','research',passed>=3,f'{best}: {passed}/4 median gates; '+'; '.join(details),HERE/'ablation')
    except (OSError,KeyError,ValueError,ArithmeticError) as e:
        gate('wait+notify<=barrier','research',False,str(e),HERE/'ablation')


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
        window_ok=len(windows)==12 and all(
            r['measured_l2_ms']*1e6>=max(r['cp_corrected_ns'],r['queue_lb_ns'])
            and min(r['cp_split_'+n+'_ns'] for n in ('task','wait','prerun_barrier','publish','gap'))>=0
            and abs(sum(r['cp_split_'+n+'_ns'] for n in ('task','wait','prerun_barrier','publish','gap'))-r['cp_reconstructed_ns'])<=.01*r['cp_reconstructed_ns']
            and r['cp_split_wait_ns']<=r['kernel_span_ns'] for r in windows)
        gate('A-window','report',window_ok,'12 window floor/split checks; '+'; '.join(f'{r["model"]}/s{r["seq"]}/{r["config"]}: cp={r["cp_corrected_ns"]} hol={r["hol_reclaimable_ns"]}' for r in windows),REPO/'docs/experiments/WINDOW/raw/final')
    except (OSError,ValueError,KeyError) as e:
        for name in ('A-a','A-b','A-c'):
            gate(name,'hard',False,str(e),REPO/'docs/experiments/TRACE_V2/analyze.py')
    check_litmus()
    checked('B-a','hard',lambda:audit_raw.five_arm(REPO/'docs/experiments/FENCE/raw'),REPO/'docs/experiments/FENCE/raw/paired')
    def order():
        sequence=('23c5d618','03053089','204df980','e6dd44bc')
        for a,b in zip(sequence,sequence[1:]):
            subprocess.run(['git','merge-base','--is-ancestor',a,b],cwd=REPO,check=True)
        frozen=subprocess.check_output(['git','show','03053089:docs/experiments/SYNC_V3/targets.tsv'],cwd=REPO)
        if frozen!=(HERE/'targets.tsv').read_bytes():raise ValueError('targets changed after their dedicated freeze commit')
        return ('Resumed prerequisite order A -> frozen targets -> completed B -> C1 is valid. '
                'Original 66ae000e violated H4 and was reverted at 8299ccaa; historical violation is not erased. '
                'Correction follows the user-authorized local repair rule.')
    checked('B-c/H4-resumed','hard',order,'git: 23c5d618 -> 03053089 -> 204df980 -> e6dd44bc')
    for name in ('c1','c2'):
        checked(name.upper()+'-correctness','hard',lambda n=name:audit_raw.correctness(HERE/n),HERE/name/'correctness')
        checked(name.upper()+'-SEQSCAN','hard',lambda n=name:audit_raw.correctness(HERE/n,subset=True),HERE/name/'seqscan')
    def deadlock():
        detail=audit_raw.correctness(HERE/'c2_dependency')
        counts=[]
        for m in ('gqa2','mha4'):
            for seq in (4,128):
                dump=HERE/'c2_dependency/trace'/f'{m}_s{seq}'
                status,meta=audit_raw.result(dump/'run.log')
                if status!='PASS' or meta['exit_code']:raise ValueError(f'{dump}: failed witness execution')
                n=len(trace_dependency.witnesses(dump))
                if not n:raise ValueError(f'{dump}: no retained next-slot dependency')
                counts.append(f'{m}/s{seq}={n} witnesses')
        return detail+'; '+', '.join(counts)
    checked('C2-deadlock','hard',deadlock,HERE/'c2_dependency')
    for w in (2,4):
        checked(f'C3a-W{w}','hard',lambda w=w:audit_raw.correctness(HERE/f'local{w}'),HERE/f'local{w}/correctness')
        checked(f'W{w}-control','hard',lambda w=w:audit_raw.correctness(HERE/f'window{w}'),HERE/f'window{w}/correctness')
    checked('RED-shard-composition','hard',lambda:audit_raw.correctness(HERE/'sharded_red'),HERE/'sharded_red/correctness')
    checked('C3b-degeneracy','hard',cluster_degeneracy,HERE/'sass_identity')
    checked('C3b-sm120-compile','report',cluster_compile,HERE/'cluster_compile')
    checked('D-a','hard',lambda:audit_raw.correctness(REPO/'docs/experiments/CHAIN2/final',chain=True),REPO/'docs/experiments/CHAIN2/final/correctness')
    checked('D-b','hard',lambda:audit_raw.chain_paths(REPO/'docs/experiments/CHAIN2/final'),REPO/'docs/experiments/CHAIN2/final/replay/path')
    checked('D-c','report',chain_comparison,REPO/'docs/experiments/CHAIN2/final/paired')
    checked('D-d','report',lambda:audit_raw.chain_rejections(REPO/'docs/experiments/CHAIN2/final'),REPO/'docs/experiments/CHAIN2/final/on/rejected_extensions.tsv')
    checked('E-frozen-targets','report',lambda:audit_raw.targets(analyze),HERE/'targets_raw/dump')
    checked('E-target-positions','report',audit_raw.target_positions,HERE/'target_positions')
    checked('window-probe-coverage','hard',verify_window_probe.verify,HERE/'window_probe_fix')
    checked('C-five-arm/ablation','report',matrix_measurements,HERE/'ablation')
    checked('real-width-ablation','report',lambda:audit_raw.realwidth(HERE/'realwidth'),HERE/'realwidth')
    checked('C1-MEMBAR','report',membar,HERE/'c1/sass')
    checked('C-instruction-counts','report',instruction_counts,HERE/'c2/sass')
    checked('C3a-neither-barriers','report',local_probe_barriers,HERE/'local_probe_sass')
    checked('B-b','report',fence_pricing,REPO/'docs/experiments/FENCE/raw/paired')
    checked('C3a-window-gain','report',window_gain,HERE/'ablation')
    checked('sm120-self-check','report',self_checks,HERE/'self_check')
    research_gate()
    hard=[r for r in results if r[1]=='hard']
    print(f'\n{sum(r[2] for r in hard)}/{len(hard)} hard gates pass; '
          f'{sum(r[2] for r in results if r[1]=="report")}/{sum(r[1]=="report" for r in results)} report gates pass',flush=True)
    return int(any(not r[2] for r in hard))


if __name__=='__main__':
    raise SystemExit(main())
