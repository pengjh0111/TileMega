#!/usr/bin/env python3
"""§4.5 A-c: the two reference models still pass at HEAD with every switch off.

Two halves, both built with the default macros -- no prefetch runtime, no
per-stage kappa table, no segmentation -- because A-c asks whether round seven
broke what round five and six already had, not whether the new arms work:

  * the four cells (gqa2/mha4 x seq in {4,128}), solved at HEAD, 50 rounds each;
  * the SEQSCAN subset, the twelve R5 cases, 50 rounds each.  Their sources are
    the projections B1-a already verified byte-identical to JOINT's plans once
    the guarded field is stripped, and with TILEMEGA_PREFETCH_RUNTIME left
    undefined that field compiles out, so this builds R5's plan exactly.

Every round is a fresh process under the GPU lock.  A failing round is counted
and the campaign continues, so one bad cell does not cost the other eleven.
"""
import argparse,concurrent.futures,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure
sys.path.insert(0,str(REPO/'docs/experiments/PIPELINE'))
import seqscan as sq
CELLS=sq.CELLS
CASES=sq.CASES

def plan_spec(source):
    """The plan's own kappa and residency, as the solver wrote them into it."""
    import re
    text=Path(source).read_text()
    macro=lambda n,d:(lambda m:m[1] if m else d)(re.search(r'^#define '+n+r' (\d+)$',text,re.M))
    return dict(source=str(source),kappa=macro('TILEMEGA_EVENT_KAPPA','1'),
                residency=macro('TILEMEGA_RESIDENCY_CAP','0'),placement_macro='0')

def units(out,plans,half='both'):
    """(cell dir, model, seq, past, arm, spec) for all sixteen builds."""
    for c in CELLS:
        model,seq=c.split('_s')[0],int(c.split('_s')[1])
        cell=out/c
        if half in ('both','reference'):
            yield cell,model,seq,3,'default',plan_spec(plans/c/'auto.cu')
        if half=='reference':continue
        placement=sq.choice(c)['placement']
        for s,p in CASES:
            src=REPO/'docs/experiments/PIPELINE/raw/seqscan'/c/sq.case_name(s,p)/(placement+'.cu')
            yield cell,model,s,p,f'seqscan_{sq.case_name(s,p)}',dict(sq.choice(c),
                source=str(src),placement_macro='0')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['build','run','report'])
    ap.add_argument('--out',type=Path,default=HERE/'regression')
    ap.add_argument('--plans',type=Path,default=HERE/'topk')
    ap.add_argument('--rounds',type=int,default=50);ap.add_argument('--jobs',type=int,default=8)
    ap.add_argument('--arch',default='sm_89')
    ap.add_argument('--half',choices=['both','reference','seqscan'],default='both')
    a=ap.parse_args()
    if a.action=='build':
        work=[u for u in units(a.out,a.plans,a.half) if not (u[0]/'bin'/u[4]).exists()]
        for u in work:u[0].mkdir(parents=True,exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs=[pool.submit(measure.build,cell,model,arm,spec,a.arch)
                  for cell,model,_,_,arm,spec in work]
            bad=sum(j.result()!=0 for j in jobs)
        print('A-c BUILD units',len(work),'failed',bad,flush=True)
        return 1 if bad else 0
    if a.action=='run':
        session=str(time.time_ns());work=list(units(a.out,a.plans,a.half))
        for i in range(a.rounds):
            # rotate the unit order every round: an ordering effect hits all alike
            for j in range(len(work)):
                cell,model,seq,past,arm,_=work[(i+j)%len(work)]
                folder=cell/'correctness'/arm
                if (folder/f'r{i}.log').exists():continue
                try:measure.run(cell,model,seq,arm,folder,i,j,session,past=past)
                except Exception as e:print('A-c FAIL',cell.name,arm,i,e,flush=True)
            print('A-c round',i,'done',flush=True)
        return 0
    rows=[]
    for cell,model,seq,past,arm,spec in units(a.out,a.plans,a.half):
        logs=sorted((cell/'correctness'/arm).glob('r*.log'),key=lambda p:int(p.stem[1:]))
        passing=sum('RESULT status=PASS' in p.read_text() for p in logs)
        binaries={json.loads(p.with_suffix('.json').read_text())['binary_sha256'] for p in logs}
        rows.append(dict(cell=cell.name,arm=arm,seq=seq,past=past,rounds=len(logs),
                         passing=passing,binaries=len(binaries),
                         source=spec['source'],kappa=spec['kappa'],residency=spec['residency']))
    columns=list(rows[0])
    with (a.out/'regression.tsv').open('w') as f:
        f.write('\t'.join(columns)+'\n')
        for r in rows:f.write('\t'.join(str(r[c]) for c in columns)+'\n')
    ref=[r for r in rows if r['arm']=='default'];sub=[r for r in rows if r['arm']!='default']
    for label,group in (('reference',ref),('seqscan',sub)):
        good=sum(r['passing']==r['rounds'] and r['rounds']==a.rounds for r in group)
        print('A-c',label,'arms',len(group),'passing',good,
              'rounds',sum(r['rounds'] for r in group),
              'passing_rounds',sum(r['passing'] for r in group),flush=True)
    ok=all(r['passing']==r['rounds']==a.rounds for r in rows)
    print('A-c RESULT',('PASS' if ok else 'FAIL'),'arms',len(rows),flush=True)
    return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
