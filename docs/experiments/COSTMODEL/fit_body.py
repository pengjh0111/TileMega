#!/usr/bin/env python3
"""Fit backend work features from fresh raw phase dumps, never L2 timings.

Each stage/shape/cell contributes one median observation. This prevents the
largest real-width task domain from silently weighting the regression more
heavily. Phase-instrumentation exposure remains a reported calibration limit.
"""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path
from analyze_kloop import write
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw_nodes(dump):
    slots={int(r['slot']):r for r in csv.DictReader((dump/'slots.tsv').open(),delimiter='\t')}
    result=[]
    for row in csv.DictReader((dump/'phases.tsv').open(),delimiter='\t'):
        r={k:int(v) for k,v in row.items()};slot=slots[r['slot']]
        cycles=r['run_end_cycles']-r['run_begin_cycles'];ns=r['run_end_ns']-r['run_begin_ns']
        if cycles<=0 or ns<0:raise RuntimeError('invalid raw task interval')
        scale=ns/cycles;main=r['mainloop_end_cycles']-r['first_operand_ready_cycles']
        measured=r['kind']==0;loop=r['loop_end_cycles']-r['loop_begin_cycles'] if measured else 0
        body=loop-r['operand_wait_cycles'] if measured else 0;fixed=main-loop if measured else 0
        if measured and (not r['k_iterations'] or min(loop,body,fixed)<0):raise RuntimeError('loop closure')
        r.update(stage=int(slot['stage']),kloop_measured=int(measured),run_ns=ns,run_cycles=cycles,
                 mainloop_cycles=main,setup_cycles=r['setup_end_cycles']-r['run_begin_cycles'],
                 epilogue_cycles=r['run_end_cycles']-r['mainloop_end_cycles'],kloop_body_ns=body*scale,
                 kloop_wait_ns=r['operand_wait_cycles']*scale,kloop_fixed_ns=fixed*scale)
        result.append(r)
    return result
def nnls(x,y):
    scales=[max(abs(r[j]) for r in x) or 1 for j in range(len(x[0]))]
    z=[[v/scales[j] for j,v in enumerate(r)] for r in x]
    beta=[0.]*len(scales);residual=list(y)
    for _ in range(10000):
        change=0
        for j in range(len(beta)):
            denom=sum(r[j]**2 for r in z)
            if not denom:continue
            step=sum(r[j]*v for r,v in zip(z,residual))/denom
            value=max(0,beta[j]+step);delta=value-beta[j];beta[j]=value;change=max(change,abs(delta))
            residual=[v-r[j]*delta for r,v in zip(z,residual)]
        if change<1e-9:break
    return [v/s for v,s in zip(beta,scales)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',nargs='+',type=Path,default=[HERE/'raw_kloop',HERE/'raw_control_kloop']);ap.add_argument('--target',type=Path,default=REPO/'configs/targets/sm_89.json');ap.add_argument('--out',type=Path,default=HERE/'body_fit');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[];scalar=[];sources={}
    for root in a.raw:
        for cell in sorted(root.glob('*_s*')):
            specs=json.loads((cell/'specs.json').read_text())
            for arm,s in specs.items():
                dump=cell/'phase'/arm/'dump'
                if not (dump/'phases.tsv').exists():raise RuntimeError('missing '+str(dump))
                nodes=raw_nodes(dump)
                sources[str(dump/'phases.tsv')]=sha(dump/'phases.tsv')
                grouped=collections.defaultdict(list)
                for n in nodes:grouped[n['stage']].append(n)
                for stage,ns in grouped.items():
                    if not ns[0]['kloop_measured']:
                        scalar.append(statistics.median((n['setup_cycles']+n['epilogue_cycles'])*n['run_ns']/n['run_cycles'] for n in ns));continue
                    med=lambda key:statistics.median(n[key] for n in ns)
                    iterations=med('k_iterations')
                    if any(n['k_iterations']!=iterations for n in ns):raise RuntimeError('varying K work within stage needs per-work grouping')
                    rows.append(dict(cell=cell.name,cohort=root.name,arm=arm,stage=stage,tile_m=ns[0]['tile_m'],tile_n=ns[0]['tile_n'],tile_k=ns[0]['tile_k'],stages=int(s.get('stages',2)),iterations=iterations,operand_elements=ns[0]['operand_bytes']/2,
                        fixed_ns=statistics.median((n['run_cycles']-n['mainloop_cycles'])*n['run_ns']/n['run_cycles'] for n in ns),
                        loop_body_ns=med('kloop_body_ns')/iterations,loop_wait_ns=med('kloop_wait_ns')/iterations,loop_fixed_ns=med('kloop_fixed_ns'),samples=len(ns),dump=str(dump)))
    if not rows:raise RuntimeError('no raw calibration rows')
    features={
      'fixed':lambda r:[1,r['tile_m']*r['tile_n'],r['operand_elements']*r['stages']],
      'loop_body':lambda r:[1,r['tile_m']*r['tile_n']*r['tile_k']],
      'loop_wait':lambda r:[1,r['operand_elements']],
      'loop_fixed':lambda r:[1,r['tile_m']*r['tile_n']]}
    fit={};errors={}
    for part,feature in features.items():
        x=[feature(r) for r in rows];y=[r[part+'_ns'] for r in rows];coeff=nnls(x,y);fit[part]=coeff
        residual=[]
        for r,v,w in zip(rows,x,y):
            prediction=sum(c*f for c,f in zip(coeff,v));r[part+'_predicted_ns']=prediction;residual.append(abs(prediction-w)/max(1,w))
        errors[part]=dict(p50=statistics.median(residual),p90=sorted(residual)[math.ceil(.9*len(residual))-1],max=max(residual))
    fit.update(scalar_fixed_ns=statistics.median(scalar),samples=sum(r['samples'] for r in rows),source=str(a.out/'sources.json'))
    target=json.loads(a.target.read_text());target['calibration_by_dtype']['bf16']['task_body']=fit
    (a.out/'target.json').write_text(json.dumps(target,indent=2)+'\n');(a.out/'fit.json').write_text(json.dumps(fit,indent=2)+'\n')
    (a.out/'sources.json').write_text(json.dumps(dict(raw_phase_sha256=sources,target=str(a.target),target_sha256=sha(a.target),scope='raw instrumented TaskBody service; not a fit to end-to-end L2 timings'),indent=2)+'\n')
    (a.out/'errors.json').write_text(json.dumps(errors,indent=2)+'\n');write(a.out/'observations.tsv',rows)
    print(json.dumps(dict(rows=len(rows),samples=fit['samples'],coefficients=fit,residuals=errors),indent=2))
if __name__=='__main__':main()
