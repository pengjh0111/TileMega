#!/usr/bin/env python3
"""R6 production-search artifacts, fresh-process rotated pairs and correctness.

Usage: run.py ACTION [--models gqa2 mha4 real] [--seqs 4 128]
Actions: search, build, pilot, freeze, measure, correctness, trace_build, trace.
The reference driver here is sm_89; the sm_120 runner must regenerate pinned
R5 champion plans on its own target before using them as a control.
"""
import argparse,concurrent.futures,csv,json,os,shutil,statistics,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure as r5

def rows(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def specs(cell,m,seq):
    selected=json.loads((REPO/f'docs/experiments/JOINT/raw/{m}_s{seq}/choice.json').read_text())['choice']
    result={'control':dict(source=str(r5.source(m)),kappa='1',residency='0',placement_macro='5'),
            'champion':dict(selected,placement_macro='0')}
    top=rows(cell/'auto.cu.top3.tsv')
    if len(top)!=3:raise RuntimeError('search did not deliver three candidates')
    for r in top:
        result['top'+r['rank']]=dict(source=str(Path(r['source']).resolve()),kappa=r['kappa'],residency=r['residency'],placement_macro='0')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['search','build','pilot','freeze','measure','correctness','trace_build','trace'])
    ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--raw',type=Path,default=HERE/'reduced');ap.add_argument('--capacity',type=int,default=9);ap.add_argument('--jobs',type=int,default=2)
    a=ap.parse_args();a.raw.mkdir(parents=True,exist_ok=True);session=str(time.time_ns())
    cells=[(a.raw/f'{m}_s{s}',m,s) for m in a.models for s in a.seqs]
    if a.action=='search':
        def solve(cell,m,s):
            cell.mkdir(parents=True,exist_ok=True)
            if (cell/'auto.cu').exists():raise RuntimeError('refusing overwrite search')
            source=REPO/(f'docs/experiments/SEQSCAN/raw/export/{m}.json' if m!='real' else f'docs/experiments/REALMODEL/raw/work/r2sim_s{s}/model.json')
            cmd=[str(REPO/'build-portable/tools/tilemega-compile'),str(source),str(cell/'auto.cu'),'--solve',str(REPO/'docs/experiments/COSTMODEL/event_fit/target.json'),'--seq',str(s),'--past','3','--search-capacity',str(a.capacity),'--search-domain',str(REPO/'docs/experiments/COSTMODEL/event_fit/search_domain.json'),'--dump-cg',str(cell/'auto.mlir'),'--hop-curve',str(REPO/'docs/experiments/SIMULATOR/hop_ns.tsv')]
            (cell/'search_command.json').write_text(json.dumps(dict(command=cmd,session=session),indent=2)+'\n')
            with (cell/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures=[pool.submit(solve,*c) for c in cells]
            for f in futures:f.result()
        return
    if a.action in ('build','trace_build'):
        free=shutil.disk_usage(a.raw).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192:raise RuntimeError('disk budget')
        jobs=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for cell,m,s in cells:
                arms=specs(cell,m,s);(cell/'specs.json').write_text(json.dumps(arms,indent=2)+'\n')
                if a.action=='trace_build':
                    choice=json.loads((cell/'choice.json').read_text())['arm'];arms={k:arms[k] for k in ('control','champion',choice)}
                for arm,spec in arms.items():jobs.append(pool.submit(r5.build,cell,m,arm,spec,'sm_89',a.action=='trace_build'))
            if any(f.result()!=0 for f in jobs):raise RuntimeError('candidate build failed; inspect and explicitly exclude')
        return
    for cell,m,s in cells:
        arms=specs(cell,m,s);excluded_path=cell/'excluded.json';excluded=json.loads(excluded_path.read_text()) if excluded_path.exists() else {}
        if a.action=='freeze':
            valid=[k for k in arms if k.startswith('top') and k not in excluded]
            medians={k:statistics.median(r5.timing(cell/'pilot'/k/f'r{i}.log')['l2_ms'] for i in range(5)) for k in valid}
            if not medians:raise RuntimeError('all new candidates rejected')
            if (cell/'choice.json').exists():raise RuntimeError('choice already frozen')
            arm=min(medians,key=medians.get);(cell/'choice.json').write_text(json.dumps(dict(arm=arm,pilot_medians=medians,choice=arms[arm],frozen_ns=time.time_ns(),excluded=excluded),indent=2)+'\n');continue
        if a.action in ('pilot','measure'):
            if a.action=='measure' and not (cell/'choice.json').exists():raise RuntimeError('freeze before confirmatory measurement')
            order=list(arms)
            for i in range(5 if a.action=='pilot' else 25):
                for j in range(len(order)):
                    arm=order[(i+j)%len(order)]
                    if arm in excluded:continue
                    try:r5.run(cell,m,s,arm,cell/a.action/arm,i,j,session)
                    except RuntimeError as e:
                        if a.action!='pilot' or arm in ('control','champion'):raise
                        excluded[arm]=dict(reason=str(e),log=str(cell/a.action/arm/f'r{i}.log'),time_ns=time.time_ns())
                        excluded_path.write_text(json.dumps(excluded,indent=2)+'\n')
            print(a.action,m,s,'completed',flush=True)
        elif a.action=='correctness':
            arm=json.loads((cell/'choice.json').read_text())['arm']
            for i in range(50):r5.run(cell,m,s,arm,cell/'correctness',i,0,session)
            print('CORRECTNESS',m,s,'50/50',flush=True)
        elif a.action=='trace':
            arm=json.loads((cell/'choice.json').read_text())['arm']
            for j,k in enumerate(('control','champion',arm)):r5.run(cell,m,s,k+'_trace',cell/'trace'/k,0,j,session,True)
if __name__=='__main__':main()
