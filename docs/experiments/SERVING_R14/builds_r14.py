#!/usr/bin/env python3
"""Pinned R14 builds and exact-source trace variants, each with an identity."""
import argparse,concurrent.futures,copy,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
from pin_case import pin,classes
from fixed_builds import run_bounded
from tilemega.build.identity import snapshot,generate,verify,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def compile_variant(job):
    base=Path(job['base_so']);out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);so=out/'plan.so'
    verify(base)
    command=shlex.split(Path(str(base)+'.build_command.txt').read_text())
    command[command.index('-o')+1]=str(so)
    source=next(i for i,arg in enumerate(command) if not arg.startswith('-') and arg.endswith('.cu'))
    command[source]=str(so)+'.cu'
    for name,value in job['defines'].items():
        command=[arg for arg in command if not arg.startswith('-D'+name+'=')]
        command+=['-D'+name+'='+str(value)]
    for suffix in ('.cu','.plan.json','.classes.tsv','.selected_classes.tsv'):
        if Path(str(base)+suffix).exists():shutil.copyfile(str(base)+suffix,str(so)+suffix)
    snapshot(ROOT,str(so)+'.source.json',os.environ['TILEMEGA_BIN'])
    Path(str(so)+'.build_command.txt').write_text(shlex.join(command)+'\n')
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with Path(str(so)+'.ptxas.log').open('w') as log:code=run_bounded(command,log,3600)
    if code:raise RuntimeError(f'nvcc exit {code}')
    generate(so,ROOT,str(so)+'.source.json')
    return so

def one(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);so=out/'plan.so'
    record=dict(job,exit_code=1,so=str(so),placeholder_measurement=True)
    try:
        if job.get('base_so'):so=compile_variant(job)
        else:
            manifest=Path(job['manifest']);data=json.loads(manifest.read_text())
            for change in job.get('gemm_overrides',[]):data['gemms'][change['index']].update(change['values'])
            pinned=out/'source_manifest.json';pinned.write_text(json.dumps(data,indent=2)+'\n')
            recipe=pin(pinned,job['classes'],job.get('target_classes',job['classes']),out,job['overrides'])
            phase=job.get('phase','decode');interval='0:0' if phase=='prefill' else '64:1086'
            command=[os.environ['TILEMEGA_BIN'],'compile',job['export'],str(so),'--serving',phase,
                '--batch',str(job['batch']),'--past-range',interval,'--capacity','1088','--solver','skeleton',
                '--solve',job['target'],'--runtime-target',job['target'],'--emit','serving',
                '--search-passes','1','--top-m','1','--search-jobs','3','--search-budget-ms','60000',
                '--handoff','off','--candidate-mode','L1','--candidate-loop','0',
                '--dump-cg',str(out/'selected.mlir'),'--measure-cmd',shlex.join([sys.executable,str(HERE/'measure_stub.py')])]+recipe['options']
            (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            with (out/'build.log').open('w') as log:code=run_bounded(command,log,3600)
            if code:raise RuntimeError(f'compile exit {code}; see build.log')
            if sorted(classes(str(so)+'.classes.tsv').values())!=recipe['expected_partition']:
                raise ValueError('rebuilt class partition changed')
        identity=verify(so)
        record.update(exit_code=0,artifact_id=identity['artifact_id'],sha256=sha(so),
                      identity=str(so)+'.identity.json',trace=identity['trace'])
    except Exception as error:record['error']=repr(error)
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(job['label'],record['exit_code'],flush=True)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();jobs=json.loads(a.jobs.read_text());rows=[]
    # Variants only enter after their base succeeds; independent plans use -j3.
    for variants in (False,True):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            rows+=list(pool.map(one,[j for j in jobs if bool(j.get('base_so'))==variants]))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n')
    if any(r['exit_code'] for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
