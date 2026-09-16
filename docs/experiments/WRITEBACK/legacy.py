#!/usr/bin/env python3
"""R6 legacy queue identity and complete requested SEQSCAN subset.
Builds baseline/head with switches off; no correctness tolerance changes.
"""
import argparse,concurrent.futures,csv,fcntl,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'legacy'
BASE='bad8a0d9b17804b73afe00a6d545dcea72cc6cbb'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(m,s,p,arm,out,dump=None):
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('refusing overwrite '+str(out))
    binary=OUTPUT/'bin'/f'{m}_{arm}'
    cmd=[str(binary),str(REPO/f'docs/experiments/SEQSCAN/raw/fixture/{m}_s{s}_p{p}')]
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
    if dump:dump.mkdir(parents=True,exist_ok=True);env['TILEMEGA_PLAN_DUMP']=str(dump)
    with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
        r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
    out.write_text(r.stdout+r.stderr)
    out.with_suffix('.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,binary_sha256=sha(binary),started_ns=start,elapsed_ns=time.time_ns()-start,environment={k:v for k,v in env.items() if k.startswith('TILEMEGA_')}),indent=2)+'\n')
    if r.returncode or 'RESULT status=PASS' not in out.read_text():raise RuntimeError('correctness regression '+str(out))
def main():
    global OUTPUT
    a=argparse.ArgumentParser();a.add_argument('action',choices=['build','identity','seqscan']);a.add_argument('--out',type=Path,default=HERE/'legacy_matched');a.add_argument('--baseline-tree',type=Path,default=Path('/tmp/tilemega-r6-baseline'));a=a.parse_args()
    root=a.out;OUTPUT=root;root.mkdir(exist_ok=True)
    if a.action=='build':
        free=shutil.disk_usage(root).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192:raise RuntimeError('disk budget')
        # TargetSpec evolved in R6: baseline headers MUST link the baseline
        # host library, even though the legacy materializer itself is unchanged.
        baseline=a.baseline_tree
        if not (baseline/'build/libtilemega.a').is_file():raise RuntimeError('build the matched baseline host library first')
        (root/'bin').mkdir(exist_ok=True)
        def build(m,arm):
            source=REPO/f'docs/experiments/SEQSCAN/raw/src/{m}.cu';binary=root/'bin'/f'{m}_{arm}'
            cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-DTILEMEGA_EVENT_KAPPA=1','-I'+str((baseline if arm=='base' else REPO)/'include'),*['-I'+str(REPO/p) for p in ('third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test')],str(source),str(baseline/'build/libtilemega.a' if arm=='base' else REPO/'build-portable/libtilemega.a'),'-L/usr/local/cuda/lib64','-lcudart','-o',str(binary)]
            with (root/f'{m}_{arm}.build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            (root/f'{m}_{arm}.build.json').write_text(json.dumps(dict(base=BASE,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=cmd,source_sha256=sha(source),binary_sha256=sha(binary)),indent=2)+'\n')
            print('BUILD',m,arm,'PASS',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(build,m,arm) for m in ('gqa2','mha4') for arm in ('base','head')]
            for job in jobs:job.result()
    elif a.action=='identity':
        rows=[]
        for m in ('gqa2','mha4'):
            for s in (4,128):
                cell=root/'identity'/f'{m}_s{s}'
                for arm in ('base','head'):run(m,s,3,arm,cell/(arm+'.log'),cell/arm)
                for name in ('schedule.tsv','waits.tsv','events.tsv'):
                    before=(cell/'base'/name).read_bytes();after=(cell/'head'/name).read_bytes()
                    rows.append(dict(model=m,seq=s,table=name,bytes=len(after),identical=before==after,base_sha256=sha(cell/'base'/name),head_sha256=sha(cell/'head'/name)))
        (root/'identity.json').write_text(json.dumps(rows,indent=2)+'\n')
        if not all(r['identical'] for r in rows):raise RuntimeError('legacy tables differ')
        print('LEGACY_IDENTITY PASS tables='+str(len(rows)),flush=True)
    else:
        for m in ('gqa2','mha4'):
            for s in (4,128,2048):
                for p in (0,512):
                    for i in range(50):run(m,s,p,'head',root/'seqscan'/f'{m}_s{s}_p{p}'/f'r{i}.log')
                    print('SEQSCAN',m,s,p,'50/50',flush=True)
if __name__=='__main__':main()
