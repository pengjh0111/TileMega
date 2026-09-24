#!/usr/bin/env python3
"""Compile and measure shortlisted R9 candidates in ten fresh processes.

Example: measure.py --source arm/selected.cu --fixture /path/to/fixture --top3
Each build has a command JSON and ptxas log. Each process has a raw log and
command/environment metadata. The winner is the internally bit-exact candidate
with the smallest median L2 time; CPU golden is deliberately not this gate.
GPU runs take an advisory lock shared by all R9 runners on this host.
"""
import argparse,csv,fcntl,hashlib,json,os,pathlib,re,shutil,statistics,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[3]
def run(cmd, log, env=None, timeout=None):
    cmd=list(map(str,cmd));start=time.time_ns()
    executable=shutil.which(cmd[0]);digest=hashlib.sha256(pathlib.Path(executable).read_bytes()).hexdigest() if executable else None
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    with log.open('w') as f:
        try:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);code=p.returncode
        except subprocess.TimeoutExpired:code=124
    log.with_suffix('.command.json').write_text(json.dumps(dict(command=cmd,executable_sha256=digest,head=head,started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=code,environment={k:v for k,v in (env or {}).items() if k.startswith('TILEMEGA_')}),indent=2)+'\n')
    return code

def parse(text):
    hashes=re.search(r'E2E_HASH l05=(\w+) l1=(\w+) l2=(\w+)',text)
    good=bool(hashes and len(set(hashes.groups()))==1 and 'l1_vs_l05_mismatch=0' in text and 'l2_vs_l1_mismatch=0' in text)
    row=re.search(r'E2E_TIME l05_ms=([\d.]+) l1_ms=([\d.]+).*? l2_ms=([\d.]+)',text)
    return good,tuple(map(float,row.groups())) if row else None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=pathlib.Path,required=True);ap.add_argument('--fixture',type=pathlib.Path,required=True);ap.add_argument('--top3',action='store_true');a=ap.parse_args()
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')};env.update(TILEMEGA_WARMUP='2',TILEMEGA_REPEAT='10')
    sources=[a.source];shortlist=[]
    if a.top3:
        shortlist=list(csv.DictReader(pathlib.Path(str(a.source)+'.top3.tsv').open(),delimiter='\t'))
        sources=[pathlib.Path(row['source']) for row in shortlist]
    records=[]
    for source in sources:
        out=pathlib.Path(str(source)+'.measurement');out.mkdir(parents=True,exist_ok=True)
        free=shutil.disk_usage(out).free//2**20;need=8192
        (out/'disk.json').write_text(json.dumps(dict(NEED_MIB=need,FREE_MIB=free)))
        if free<need:raise RuntimeError('disk budget')
        binary=out/'kernel'
        cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-Xptxas=-v','-DTILEMEGA_MIDPOINT_REFINE=0']
        for sub in ['include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test']:cmd+=['-I'+str(ROOT/sub)]
        cmd += [source,ROOT/'build-portable/libtilemega.a','-L/usr/local/cuda/lib64','-lcudart','-o',binary]
        if run(cmd,out/'build.log',env):records.append(dict(source=str(source),status='build_failed'));continue
        samples=[];passed=0
        with open('/tmp/tilemega-r9-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run(['nvidia-smi','-q'],out/'device_before.log',env)
            for i in range(10):
                log=out/f'process_{i:02}.log';code=run([binary,a.fixture],log,env,timeout=600)
                good,timing=parse(log.read_text());passed+=good
                if timing:samples.append(timing)
                print(f'PROCESS source={source} round={i} internal={int(good)} exit={code}',flush=True)
            run(['nvidia-smi','-q'],out/'device_after.log',env)
        record=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),passed=passed,samples=len(samples),status='ok' if passed==10 and len(samples)==10 else 'internal_failed')
        if samples:
            record.update(zip(['l05_ms','l1_ms','l2_ms'],map(statistics.median,zip(*samples))))
        records.append(record)
    valid=[r for r in records if r['status']=='ok'];winner=min(valid,key=lambda r:r['l2_ms']) if valid else None
    if winner and a.top3:
        chosen=next(row for row in shortlist if row['source']==winner['source'])
        for field,extension in [('source','.cu'),('cg','.mlir')]:
            shutil.copy2(chosen[field],str(a.source)+'.measured'+extension)
        winner['key']=chosen['key']
    pathlib.Path(str(a.source)+'.measurement.json').write_text(json.dumps(dict(candidates=records,winner=winner),indent=2)+'\n')
    if not valid:raise SystemExit(1)
if __name__=='__main__':main()
