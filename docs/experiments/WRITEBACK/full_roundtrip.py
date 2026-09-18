#!/usr/bin/env python3
"""Compare all executable host tables after CG writeback versus direct injection.

Run from the repository root. Five seq points use the same fixed geometry,
past=3, grid=128 and kappa=1 as interval_closure. The direct tables are freshly
re-solved; CUDA executes both carriers with unchanged correctness checks.
This is a table-equivalence check, not a performance or synchronization claim.
"""
import argparse,csv,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def execute(command,folder,env=None,gpu=False):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'command.json').exists():raise RuntimeError('refusing overwrite '+str(folder))
    meta=dict(command=command,started_ns=time.time_ns())
    if gpu:
        meta['binary_sha256']=sha(Path(command[0]))
        meta['environment']={k:v for k,v in env.items() if k.startswith('TILEMEGA_')}
    with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
        if gpu:fcntl.flock(lock,fcntl.LOCK_EX)
        meta['started_ns']=time.time_ns()
        with (folder/'run.log').open('w') as log:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    meta.update(exit_code=result.returncode,elapsed_ns=time.time_ns()-meta['started_ns'])
    (folder/'command.json').write_text(json.dumps(meta,indent=2)+'\n')
    result.check_returncode()
    if gpu and 'RESULT status=PASS' not in (folder/'run.log').read_text():
        raise RuntimeError('numerical failure '+str(folder))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'full_roundtrip')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(out).free//2**20
    print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
    if free<8192:raise RuntimeError('insufficient compile space')
    driver=out/'bin/resolve'
    execute(['python3',str(HERE.parent/'JOINT2/build_driver.py'),str(HERE/'interval_check.cpp'),str(driver)],out/'resolve_build')
    execute([str(driver),str(HERE/'interval_closure/gqa2.mlir'),str(HERE.parent/'COSTMODEL/event_fit/target.json'),str(out/'direct_s')],out/'resolve')
    template=json.loads((HERE/'interval_closure/build/interval.json').read_text())['command']
    original=str(HERE/'interval_closure/gqa2.cu')
    for arm,source in [('cg',original),('direct',str(HERE/'direct_host.cu'))]:
        command=template.copy();command[command.index(original)]=source
        command[command.index('-o')+1]=str(out/'bin'/arm)
        execute(command,out/('build_'+arm))
    rows=[]
    for seq in range(1,6):
        for arm in ('cg','direct'):
            folder=out/arm/f's{seq}';folder.mkdir(parents=True,exist_ok=True)
            env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
            env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1',TILEMEGA_PLAN_DUMP=str(folder))
            command=[str(out/'bin'/arm),str(HERE/'interval_closure/fixtures'/f's{seq}')]
            if arm=='direct':command += [str(seq),'3','128',str(out/f'direct_s{seq}.tsv')]
            execute(command,folder,env,True)
        for name in ('schedule.tsv','waits.tsv','events.tsv'):
            a=out/'cg'/f's{seq}'/name;b=out/'direct'/f's{seq}'/name
            same=a.read_bytes()==b.read_bytes()
            rows.append(dict(seq=seq,table=name,bytes=a.stat().st_size,identical=int(same),cg_sha256=sha(a),direct_sha256=sha(b)))
            if not same:raise RuntimeError(f'full table differs seq={seq} {name}')
        print('FULL_ROUNDTRIP',seq,'schedule/waits/events byte-identical',flush=True)
    with (out/'comparisons.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    print('FULL_ROUNDTRIP_PASS 15/15 tables; 10/10 fresh processes',flush=True)
if __name__=='__main__':main()
