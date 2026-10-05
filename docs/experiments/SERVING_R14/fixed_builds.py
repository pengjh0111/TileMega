#!/usr/bin/env python3
"""Thirty prescribed fixed builds; failed ablation arms do not stop siblings."""
import argparse,concurrent.futures,hashlib,json,os,shlex,signal,subprocess,sys
from pathlib import Path
from pin_case import pin,classes
from gpu_guard import descendants
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def run_bounded(command,log,timeout=1800):
    # Keep the scheduler's session; GNU timeout can create additional groups.
    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    try:return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        for pid in descendants(process.pid):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        process.wait();return 124

def one(job,resume=False,retry_failed=False):
    folder=HERE/'raw/B0b'/job['cell']/job['label'];folder.mkdir(parents=True,exist_ok=True);so=folder/'plan.so'
    existing=folder/'record.json'
    if resume and existing.exists():
        prior=json.loads(existing.read_text())
        if prior.get('exit_code')==0 and so.exists() and hashlib.sha256(so.read_bytes()).hexdigest()==prior.get('sha256'):
            print(job['cell'],job['label'],'reuse',flush=True);return prior
        if not retry_failed or 'measured geometry differs within class' in prior.get('error',''):
            print(job['cell'],job['label'],'retained rejection',flush=True);return prior
    try:
        record=pin(job['manifest'],job['classes'],job['target_classes'],folder,job['overrides'])
        cmd=[os.environ['TILEMEGA_BIN'],'compile',job['export'],str(so),'--serving','decode','--batch',str(job['batch']),'--past-range','64:1086','--capacity','1088','--solver','skeleton','--solve','/root/r14_work/target_r12b.json','--runtime-target','/root/r14_work/target_r12b.json','--emit','serving','--search-passes','1','--top-m','1','--search-jobs','3','--search-budget-ms','60000','--handoff','off','--kphase-mask','31','--v3-poll-ns','0','--dump-cg',str(folder/'selected.mlir'),'--measure-cmd',shlex.join([sys.executable,str(HERE/'measure_stub.py')])]+record['options']
        (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (folder/'build.log').open('w') as f:code=run_bounded(cmd,f)
        result=dict(job,exit_code=code,so=str(so),placeholder_measurement=True)
        if code==0:
            floor_command=[os.environ['TILEMEGA_BIN'],'inspect','request-floor',str(folder/'selected.mlir'),'/root/r14_work/target_r12b.json',str(job['batch']),'64','1086',str(folder/'floor.json'),str(folder/'floor.tsv')]
            with (folder/'floor.log').open('w') as f:subprocess.run(floor_command,stdout=f,stderr=subprocess.STDOUT,check=True)
            actual_partition=sorted(classes(str(so)+'.classes.tsv').values())
            if actual_partition!=record['expected_partition']:
                raise ValueError('rebuilt and same-DN donor class GEMM sets differ')
            manifest=json.loads(Path(str(so)+'.plan.json').read_text())
            result.update(sha256=hashlib.sha256(so.read_bytes()).hexdigest(),manifest_result=manifest,residency_same=manifest['residency']==record['source_residency'],grid_same=manifest['grid']==record['source_grid'])
    except Exception as e:result=dict(job,exit_code=1,error=str(e),so=str(so),placeholder_measurement=True)
    (folder/'record.json').write_text(json.dumps(result,indent=2)+'\n');print(job['cell'],job['label'],result['exit_code'],flush=True);return result

def trace_command(build_command,destination):
    command=shlex.split(build_command)
    # The final .so may reuse a candidate binary; replace nvcc's actual -o.
    output=command.index('-o')+1
    command[output]=str(destination)
    command+=['-DTILEMEGA_TRACE_V2=1','-DTILEMEGA_PAGE_TRACE=1']
    return command

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    jobs=json.loads(a.jobs.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(lambda job:one(job,a.resume,a.retry_failed),jobs))
    # Trace is a second compilation of the exact P-base source and flags.
    for result in list(results):
        if result['label']!='P-base' or result['exit_code']:continue
        source=Path(result['so']);folder=HERE/'raw/B0b'/result['cell']/'P-trace';folder.mkdir(parents=True,exist_ok=True);dest=folder/'plan.so'
        record=dict(result,label='P-trace',so=str(dest),exit_code=1)
        try:
            cmd=trace_command(Path(str(source)+'.build_command.txt').read_text(),dest)
            (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            with (folder/'build.log').open('w') as log:code=run_bounded(cmd,log)
            record['exit_code']=code
            if code==0:
                if not dest.is_file():raise FileNotFoundError('nvcc succeeded without the requested trace output')
                Path(str(dest)+'.plan.json').write_text(Path(str(source)+'.plan.json').read_text());record['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
        except Exception as e:record.update(exit_code=1,error=str(e))
        (folder/'record.json').write_text(json.dumps(record,indent=2)+'\n');results.append(record)
    (HERE/'fixed_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('fixed builds complete: '+str(sum(r['exit_code']==0 for r in results))+'/'+str(len(results)))
if __name__=='__main__':main()
