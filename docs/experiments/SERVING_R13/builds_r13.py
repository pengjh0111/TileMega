#!/usr/bin/env python3
"""Fixed geometry builds; each rejection is retained without stopping siblings."""
import argparse,concurrent.futures,copy,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
from pin_case import pin,classes
from fixed_builds import run_bounded,trace_command
HERE=Path(__file__).resolve().parent
TARGET=Path('/root/r13_work/target_r12b.json')

def project(data,kind,page_bytes=16384):
    data=copy.deepcopy(data)
    if kind=='paged':
        data.update(pg='pages',residency=1,kappa=1)
        for g in data['gemms']:
            g['stages']=2;g['split_k']=1
            g['tile_n']=min(g['tile_n'],page_bytes//(2*g['tile_k']))
    elif kind=='head':
        g=data['gemms'][-1];g.update(tile_n=128,tile_k=64,stages=4,split_k=1)
    elif kind!='identity':raise ValueError(kind)
    return data

def one(job):
    folder=Path(job['out']);folder.mkdir(parents=True,exist_ok=True);so=folder/'plan.so'
    result=dict(job,exit_code=1,so=str(so),placeholder_measurement=True,
                compiler_commit=os.environ['TILEMEGA_R13_COMPILER_COMMIT'],
                target=str(TARGET),target_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest())
    try:
        source=json.loads(Path(job['manifest']).read_text())
        data=project(source,job.get('projection','identity'))
        pinned=folder/'source_manifest.json';pinned.write_text(json.dumps(data,indent=2)+'\n')
        recipe=pin(pinned,job['classes'],job.get('target_classes',job['classes']),folder,job['overrides'])
        phase=job['phase'];interval='0:0' if phase=='prefill' else '64:1086'
        cmd=[os.environ['TILEMEGA_BIN'],'compile',job['export'],str(so),
             '--serving',phase,'--batch',str(job['batch']),'--past-range',interval,
             '--capacity','1088','--solver','skeleton','--solve',str(TARGET),
             '--runtime-target',str(TARGET),'--emit','serving','--search-passes','1',
             '--top-m','1','--search-jobs','3','--search-budget-ms','60000',
             '--handoff','off','--dump-cg',str(folder/'selected.mlir'),
             '--measure-cmd',shlex.join([sys.executable,str(HERE/'measure_stub.py')])]+recipe['options']
        if phase=='decode':cmd+=['--kphase-mask','31','--v3-poll-ns','0']
        (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (folder/'build.log').open('w') as log:code=run_bounded(cmd,log,3600)
        result['exit_code']=code
        if code==0:
            actual=json.loads(Path(str(so)+'.plan.json').read_text())
            result.update(sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
                source_sha256=subprocess.check_output([os.environ['TILEMEGA_BIN'],'version','--json'],text=True).strip(),
                manifest_result=actual,residency_same=actual['residency']==data['residency'],
                grid_same=actual['grid']==data['grid'])
            if sorted(classes(str(so)+'.classes.tsv').values())!=recipe['expected_partition']:
                raise ValueError('rebuilt class partition differs from pinned donor')
        else:result['reason']='compile rejected fixed geometry; see build.log'
    except Exception as error:result.update(exit_code=1,reason=repr(error))
    (folder/'record.json').write_text(json.dumps(result,indent=2)+'\n')
    print(job['cell'],job['label'],result['exit_code'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    jobs=json.loads(a.jobs.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,jobs))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n')
    print('fixed builds: '+str(sum(r['exit_code']==0 for r in rows))+'/'+str(len(rows)))
if __name__=='__main__':main()
