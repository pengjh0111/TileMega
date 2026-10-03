#!/usr/bin/env python3
"""Five fresh loading processes, with one bounded stability extension."""
import argparse,json,statistics,subprocess
from pathlib import Path

def aggregate(paths):
    rows=[];methods={}
    for path in paths:
        data=json.loads(path.read_text());points=data['points']
        reference=next((p for p in points if p.get('suite')=='MB-1a-reproduction'),None)
        if reference is None:raise ValueError('missing calibration launch configuration: '+str(path))
        def calibration(p):
            return (p.get('suite')=='MB-1a' and p.get('method')==1 and
                    p.get('allocation')=='A' and p.get('param')==0 and
                    p.get('grid')==reference['grid'] and p.get('threads')==reference['threads'])
        base=[p['gbps'] for p in points if p.get('suite')=='MB-1a' and
              calibration(p) and
              1024*2**20<=p.get('working_set_bytes',0)<=2048*2**20]
        old=[p['gbps'] for p in points if p.get('suite')=='MB-1a' and
              calibration(p) and
              p.get('working_set_bytes') in (512*2**20,1024*2**20,2048*2**20)]
        row=dict(path=str(path),pid=data['pid'],contaminated=data['contaminated'],
                 calibration_median_gbps=statistics.median(base),
                 original_three_point_median_gbps=statistics.median(old))
        rows.append(row)
        if not data['contaminated']:
            for p in points:
                if 'gbps' in p:methods[str(p.get('method'))]=max(methods.get(str(p.get('method')),0),p['gbps'])
    clean=[r['calibration_median_gbps'] for r in rows if not r['contaminated']]
    median=statistics.median(clean) if clean else None
    return dict(processes=rows,calibration_median_gbps=median,
                calibration_definition='method 1; allocation A; 1, 1.5, 2 GiB; median within each process then across processes',
                original_definition='512 MiB, 1 GiB, 2 GiB also reported per process',
                process_range_gbps=max(clean)-min(clean) if clean else None,
                process_range_relative=(max(clean)-min(clean))/median if clean else None,
                method_max_gbps=methods,maximum_gbps=max(methods.values()) if methods else None)

def write_target(source,out,value,results):
    data=json.loads(source.read_text())
    data['calibration']['pipelines']['dram_gbps']=value
    data['calibration_by_dtype']['bf16']['pipelines']['dram_gbps']=value
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2)+'\n')
    Path(str(out)+'.calibration.json').write_text(json.dumps(
        dict(method=results['calibration_definition'],source=str(source),
             dram_gbps=value,processes=results['processes']),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path)
    p.add_argument('--reaggregate',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--target',type=Path)
    p.add_argument('--target-out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.reaggregate:
        paths=sorted(a.reaggregate.glob('process*/loadbench.json'))
        if not paths:raise ValueError('no completed loading processes')
        result=aggregate(paths)
        result['source_directory']=str(a.reaggregate)
        result['correction']='calibration median uses its actual grid and thread count, excluding the occupancy sweep'
        (a.out/'dram_ceiling_corrected.json').write_text(json.dumps(result,indent=2)+'\n')
        print('calibration ceiling reaggregated without GPU execution');return
    if a.binary is None:p.error('--binary is required without --reaggregate')
    files=[];best=None
    for number in range(10):
        folder=a.out/f'process{number}';folder.mkdir(exist_ok=True);raw=folder/'loadbench.json'
        cmd=[str(a.binary),'--suite','a','--out',str(raw)]
        if best is not None:
            cmd+=['--ceiling-replay']
            for key in ('method','param','grid','threads'):cmd+=['--best-'+key,str(best[key])]
        (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (folder/'stdout.log').open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:raise SystemExit(code)
        data=json.loads(raw.read_text());files.append(raw)
        if data['contaminated']:raise SystemExit(75)
        if best is None:best=max((r for r in data['points'] if r.get('suite')=='MB-1a'),key=lambda r:r['gbps'])
        result=aggregate(files);result['best_config']={k:best[k] for k in ('method','param','grid','threads')}
        (a.out/'dram_ceiling.json').write_text(json.dumps(result,indent=2)+'\n')
        if number==4 and result['process_range_relative']<=.03:break
    if a.target:
        if not a.target_out:p.error('--target-out is required with --target')
        write_target(a.target,a.target_out,result['calibration_median_gbps'],result)
    print(json.dumps({k:v for k,v in result.items() if k!='processes'}))
if __name__=='__main__':main()
