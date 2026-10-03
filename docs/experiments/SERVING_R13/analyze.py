#!/usr/bin/env python3
"""CPU-only R13 tables; missing evidence never becomes a passing result."""
import argparse,csv,json,math,re,statistics
from collections import defaultdict
from pathlib import Path
from ledger import read,write,stages,steps
HERE=Path(__file__).resolve().parent
CELLS=('llama_B1','llama_B16','qwen3_B1','qwen3_B16')
GROUPS=[['B0','B0-noev','B0l','NL2e','NL2g','NL2r'],['PR_L1','PR_L2','PR_L2l'],
        ['PS_L1','PS_L2','PS_L2l'],['PSA_L1','PSA_L2','PSA_L2l']]
def quantile(values,q):
    values=sorted(values);n=q*(len(values)-1);i=int(n)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(n-i) if values else None
def stats(v):return dict(median=statistics.median(v),range=max(v)-min(v),samples=json.dumps(v))
def slope(points):
    x=statistics.mean(a for a,b in points);y=statistics.mean(b for a,b in points)
    denominator=sum((a-x)**2 for a,b in points)
    return 100000*sum((a-x)*(b-y) for a,b in points)/denominator if denominator else 0

def collected(root):
    raw=[];past=[];tokens={}
    for path in (root/'raw').glob('*/**/round*.json'):
        try:data=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if 'arms' not in data or data.get('invalidated'):continue
        matrix=path.relative_to(root/'raw').parts[0].split('_')[0]
        for label,record in data['arms'].items():
            if record.get('exit_code') or 'e2e_seconds' not in record:continue
            out=Path(record['out']);cell=data['cell'];rnd=data['round']
            row=dict(matrix=matrix,cell=cell,arm=label,round=rnd,
                     ttft_s=record['ttft_seconds'],e2e_s=record['e2e_seconds'],
                     tpot_s=(record['e2e_seconds']-record['ttft_seconds'])/1023,
                     tpot_p50_s=record.get('tpot_p50_seconds'),tpot_p90_s=record.get('tpot_p90_seconds'),
                     tok_s=int(cell.split('_B')[1])*1024/record['e2e_seconds'],source=str(path))
            vl=data['arms'].get('vllm',{})
            if vl.get('exit_code')==0 and vl.get('e2e_seconds'):
                row['tm_vllm']=vl['e2e_seconds']/row['e2e_s']
            times=out/'step_times.tsv'
            if label=='vllm' and not times.exists():times=out/f"B{cell.split('_B')[1]}"/'step_times.tsv'
            if times.exists():
                points=[(63+int(x['step']),float(x['gpu_ms'])) for x in read(times) if int(x['step'])>0]
                if len(points)>1:row['slope_us_per_100_token']=slope(points)
                buckets=defaultdict(list)
                for position,value in points:
                    lo=64+128*((position-64)//128)
                    if 64<=lo<1088:buckets[lo].append(value)
                for lo,values in buckets.items():past.append(dict(matrix=matrix,cell=cell,arm=label,round=rnd,past_lo=lo,mean_ms=statistics.mean(values)))
            generated=next((r for r in record.get('runs',record.get('generation_runs',[])) if not r.get('warmup') and r.get('N')==1024),{})
            tok=generated.get('tokens');tp=out/generated.get('tokens_file','tokens_N1024_run1.json')
            if tok is None and tp.exists():tok=json.loads(tp.read_text())
            if tok is not None:tokens[(matrix,cell,label,rnd)]=tok
            raw.append(row)
    return raw,past,tokens

def medians(raw):
    grouped=defaultdict(list);result=[]
    for r in raw:grouped[(r['matrix'],r['cell'],r['arm'])].append(r)
    for (matrix,cell,arm),records in sorted(grouped.items()):
        row=dict(matrix=matrix,cell=cell,arm=arm,rounds=len(records))
        for metric in ('ttft_s','e2e_s','tpot_s','tpot_p50_s','tpot_p90_s','tok_s','tm_vllm','slope_us_per_100_token'):
            values=[r[metric] for r in records if r.get(metric) is not None]
            if values:
                summary=stats(values)
                for k,v in summary.items():row[metric+'_'+k]=v
        result.append(row)
    return result

def trace_v2(folder):
    slotfile=folder/'slots.tsv'
    if not slotfile.exists():return [],[],[]
    slots=[{k:int(v) for k,v in r.items()} for r in read(slotfile)]
    waits={int(r['wait_index']):int(r['event_index']) for r in read(folder/'waits.tsv')}
    events={int(r['event_index']):int(r['publish_ns']) for r in read(folder/'events.tsv')}
    result=[];hol=[]
    for slot in slots:
        if not slot['run_begin']:continue
        rows=[waits[i] for i in range(slot['wait_begin_idx'],slot['wait_begin_idx']+slot['wait_count'])]
        arrival=max([events.get(i,0) for i in rows],default=0)
        result.append(dict(source=str(folder),worker=slot['worker'],stage=slot['stage'],task=slot['logical_task'],
                           wait_ns=slot['ready']-slot['wait_begin'],
                           setup_ns=slot['run_begin']-slot['ready'],run_ns=slot['run_end']-slot['run_begin'],
                           publication_ns=slot['publish_end']-slot['run_end'],
                           all_dependencies_arrived=arrival,publish_ns=slot['publish_end']))
        # FIFO blocking: later slot dependencies complete before the current
        # wait ends. Empty waits are ready at the worker's first wait start.
        for later in slots:
            if later['worker']!=slot['worker'] or later['slot']<=slot['slot']:continue
            later_rows=[waits[i] for i in range(later['wait_begin_idx'],later['wait_begin_idx']+later['wait_count'])]
            ready=max([events.get(i,0) for i in later_rows],default=slot['wait_begin'])
            if ready and ready<slot['ready']:
                overlap=max(0,slot['ready']-max(slot['wait_begin'],ready))
                if overlap:hol.append(dict(source=str(folder),worker=slot['worker'],blocked_slot=slot['slot'],ready_later_slot=later['slot'],hol_ns=overlap))
    reducers=[]
    if (folder/'reducers.tsv').exists():
        for r in read(folder/'reducers.tsv'):
            reducer={k:int(v) for k,v in r.items()};reducer['source']=str(folder)
            owner=next((s for s in slots if s['stage']==reducer['producer_stage'] and s['logical_task']==reducer['producer_task']),None)
            reducer['owner_found']=owner is not None
            # The reducer completes at its own release publication, not at the
            # skipped queue stage or at an unrelated aggregate event.
            reducer['complete_ns']=reducer['publish_ns'];reducers.append(reducer)
    return result,hol,reducers

def resources(root):
    output=[]
    for path in (root/'raw').glob('B0b/**/*.ptxas.log'):
        name=None
        for line in path.read_text(errors='replace').splitlines():
            match=re.search(r"(?:Compiling entry function|Function properties for)\s+'?([^'\s]+)",line)
            if match:name=match.group(1)
            if name and ('Used ' in line or 'bytes stack frame' in line):
                output.append(dict(source=str(path),function=name,resources=line.strip()))
    return output

def runtime_structure(root):
    output=[]
    regex=r'E2E_STAGES runtime=(\d+) queued=(\d+) elided=(\d+) tasks=(\d+) fine_events=(\d+) aggregate_events=(\d+)'
    for path in (root/'raw').rglob('*.log'):
        matches=list(re.finditer(regex,path.read_text(errors='replace')))
        if matches:
            output.append(dict(source=str(path),**dict(zip(('runtime','queued','elided','tasks','fine_events','aggregate_events'),map(int,matches[-1].groups())))))
    return output

def microbench(root):
    summary=root/'raw/MB-1a/dram_ceiling.json';ceiling=None;output=[]
    if summary.exists():
        data=json.loads(summary.read_text());ceiling=data['maximum_gbps']
        output.append(dict(suite='MB-1a',maximum_gbps=ceiling,processes=len(data['processes']),
                           process_range_gbps=data['process_range_gbps'],process_range_relative=data['process_range_relative'],
                           contaminated=sum(p['contaminated'] for p in data['processes']),calibration_median_gbps=data['calibration_median_gbps'],source=str(summary)))
    for suite in ('b','c','d','e','f'):
        path=root/f'raw/MB-1{suite}/loadbench.json'
        if not path.exists():continue
        data=json.loads(path.read_text());groups=defaultdict(list)
        for point in data['points']:groups[point['suite']].append(point)
        for label,points in groups.items():
            rates=[p['gbps'] for p in points if 'gbps' in p]
            row=dict(suite=label,maximum_gbps=max(rates) if rates else None,contaminated=data.get('contaminated'),source=str(path),points=len(points))
            if rates:
                for name,rate in dict(calibration=981.6,bf16=884.5,maximum=ceiling).items():
                    if rate:row['ratio_'+name]=max(rates)/rate
            output.append(row)
    return output,ceiling

def correctness(root,tokens):
    output=[]
    for group in GROUPS:
        for matrix in ('A2','B2','B3','B4','D2'):
            for cell in CELLS:
                values=[(k,t) for k,t in tokens.items() if k[0]==matrix and k[1]==cell and k[2] in group]
                if len(values)<2:continue
                reference=values[0][1]
                for key,value in values[1:]:
                    count=sum(a!=b for ar,br in zip(reference,value) for a,b in zip(ar,br))
                    shape_same=[len(r) for r in reference]==[len(r) for r in value]
                    output.append(dict(matrix=matrix,cell=cell,arm=key[2],round=key[3],reference=values[0][0][2],mismatches=count,shape_equal=shape_same,pass_=count==0 and shape_same))
    for path in (root/'raw').rglob('*.json'):
        if path.name not in ('smoke.json','results.json','summary.json','smokes.json','check.json','hf_check.json'):continue
        try:record=json.loads(path.read_text())
        except (OSError,ValueError):continue
        output.append(dict(source=str(path),record=json.dumps(record,separators=(',',':'))))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE);a=p.parse_args();root=a.root;out=root/'results';out.mkdir(parents=True,exist_ok=True)
    raw,past,tokens=collected(root);median=medians(raw);mb,ceiling=microbench(root)
    write(out/'measurements.tsv',raw);write(out/'past.tsv',past)
    write(out/'T1.tsv',[r for r in median if r['matrix']=='A1'])
    write(out/'T2.tsv',[r for r in median if r['matrix'] in ('A2','B4')]);write(out/'T2_resources.tsv',resources(root));write(out/'runtime_structure.tsv',runtime_structure(root))
    write(out/'T3.tsv',mb);stage_rows=[];step_rows=[];task_rows=[];hol=[];reducers=[]
    for folder in sorted({p.parent for p in (root/'raw').rglob('stage_trace.tsv')}):
        cell=next((c for c in CELLS if c in str(folder)),None)
        if cell and ceiling:stage_rows+=stages(folder,int(cell.split('_B')[1]),{'884_5':884.5,'981_6':981.6,'measured':ceiling})
    for folder in sorted({p.parent for p in (root/'raw').rglob('step_trace.tsv')}):
        for row in steps(folder):row['source']=str(folder);step_rows.append(row)
    for folder in sorted({p.parent for p in (root/'raw').rglob('slots.tsv')}):
        task,blocked,elided=trace_v2(folder);task_rows+=task;hol+=blocked;reducers+=elided
    write(out/'T4.tsv',stage_rows);write(out/'T5.tsv',[r for r in median if r['matrix']=='B2']);write(out/'T5_tasks.tsv',task_rows);write(out/'T5_hol.tsv',hol);write(out/'T7_reducers.tsv',reducers)
    write(out/'T6.tsv',[r for r in median if r['matrix']=='B3']);write(out/'T6_steps.tsv',step_rows)
    for name in ('T7','T8','T9','T10','T11','T12'):
        rows=[r for r in median if r['matrix']=='D2'] if name=='T10' else correctness(root,tokens) if name=='T11' else []
        write(out/(name+'.tsv'),rows)
    canaries=[]
    for r in raw:
        base='vllm' if r['matrix'] in ('A1','D2') else 'B0'
        sample=next((x for x in raw if x['matrix']==r['matrix'] and x['cell']==r['cell'] and x['round']==r['round'] and x['arm']==base),None)
        series=[x['tpot_s'] for x in raw if x['matrix']==r['matrix'] and x['cell']==r['cell'] and x['arm']==base]
        if sample and series and abs(sample['tpot_s']/statistics.median(series)-1)>.02:
            canaries.append(dict(matrix=r['matrix'],cell=r['cell'],round=r['round'],canary=base,action='eligible for one registered rerun'))
    write(out/'canaries.tsv',list({json.dumps(r,sort_keys=True):r for r in canaries}.values()))
    print(json.dumps(dict(measurements=len(raw),stage_rows=len(stage_rows),task_rows=len(task_rows),ceiling=ceiling)))
if __name__=='__main__':main()
