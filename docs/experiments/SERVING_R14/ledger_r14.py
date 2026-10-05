#!/usr/bin/env python3
"""Identity-checked trace diagnostics. Instrumented times never become E2E."""
import argparse,csv,json,statistics
from pathlib import Path
from tilemega.build.identity import verify,digest

def linear_fit(rows):
    if len(rows)<2:return dict(slope_ns_per_byte=None,intercept_ns=None)
    x=[r['bytes'] for r in rows];y=[r['run_ns'] for r in rows]
    mx=statistics.mean(x);my=statistics.mean(y);xx=sum((a-mx)**2 for a in x)
    if not xx:return dict(slope_ns_per_byte=None,intercept_ns=my)
    slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/xx
    return dict(slope_ns_per_byte=slope,intercept_ns=my-slope*mx)

def execution(record):
    i=record['execution_identity']
    if not i or i['execution_id']!=digest({k:v for k,v in i.items() if k!='execution_id'}):
        raise ValueError('missing or modified execution identity')
    return i

def trace_diagnostic(root):
    groups={}
    for f in root.glob('raw/P0_trace_round*/qwen3_B16/round*.json'):
        result=json.loads(f.read_text())
        for label,record in result['arms'].items():
            if record['exit_code']:raise ValueError(f'{label}: unsuccessful diagnostic arm')
            i=execution(record);trace=label!='base'
            if i['trace']!=trace:raise ValueError('diagnostic instrumentation identity mismatch')
            target=groups.setdefault(label,dict(tpot=[],identity=i))
            if target['identity']['execution_id']!=i['execution_id']:raise ValueError('mixed artifacts across rounds')
            target['tpot'].append((record['e2e_seconds']-record['ttft_seconds'])/1023)
    if set(groups)!={'base','timer_only','stores_only','full_trace'}:raise ValueError('incomplete diagnostic arms')
    if any(len(x['tpot'])!=3 for x in groups.values()):raise ValueError('need three paired diagnostic rounds')
    base=statistics.median(groups['base']['tpot'])
    for label,row in groups.items():
        row.update(median_seconds=statistics.median(row['tpot']),range_seconds=max(row['tpot'])-min(row['tpot']))
        row['relative_to_uninstrumented']=row['median_seconds']/base-1
    return groups

def tasks(path):
    report={}
    with path.open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            # Subphase intervals are disjoint, measured on the same sampled CTA.
            kind=row['kind'];value=dict(bytes=int(row['bytes']),run_ns=int(row['run_end'])-int(row['run_begin']))
            for key in ('query_ns','first_page_wait_ns','later_page_wait_ns','wave_compute_ns','la_ns','epilogue_ns'):
                if key in row:value[key]=int(row[key])
            if value['run_ns']<0:raise ValueError('negative task span')
            value['effective_gbps']=value['bytes']/value['run_ns'] if value['run_ns'] else None
            report.setdefault(kind,[]).append(value)
    return {kind:dict(samples=rows,fit=linear_fit(rows)) for kind,rows in report.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic-root',type=Path);p.add_argument('--tasks',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=trace_diagnostic(a.diagnostic_root) if a.diagnostic_root else tasks(a.tasks)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
