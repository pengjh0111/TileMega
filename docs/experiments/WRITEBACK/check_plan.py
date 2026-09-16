#!/usr/bin/env python3
"""Compare written CG pi/sigma with the generated host's raw queue dump."""
import argparse,csv,difflib,json,re
from pathlib import Path

def check(cg,dump,out):
    text=cg.read_text()
    table=re.search(r'tilemega\.placement_table = \{(.*?)\}, tilemega\.',text,re.S)
    if not table:raise RuntimeError('this checker requires a materialized CG table')
    arrays={}
    for key in ('worker','slot'):
        match=re.search(key+r' = array<i64: ([^>]+)>',table.group(1))
        if not match:raise RuntimeError('missing '+key)
        arrays[key]=[int(x) for x in match.group(1).split(',')]
    rows=list(csv.DictReader((dump/'schedule.tsv').open(),delimiter='\t'))
    nodes=sorted(rows,key=lambda r:(int(r['stage']),int(r['logical_task'])))
    if len(nodes)!=len(arrays['worker']):raise RuntimeError('CG/host node count differs')
    wanted=['stage\tlogical_task\tworker\tslot\n'];actual=wanted.copy()
    for i,r in enumerate(nodes):
        prefix=f"{r['stage']}\t{r['logical_task']}\t"
        wanted.append(prefix+f"{arrays['worker'][i]}\t{arrays['slot'][i]}\n")
        actual.append(prefix+f"{r['worker']}\t{r['slot']}\n")
    out.mkdir(parents=True,exist_ok=True)
    (out/'cg_plan.tsv').write_text(''.join(wanted));(out/'host_plan.tsv').write_text(''.join(actual))
    diff=''.join(difflib.unified_diff(wanted,actual,fromfile=str(cg),tofile=str(dump/'schedule.tsv')))
    (out/'plan.diff').write_text(diff)
    result=dict(status='FAIL' if diff else 'PASS',nodes=len(nodes),cg=str(cg),dump=str(dump),diff_bytes=len(diff))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if diff:raise RuntimeError('writeback round trip differs')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cg',type=Path);p.add_argument('dump',type=Path);p.add_argument('out',type=Path);a=p.parse_args();check(a.cg,a.dump,a.out)
