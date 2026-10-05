#!/usr/bin/env python3
"""Replay only trace summaries; keep the expensive task/HOL analysis unchanged."""
import json, statistics, re
from collections import defaultdict
from pathlib import Path
from analyze import trace_cell
from ledger import read, write
from page_chain import pages, chain
from cm_report import collect

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def main():
    rows=read(OUT/'T4.tsv')
    changed=0
    for row in rows:
        cell=trace_cell(row['source'])
        if cell!=row['cell']:
            changed+=1
            ratio=int(cell.split('_B')[1])/int(row['cell'].split('_B')[1])
            if row['kind']=='kFusedAttention':row['bytes']=str(int(int(row['bytes'])*ratio))
        row['cell']=cell
        elapsed=float(row['duration_ns']);byte_count=int(row['bytes'])
        row['gbps']=byte_count/elapsed if elapsed else None
        for label in ('884_5','981_6','measured'):
            if label=='measured':
                ceiling=json.loads((HERE/'raw/MB-1a-complete/dram_ceiling.json').read_text())['maximum_gbps']
            else:ceiling=float(label.replace('_','.'))
            row['floor_ns_'+label]=byte_count/ceiling
            row['excess_ns_'+label]=elapsed-byte_count/ceiling
    write(OUT/'T4.tsv',rows)
    write(OUT/'T4_top_excess.tsv',sorted(rows,key=lambda r:r['excess_ns_measured'],reverse=True)[:10])
    groups=defaultdict(list)
    for row in rows:
        layer=re.search(r'l(\d+)\.',row.get('name',''))
        kind=next((k for k in ('qkv','gate_up','down','lm_head','o')
                   if re.search(r'(?:^|[.])'+k+r'(?:[.]|$)',row.get('name',''))),row['kind'])
        groups[(row['cell'],row['source'],row['past'],row['iteration'],kind,layer.group(1) if layer else '')].append(row)
    aggregate=[]
    for key,values in groups.items():
        row=dict(zip(('cell','source','past','iteration','kind','layer'),key))
        for field in ('duration_ns','bytes','tail_ns','excess_ns_884_5','excess_ns_981_6','excess_ns_measured'):
            row[field]=sum(float(v[field]) for v in values)
        row['gbps']=row['bytes']/row['duration_ns'] if row['duration_ns'] else None
        aggregate.append(row)
    write(OUT/'T4_kinds_layers.tsv',aggregate)
    ceiling=json.loads((HERE/'raw/MB-1a-complete/dram_ceiling.json').read_text())['maximum_gbps']
    page_rows=[]
    for folder in sorted({p.parent for p in (HERE/'raw').rglob('page_trace.tsv')}):
        cell=trace_cell(folder)
        if not cell:continue
        model,b=cell.split('_B')
        floor=json.loads((HERE/f'raw/inputs/{model}_decode_B{b}_floor.json').read_text())
        point=next(p for p in floor['points'] if p['past']==575)
        for row in pages(folder/'page_trace.tsv',point['dram_ns']*884.5010943,
                         {'884_5':884.5,'981_6':981.6,'measured':ceiling}):
            row['cell']=cell;page_rows.append(row)
        if (folder/'slots.tsv').exists():
            links,_=chain(folder)
            write(OUT/(folder.name+'_'+cell+'_chain.tsv'),links)
    write(OUT/'T7.tsv',page_rows)
    chains=read(OUT/'T7_chains.tsv')
    for row in chains:row['cell']=trace_cell(row['source'])
    write(OUT/'T7_chains.tsv',chains)
    collect(HERE,OUT)
    (HERE/'raw/final_review/trace_replay.json').write_text(json.dumps(dict(
        corrected_stage_rows=changed,stage_rows=len(rows),page_rows=len(page_rows),
        changes='exact batch-path matching; attention bytes and floors recomputed; CM joins repeated',
        phase_c_preregistration='original trigger inputs/decision retained; no optimization reselected'),indent=2)+'\n')
    print('Replayed trace cell labels and model joins without rerunning GPU or HOL analysis')

if __name__=='__main__':main()
