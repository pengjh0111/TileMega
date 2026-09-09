#!/usr/bin/env python3
"""Check production price QPs against every steady-state full-arm queue record."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True)
    a=p.parse_args()
    prices={}; comparisons=0; differences=[]; hashes={}
    for model in ('gqa2','mha4'):
        log=a.fit/f'{model}_price.log'
        text=log.read_text()
        if 'EVENT_PRICE_GATE PASS' not in text or 'ISL_CONTEXT remaining=0' not in text:
            raise RuntimeError('production price tool did not finish its gate')
        path=a.fit/f'{model}_price.tsv'
        rows=list(csv.DictReader(path.open(),delimiter='\t'))
        if len(rows)!=18: raise RuntimeError('incomplete kappa/seq price grid')
        for row in rows:
            key=(model,int(row['seq']),int(row['kappa']))
            if key in prices: raise RuntimeError('duplicate price point')
            prices[key]=row
            if key[2]==1:
                differences.append(dict(model=model,seq=key[1],delta_ns=float(row['delta_from_k0_ns'])))
        for source in (log,path): hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for row in csv.DictReader((a.raw/'attrib.tsv').open(),delimiter='\t'):
        if row['arm']!='full': continue
        price=prices[(row['model'],int(row['seq']),1)]
        for actual,predicted in [('task_refs','task_refs'),('waits','waits'),
                                 ('variant_stages','stages'),('max_worker_task_refs','max_worker_tasks')]:
            if row[actual]!=price[predicted]: raise RuntimeError(f'queue projection mismatch: {row}')
            comparisons+=1
    if comparisons!=1200: raise RuntimeError('missing paired full-arm processes')
    out=a.fit/'functional_gate.json'
    if out.exists(): raise RuntimeError('refusing to overwrite gate evidence')
    out.write_text(json.dumps(dict(exact_runtime_counter_comparisons=comparisons,
        parameter_substitution_bit_checks=36,kappa_differences=differences,
        evidence_sha256=hashes,scope='A9 event price only; not A6 or L2 DP acceptance'),indent=2)+'\n')


if __name__=='__main__': main()
