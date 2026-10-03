#!/usr/bin/env python3
"""Existing-kernel equivalence; a failed check is recorded, never hidden."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    root=HERE/'raw';pairs=[('nonpaged_reference',root/'A0/llama_B1/N-R12b-noWD/plan.so',root/'B0b/llama_B1/N-R12b-noWD-pdl-off/plan.so',False),
        ('paged_reference',root/'A0/llama_B1/P-R12bN-noWD/plan.so',root/'B0b/llama_B1/P-R12bN-noWD/plan.so',False),
        ('sm89_pdl',root/'B0b/llama_B1/N-R12b-noWD-pdl-off/plan.so',root/'B0b/llama_B1/N-R12b-noWD/plan.so',True)]
    rows=[]
    for name,reference,candidate,ignore in pairs:
        if not reference.exists() or not candidate.exists():rows.append(dict(name=name,status='missing'));continue
        cmd=[sys.executable,str(HERE/'compare_kernels.py'),'--reference',str(reference),'--candidate',str(candidate),'--out',str(a.out/name)]
        if ignore:cmd+=['--ignore-pdl-macro']
        with (a.out/(name+'.log')).open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        rows.append(dict(name=name,exit_code=code))
    (a.out/'identity_results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
    return int(any(r.get('exit_code')!=0 for r in rows))
if __name__=='__main__':raise SystemExit(main())
