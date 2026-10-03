#!/usr/bin/env python3
"""Enforce the registered B0 baseline stop before Phase B timing."""
import argparse,json,statistics
from pathlib import Path
from analyze import collected
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw,_,_=collected(HERE);rows=[]
    for cell in ('llama_B1','llama_B16','qwen3_B1','qwen3_B16'):
        values={label:[r['tpot_s'] for r in raw if r['matrix']=='A1' and r['cell']==cell and r['arm']==label] for label in ('R12bN_L1','B0-pfR12b')}
        if min(map(len,values.values()))<3:
            rows.append(dict(cell=cell,status='missing three clean rounds'));continue
        change=statistics.median(values['B0-pfR12b'])/statistics.median(values['R12bN_L1'])-1
        rows.append(dict(cell=cell,tpot_rel=change,kill=change>.01,samples=values))
    passed=len(rows)==4 and all(r.get('kill') is False for r in rows)
    (a.out/'baseline_stop.json').write_text(json.dumps(dict(pass_=passed,rows=rows),indent=2)+'\n')
    print('registered B0 baseline stop: '+('PASS' if passed else 'requires diagnosis; Phase B timing held'))
    return 0 if passed else 3
if __name__=='__main__':raise SystemExit(main())
