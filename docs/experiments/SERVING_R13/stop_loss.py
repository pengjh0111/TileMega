#!/usr/bin/env python3
"""Record the registered B0 threshold; optionally continue data collection."""
import argparse,json,statistics
from pathlib import Path
from analyze import collected
HERE=Path(__file__).resolve().parent

def outcome(rows,advisory=False):
    complete=len(rows)==4 and all('kill' in r for r in rows)
    passed=complete and all(r['kill'] is False for r in rows)
    # The user relaxed only performance gating. Missing clean evidence still
    # blocks the queue, and the original threshold result remains unchanged.
    allowed=complete and (passed or advisory)
    return dict(pass_=passed,collection_allowed=allowed,advisory=advisory,
                threshold_rel=.01,rows=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--advisory',action='store_true',help='record regressions without blocking collection')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw,_,_=collected(HERE);rows=[]
    for cell in ('llama_B1','llama_B16','qwen3_B1','qwen3_B16'):
        values={label:[r['tpot_s'] for r in raw if r['matrix']=='A1' and r['cell']==cell and r['arm']==label] for label in ('R12bN_L1','B0-pfR12b')}
        if min(map(len,values.values()))<3:
            rows.append(dict(cell=cell,status='missing three clean rounds'));continue
        change=statistics.median(values['B0-pfR12b'])/statistics.median(values['R12bN_L1'])-1
        rows.append(dict(cell=cell,tpot_rel=change,kill=change>.01,samples=values))
    report=outcome(rows,a.advisory)
    (a.out/'baseline_stop.json').write_text(json.dumps(report,indent=2)+'\n')
    status='PASS' if report['pass_'] else 'WARNING; user-authorized collection continues' if report['collection_allowed'] else 'requires diagnosis; Phase B timing held'
    print('registered B0 baseline stop: '+status)
    return 0 if report['collection_allowed'] else 3
if __name__=='__main__':raise SystemExit(main())
