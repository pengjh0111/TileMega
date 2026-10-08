#!/usr/bin/env python3
"""One bounded recollection of noisy D1 finalist rounds, without rebuilding."""
import argparse,json,statistics
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run
from tilemega.build.identity import verify,sha
from tilemega.serving.integrated_selection import integrated_ms,PASTS

def canary(samples):
    median=statistics.median(samples)
    return [i for i,value in enumerate(samples) if abs(value/median-1)>.02]

def finalists(choice):
    rows=[r for r in choice['candidates'] if len(r.get('samples_ms',[]))==3 and not r.get('error')]
    if len(rows)!=3:raise ValueError('bounded recollection requires exactly three finalists')
    if not any(r['mode']=='L1' and not r['loop'] for r in rows):raise ValueError('missing L1 canary')
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--batch',type=str,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    choice=json.loads(a.input.read_text())[a.batch]['decode_pg_choice'];rows=finalists(choice)
    original=[dict(mode=r['mode'],loop=r['loop'],library=r['library'],samples_ms=r['samples_ms']) for r in rows]
    identities={r['library']:verify(r['library'])['artifact_id'] for r in rows}
    write(a.out/'inputs.json',dict(input=str(a.input),input_sha256=sha(a.input),batch=a.batch,original_finalists=original,artifacts=identities,recollection_limit=1,reason='D1 B16 confirmation outliers despite accepted final GPU guard'))
    measured=[dict(row,samples_ms=[],measurements=[]) for row in original]
    for round_index in range(3):
        shift=round_index%len(rows)
        for index in list(range(len(rows)))[shift:]+list(range(len(rows)))[:shift]:
            row=rows[index];out=a.out/f"{row['mode']}_loop{row['loop']}_r{round_index}"
            run([PYTHON,'-m','tilemega.serving.measure_candidate','--so',row['library'],'--model',a.model,
                 '--batch',a.batch,'--past-list',','.join(map(str,PASTS)),'--mode',row['mode'],
                 '--loop',row['loop'],'--loop-steps','64','--warmup','8','--guard-wait-s','1800','--out',out],out/'run.log',2100)
            record=json.loads((out/'measurements.json').read_text())
            if verify(row['library'])['artifact_id']!=identities[row['library']]:raise ValueError('artifact changed during recollection')
            values=record['modes'][row['mode']]['by_past']
            execution=[v['execution_identity'] for v in values.values()]
            if any(e!=execution[0] or e['trace'] or e['artifact_id']!=identities[row['library']] for e in execution):raise ValueError('invalid execution identity')
            if any(bool(v['decode_loop_used'])!=bool(row['loop']) for v in values.values()):raise ValueError('loop mode mismatch')
            score=integrated_ms(values);measured[index]['samples_ms'].append(score)
            measured[index]['measurements'].append(dict(round=round_index,score_ms=score,path=str(out/'measurements.json'),by_past=values,execution_identity=execution[0]))
    baseline=next(r for r in measured if r['mode']=='L1' and not r['loop'])
    marked=canary(baseline['samples_ms'])
    for row in measured:
        row['median_ms']=statistics.median(row['samples_ms']);row['range_ms']=max(row['samples_ms'])-min(row['samples_ms'])
    winner=min(measured,key=lambda row:row['median_ms'])
    write(a.out/'result.json',dict(recollection_count=1,canary_rounds=marked,accepted=not marked,
          candidates=measured,observed_winner=winner,selection_updated=False,
          limitation='An accepted guard cannot by itself exclude short unobserved interference; no automatic cache/sidecar/default rewrite.'))
    print('D1 recollection '+('accepted' if not marked else 'still unstable; no further recollection authorized'))

if __name__=='__main__':main()
