#!/usr/bin/env python3
"""Attach measured 600-second search winners and paired comparisons to AB5."""
import argparse
import csv
import itertools
import json
from pathlib import Path

from sm90d1_common import read,sha
from sm90r6_merge_analysis import MODES,MODELS,identity,write
from sm90r6_pair_winners import summarize


def paired(root,name,coordinates):
    directory=root/'ab5/pair-v1'/name;path=directory/'result.json';data=read(path)
    distinct=len({identity(value) for value in coordinates.values()})
    if distinct==1:
        if data['pairs'] or data['reason']!='all three P-SEL winners have identical coordinates':
            raise ValueError('identical AB5 winners should retain the no-pair decision')
        return data,path
    measurement=directory/'measurement';inputs=read(measurement/'inputs.json')
    if any(sha(path)!=digest for path,digest in inputs.items()):
        raise ValueError('AB5 paired comparison input changed')
    aliases=data['aliases']
    if set(aliases)!=set(MODES) or any(coordinates[mode]!=coordinates[alias] for mode,alias in aliases.items()):
        raise ValueError('AB5 pair aliases do not represent selected coordinates')
    representatives=list(dict.fromkeys(aliases.values()))
    if len(representatives)!=distinct:raise ValueError('duplicate paired coordinate representatives')
    raw=read(data['raw']);gpus={row['gpu'] for row in raw['rows']}
    if gpus!={data['gpu']} or data['gpu'] not in (0,1,2,3,4,5,7):
        raise ValueError('AB5 P-PAIR must use one eligible card')
    if summarize(raw['rows'],list(itertools.combinations(representatives,2)),data['model'])!=data['pairs']:
        raise ValueError('AB5 pair summaries do not reproduce')
    return data,path


def merge(root,curve):
    with curve.open() as stream:rows=list(csv.DictReader(stream))
    rows=[row for row in rows if row['kind']!='measured_winner']
    columns=('measurement_protocol','measurement_gpu','range_pct','pair_comparisons','pair_status',
             'measurement_raw','measurement_sha256','pair_raw','pair_sha256')
    for row in rows:
        for column in columns:row[column]=''
    for model in MODELS:
        for batch in (1,16):
            name=model+'-B'+str(batch);selected={};coordinates={}
            searches=[json.loads(line) for line in (root/'ab5/budget-v1'/name/'searches.jsonl').read_text().splitlines() if line.strip()]
            for mode in MODES:
                if mode=='reuse':
                    build=root/'a0/finalists-v1'/name
                    psel=root/'a0'/('psel-v2' if name=='olmoe_1b_7b-B16' else 'psel-v1')/name
                else:
                    build=root/'ab5/finalists-v1'/name/mode/'build';psel=root/'ab5/psel-v1'/name/mode
                winner=read(psel/'result.json');rank=winner['winner_rank'];binary=build/f'candidate-{rank}'/'decode.so'
                artifact=read(str(binary)+'.identity.json');coordinate=artifact['plan']['native_dense']
                if artifact['binary_sha256']!=sha(binary) or winner['winner_binary_sha256']!=sha(binary):
                    raise ValueError('AB5 selected binary changed')
                search=[s for s in searches if s['analysis']==mode and s['input']['budget_ms']==600000]
                if len(search)!=1 or search[0]['finalists'][rank]['coordinates']!=coordinate:
                    raise ValueError('AB5 winner is not a retained 600-second finalist')
                raw=read(psel/'unit/result.json');placement=read(psel/'unit/unit.json')
                label='c'+str(rank)
                if raw['arms']!=winner['arms'] or raw['winner']!=label or {r['gpu'] for r in raw['rows']}!={placement['gpu']}:
                    raise ValueError('AB5 P-SEL summaries or card identity changed')
                matching=[r for r in rows if r['model']==model and int(r['batch'])==batch and
                          r['analysis']==mode and int(r['budget_ms'])==600000 and r['kind']=='finalist' and
                          r['candidate_sha256']==identity(coordinate)]
                if len(matching)!=1:raise ValueError('budget curve lacks the measured finalist')
                row=dict(matching[0]);arm=winner['arms'][label]
                if arm['processes']!=2:raise ValueError('P-SEL requires two fresh processes per candidate')
                row.update(kind='measured_winner',measured_tpot_ms=arm['tpot_ms'],
                    measurement_protocol='P-SEL; two processes per candidate',measurement_gpu=placement['gpu'],
                    range_pct=arm['range_pct'],measurement_raw=str((psel/'result.json').relative_to(root)),
                    measurement_sha256=sha(psel/'result.json'))
                selected[mode]=row;coordinates[mode]=coordinate
            comparison,pair_path=paired(root,name,coordinates)
            for mode,row in selected.items():
                alias=comparison.get('aliases',{}).get(mode,mode)
                row['pair_comparisons']=json.dumps([p for p in comparison['pairs'] if alias in p['arms']],sort_keys=True,separators=(',',':'))
                row['pair_status']=comparison['reason']
                row['pair_raw']=str(pair_path.relative_to(root));row['pair_sha256']=sha(pair_path)
                rows.append(row)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--curve',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=merge(a.root,a.curve);write(a.out,rows)
    print(json.dumps(dict(rows=len(rows),measured_winners=24)))
