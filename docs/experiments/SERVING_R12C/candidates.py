#!/usr/bin/env python3
"""Rotate the existing candidate protocol over registered plans and pasts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arms',type=Path,required=True);p.add_argument('--cell',required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[r for r in json.loads(a.arms.read_text())[a.cell] if r['label'] in ('R12bN_L2','R12bP','S1P','S1N') and r.get('available',True)]
i=a.round%len(rows);rows=rows[i:]+rows[:i];codes={}
for row in rows:
    out=a.out/row['label'];cmd=[row['python'],'-m','tilemega.serving.measure_candidate','--so',row['decode'],'--model',row['model_path'],'--batch',str(row['batch']),'--past-list','192,575,1000','--mode','L2','--out',str(out)]
    code=subprocess.run(cmd,cwd=row['root'],env=dict(os.environ,PYTHONPATH=str(Path(row['root'])/'python'))).returncode;codes[row['label']]=code
    if code==75:raise SystemExit(75)
a.out.mkdir(parents=True,exist_ok=True);(a.out/'status.json').write_text(json.dumps(codes)+'\n');print(json.dumps(codes))
